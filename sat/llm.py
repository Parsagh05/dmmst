"""End-to-end LLM survival model (paper §3).

A causal language model reads the patient record as text and answers with one token per
time unit (the duration cuts of the label transform):

    [N]   no event in this unit          [C]   no longer observed (censored)
    [Ek]  event k in this unit           [Ei+Ej]  several events in the same unit
                                          (combination tokens found in the training data)

e.g. ``[N] [N] [E1] [N] [C]`` (paper: "NNE1NC"). Training is token-level cross-entropy on
the answer only (Eq. 8). At inference the answer prefix is teacher-forced to all ``[N]``
(no event so far); at unit t the next-token distribution gives, after removing the
censoring mass,

    h_k(t) = sum_{tokens containing Ek} p_t(token) / (1 - p_t([C]))

the event-k hazard along the event-free path, so S_k(t) = prod_{m<=t} (1 - h_k(m)) and
CIF_k = 1 - S_k. These curves are scored with exactly the metrics, splits and horizons of
every other model.

    python -m sat.llm experiments=multievent/survival <dataset overrides> modelname=llm \\
        llm_init=pretrained llm_model=distilgpt2        # or llm_init=scratch

Training losses (``llm_loss``), both from the paper:

* ``ce``      - token-level cross-entropy on the answer (Eq. 8);
* ``deephit`` - the DeepHit-style likelihood the paper proposes as an alternative,
  written with the same per-unit event hazards h_k(t) used at inference (so training
  and inference describe one distribution): an event k observed in unit s contributes
  -log[S_k(s-1) h_k(s)], a subject censored in unit s contributes -log S_k(s), with
  S_k(s) = prod_{m<=s} (1 - h_k(m)) (cause-specific, one term per event).

``llm_illegal_coeff`` > 0 adds the paper's "illegal sequence" penalty: in this answer
format a [C] can only be followed by [C], so the penalty is the probability the model
puts on any other token at a position whose prefix already contains [C].

The censoring token's probability is removed when reading hazards rather than treated
as an outcome (independent censoring). Any Hugging Face causal LM works as the backbone
(GPT-2, Qwen2.5-0.5B, ...); ``llm_lora_r`` > 0 trains LoRA adapters (plus the embedding
and output layers, which hold the new answer tokens) instead of all weights.
"""

__authors__ = ["Parsa"]
__status__ = "Development"

import json
import math
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from sat.utils import config, logging, rand

logger = logging.get_default_logger()


# ------------------------------------------------------------------ text format
def record_text(x: str, numerics, modality) -> str:
    """'x_age 0.53, c_sex_F, VISIT -3.20, E11, LAB_eGFR -1.10, ...'"""
    parts = []
    for tok, v, m in zip(x.split(), numerics, modality):
        parts.append(f"{tok} {float(v):.2f}" if int(m) == 1 else tok)
    return "Patient: " + ", ".join(parts) + ". Outcome:"


def answer_tokens(events, durations, cuts):
    """One label per unit (cuts[t-1], cuts[t]]: sets of events, 'N', or 'C'."""
    events = np.atleast_1d(np.asarray(events, dtype=int))  # single-event data: scalars
    durations = np.atleast_1d(np.asarray(durations, dtype=float))
    end = durations.max()  # end of observation (censoring time or last event)
    out = []
    for t in range(1, len(cuts)):
        lo, hi = cuts[t - 1], cuts[t]
        if lo >= end:
            out.append("C")
            continue
        hit = [k for k in range(len(events)) if events[k] == 1 and lo < durations[k] <= hi]
        out.append("N" if not hit else "+".join(f"E{k + 1}" for k in hit))
    return out


def _vocab(train_answers, K):
    labels = {"N", "C"} | {f"E{k + 1}" for k in range(K)}
    for ans in train_answers:
        labels.update(ans)
    return sorted(labels, key=lambda s: (s.count("+"), s))


def _map_unknown(label, vocab):
    """A combination not seen in training -> its first event (keeps the event)."""
    return label if label in vocab else label.split("+")[0]


# ------------------------------------------------------------------ data
class _Encoded(torch.utils.data.Dataset):
    def __init__(self, prompts, answers, tok, special, max_prompt, units=None):
        self.items = []
        self.units = units  # [(unit (K,), observed (K,))] for the deephit loss
        for p, a in zip(prompts, answers):
            ids = tok(p, add_special_tokens=False)["input_ids"]
            if len(ids) > max_prompt:  # keep the head (static) and the most recent history
                ids = ids[:32] + ids[-(max_prompt - 32):]
            ans = [special[s] for s in a]
            self.items.append((ids, ans))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, ans = self.items[i]
        out = {"prompt": ids, "answer": ans}
        if self.units is not None:
            out["event_unit"], out["event_observed"] = self.units[i]
        return out


def _collate(pad_id):
    """Right-padded prompt+answer; `answer_pos` = positions whose next-token prediction
    is an answer token (the last prompt token and the answer tokens but the last)."""

    def fn(batch):
        L = max(len(b["prompt"]) + len(b["answer"]) for b in batch)
        T = len(batch[0]["answer"])
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        pos = torch.zeros((len(batch), T), dtype=torch.long)
        for i, b in enumerate(batch):
            seq = b["prompt"] + b["answer"]
            ids[i, : len(seq)] = torch.tensor(seq)
            att[i, : len(seq)] = 1
            pos[i] = len(b["prompt"]) - 1 + torch.arange(T)
        labels = torch.tensor([b["answer"] for b in batch], dtype=torch.long)
        out = {"input_ids": ids, "attention_mask": att, "answer_pos": pos, "labels": labels}
        if "event_unit" in batch[0]:
            out["event_unit"] = torch.tensor(np.stack([b["event_unit"] for b in batch]), dtype=torch.long)
            out["event_observed"] = torch.tensor(
                np.stack([b["event_observed"] for b in batch]), dtype=torch.float32)
        return out

    return fn


def _lm(model):
    """The causal LM itself, also when wrapped by PEFT (LoRA)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def answer_logits(model, input_ids, attention_mask, answer_pos):
    """Next-token logits at the answer positions only. Applying the LM head at every
    prompt position (50k-way softmax x ~800 positions) is what made a naive run ~50x
    slower; the loss is unchanged because only answer positions carry labels."""
    lm = _lm(model)
    h = lm.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    h = torch.gather(h, 1, answer_pos[:, :, None].expand(-1, -1, h.shape[-1]))
    return lm.get_output_embeddings()(h)  # (b, T, V)


def event_hazards(logits, ans_ids, contains, c_idx):
    """Per-unit event hazards from next-token logits restricted to the answer
    vocabulary: h_k(t) = p_t(tokens with Ek) / (1 - p_t([C])). logits: (b, T, V).
    Returns (hazards (b, T, K), answer-token probabilities (b, T, |answer vocab|))."""
    p = torch.softmax(logits[:, :, ans_ids].float(), dim=-1)
    haz = (p @ contains) / (1.0 - p[:, :, c_idx : c_idx + 1]).clamp_min(1e-6)
    return haz.clamp(1e-7, 1 - 1e-7), p


# ------------------------------------------------------------------ inference
@torch.no_grad()
def survival_curves(model, enc: _Encoded, special, vocab, K, T, device, batch_size=64):
    """(n, K, T+1) survival on the cut grid (column 0 = time 0)."""
    model.eval()
    ans_ids = torch.tensor([special[s] for s in vocab], device=device)
    n_id = special["N"]
    contains = torch.zeros((len(vocab), K), device=device)
    for j, s in enumerate(vocab):
        for part in s.split("+"):
            if part.startswith("E"):
                contains[j, int(part[1:]) - 1] = 1.0
    c_idx = vocab.index("C")
    out = []
    model = _lm(model)
    for s in range(0, len(enc), batch_size):
        chunk = enc.items[s : s + batch_size]
        L = max(len(ids) for ids, _ in chunk) + T
        ids = torch.full((len(chunk), L), n_id, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        pos = []
        for i, (p, _) in enumerate(chunk):
            ids[i, : len(p)] = torch.tensor(p)
            att[i, : len(p) + T] = 1
            pos.append(len(p) - 1)
        idx = torch.tensor(pos, device=device)[:, None] + torch.arange(T, device=device)[None, :]
        lg = answer_logits(model, ids.to(device), att.to(device), idx)  # (b, T, V)
        haz, _ = event_hazards(lg, ans_ids, contains, c_idx)  # (b, T, K)
        surv = torch.cumprod(1.0 - haz, dim=1).permute(0, 2, 1)  # (b, K, T)
        out.append(torch.cat([torch.ones_like(surv[:, :, :1]), surv], dim=2).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ main
def deephit_loss(haz, unit, observed):
    """Cause-specific DeepHit-style likelihood on per-unit hazards.

    haz: (b, T, K) hazards; unit: (b, K) 0-based unit of each event / censoring time;
    observed: (b, K) 1 if event k was observed in that unit."""
    log_s = torch.cumsum(torch.log1p(-haz), dim=1)  # log S_k(t), t = 1..T
    log_s_prev = torch.cat([torch.zeros_like(log_s[:, :1]), log_s[:, :-1]], dim=1)
    idx = unit.clamp(0, haz.shape[1] - 1)[:, None, :]  # (b, 1, K)

    def at(x):
        return torch.gather(x, 1, idx)[:, 0, :]

    ll = observed * (at(log_s_prev) + torch.log(at(haz))) + (1 - observed) * at(log_s)
    return -ll.sum(dim=1).mean()


def illegal_mass(probs, labels, c_label):
    """Mean probability on non-[C] tokens at positions whose (true) prefix has a [C].
    probs: (b, T, |answer vocab|); labels: (b, T) positions in the answer vocabulary."""
    seen_c = torch.cumsum((labels == c_label).long(), dim=1)
    after_c = torch.cat([torch.zeros_like(seen_c[:, :1]), seen_c[:, :-1]], dim=1) > 0
    if not after_c.any():
        return probs.sum() * 0.0
    return (1.0 - probs[..., c_label])[after_c].mean()


def event_units(events, durations, cuts):
    """(K,) 0-based unit of each event / censoring time and whether it was observed;
    unit t covers (cuts[t], cuts[t + 1]], as in answer_tokens."""
    events = np.atleast_1d(np.asarray(events, dtype=int))
    durations = np.atleast_1d(np.asarray(durations, dtype=float))
    unit = np.clip(np.searchsorted(cuts, durations, side="left") - 1, 0, len(cuts) - 2)
    return unit.astype(int), events.astype(float)


def _answer_trainer_cls():
    from transformers import Trainer

    class AnswerTrainer(Trainer):
        """Answer-token loss: cross-entropy (Eq. 8) or the DeepHit-style likelihood,
        optionally plus the illegal-sequence penalty."""

        loss_kind = "ce"
        illegal_coeff = 0.0
        ans_ids = contains = None
        c_idx = 0

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            logits = answer_logits(model, inputs["input_ids"], inputs["attention_mask"], inputs["answer_pos"])
            dev = logits.device
            ans_ids, contains = self.ans_ids.to(dev), self.contains.to(dev)
            if self.loss_kind == "ce":
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), inputs["labels"].reshape(-1))
                probs = torch.softmax(logits[:, :, ans_ids].float(), dim=-1)
            else:
                haz, probs = event_hazards(logits, ans_ids, contains, self.c_idx)
                loss = deephit_loss(haz, inputs["event_unit"], inputs["event_observed"])
            if self.illegal_coeff > 0:
                lab = (inputs["labels"][..., None] == ans_ids[None, None, :]).float().argmax(-1)
                loss = loss + self.illegal_coeff * illegal_mass(probs, lab, self.c_idx)
            return (loss, {"logits": logits}) if return_outputs else loss

    return AnswerTrainer


@rand.seed
def _llm(cfg: DictConfig):
    from transformers import (AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback,
                              GPT2Config, GPT2LMHeadModel, TrainingArguments)

    _AnswerTrainer = _answer_trainer_cls()

    from sat.data import splitter
    from sat.evaluate.multievent_metrics import _interp_cif
    from sat.evaluate.shared import evaluate_curves, references, write_metrics

    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = int(cfg.data.num_events)
    save = cfg.data.label_transform.save_dir
    cuts = pd.read_csv(f"{save}/duration_cuts.csv", header=None, names=["c"]).c.values.astype(float)
    T = len(cuts) - 1

    ds = splitter.StreamingKFoldSplitter(
        id_field=cfg.data.id_col, k=None, val_ratio=cfg.data.validation_ratio,
        test_ratio=cfg.data.test_ratio, test_split_strategy="hash",
        split_names=cfg.data.splits, split_seed=cfg.get("split_seed"),
    ).load_split(cfg=cfg.data.load)
    tr, va, te = (ds[s] for s in cfg.data.splits)

    def texts(split):
        return [record_text(x, v, m) for x, v, m in zip(split["x"], split["numerics"], split["modality"])]

    def answers(split):
        return [answer_tokens(e, d, cuts) for e, d in zip(split[cfg.data.event_col], split[cfg.data.duration_col])]

    a_tr, a_va = answers(tr), answers(va)
    vocab = _vocab(a_tr, K)
    a_va = [[_map_unknown(s, vocab) for s in a] for a in a_va]
    logger.info(f"answer vocabulary ({len(vocab)}): {vocab}; T={T} units")

    tok = AutoTokenizer.from_pretrained(cfg.llm_model)
    if cfg.llm_init == "scratch":  # small BPE fitted on the training records, not GPT-2's 50k
        tok = tok.train_new_from_iterator(texts(tr), vocab_size=cfg.llm_scratch_vocab)
    tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": [f"[{s}]" for s in vocab]})
    special = {s: tok.convert_tokens_to_ids(f"[{s}]") for s in vocab}

    if cfg.llm_init == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(cfg.llm_model, torch_dtype=torch.float32)
    elif cfg.llm_init == "scratch":
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(tok), n_positions=cfg.llm_max_prompt_tokens + T + 8,
            n_embd=cfg.llm_scratch_hidden, n_layer=cfg.llm_scratch_layers, n_head=cfg.llm_scratch_heads,
            bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id))
    else:
        raise ValueError(f"llm_init must be pretrained|scratch, got {cfg.llm_init}")
    model.resize_token_embeddings(len(tok))
    n_pos = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", 2048)
    if int(cfg.get("llm_lora_r", 0)) > 0:
        from peft import LoraConfig, get_peft_model
        from peft.tuners.lora import torchao as _peft_torchao

        # Kaggle ships torchao 0.10, which peft rejects with ImportError while checking
        # for torchao-quantised layers; we use none, so report torchao as absent
        _peft_torchao.is_torchao_available = lambda: False

        # the new answer tokens live in the input embeddings and the output layer, so
        # those are trained in full; everything else through rank-r adapters
        emb = model.get_input_embeddings()
        out_layer = model.get_output_embeddings()
        names = {m: n for n, m in model.named_modules()}
        model = get_peft_model(model, LoraConfig(
            r=int(cfg.llm_lora_r), lora_alpha=2 * int(cfg.llm_lora_r), lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM",
            modules_to_save=[names[emb].split(".")[-1], names[out_layer].split(".")[-1]],
        ))
    model.to(device)

    loss_kind = str(cfg.get("llm_loss", "ce"))
    if loss_kind not in ("ce", "deephit"):
        raise ValueError(f"llm_loss must be ce|deephit, got {loss_kind}")

    def units(split):
        if loss_kind != "deephit":
            return None
        return [event_units(e, d, cuts) for e, d in
                zip(split[cfg.data.event_col], split[cfg.data.duration_col])]

    max_prompt = min(cfg.llm_max_prompt_tokens, n_pos - T - 1)
    e_tr = _Encoded(texts(tr), a_tr, tok, special, max_prompt, units(tr))
    e_va = _Encoded(texts(va), a_va, tok, special, max_prompt, units(va))
    e_va_inf = _Encoded(texts(va), [["N"] * T] * len(va), tok, special, max_prompt)
    e_te = _Encoded(texts(te), [["N"] * T] * len(te), tok, special, max_prompt)

    out_dir = Path(f"{cfg.modelhub}/{cfg.dataset}/{cfg.modelname}")
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = max(1, math.ceil(len(e_tr) / cfg.llm_batch_size))
    args = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=cfg.llm_epochs,
        per_device_train_batch_size=cfg.llm_batch_size, per_device_eval_batch_size=cfg.llm_batch_size,
        learning_rate=cfg.llm_learning_rate, weight_decay=0.01, warmup_steps=min(100, steps_per_epoch),
        lr_scheduler_type="linear", eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_strategy="epoch", report_to=[], seed=int(cfg.seed or 0), save_safetensors=False,
        remove_unused_columns=False, fp16=device == "cuda", dataloader_pin_memory=device == "cuda",
        gradient_accumulation_steps=int(cfg.get("llm_grad_accum", 1)),
        max_steps=int(cfg.get("llm_max_steps", -1)),  # > 0 only for smoke tests
        prediction_loss_only=True, label_names=["labels"],
    )
    contains = torch.zeros((len(vocab), K))
    for j, lab in enumerate(vocab):
        for part in lab.split("+"):
            if part.startswith("E"):
                contains[j, int(part[1:]) - 1] = 1.0
    _AnswerTrainer.loss_kind = loss_kind
    _AnswerTrainer.illegal_coeff = float(cfg.get("llm_illegal_coeff", 0.0))
    _AnswerTrainer.ans_ids = torch.tensor([special[x] for x in vocab])
    _AnswerTrainer.contains = contains
    _AnswerTrainer.c_idx = vocab.index("C")
    trainer = _AnswerTrainer(model=model, args=args, train_dataset=e_tr, eval_dataset=e_va,
                      data_collator=_collate(tok.pad_token_id),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.llm_patience)])
    trainer.train()

    def curves(enc):
        return survival_curves(trainer.model, enc, special, vocab, K, T, device)

    def surv_at(S):
        """Per-unit hazards are constant within a unit: log-linear between the cuts."""
        n = len(S)

        def f(times):
            return np.stack([
                np.stack([1.0 - _interp_cif(cuts, 1.0 - S[:, k], np.full(n, t)) for t in times], 1)
                for k in range(K)], axis=1)
        return f

    def labels(split):
        e = np.asarray(split[cfg.data.event_col], dtype=float).reshape(len(split), K)
        d = np.asarray(split[cfg.data.duration_col], dtype=float).reshape(len(split), K)
        return references(e, d)

    answer_loss = float(trainer.evaluate()["eval_loss"])
    s_va, s_te = curves(e_va_inf), curves(e_te)
    val = evaluate_curves(cfg, surv_at(s_va), labels(va)) | {"answer_loss": answer_loss}
    test = evaluate_curves(cfg, surv_at(s_te), labels(te))
    write_metrics(out_dir, test, val)
    (out_dir / "answer_vocab.json").write_text(json.dumps(vocab))
    for k in range(K):
        pd.DataFrame(s_te[:, k], columns=[f"t{j}" for j in range(T + 1)]).assign(
            id=te[cfg.data.id_col]).to_csv(out_dir / f"survival{k}.csv", index=False)
    logger.info(f"LLM ({cfg.llm_model}, {cfg.llm_init}, loss={loss_kind}): "
                f"test C_td {test.get('ctd_weighted_avg'):.4f} -> {out_dir}")
    return test


@hydra.main(version_base=None, config_path="../conf", config_name="finetune.yaml")
def llm(cfg: DictConfig) -> None:
    config.Config()
    _llm(cfg)


if __name__ == "__main__":
    llm()
