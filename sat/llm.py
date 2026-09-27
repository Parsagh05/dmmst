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

Deviations from the paper's sketch, both deliberate: the censoring token's probability is
removed rather than treated as an outcome (independent censoring), and the optional
DeepHit-style loss and "illegal sequence" penalties are not implemented.
"""

__authors__ = ["Dominik Dahlem"]
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
    events = np.asarray(events, dtype=int)
    durations = np.asarray(durations, dtype=float)
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
    def __init__(self, prompts, answers, tok, special, max_prompt):
        self.items = []
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
        return {"prompt": ids, "answer": ans}


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
        return {"input_ids": ids, "attention_mask": att, "answer_pos": pos, "labels": labels}

    return fn


def answer_logits(model, input_ids, attention_mask, answer_pos):
    """Next-token logits at the answer positions only. Applying the LM head at every
    prompt position (50k-way softmax x ~800 positions) is what made a naive run ~50x
    slower; the loss is unchanged because only answer positions carry labels."""
    h = model.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    h = torch.gather(h, 1, answer_pos[:, :, None].expand(-1, -1, h.shape[-1]))
    return model.get_output_embeddings()(h)  # (b, T, V)


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
        p = torch.softmax(lg[:, :, ans_ids], dim=-1)  # restricted to the answer vocabulary
        p_event = p @ contains  # (b, T, K)
        haz = (p_event / (1.0 - p[:, :, c_idx : c_idx + 1]).clamp_min(1e-6)).clamp(0, 1)
        surv = torch.cumprod(1.0 - haz, dim=1).permute(0, 2, 1)  # (b, K, T)
        out.append(torch.cat([torch.ones_like(surv[:, :, :1]), surv], dim=2).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ main
def _answer_trainer_cls():
    from transformers import Trainer

    class AnswerTrainer(Trainer):
        """Token-level cross-entropy over the answer tokens (paper Eq. 8)."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            logits = answer_logits(model, inputs["input_ids"], inputs["attention_mask"], inputs["answer_pos"])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), inputs["labels"].reshape(-1))
            return (loss, {"logits": logits}) if return_outputs else loss

    return AnswerTrainer


@rand.seed
def _llm(cfg: DictConfig):
    from transformers import (AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback,
                              GPT2Config, GPT2LMHeadModel, TrainingArguments)

    _AnswerTrainer = _answer_trainer_cls()

    from sat.data import splitter
    from sat.evaluate.multievent_metrics import WithinSubjectOrdering
    from sat.evaluate.survtrace_metrics import SurvivalMAEMetrics, SurvTRACEMetrics

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
        model = AutoModelForCausalLM.from_pretrained(cfg.llm_model)
    elif cfg.llm_init == "scratch":
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(tok), n_positions=cfg.llm_max_prompt_tokens + T + 8,
            n_embd=cfg.llm_scratch_hidden, n_layer=cfg.llm_scratch_layers, n_head=cfg.llm_scratch_heads,
            bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id))
    else:
        raise ValueError(f"llm_init must be pretrained|scratch, got {cfg.llm_init}")
    model.resize_token_embeddings(len(tok))
    model.to(device)

    max_prompt = min(cfg.llm_max_prompt_tokens, model.config.n_positions - T - 1)
    e_tr = _Encoded(texts(tr), a_tr, tok, special, max_prompt)
    e_va = _Encoded(texts(va), a_va, tok, special, max_prompt)
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
        prediction_loss_only=True, label_names=["labels"],
    )
    trainer = _AnswerTrainer(model=model, args=args, train_dataset=e_tr, eval_dataset=e_va,
                      data_collator=_collate(tok.pad_token_id),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.llm_patience)])
    trainer.train()

    surv = survival_curves(trainer.model, e_te, special, vocab, K, T, device)
    predictions = np.stack([np.zeros_like(surv), 1.0 - surv, surv], axis=1)
    references = np.zeros((len(te), 4 * K))
    references[:, K : 2 * K] = np.asarray(te[cfg.data.event_col], dtype=float).reshape(-1, K)
    references[:, 3 * K : 4 * K] = np.asarray(te[cfg.data.duration_col], dtype=float).reshape(-1, K)

    cuts_file, train_file = f"{save}/duration_cuts.csv", f"{save}/transformed_train_labels.csv"
    metrics = {"eval_answer_ce": float(trainer.evaluate()["eval_loss"])}
    modules = [SurvTRACEMetrics(cfg.data, cuts_file, train_file, cfg.get("per_event_horizons", False)),
               SurvivalMAEMetrics(cfg.data, cuts_file, train_file)]
    if K > 1:
        modules.append(WithinSubjectOrdering(cfg.data, cuts_file, train_file))
    for m in modules:
        metrics.update(m.compute(predictions, references))

    payload = {"test": {k: {"mean": float(v), "variance": 0.0, "sd": 0.0} for k, v in metrics.items()}}
    payload["validation"] = {"eval_answer_ce": payload["test"]["eval_answer_ce"]}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=4))
    (out_dir / "answer_vocab.json").write_text(json.dumps(vocab))
    for k in range(K):
        pd.DataFrame(surv[:, k], columns=[f"t{j}" for j in range(T + 1)]).assign(
            id=te[cfg.data.id_col]).to_csv(out_dir / f"survival{k}.csv", index=False)
    logger.info(f"LLM metrics -> {out_dir / 'metrics.json'}")
    for k, v in metrics.items():
        logger.info(f"  {k} = {v:.4f}")
    return metrics


@hydra.main(version_base=None, config_path="../conf", config_name="finetune.yaml")
def llm(cfg: DictConfig) -> None:
    config.Config()
    _llm(cfg)


if __name__ == "__main__":
    llm()
