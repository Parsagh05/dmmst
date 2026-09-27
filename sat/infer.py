"""Inference of a transformer model for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from logging import DEBUG, ERROR
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from logdecorator import log_on_end, log_on_error, log_on_start
from omegaconf import DictConfig
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from sat.models import heads
from sat.models.heads.embeddings import TokenEmbedding
from sat.models.utils import get_device, load_model
from sat.utils import config, logging, rand, tokenizing
from sat.utils.output import write_interpolation, write_output

logger = logging.get_default_logger()


@rand.seed
def _infer(cfg: DictConfig) -> None:
    device_str, device = get_device()
    logger.info(f"Running models on device: {device_str}")

    mtlConfig = hydra.utils.instantiate(cfg.tasks.config)
    model = heads.MTLForSurvival(mtlConfig)

    model_path = Path(f"{cfg.trainer.training_arguments.output_dir}/pytorch_model.bin")
    logger.info(f"Load model: {model_path}")

    model = load_model(model_path, model)
    model.to(device)
    model.eval()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(f"{cfg.tokenizers.tokenizer_dir}/tokenizer.json")),
        pad_token=cfg.tokenizers.pad_token,
        mask_token=cfg.tokenizers.mask_token,
    )

    logger.debug("Load data")

    if cfg.infer_data:
        dataset = load_dataset(path="json", data_files={"infer": cfg.infer_data})
    else:
        logger.warn(
            "For inference a datafile has to be specified using the 'infer_data' config key"
        )
        return

    if cfg.token_emb == TokenEmbedding.BERT.value:
        # identical to finetune: the model was trained with a [CLS] token
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single="$A [CLS]" if cfg.tokenizers.cls_model_type == "GPT" else "[CLS] $A",
            special_tokens=[("[CLS]", tokenizer.convert_tokens_to_ids("[CLS]"))],
        )

    def tokenize_function(examples):
        text = examples[cfg.tokenizers.tokenize_column]
        if cfg.tokenizers.is_split_into_words:
            text = [t.split() for t in text]
        return tokenizer(
            text=text,
            max_length=cfg.tokenizers.max_seq_length,
            padding=cfg.tokenizers.padding_args.padding,
            truncation=cfg.tokenizers.do_truncation,
            is_split_into_words=cfg.tokenizers.is_split_into_words,
            pad_to_multiple_of=cfg.tokenizers.padding_args.pad_to_multiple_of,
        )

    dataset = dataset["infer"]
    if cfg.select_id:
        dataset = dataset.filter(lambda x: x[cfg.data.id_col] == cfg.select_id)
    ids = dataset[cfg.data.id_col]

    dataset = dataset.map(tokenize_function, batched=True)
    if "numerics" in dataset.column_names:
        # identical to finetune: align the numeric values with the (padded,
        # truncated, possibly [CLS]-prefixed) token sequence
        dataset = dataset.map(
            tokenizing.numerics_padding_and_truncation,
            fn_kwargs={
                "max_seq_length": cfg.tokenizers.max_seq_length,
                "truncation_direction": cfg.tokenizers.truncation_args.direction,
                "padding_direction": cfg.tokenizers.padding_args.direction,
                "token_emb": cfg.token_emb,
            },
        )
    else:
        logger.warning(
            "No 'numerics' column in infer_data: numeric features will not reach the model."
        )

    # The HF "survival-analysis" pipeline is bypassed: it was never registered here,
    # re-tokenised the text with the wrong split flag (collapsing the batch) and never
    # forwarded `numerics`, so predictions ignored the patients' actual values.
    cols = [c for c in ("input_ids", "attention_mask", "numerics") if c in dataset.column_names]
    dataset.set_format(type="torch", columns=cols)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(dataset), 512):
            batch = dataset[start : start + 512]
            inputs = {k: batch[k].to(device) for k in cols}
            if "numerics" in inputs:
                inputs["numerics"] = inputs["numerics"].float()
            out = model(**inputs)
            # (logits, hazard, risk, survival, ...) without loss - the layout
            # write_output expects for a plain tuple
            fields = ("logits", "hazard", "risk", "survival", "time_to_event", "event")
            chunks.append(
                tuple(
                    out[f].detach().cpu() if torch.is_tensor(out[f]) else out[f]
                    for f in fields
                    if out.get(f) is not None
                )
            )
    output = tuple(
        torch.cat([c[i] for c in chunks]) if torch.is_tensor(chunks[0][i]) else chunks[0][i]
        for i in range(len(chunks[0]))
    )

    logger.info(f"Write prediction to {cfg.trainer.training_arguments.output_dir}")
    write_output(
        predictions=output,
        metrics=None,
        cfg=cfg,
        output_dir=cfg.trainer.training_arguments.output_dir,
        ids=ids,
        events=None,
        durations=None,
        model=model,
        prefix="inference_",
    )
    write_interpolation(
        cfg=cfg,
        predictions=output,
        ids=ids,
        output_dir=cfg.trainer.training_arguments.output_dir,
        is_survival=model.is_survival,
    )


@log_on_start(DEBUG, "Start inference...", logger=logger)
@log_on_error(
    ERROR,
    "Error during inference: {e!r}",
    logger=logger,
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "done!", logger=logger)
@hydra.main(version_base=None, config_path="../conf", config_name="infer.yaml")
def infer(cfg: DictConfig) -> None:
    config.Config()
    _infer(cfg)


if __name__ == "__main__":
    infer()
