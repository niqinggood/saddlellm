"""Completion-labelled SFT for research datasets with explicit token masks."""

# ruff: noqa: N999 -- Matches the repository's public module naming convention.

from __future__ import annotations

from pathlib import Path


def train_labelled_sft(model, tokenizer, dataset, arguments, output_path: str):
    """Train supplied labels without rewriting a task's chat template.

    Every row must contain a real answer token after the causal shift. The task
    owns tokenization; SaddleLLM owns Trainer execution, persistence and logs.
    """
    from transformers import DataCollatorForSeq2Seq, Trainer

    if not len(dataset):
        raise ValueError("Labelled SFT requires non-empty data")
    supervised = 0
    for row in dataset:
        if len(row["input_ids"]) != len(row["labels"]):
            raise ValueError("input_ids and labels must have the same length")
        count = sum(label != -100 for label in row["labels"][1:])
        if not count:
            raise ValueError("SFT row has no supervised answer after causal shift")
        supervised += count
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
    )
    result = trainer.train()
    destination = Path(output_path)
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(str(destination))
    trainer.save_state()
    return {
        "train_metrics": dict(result.metrics),
        "global_step": trainer.state.global_step,
        "log_history": trainer.state.log_history,
        "supervised_tokens_per_epoch": supervised,
        "engine": "saddlellm.training.LabelledSFT",
    }
