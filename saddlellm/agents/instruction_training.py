"""Explicit chat formatting, assistant targets and content partition evidence."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping


PARTITION_SCHEMA = "saddle.text-partition/v1"


def _normalized(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def content_hash(value):
    if isinstance(value, Mapping):
        value = value.get("sourceIdentity", value.get("text", ""))
    return hashlib.sha256(_normalized(value).encode("utf-8")).hexdigest()


def partition_evidence(train, evaluation, tokenizer=None, max_length=None):
    train_hashes = [content_hash(value) for value in train]
    eval_hashes = [content_hash(value) for value in evaluation]
    evidence = {"schemaVersion": PARTITION_SCHEMA, "normalization": "NFKC-and-whitespace/v1",
            "trainRecordHashes": train_hashes, "evalRecordHashes": eval_hashes,
            "trainSamples": len(train_hashes), "evalSamples": len(eval_hashes),
            "trainUniqueSamples": len(set(train_hashes)), "evalUniqueSamples": len(set(eval_hashes)),
            "overlapUniqueSamples": len(set(train_hashes) & set(eval_hashes))}
    if tokenizer is not None and max_length is not None:
        def token_hash(record):
            if isinstance(record, Mapping):
                ids, labels = record["input_ids"][:max_length], record["labels"][:max_length]
            else:
                ids = tokenizer(record, truncation=True, max_length=max_length)["input_ids"]
                labels = ids
            return hashlib.sha256(json.dumps({"input_ids": ids, "labels": labels}, separators=(",", ":")).encode()).hexdigest()
        evidence["trainTokenHashes"] = [token_hash(record) for record in train]
        evidence["evalTokenHashes"] = [token_hash(record) for record in evaluation]
        evidence["maxLength"] = max_length
        evidence["overlapTokenSequences"] = len(set(evidence["trainTokenHashes"]) & set(evidence["evalTokenHashes"]))
    return evidence


def training_format(config):
    mode = str(config.get("trainingTextFormat") or "raw").strip().lower()
    if mode not in {"raw", "chat_template"}:
        raise ValueError("trainingTextFormat must be raw or chat_template")
    response_only = str(config.get("responseOnlyLoss", False)).strip().lower() in {"1", "true", "yes", "on"}
    if response_only and mode != "chat_template":
        raise ValueError("responseOnlyLoss requires trainingTextFormat=chat_template")
    return mode, response_only


def _messages(record):
    messages = record.get("messages")
    if messages is None:
        instruction, extra = record.get("instruction"), record.get("input")
        answer = record.get("output", record.get("response"))
        if not isinstance(instruction, str) or not instruction.strip() or not isinstance(answer, str) or not answer.strip():
            raise ValueError("chat_template training needs messages or non-empty instruction/output records")
        if extra is not None and not isinstance(extra, str):
            raise ValueError("instruction input must be text")
        messages = [{"role": "user", "content": instruction + ("\n\n" + extra if extra else "")},
                    {"role": "assistant", "content": answer}]
    if not isinstance(messages, list) or not messages or len(messages) > 64:
        raise ValueError("training messages must contain 1 to 64 text messages")
    expected, normalized = "user", []
    for index, item in enumerate(messages):
        if not isinstance(item, Mapping) or set(item) - {"role", "content"}:
            raise ValueError("training messages only support role and text content")
        role, content = item.get("role"), item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("training messages must have non-empty text content")
        if role == "system":
            if index != 0:
                raise ValueError("system message must be first")
        elif role != expected:
            raise ValueError("training messages must alternate user and assistant")
        else:
            expected = "assistant" if role == "user" else "user"
        normalized.append({"role": role, "content": content})
    if normalized[-1]["role"] != "assistant":
        raise ValueError("training messages must end with an assistant response")
    return normalized


def format_training_record(record, tokenizer, response_only=False):
    if not isinstance(record, Mapping):
        raise ValueError("chat_template training requires structured instruction/messages records")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The actual model tokenizer has no saved chat template; no generic Instruction/Response fallback is used")
    messages = _messages(record)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if not isinstance(text, str) or not text:
        raise ValueError("saved chat template returned empty training text")
    kwargs = {"add_special_tokens": False}
    if response_only:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("responseOnlyLoss requires a fast tokenizer with exact character offsets")
        kwargs["return_offsets_mapping"] = True
    encoded = tokenizer(text, **kwargs)
    ids = list(encoded["input_ids"])
    labels = ids.copy()
    if response_only:
        spans = []
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prompt = tokenizer.apply_chat_template(messages[:index], tokenize=False, add_generation_prompt=True)
            completed = tokenizer.apply_chat_template(messages[:index + 1], tokenize=False, add_generation_prompt=False)
            if not text.startswith(prompt) or not text.startswith(completed) or not completed.startswith(prompt):
                raise ValueError("chat template does not preserve an exact assistant prefix; cannot safely mask response targets")
            spans.append((len(prompt), len(completed)))
        labels = [-100] * len(ids)
        for index, (start, stop) in enumerate(encoded["offset_mapping"]):
            if start == stop:
                continue
            for begin, end in spans:
                if start < begin < stop or start < end < stop:
                    raise ValueError("token crosses an assistant response boundary; cannot safely mask prompt tokens")
                if begin <= start and stop <= end:
                    labels[index] = ids[index]
                    break
        if not any(label != -100 for label in labels[1:]):
            raise ValueError("chat_template record has no assistant target tokens")
    return {"text": text, "input_ids": ids, "labels": labels,
            "sourceIdentity": json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "trainingTextFormat": "chat_template", "responseOnlyLoss": response_only}


def formatted_token_batch(tokenizer, records, max_length, device):
    import torch
    if max_length < 2 or not records:
        raise ValueError("token batch needs records and maxLength >= 2")
    if tokenizer.pad_token_id is None:
        raise ValueError("training tokenizer has no padding token")
    ids, masks, labels = [], [], []
    for record in records:
        tokens = list(record["input_ids"][:max_length])
        targets = list(record["labels"][:max_length])
        if not any(value != -100 for value in targets[1:]):
            raise ValueError("maxLength truncates every assistant target token; increase maxLength or shorten the prompt")
        padding = max_length - len(tokens)
        ids.append(tokens + [tokenizer.pad_token_id] * padding)
        masks.append([1] * len(tokens) + [0] * padding)
        labels.append(targets + [-100] * padding)
    return {"input_ids": torch.tensor(ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device)}
