"""Synthetic microbenchmarks for repetition penalty and text-stop control.

No model forward, HTTP, tokenizer encoding or trained weights are measured.
Stop inputs use a cleanup-free ByteLevel tokenizer and fixed synthetic tokens.
"""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from saddlellm.models.NativeGeneration import NativeGenerationSession
from saddlellm.runtime.GenerationStop import StopStringRows


def legacy_penalty(session, logits, out):
    for index in range(out.shape[0]):
        ids = torch.unique(out[index])
        values = logits[index, ids]
        logits[index, ids] = torch.where(
            values < 0,
            values * session.repetition_penalty,
            values / session.repetition_penalty,
        )


class CountedStops(StopStringRows):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decode_calls = self.decoded_tokens = 0

    def _decode(self, ids):
        self.decode_calls += 1
        self.decoded_tokens += len(ids)
        return super()._decode(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--stop-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if (
        min(
            args.batch_size,
            args.vocab_size,
            args.prompt_tokens,
            args.decode_tokens,
            args.stop_tokens,
            args.repeats,
        )
        < 1
    ):
        parser.error("sizes, token counts and repeats must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(173)

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def measure(function):
        function()
        synchronize()
        started = time.perf_counter()
        for _ in range(args.repeats):
            result = function()
        synchronize()
        return result, (time.perf_counter() - started) * 1000 / args.repeats

    prompt = torch.randint(
        args.vocab_size, (args.batch_size, args.prompt_tokens), device=args.device
    )
    tokens = torch.randint(
        args.vocab_size, (args.decode_tokens, args.batch_size, 1), device=args.device
    )
    raw_logits = torch.randn(args.batch_size, args.vocab_size, device=args.device)
    active = torch.ones(args.batch_size, dtype=torch.bool, device=args.device)
    dummy_model = SimpleNamespace(
        eval=lambda: None,
        config=SimpleNamespace(pad_token_id=0, eos_token_id=[]),
        layers=[],
    )

    def penalties(reference):
        with (
            torch.inference_mode(),
            NativeGenerationSession(
                dummy_model,
                prompt,
                max_new_tokens=args.decode_tokens,
                repetition_penalty=1.1,
                use_cache=False,
            ) as session,
        ):
            for token in tokens.unbind(0):
                logits = raw_logits.clone()
                history, _ = session.buffers.views()
                if reference:
                    legacy_penalty(session, logits, history)
                else:
                    session._apply_repetition_penalty(logits, history)
                session.buffers.append(token, active)
                if not reference:
                    session._record_repetition_tokens(token)
            return logits

    reference_scores, reference_ms = measure(lambda: penalties(True))
    optimized_scores, optimized_ms = measure(lambda: penalties(False))
    if not torch.equal(reference_scores, optimized_scores):
        raise RuntimeError("repetition penalty final scores differ")
    report = {
        "scope": (
            "synthetic repetition-penalty and stop-check microbenchmarks; "
            "not model throughput"
        ),
        "device": args.device,
        "batch_size": args.batch_size,
        "vocab_size": args.vocab_size,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "repeats": args.repeats,
        "repetition_penalty": {
            "reference_ms": reference_ms,
            "optimized_ms": optimized_ms,
            "final_scores_equal": True,
        },
    }

    vocab = {"[UNK]": 0}
    vocab.update(
        {char: index + 1 for index, char in enumerate(bytes_to_unicode().values())}
    )
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        clean_up_tokenization_spaces=False,
    )
    repeated = torch.full(
        (args.batch_size, args.stop_tokens + 1),
        vocab["x"],
        dtype=torch.long,
        device=args.device,
    )
    stops_by_row = [["<STOP>", "<END>", "<DONE>"]] * args.batch_size

    def check_stops(incremental):
        checker = CountedStops(
            tokenizer, 1, stops_by_row, allow_incremental=incremental
        )
        decisions = [
            checker(repeated[:, : count + 1])
            for count in range(1, args.stop_tokens + 1)
        ]
        return torch.stack(decisions), checker

    (reference_decisions, reference_checker), reference_ms = measure(
        lambda: check_stops(False)
    )
    (optimized_decisions, optimized_checker), optimized_ms = measure(
        lambda: check_stops(True)
    )
    if not torch.equal(reference_decisions, optimized_decisions):
        raise RuntimeError("text stop decisions differ")
    report["text_stops"] = {
        "tokens_per_row": args.stop_tokens,
        "stop_strings_per_row": len(stops_by_row[0]),
        "reference_ms": reference_ms,
        "optimized_ms": optimized_ms,
        "reference_decoded_tokens": reference_checker.decoded_tokens,
        "optimized_decoded_tokens": optimized_checker.decoded_tokens,
        "reference_decode_calls": reference_checker.decode_calls,
        "optimized_decode_calls": optimized_checker.decode_calls,
        "decisions_equal": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
