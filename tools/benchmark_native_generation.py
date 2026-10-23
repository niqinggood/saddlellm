"""Offline synthetic-model benchmark for native generation fast paths.

Compares full prefill logits + concatenated token/mask buffers with current
generation. Both modes use the same geometric KV cache and model weights.
This is not a benchmark of a trained production model.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from unittest.mock import patch

import torch

import saddlellm.models.SaddleModeling as modeling


class ConcatenatedBuffers:
    """Reference behavior before token/mask storage was preallocated."""

    def __init__(self, tokens, mask, max_new_tokens):
        self.tokens, self.mask = tokens, mask

    def views(self):
        return self.tokens, self.mask

    def append(self, token, active):
        self.tokens = torch.cat([self.tokens, token], dim=1)
        self.mask = torch.cat([self.mask, active.to(self.mask.dtype)[:, None]], dim=1)

    def sequences(self):
        return self.tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--prompt-tokens', type=int, default=256)
    parser.add_argument('--decode-tokens', type=int, default=32)
    parser.add_argument('--vocab-size', type=int, default=8192)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.decode_tokens, args.repeats) < 1:
        parser.error('token counts and repeats must be positive')
    if args.vocab_size < 4:
        parser.error('vocab-size must be at least 4')
    torch.set_num_threads(1)
    torch.manual_seed(173)
    model = modeling.SaddleForCausalLM(modeling.SaddleModelConfig(
        vocab_size=args.vocab_size, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=args.prompt_tokens + args.decode_tokens,
        attention_backend='sdpa',
    )).to(args.device).eval()
    prompt = torch.randint(3, args.vocab_size, (1, args.prompt_tokens),
                           device=args.device)

    def synchronize():
        if args.device == 'cuda':
            torch.cuda.synchronize()

    report = {
        'scope': 'synthetic untrained model, full generation, same KV cache',
        'device': args.device, 'dtype': str(next(model.parameters()).dtype),
        'prompt_tokens': args.prompt_tokens, 'decode_tokens': args.decode_tokens,
        'vocab_size': args.vocab_size, 'hidden_size': 64, 'layers': 2,
        'repeats': args.repeats,
    }
    sequences = {}
    for mode in ('reference', 'optimized'):
        context = (patch.object(modeling, 'GenerationBuffers', ConcatenatedBuffers)
                   if mode == 'reference' else nullcontext())
        with context, torch.inference_mode():
            def generate():
                return model.generate(
                    prompt, max_new_tokens=args.decode_tokens,
                    do_sample=False, eos_token_id=[],
                    logits_to_keep=0 if mode == 'reference' else 1,
                )

            generate()  # warm up kernels and RoPE tables
            synchronize()
            base_memory = None
            if args.device == 'cuda':
                base_memory = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(args.repeats):
                output = generate()
            synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000 / args.repeats
            peak_extra = (torch.cuda.max_memory_allocated() - base_memory
                          if base_memory is not None else None)
            sequences[mode] = output.cpu()
            del output
            positions = []
            handle = model.lm_head.register_forward_pre_hook(
                lambda _, inputs: positions.append(inputs[0].shape[1])
            )
            try:
                generate()
            finally:
                handle.remove()
            report[mode] = {
                'mean_generation_ms': elapsed_ms,
                'projected_positions': sum(positions),
                'materialized_logit_elements': sum(positions) * args.vocab_size,
                'peak_extra_cuda_bytes': peak_extra,
            }
    if not torch.equal(sequences['reference'], sequences['optimized']):
        raise RuntimeError('benchmark modes produced different token sequences')
    report['outputs_equal'] = True
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
