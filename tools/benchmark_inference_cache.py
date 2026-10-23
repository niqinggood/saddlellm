"""Measure cache append cost, not full-model inference throughput.

Run from the project environment:
python tools/benchmark_inference_cache.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from saddlellm.models.InferenceCache import InferenceLayerCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--prompt-tokens', type=int, default=128)
    parser.add_argument('--decode-tokens', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.decode_tokens, args.repeats) < 1:
        parser.error('token counts and repeats must be positive')
    torch.set_num_threads(1)
    prompt = torch.randn(1, 8, args.prompt_tokens, 64, device=args.device)
    token = torch.randn(1, 8, 1, 64, device=args.device)

    def synchronize():
        if args.device == 'cuda':
            torch.cuda.synchronize()

    def run(mode):
        allocations = 1
        copied = 0
        with torch.inference_mode():
            if mode == 'geometric':
                cache = InferenceLayerCache(args.prompt_tokens + args.decode_tokens)
                result = cache.append(prompt, prompt)
            else:
                result = (prompt, prompt)
            for _ in range(args.decode_tokens):
                length = result[0].shape[2]
                if mode == 'geometric':
                    old_capacity = cache.capacity
                    result = cache.append(token, token)
                    if cache.capacity != old_capacity:
                        allocations += 1
                        copied += 2 * length * token.numel() * token.element_size()
                else:
                    result = tuple(torch.cat([value, token], dim=2) for value in result)
                    allocations += 1
                    copied += 2 * length * token.numel() * token.element_size()
            return result, allocations, copied

    report = {'device': args.device, 'prompt_tokens': args.prompt_tokens,
              'decode_tokens': args.decode_tokens, 'scope': 'one layer KV append only'}
    references = {}
    for mode in ('dynamic', 'geometric'):
        run(mode)  # warm up
        synchronize()
        start = time.perf_counter()
        for _ in range(args.repeats):
            result, allocations, copied = run(mode)
        synchronize()
        report[mode] = {
            'mean_ms': (time.perf_counter() - start) * 1000 / args.repeats,
            'storage_allocations': allocations,
            'history_copy_bytes': copied,
        }
        references[mode] = result
    for old, new in zip(references['dynamic'], references['geometric']):
        torch.testing.assert_close(old, new)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
