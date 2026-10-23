"""Generate labelled synthetic typed questions for end-to-end verification.

This controlled fixture tests learning and calibration plumbing, not general
language understanding. State groups never cross the four data splits.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def prepare(output, *, train_states=96, heldout_states=24, seed=42):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    colors, intensities = ["red", "blue", "green"], ["low", "medium", "high"]
    question = "What color is the signal?"
    questions = [
        {
            "task_id": "color",
            "type": "choice",
            "question": question,
            "choices": [{"id": color, "text": color} for color in colors],
        },
        {"task_id": "red", "type": "noul", "question": "Is the signal red?"},
        {
            "task_id": "intensity",
            "type": "score",
            "question": "Rate the signal intensity.",
            "levels": [
                {"id": intensity, "text": intensity, "value": i}
                for i, intensity in enumerate(intensities)
            ],
        },
    ]
    for split in ("train", "validation", "calibration", "test"):
        rows = []
        count = train_states if split == "train" else heldout_states
        for i in range(count):
            # Independent randomized case handles carry no target information.
            handle = f"{rng.getrandbits(64):016x}"
            color, intensity = colors[i % 3], intensities[(i // 3) % 3]
            ambiguous = i % 12 == 11
            state = f"Color: {'unknown' if ambiguous else color}. Intensity: {intensity}. Case: {handle}."
            for j, item in enumerate(questions):
                row = {
                    **item,
                    "id": f"{split}-{i}-{j}",
                    "state_id": handle,
                    "state": state,
                }
                if j == 0 and ambiguous:
                    row["target_probs"] = {c: 1 / 3 for c in colors}
                elif j == 1 and ambiguous:
                    row["target_probs"] = {"false": 2 / 3, "true": 1 / 3}
                else:
                    row["label"] = (
                        color if j == 0 else color == "red" if j == 1 else intensity
                    )
                if j == 0 and i % 2:
                    row["choices"] = list(reversed(item["choices"]))
                rows.append(row)
            if split == "test":
                rows.append(
                    {
                        "id": f"test-{i}-unseen",
                        "state_id": handle,
                        "state": state,
                        "task_id": "color-unseen-question",
                        "type": "choice",
                        "question": "Which color does the signal have?",
                        "choices": questions[0]["choices"],
                        **(
                            {"target_probs": {c: 1 / 3 for c in colors}}
                            if ambiguous
                            else {"label": color}
                        ),
                    }
                )
        (root / (split + ".jsonl")).write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    descriptor = {
        "schema_version": 2,
        "task_type": "typed",
        "source": "synthetic controlled signal fixture",
        "purpose": "Training/inference integration; no claim of general decision quality",
        "splits": {
            name: name + ".jsonl"
            for name in ("train", "validation", "calibration", "test")
        },
    }
    (root / "dataset.json").write_text(
        json.dumps(descriptor, indent=2) + "\n", encoding="utf-8"
    )
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-states", type=int, default=96)
    parser.add_argument("--heldout-states", type=int, default=24)
    args = parser.parse_args()
    print(
        prepare(
            args.output,
            train_states=args.train_states,
            heldout_states=args.heldout_states,
        )
    )
