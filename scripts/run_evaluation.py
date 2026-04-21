"""
run_evaluation.py
-----------------
Evaluate the knowledge base pipeline against a test set.

Run with:
  python scripts/run_evaluation.py --test-set data/eval_set.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from src.evaluation.evaluator import EvalSample, run_evaluation

log = structlog.get_logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default="data/eval_set.json")
    parser.add_argument("--output", default="docs/eval_report.json")
    args = parser.parse_args()

    raw = json.loads(Path(args.test_set).read_text())
    samples = [
        EvalSample(
            question=s["question"],
            ground_truth=s["ground_truth"],
            source_type=s.get("source_type"),
        )
        for s in raw
    ]

    log.info("evaluation_started", samples=len(samples))
    summary = run_evaluation(samples, output_path=args.output)

    print("\n--- Evaluation Summary ---")
    print(f"  Samples          : {summary['total_samples']}")
    print(f"  Avg groundedness : {summary['avg_groundedness']} / 5")
    print(f"  Avg relevance    : {summary['avg_relevance']} / 5")
    print(f"  Avg coherence    : {summary['avg_coherence']} / 5")
    print(f"  Overall avg      : {summary['avg_score']} / 5")
    print(f"  Total tokens     : {summary['total_tokens']:,}")
    print(f"  Total cost       : ${summary['total_cost_usd']:.4f}")
    print(f"  Cost per query   : ${summary['cost_per_query_usd']:.5f}")
    print(f"\n  Report saved to  : {args.output}")


if __name__ == "__main__":
    main()