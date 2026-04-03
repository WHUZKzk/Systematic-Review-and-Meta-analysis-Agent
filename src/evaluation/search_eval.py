"""
search_eval.py — Evaluation of the Search Stage (Stage 1).

Metrics computed against bench_review.json ground truth:

  Recall  = |retrieved ∩ ground_truth| / |ground_truth| × 100%
  NNS     = |candidate_pool| / |ground_truth|   (Number Needed to Screen)
  Precision = |retrieved ∩ ground_truth| / |candidate_pool| × 100%

Usage (programmatic):
    from evaluation.search_eval import SearchEvaluator
    ev = SearchEvaluator("bench_review.json")
    report = ev.evaluate(candidate_pool)   # list[dict] with 'pmid' field
    ev.print_report(report)

Usage (CLI):
    python -m evaluation.search_eval \
        --bench bench_review.json \
        --pool  outputs/candidate_pool.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SearchEvalReport:
    # Ground truth
    n_ground_truth: int
    ground_truth_pmids: Set[str]

    # System output
    n_candidate_pool: int

    # Retrieved = pool ∩ ground truth
    retrieved_pmids: Set[str]
    missed_pmids: Set[str]

    # Metrics
    recall: float            # %
    precision: float         # %
    nns: float               # ratio

    # Per-study detail (for missed papers)
    missed_details: List[Dict[str, Any]] = field(default_factory=list)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class SearchEvaluator:
    """
    Evaluates a SearchResult / candidate pool against a ground-truth benchmark.

    Args:
        bench_path: path to bench_review.json
    """

    def __init__(self, bench_path: str | Path):
        self._bench_path = Path(bench_path)
        self._ground_truth: List[Dict] = self._load_bench()

    def _load_bench(self) -> List[Dict]:
        with open(self._bench_path, encoding="utf-8") as fh:
            data = json.load(fh)
        # Accept list[{PMID, title, ...}] or dict keyed by PMID
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            records = []
            for pmid, val in data.items():
                if isinstance(val, dict):
                    val.setdefault("PMID", str(pmid))
                    records.append(val)
                else:
                    records.append({"PMID": str(pmid)})
            return records
        raise ValueError(f"Unexpected bench format in {self._bench_path}")

    @property
    def ground_truth_pmids(self) -> Set[str]:
        return {str(r.get("PMID", "")).strip() for r in self._ground_truth if r.get("PMID")}

    def evaluate(self, candidate_pool: List[Dict[str, Any]]) -> SearchEvalReport:
        """
        Args:
            candidate_pool: list of dicts each containing at minimum a 'pmid' key.
                            Also accepts 'PMID' (case-insensitive).

        Returns:
            SearchEvalReport with Recall, Precision, NNS and missed-paper detail.
        """
        # Normalize pool PMIDs
        pool_pmids: Set[str] = set()
        for rec in candidate_pool:
            pmid = str(rec.get("pmid") or rec.get("PMID") or "").strip()
            if pmid:
                pool_pmids.add(pmid)

        gt_pmids = self.ground_truth_pmids
        retrieved = gt_pmids & pool_pmids
        missed    = gt_pmids - pool_pmids

        n_gt   = len(gt_pmids)
        n_pool = len(pool_pmids)

        recall    = (len(retrieved) / n_gt * 100) if n_gt else 0.0
        precision = (len(retrieved) / n_pool * 100) if n_pool else 0.0
        nns       = (n_pool / n_gt) if n_gt else float("inf")

        # Build missed-paper detail from bench data
        gt_index = {str(r.get("PMID", "")).strip(): r for r in self._ground_truth}
        missed_details = [
            {
                "pmid": pmid,
                "title": gt_index.get(pmid, {}).get("title", "N/A"),
                "pico_p": gt_index.get(pmid, {}).get("PICO", {}).get("P", "")[:100],
            }
            for pmid in sorted(missed)
        ]

        return SearchEvalReport(
            n_ground_truth=n_gt,
            ground_truth_pmids=gt_pmids,
            n_candidate_pool=n_pool,
            retrieved_pmids=retrieved,
            missed_pmids=missed,
            recall=round(recall, 1),
            precision=round(precision, 1),
            nns=round(nns, 2),
            missed_details=missed_details,
        )

    def evaluate_from_file(self, pool_path: str | Path) -> SearchEvalReport:
        """Load candidate_pool from a JSON file and evaluate."""
        with open(pool_path, encoding="utf-8") as fh:
            pool = json.load(fh)
        # Support {candidate_pool: [...]} wrapper produced by SearchPipeline
        if isinstance(pool, dict):
            pool = pool.get("candidate_pool", pool.get("results", list(pool.values())))
        return self.evaluate(pool)

    @staticmethod
    def print_report(report: SearchEvalReport, verbose: bool = True) -> None:
        sep = "=" * 60
        print(sep)
        print("  SEARCH STAGE EVALUATION REPORT")
        print(sep)
        print(f"  Ground truth papers   : {report.n_ground_truth}")
        print(f"  Candidate pool size   : {report.n_candidate_pool}")
        print(f"  Successfully retrieved: {len(report.retrieved_pmids)}")
        print(f"  Missed                : {len(report.missed_pmids)}")
        print(sep)
        print(f"  Recall                : {report.recall:.1f}%")
        print(f"  Precision             : {report.precision:.1f}%")
        print(f"  NNS (pool / GT)       : {report.nns:.2f}")
        print(sep)
        if verbose and report.missed_pmids:
            print("  MISSED PAPERS:")
            for d in report.missed_details:
                print(f"    PMID {d['pmid']}: {d['title'][:70]}")
                if d["pico_p"]:
                    print(f"      P: {d['pico_p']}...")
        print(sep)

    def save_report(self, report: SearchEvalReport, out_path: str | Path) -> None:
        out = {
            "n_ground_truth": report.n_ground_truth,
            "n_candidate_pool": report.n_candidate_pool,
            "n_retrieved": len(report.retrieved_pmids),
            "n_missed": len(report.missed_pmids),
            "recall_pct": report.recall,
            "precision_pct": report.precision,
            "nns": report.nns,
            "retrieved_pmids": sorted(report.retrieved_pmids),
            "missed_pmids": sorted(report.missed_pmids),
            "missed_details": report.missed_details,
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        logger.info("Search eval report saved → %s", out_path)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate search stage against bench_review.json")
    parser.add_argument("--bench", default="bench_review.json",
                        help="Path to bench_review.json")
    parser.add_argument("--pool", required=True,
                        help="Path to candidate_pool.json (SearchPipeline output)")
    parser.add_argument("--out", default=None,
                        help="Optional path to save JSON report")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress missed-paper detail")
    args = parser.parse_args()

    ev = SearchEvaluator(args.bench)
    report = ev.evaluate_from_file(args.pool)
    ev.print_report(report, verbose=not args.quiet)
    if args.out:
        ev.save_report(report, args.out)


if __name__ == "__main__":
    _cli()
