"""
screening_eval.py — Evaluation of the Screening Stage (Stage 2).

Metrics computed against bench_review.json ground truth
(all bench PMIDs are treated as INCLUDE ground truth):

  Recall              = TP / (TP + FN)  × 100%
  Precision           = TP / (TP + FP)  × 100%
  F1                  = 2 × P × R / (P + R)
  Cohen's κ           = (Po - Pe) / (1 - Pe)   [reviewer_a vs reviewer_b]
  Adjudication Acc.   = adj_correct / adj_total × 100%

where:
  TP = included by system AND in ground truth
  FP = included by system but NOT in ground truth
  FN = in ground truth but excluded/uncertain by system

Usage (programmatic):
    from evaluation.screening_eval import ScreeningEvaluator
    ev = ScreeningEvaluator("bench_review.json")
    report = ev.evaluate(screening_result)
    ev.print_report(report)

Usage (CLI):
    python -m evaluation.screening_eval \
        --bench  bench_review.json \
        --result outputs/screening_result.json
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ── Kappa helper (standalone, no pipeline import needed) ──────────────────────

def _cohen_kappa(
    decisions_a: List[str], decisions_b: List[str],
    categories: Optional[List[str]] = None,
) -> Tuple[float, float, str]:
    """
    Compute Cohen's unweighted kappa.

    Returns: (kappa, observed_agreement, label)
    """
    if categories is None:
        categories = ["INCLUDE", "EXCLUDE", "UNCERTAIN"]
    n = len(decisions_a)
    if n == 0:
        return 0.0, 0.0, "poor"

    cat_index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    # Confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(decisions_a, decisions_b):
        i = cat_index.get(a, -1)
        j = cat_index.get(b, -1)
        if i >= 0 and j >= 0:
            matrix[i][j] += 1

    # Observed agreement
    po = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement
    row_sums = [sum(matrix[i]) for i in range(k)]
    col_sums = [sum(matrix[r][c] for r in range(k)) for c in range(k)]
    pe = sum((row_sums[i] / n) * (col_sums[i] / n) for i in range(k))

    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0

    # Landis & Koch (1977) categories
    if kappa < 0:
        label = "poor"
    elif kappa < 0.20:
        label = "slight"
    elif kappa < 0.40:
        label = "fair"
    elif kappa < 0.60:
        label = "moderate"
    elif kappa < 0.80:
        label = "substantial"
    else:
        label = "almost_perfect"

    return round(kappa, 4), round(po, 4), label


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ScreeningEvalReport:
    # Ground truth
    n_ground_truth: int
    ground_truth_pmids: Set[str]

    # Screening outcome counts
    n_included_by_system: int
    n_tp: int           # INCLUDE ∩ ground_truth
    n_fp: int           # INCLUDE but NOT ground_truth
    n_fn: int           # ground_truth but NOT included
    n_tn: int           # NOT ground_truth and NOT included (approx)

    # Core metrics
    recall: float       # %
    precision: float    # %
    f1: float           # 0-1

    # Inter-rater agreement (reviewer_a vs reviewer_b)
    kappa: Optional[float]
    kappa_label: Optional[str]
    observed_agreement: Optional[float]
    n_disagreements: int

    # Adjudication accuracy
    adj_total: int
    adj_correct: int
    adj_accuracy: Optional[float]   # %

    # Detail lists
    tp_pmids: Set[str] = field(default_factory=set)
    fp_pmids: Set[str] = field(default_factory=set)
    fn_pmids: Set[str] = field(default_factory=set)
    fn_details: List[Dict[str, Any]] = field(default_factory=list)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class ScreeningEvaluator:
    """
    Evaluates a ScreeningResult against bench_review.json ground truth.

    All papers listed in bench_review.json are treated as ground-truth INCLUDE.
    Papers in the candidate pool that are NOT in the bench are treated as
    ground-truth EXCLUDE.

    Args:
        bench_path: path to bench_review.json
    """

    def __init__(self, bench_path: str | Path):
        self._bench_path = Path(bench_path)
        self._ground_truth = self._load_bench()

    def _load_bench(self) -> List[Dict]:
        with open(self._bench_path, encoding="utf-8") as fh:
            data = json.load(fh)
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

    def evaluate(
        self,
        included_studies: List[Dict],
        excluded_studies: List[Dict],
        uncertain_studies: Optional[List[Dict]] = None,
        screening_report: Optional[Dict] = None,
    ) -> ScreeningEvalReport:
        """
        Evaluate from ScreeningResult components.

        Args:
            included_studies:  ScreeningResult.included_studies
            excluded_studies:  ScreeningResult.excluded_studies
            uncertain_studies: ScreeningResult.uncertain_studies (treated as excluded for recall)
            screening_report:  ScreeningResult.screening_report (used to extract kappa + adjudication data)
        """
        uncertain_studies = uncertain_studies or []
        gt_pmids = self.ground_truth_pmids

        included_pmids: Set[str] = {
            str(r.get("pmid") or r.get("PMID", "")).strip()
            for r in included_studies
        } - {""}
        excluded_pmids: Set[str] = {
            str(r.get("pmid") or r.get("PMID", "")).strip()
            for r in excluded_studies
        } - {""}
        uncertain_pmids: Set[str] = {
            str(r.get("pmid") or r.get("PMID", "")).strip()
            for r in uncertain_studies
        } - {""}

        # NOTE: uncertain = Tier2 full-text required, counted as not-yet-included for Recall
        all_system_pmids = included_pmids | excluded_pmids | uncertain_pmids

        tp = gt_pmids & included_pmids
        fp = included_pmids - gt_pmids
        fn = gt_pmids - included_pmids          # missed (excluded or uncertain)
        tn_approx = (all_system_pmids - gt_pmids) - fp   # excluded true negatives

        recall    = (len(tp) / len(gt_pmids) * 100) if gt_pmids else 0.0
        precision = (len(tp) / len(included_pmids) * 100) if included_pmids else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        # -- Extract kappa from screening_report
        kappa = obs_agreement = kappa_label = None
        n_disagreements = 0
        if screening_report:
            kappa_block = screening_report.get("inter_rater_agreement", {})
            kappa = kappa_block.get("kappa")
            obs_agreement = kappa_block.get("observed_agreement")
            kappa_label = kappa_block.get("category")
            n_disagreements = screening_report.get("prisma_counts", {}).get(
                "records_adjudicated", 0
            )

        # -- Adjudication accuracy
        adj_total = adj_correct = 0
        adj_accuracy = None
        if screening_report:
            adj_total = screening_report.get("prisma_counts", {}).get(
                "records_adjudicated", 0
            )
            if adj_total > 0:
                # Count adjudicated papers where decision matches ground truth
                # Ground truth: bench PMID → INCLUDE; others → EXCLUDE
                adj_correct = 0
                adj_include = screening_report.get("adjudication", {}).get(
                    "adjudicated_include", 0
                )
                adj_exclude = screening_report.get("adjudication", {}).get(
                    "adjudicated_exclude", 0
                )
                # We know adj_include + adj_exclude = adj_total (approx)
                # Correct adj_include = adj_include that are in gt_pmids
                # We can't compute this exactly without per-paper adjudication data,
                # so we use the TP/FP split among included_studies as a proxy.
                # If the full disagreement list is available, use it:
                adj_decisions = screening_report.get("adjudication", {}).get(
                    "decisions", []
                )
                if adj_decisions:
                    for rec in adj_decisions:
                        pmid = str(rec.get("pmid", "")).strip()
                        decision = rec.get("final_decision", "")
                        gt_label = "INCLUDE" if pmid in gt_pmids else "EXCLUDE"
                        sys_label = (
                            "INCLUDE"
                            if decision in ("INCLUDE", "INCLUDE_Tier2")
                            else "EXCLUDE"
                        )
                        if gt_label == sys_label:
                            adj_correct += 1
                    adj_total = len(adj_decisions)
                    adj_accuracy = round(adj_correct / adj_total * 100, 1) if adj_total else None

        # Build fn detail from bench index
        gt_index = {str(r.get("PMID", "")).strip(): r for r in self._ground_truth}
        fn_details = [
            {
                "pmid": pmid,
                "title": gt_index.get(pmid, {}).get("title", "N/A"),
                "status": "uncertain" if pmid in uncertain_pmids else "excluded",
            }
            for pmid in sorted(fn)
        ]

        return ScreeningEvalReport(
            n_ground_truth=len(gt_pmids),
            ground_truth_pmids=gt_pmids,
            n_included_by_system=len(included_pmids),
            n_tp=len(tp),
            n_fp=len(fp),
            n_fn=len(fn),
            n_tn=len(tn_approx),
            recall=round(recall, 1),
            precision=round(precision, 1),
            f1=round(f1 / 100, 4),   # store as 0-1
            kappa=kappa,
            kappa_label=kappa_label,
            observed_agreement=obs_agreement,
            n_disagreements=n_disagreements,
            adj_total=adj_total,
            adj_correct=adj_correct,
            adj_accuracy=adj_accuracy,
            tp_pmids=tp,
            fp_pmids=fp,
            fn_pmids=fn,
            fn_details=fn_details,
        )

    def evaluate_from_file(self, result_path: str | Path) -> ScreeningEvalReport:
        """Load a saved ScreeningResult JSON and evaluate."""
        with open(result_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return self.evaluate(
            included_studies=data.get("included_studies", []),
            excluded_studies=data.get("excluded_studies", []),
            uncertain_studies=data.get("uncertain_studies", []),
            screening_report=data.get("screening_report"),
        )

    @staticmethod
    def print_report(report: ScreeningEvalReport, verbose: bool = True) -> None:
        sep = "=" * 60
        print(sep)
        print("  SCREENING STAGE EVALUATION REPORT")
        print(sep)
        print(f"  Ground truth papers   : {report.n_ground_truth}")
        print(f"  System INCLUDE count  : {report.n_included_by_system}")
        print(f"  TP (correct includes) : {report.n_tp}")
        print(f"  FP (false includes)   : {report.n_fp}")
        print(f"  FN (missed)           : {report.n_fn}")
        print(sep)
        print(f"  Recall                : {report.recall:.1f}%")
        print(f"  Precision             : {report.precision:.1f}%")
        print(f"  F1 Score              : {report.f1:.4f}")
        print(sep)

        if report.kappa is not None:
            print(f"  Cohen's κ             : {report.kappa:.4f}  ({report.kappa_label})")
            print(f"  Observed agreement    : {report.observed_agreement:.4f}")
            print(f"  Disagreements (items) : {report.n_disagreements}")
        else:
            print("  Cohen's κ             : N/A (no inter-rater data)")
        print(sep)

        if report.adj_total > 0:
            acc_str = f"{report.adj_accuracy:.1f}%" if report.adj_accuracy is not None else "N/A"
            print(f"  Adjudicated items     : {report.adj_total}")
            print(f"  Adj. correct          : {report.adj_correct}")
            print(f"  Adjudication Accuracy : {acc_str}")
        else:
            print("  Adjudication          : no disputes recorded")
        print(sep)

        if verbose and report.fn_pmids:
            print("  MISSED (FN) PAPERS:")
            for d in report.fn_details:
                print(f"    [{d['status'].upper():9s}] PMID {d['pmid']}: {d['title'][:60]}")
        print(sep)

    def save_report(self, report: ScreeningEvalReport, out_path: str | Path) -> None:
        out = {
            "n_ground_truth": report.n_ground_truth,
            "n_included_by_system": report.n_included_by_system,
            "tp": report.n_tp,
            "fp": report.n_fp,
            "fn": report.n_fn,
            "recall_pct": report.recall,
            "precision_pct": report.precision,
            "f1": report.f1,
            "kappa": report.kappa,
            "kappa_label": report.kappa_label,
            "observed_agreement": report.observed_agreement,
            "n_disagreements": report.n_disagreements,
            "adj_total": report.adj_total,
            "adj_correct": report.adj_correct,
            "adj_accuracy_pct": report.adj_accuracy,
            "tp_pmids": sorted(report.tp_pmids),
            "fp_pmids": sorted(report.fp_pmids),
            "fn_pmids": sorted(report.fn_pmids),
            "fn_details": report.fn_details,
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        logger.info("Screening eval report saved → %s", out_path)


# ── Ablation: single-agent vs multi-agent ─────────────────────────────────────

def ablation_single_vs_multi(
    bench_path: str | Path,
    reviewer_a_decisions: Dict[str, str],  # {pmid: "INCLUDE"/"EXCLUDE"/"UNCERTAIN"}
    reviewer_b_decisions: Dict[str, str],
    adjudicated_decisions: Dict[str, str],  # final after adjudication
) -> None:
    """
    Print ablation table comparing single-agent vs multi-agent on Recall & Precision.

    reviewer_a_decisions / reviewer_b_decisions: raw per-paper decisions from each agent.
    adjudicated_decisions: final system decisions (multi-agent output).
    """
    with open(bench_path, encoding="utf-8") as fh:
        bench = json.load(fh)
    gt_pmids = {str(r.get("PMID", "")).strip() for r in (bench if isinstance(bench, list) else [])}

    def _metrics(decisions: Dict[str, str]) -> Tuple[float, float]:
        included = {p for p, d in decisions.items() if d == "INCLUDE"}
        tp = len(included & gt_pmids)
        fp = len(included - gt_pmids)
        fn = len(gt_pmids - included)
        recall = tp / (tp + fn) * 100 if (tp + fn) else 0.0
        prec   = tp / (tp + fp) * 100 if (tp + fp) else 0.0
        return round(recall, 1), round(prec, 1)

    ra_r, ra_p = _metrics(reviewer_a_decisions)
    rb_r, rb_p = _metrics(reviewer_b_decisions)
    ms_r, ms_p = _metrics(adjudicated_decisions)

    # Kappa between A and B
    all_pmids = sorted(set(reviewer_a_decisions) | set(reviewer_b_decisions))
    da = [reviewer_a_decisions.get(p, "UNCERTAIN") for p in all_pmids]
    db = [reviewer_b_decisions.get(p, "UNCERTAIN") for p in all_pmids]
    kappa, _, kappa_lbl = _cohen_kappa(da, db)

    sep = "=" * 60
    print(sep)
    print("  ABLATION: SINGLE-AGENT vs MULTI-AGENT")
    print(sep)
    print(f"  {'System':<25} {'Recall':>8} {'Precision':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*10}")
    print(f"  {'Reviewer A (single)':.<25} {ra_r:>7.1f}% {ra_p:>9.1f}%")
    print(f"  {'Reviewer B (single)':.<25} {rb_r:>7.1f}% {rb_p:>9.1f}%")
    print(f"  {'Multi-agent (final)':.<25} {ms_r:>7.1f}% {ms_p:>9.1f}%")
    print(sep)
    print(f"  Inter-rater κ (A vs B): {kappa:.4f}  ({kappa_lbl})")
    print(sep)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate screening stage against bench_review.json"
    )
    parser.add_argument("--bench", default="bench_review.json")
    parser.add_argument("--result", required=True,
                        help="Path to screening_result.json saved by ScreeningPipeline")
    parser.add_argument("--out", default=None,
                        help="Optional path to save JSON report")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ev = ScreeningEvaluator(args.bench)
    report = ev.evaluate_from_file(args.result)
    ev.print_report(report, verbose=not args.quiet)
    if args.out:
        ev.save_report(report, args.out)


if __name__ == "__main__":
    _cli()
