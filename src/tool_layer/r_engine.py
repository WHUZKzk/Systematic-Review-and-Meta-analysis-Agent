"""
r_engine.py — R Statistical Engine tool (D3.2).

Executes R code for meta-analysis, forest plots, and related statistical computations.
Requires R and the 'meta' package to be installed on the system.

If R is not available, falls back to a Python-based placeholder that returns
structured results with a warning — allowing the pipeline to run without R
for testing and development purposes.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Dict, List, Optional

from agent_layer.base_agent import ToolResult

logger = logging.getLogger(__name__)

# ── R availability check ──────────────────────────────────────────────────────

def _r_available() -> bool:
    """Check if Rscript is on PATH."""
    try:
        result = subprocess.run(
            ["Rscript", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_R_AVAILABLE = _r_available()


# ── REngineError ──────────────────────────────────────────────────────────────

class REngineError(Exception):
    """Raised when R execution fails unrecoverably."""


# ── RStatisticalEngineTool ────────────────────────────────────────────────────

class RStatisticalEngineTool:
    """
    Execute R meta-analysis code and return structured results.

    Input params:
        r_code         : str — R code to execute
        output_vars    : List[str] — variable names to extract from R environment
                         (must be written to <results_file>.json via jsonlite)
        working_dir    : str — optional working directory for plot output files

    Output:
        results        : dict of extracted variable values from R
        stdout         : captured R console output
        plots          : list of generated plot file paths
        r_available    : bool — whether R was actually used

    If R is not available:
        Returns a structured placeholder with r_available=False and a warning.
    """

    tool_id = "r_engine"

    # Required R packages (checked at first call)
    _REQUIRED_PACKAGES = ["meta", "jsonlite"]

    def __init__(self, r_timeout: int = 120):
        self._r_timeout = r_timeout
        self._packages_checked = False

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        r_code: str = params.get("r_code", "")
        working_dir: Optional[str] = params.get("working_dir")
        expected_json: Optional[str] = params.get("expected_json")  # path to results JSON

        if not r_code:
            return ToolResult(tool_id=self.tool_id, success=False,
                               data={}, error="r_code is required")

        if not _R_AVAILABLE:
            logger.warning("R is not available; returning placeholder meta-analysis result")
            return ToolResult(
                tool_id=self.tool_id,
                success=True,
                data={
                    "r_available": False,
                    "warning": "R not installed; results are placeholder values",
                    "results": self._placeholder_results(),
                    "stdout": "",
                    "plots": [],
                },
            )

        return self._run_r(r_code, working_dir=working_dir, expected_json=expected_json)

    def _run_r(
        self,
        r_code: str,
        working_dir: Optional[str] = None,
        expected_json: Optional[str] = None,
    ) -> ToolResult:
        """Run R code in a subprocess and collect results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write R script
            script_path = pathlib.Path(tmpdir) / "analysis.R"
            results_path = pathlib.Path(tmpdir) / "results.json"

            # Prepend output path configuration to the user code
            preamble = textwrap.dedent(f"""
                options(warn = 1)  # warn immediately
                library(meta)
                library(jsonlite)
                .RESULTS_PATH <- "{str(results_path).replace(chr(92), '/')}"
                .PLOT_DIR <- "{(working_dir or tmpdir).replace(chr(92), '/')}"
                setwd(.PLOT_DIR)
            """)
            full_code = preamble + "\n" + r_code

            # Append results save if not already present
            if "write_json" not in r_code and "results" in r_code:
                full_code += textwrap.dedent("""
                    # Auto-save results if 'results' variable exists
                    if (exists("results")) {
                      write_json(results, .RESULTS_PATH, pretty=TRUE, auto_unbox=TRUE)
                    }
                """)

            script_path.write_text(full_code, encoding="utf-8")

            try:
                proc = subprocess.run(
                    ["Rscript", "--vanilla", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=self._r_timeout,
                    cwd=working_dir or tmpdir,
                )
            except subprocess.TimeoutExpired:
                return ToolResult(tool_id=self.tool_id, success=False,
                                   data={}, error=f"R execution timed out after {self._r_timeout}s")
            except Exception as exc:
                return ToolResult(tool_id=self.tool_id, success=False,
                                   data={}, error=f"R execution error: {exc}")

            stdout = proc.stdout
            stderr = proc.stderr

            if proc.returncode != 0:
                logger.error("R script failed (rc=%d):\n%s", proc.returncode, stderr[-2000:])
                return ToolResult(
                    tool_id=self.tool_id, success=False,
                    data={"stdout": stdout, "stderr": stderr[-1000:]},
                    error=f"R returned non-zero exit code {proc.returncode}",
                )

            # Parse JSON results
            results = {}
            json_to_read = expected_json or str(results_path)
            if pathlib.Path(json_to_read).exists():
                try:
                    with open(json_to_read, "r", encoding="utf-8") as fh:
                        results = json.load(fh)
                except json.JSONDecodeError as exc:
                    logger.warning("Failed to parse R results JSON: %s", exc)

            # Find generated plot files
            if working_dir:
                plots = [
                    str(p) for p in pathlib.Path(working_dir).glob("*.pdf")
                ] + [
                    str(p) for p in pathlib.Path(working_dir).glob("*.png")
                ]
            else:
                plots = []

            # Parse key statistics from stdout if results dict incomplete
            if not results:
                results = self._parse_stdout_stats(stdout)

            return ToolResult(
                tool_id=self.tool_id,
                success=True,
                data={
                    "r_available": True,
                    "results": results,
                    "stdout": stdout[-3000:],
                    "stderr_warnings": [
                        line for line in stderr.splitlines()
                        if line.strip().startswith("Warning")
                    ],
                    "plots": plots,
                },
            )

    @staticmethod
    def _parse_stdout_stats(stdout: str) -> Dict[str, Any]:
        """Extract common meta-analysis statistics from R console output."""
        results = {}
        patterns = {
            "i2": r"I\^2\s*=\s*([\d.]+)\s*%",
            "pooled_estimate": r"(?:Common|Random) effects model[^\n]*\n[^\n]*?(\d+\.?\d*)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, stdout, re.IGNORECASE)
            if m:
                try:
                    results[key] = float(m.group(1))
                except ValueError:
                    pass
        return results

    @staticmethod
    def _placeholder_results() -> Dict[str, Any]:
        """Placeholder results when R is not available."""
        return {
            "pooled_estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "i2": None,
            "tau2": None,
            "k": None,
            "note": "Placeholder — R not installed; run with R for actual results",
        }


# ── Convenience: run meta-analysis directly ───────────────────────────────────

class MetaAnalysisRunner:
    """
    High-level interface for running meta-analyses without writing raw R code.

    Selects the appropriate template (binary/continuous/generic) and
    populates placeholders from the extraction table.
    """

    def __init__(self, r_engine: Optional[RStatisticalEngineTool] = None):
        self._engine = r_engine or RStatisticalEngineTool()

    def run_continuous(
        self,
        studies: List[Dict],
        effect_measure: str = "MD",
        tau2_method: str = "REML",
        outcome_label: str = "Outcome",
        output_dir: Optional[pathlib.Path] = None,
    ) -> Dict[str, Any]:
        """Run continuous outcome meta-analysis (metacont)."""
        r_code = self._build_continuous_code(
            studies, effect_measure, tau2_method, outcome_label,
            output_dir=output_dir,
        )
        result = self._engine.execute({
            "r_code": r_code,
            "working_dir": str(output_dir) if output_dir else None,
        })
        return result.output

    def run_binary(
        self,
        studies: List[Dict],
        effect_measure: str = "OR",
        tau2_method: str = "REML",
        outcome_label: str = "Outcome",
        output_dir: Optional[pathlib.Path] = None,
    ) -> Dict[str, Any]:
        """Run binary outcome meta-analysis (metabin)."""
        r_code = self._build_binary_code(
            studies, effect_measure, tau2_method, outcome_label,
            output_dir=output_dir,
        )
        result = self._engine.execute({
            "r_code": r_code,
            "working_dir": str(output_dir) if output_dir else None,
        })
        return result.output

    def run_generic(
        self,
        studies: List[Dict],
        effect_measure: str = "HR",
        tau2_method: str = "REML",
        outcome_label: str = "Outcome",
        output_dir: Optional[pathlib.Path] = None,
    ) -> Dict[str, Any]:
        """Run generic (log-scale) meta-analysis (metagen)."""
        r_code = self._build_generic_code(
            studies, effect_measure, tau2_method, outcome_label,
            output_dir=output_dir,
        )
        result = self._engine.execute({
            "r_code": r_code,
            "working_dir": str(output_dir) if output_dir else None,
        })
        return result.output

    # ── Code builders ─────────────────────────────────────────────────────

    @staticmethod
    def _r_str_vec(values: List) -> str:
        return ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in values)

    def _build_continuous_code(
        self,
        studies: List[Dict],
        effect_measure: str,
        tau2_method: str,
        outcome_label: str,
        output_dir: Optional[pathlib.Path],
    ) -> str:
        labels = [s.get("study", s.get("pmid", f"Study_{i}")) for i, s in enumerate(studies)]
        means_e = [s.get("mean_e", s.get("mean_intervention", "NA")) for s in studies]
        sds_e   = [s.get("sd_e",   s.get("sd_intervention",   "NA")) for s in studies]
        ns_e    = [s.get("n_e",    s.get("n_intervention",    "NA")) for s in studies]
        means_c = [s.get("mean_c", s.get("mean_control",      "NA")) for s in studies]
        sds_c   = [s.get("sd_c",   s.get("sd_control",        "NA")) for s in studies]
        ns_c    = [s.get("n_c",    s.get("n_control",         "NA")) for s in studies]

        safe_label = re.sub(r"[^\w]", "_", outcome_label)
        plot_file = str(output_dir / f"{safe_label}_forest.pdf") if output_dir else f"{safe_label}_forest.pdf"

        return textwrap.dedent(f"""
            dat <- data.frame(
              study  = c({self._r_str_vec(labels)}),
              mean_e = c({self._r_str_vec(means_e)}),
              sd_e   = c({self._r_str_vec(sds_e)}),
              n_e    = c({self._r_str_vec(ns_e)}),
              mean_c = c({self._r_str_vec(means_c)}),
              sd_c   = c({self._r_str_vec(sds_c)}),
              n_c    = c({self._r_str_vec(ns_c)})
            )

            m <- metacont(
              n.e=dat$n_e, mean.e=dat$mean_e, sd.e=dat$sd_e,
              n.c=dat$n_c, mean.c=dat$mean_c, sd.c=dat$sd_c,
              studlab=dat$study, sm="{effect_measure}",
              method="{tau2_method}", hakn=TRUE, prediction=TRUE,
              title="{outcome_label}"
            )
            summary(m)

            pdf("{plot_file.replace(chr(92), '/')}")
            forest(m, sortvar=TE, prediction=TRUE, print.tau2=TRUE)
            dev.off()

            results <- list(
              pooled_estimate=m$TE.random, ci_lower=m$lower.random,
              ci_upper=m$upper.random, i2=m$I2*100, tau2=m$tau2,
              q_statistic=m$Q, q_p_value=m$pval.Q, k=m$k,
              prediction_lower=m$lower.predict, prediction_upper=m$upper.predict
            )
            write_json(results, .RESULTS_PATH, pretty=TRUE, auto_unbox=TRUE)
        """)

    def _build_binary_code(
        self,
        studies: List[Dict],
        effect_measure: str,
        tau2_method: str,
        outcome_label: str,
        output_dir: Optional[pathlib.Path],
    ) -> str:
        labels    = [s.get("study", s.get("pmid", f"Study_{i}")) for i, s in enumerate(studies)]
        events_e  = [s.get("event_e", s.get("events_intervention", "NA")) for s in studies]
        ns_e      = [s.get("n_e",     s.get("n_intervention", "NA"))      for s in studies]
        events_c  = [s.get("event_c", s.get("events_control", "NA"))      for s in studies]
        ns_c      = [s.get("n_c",     s.get("n_control", "NA"))           for s in studies]

        safe_label = re.sub(r"[^\w]", "_", outcome_label)
        plot_file = str(output_dir / f"{safe_label}_forest.pdf") if output_dir else f"{safe_label}_forest.pdf"

        return textwrap.dedent(f"""
            dat <- data.frame(
              study   = c({self._r_str_vec(labels)}),
              event_e = c({self._r_str_vec(events_e)}),
              n_e     = c({self._r_str_vec(ns_e)}),
              event_c = c({self._r_str_vec(events_c)}),
              n_c     = c({self._r_str_vec(ns_c)})
            )

            m <- metabin(
              event.e=dat$event_e, n.e=dat$n_e,
              event.c=dat$event_c, n.c=dat$n_c,
              studlab=dat$study, sm="{effect_measure}",
              method="{tau2_method}", hakn=TRUE, prediction=TRUE,
              title="{outcome_label}"
            )
            summary(m)

            pdf("{plot_file.replace(chr(92), '/')}")
            forest(m, sortvar=TE, prediction=TRUE, print.tau2=TRUE)
            dev.off()

            results <- list(
              pooled_estimate=exp(m$TE.random), ci_lower=exp(m$lower.random),
              ci_upper=exp(m$upper.random), i2=m$I2*100, tau2=m$tau2,
              q_statistic=m$Q, q_p_value=m$pval.Q, k=m$k
            )
            write_json(results, .RESULTS_PATH, pretty=TRUE, auto_unbox=TRUE)
        """)

    def _build_generic_code(
        self,
        studies: List[Dict],
        effect_measure: str,
        tau2_method: str,
        outcome_label: str,
        output_dir: Optional[pathlib.Path],
    ) -> str:
        labels = [s.get("study", s.get("pmid", f"Study_{i}")) for i, s in enumerate(studies)]
        tes    = [s.get("TE",     s.get("log_effect", "NA")) for s in studies]
        setes  = [s.get("seTE",   s.get("se_log",     "NA")) for s in studies]

        safe_label = re.sub(r"[^\w]", "_", outcome_label)
        plot_file = str(output_dir / f"{safe_label}_forest.pdf") if output_dir else f"{safe_label}_forest.pdf"
        refline = "0" if effect_measure in ("MD", "SMD") else "1"

        return textwrap.dedent(f"""
            dat <- data.frame(
              study = c({self._r_str_vec(labels)}),
              TE    = c({self._r_str_vec(tes)}),
              seTE  = c({self._r_str_vec(setes)})
            )

            m <- metagen(
              TE=dat$TE, seTE=dat$seTE, studlab=dat$study,
              sm="{effect_measure}", method="{tau2_method}",
              hakn=TRUE, prediction=TRUE, backtransf=TRUE,
              title="{outcome_label}"
            )
            summary(m)

            pdf("{plot_file.replace(chr(92), '/')}")
            forest(m, sortvar=TE, prediction=TRUE, print.tau2=TRUE, refline={refline})
            dev.off()

            results <- list(
              pooled_estimate=exp(m$TE.random), ci_lower=exp(m$lower.random),
              ci_upper=exp(m$upper.random), i2=m$I2*100, tau2=m$tau2,
              q_statistic=m$Q, q_p_value=m$pval.Q, k=m$k,
              prediction_lower=exp(m$lower.predict), prediction_upper=exp(m$upper.predict)
            )
            write_json(results, .RESULTS_PATH, pretty=TRUE, auto_unbox=TRUE)
        """)
