"""
extraction_tools.py — Tool layer components for the extraction stage.

Tools:
  FullTextFetcherTool       (D1.5) — Fetch PMC full-text XML; PDF fallback via local files
  DocumentStructureParser   (D2.1) — Parse full text into structured sections with Location IDs
  SemanticLocatorTool       (D2.2) — BM25/TF-IDF semantic search within parsed document
  PythonSandboxTool         (D3.1) — Execute Python standardization code in restricted sandbox
  SourceTraceabilityVerifier(D3.3) — Verify extracted value string exists at stated location
"""

from __future__ import annotations

import ast
import io
import json
import logging
import math
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import config
from agent_layer.base_agent import ToolResult

logger = logging.getLogger(__name__)

# ── HTTP helpers (shared with pubmed_tools) ────────────────────────────────────

def _get(url: str, params: Dict, retries: int = 3, timeout: int = 30) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


# ── FullTextFetcherTool ────────────────────────────────────────────────────────

class FullTextFetcherTool:
    """
    Fetch full-text XML for a PMC article.

    Tries:
      1. PMC OA API (efetch db=pmc)
      2. Local PDF file in paper_list/ directory (plain text extraction)
    """

    tool_id = "fulltext_fetcher"

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        pmid = str(params.get("pmid", ""))
        if not pmid:
            return ToolResult(tool_id=self.tool_id, success=False,
                               data={}, error="pmid is required")

        # Try PMC via efetch
        result = self._fetch_pmc_xml(pmid)
        if result:
            return ToolResult(tool_id=self.tool_id, success=True,
                               data={"pmid": pmid, "source": "pmc_xml",
                                       "full_text": result, "format": "xml"})

        # Fallback: local PDF text
        if config.FULL_TEXT_FALLBACK_PDF:
            result = self._fetch_local_pdf(pmid)
            if result:
                return ToolResult(tool_id=self.tool_id, success=True,
                                   data={"pmid": pmid, "source": "local_pdf",
                                           "full_text": result, "format": "text"})

        return ToolResult(tool_id=self.tool_id, success=False,
                           data={"pmid": pmid},
                           error=f"Full text not available for PMID {pmid}")

    def _fetch_pmc_xml(self, pmid: str) -> Optional[str]:
        """Fetch PMC XML via NCBI efetch API."""
        try:
            params = {
                "db": "pmc",
                "id": pmid,
                "rettype": "full",
                "retmode": "xml",
                "api_key": config.NCBI_API_KEY,
                "email": config.NCBI_EMAIL,
                "tool": "SR-Pipeline",
            }
            resp = _get(f"{config.PUBMED_BASE_URL}/efetch.fcgi", params)
            text = resp.text
            if "<article" in text or "<PubmedArticle" in text:
                return text
            return None
        except Exception as exc:
            logger.debug("PMC XML fetch failed for PMID %s: %s", pmid, exc)
            return None

    def _fetch_local_pdf(self, pmid: str) -> Optional[str]:
        """Try to find and read a local PDF file for the PMID."""
        for subdir in (config.PROJECT_ROOT / "paper_list").iterdir() if (
            config.PROJECT_ROOT / "paper_list"
        ).exists() else []:
            pdf_path = subdir / f"{pmid}.pdf"
            if pdf_path.exists():
                try:
                    # Try pdfminer or pdfplumber if available; fall back to basic
                    return self._extract_pdf_text(pdf_path)
                except Exception as exc:
                    logger.warning("PDF extraction failed for %s: %s", pdf_path, exc)
        return None

    @staticmethod
    def _extract_pdf_text(pdf_path: Path) -> str:
        """Extract text from PDF using available library."""
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(pages)
        except ImportError:
            pass

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf_path))
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            pass

        # Last resort: return placeholder
        return f"[PDF text extraction unavailable for {pdf_path.name}]"


# ── DocumentStructureParser ───────────────────────────────────────────────────

class DocumentStructureParser:
    """
    Parse full-text into a structured dict with Location IDs.

    For XML input: extracts <sec>, <title>, <p>, <table-wrap> elements.
    For plain text: splits on heuristic section headers.

    Each section gets a Location ID: e.g. "Methods", "Table_1", "Results_Para_3"
    """

    tool_id = "document_structure_parser"

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        full_text: str = params.get("full_text", "")
        fmt: str = params.get("format", "text")
        pmid: str = str(params.get("pmid", ""))

        if not full_text:
            return ToolResult(tool_id=self.tool_id, success=False,
                               data={}, error="full_text is required")

        if fmt == "xml":
            sections = self._parse_xml(full_text)
        else:
            sections = self._parse_text(full_text)

        return ToolResult(
            tool_id=self.tool_id,
            success=True,
            data={
                "pmid": pmid,
                "sections": sections,
                "section_count": len(sections),
            },
        )

    def _parse_xml(self, xml_text: str) -> List[Dict]:
        """Parse PMC XML into sections."""
        sections = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return self._parse_text(xml_text)

        # Extract body sections
        section_counter: Dict[str, int] = {}

        def _section_id(title: str) -> str:
            base = re.sub(r"\W+", "_", title.strip())[:30] or "Section"
            section_counter[base] = section_counter.get(base, 0) + 1
            return base if section_counter[base] == 1 else f"{base}_{section_counter[base]}"

        def _iter_sections(node: ET.Element, depth: int = 0) -> None:
            tag = node.tag.split("}")[-1] if "}" in node.tag else node.tag

            if tag == "sec":
                title_el = node.find(".//{*}title")
                title = title_el.text or "Section" if title_el is not None else "Section"
                text_parts = []
                for child in node.iter():
                    child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if child_tag in ("p", "label"):
                        if child.text:
                            text_parts.append(child.text.strip())
                section_text = " ".join(text_parts)
                if section_text:
                    sections.append({
                        "location_id": _section_id(title),
                        "title": title,
                        "text": section_text,
                        "depth": depth,
                        "type": "section",
                    })
                for child in node:
                    _iter_sections(child, depth + 1)

            elif tag == "table-wrap":
                table_num = len([s for s in sections if s["type"] == "table"]) + 1
                label_el = node.find(".//{*}label")
                label = label_el.text if label_el is not None else f"Table {table_num}"
                table_text = ET.tostring(node, encoding="unicode", method="text")
                sections.append({
                    "location_id": f"Table_{table_num}",
                    "title": label,
                    "text": table_text,
                    "depth": depth,
                    "type": "table",
                })

        for node in root.iter():
            node_tag = node.tag.split("}")[-1] if "}" in node.tag else node.tag
            if node_tag == "body":
                for child in node:
                    _iter_sections(child)
                break

        # Fallback: just dump all text
        if not sections:
            sections = self._parse_text(ET.tostring(root, encoding="unicode", method="text"))

        return sections

    def _parse_text(self, text: str) -> List[Dict]:
        """Heuristically split plain text into sections."""
        sections = []
        # Common section header patterns
        header_re = re.compile(
            r"^(?:ABSTRACT|INTRODUCTION|BACKGROUND|METHODS?|MATERIALS?\s+AND\s+METHODS?|"
            r"PARTICIPANTS?|RESULTS?|DISCUSSION|CONCLUSION|FUNDING|REFERENCES?|"
            r"SUPPLEMENTARY|APPENDIX|STATISTICAL\s+ANALYSIS)\b",
            re.IGNORECASE | re.MULTILINE,
        )

        parts = header_re.split(text)
        headers = ["Preamble"] + header_re.findall(text)

        table_re = re.compile(r"(Table\s+\d+[.:][^\n]*\n(?:[^\n]+\n){1,30})", re.IGNORECASE)

        for idx, (header, body) in enumerate(zip(headers, parts)):
            # Extract tables from body
            tables = table_re.findall(body)
            non_table = table_re.sub("", body)

            if non_table.strip():
                sections.append({
                    "location_id": re.sub(r"\W+", "_", header.strip())[:30],
                    "title": header.strip(),
                    "text": non_table.strip(),
                    "depth": 0,
                    "type": "section",
                })

            for t_idx, table_text in enumerate(tables):
                t_num = len([s for s in sections if s["type"] == "table"]) + 1
                sections.append({
                    "location_id": f"Table_{t_num}",
                    "title": f"Table {t_num}",
                    "text": table_text.strip(),
                    "depth": 0,
                    "type": "table",
                })

        return sections


# ── SemanticLocatorTool ────────────────────────────────────────────────────────

class SemanticLocatorTool:
    """
    Retrieve the top-K most relevant sections for a query using BM25-style scoring.

    Requires: document sections from DocumentStructureParser
    """

    tool_id = "semantic_locator"

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        query: str = params.get("query", "")
        sections: List[Dict] = params.get("sections", [])
        top_k: int = int(params.get("top_k", 5))

        if not query or not sections:
            return ToolResult(tool_id=self.tool_id, success=False,
                               data={}, error="query and sections are required")

        scored = self._bm25_score(query, sections)
        top_sections = sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]

        return ToolResult(
            tool_id=self.tool_id,
            success=True,
            data={"query": query, "results": top_sections},
        )

    def _bm25_score(self, query: str, sections: List[Dict]) -> List[Dict]:
        """Compute BM25 scores for sections against the query."""
        k1, b = 1.5, 0.75
        query_terms = re.findall(r"\w+", query.lower())

        # Compute IDF
        n = len(sections)
        df: Dict[str, int] = {}
        tokenized_docs = []
        for sec in sections:
            tokens = re.findall(r"\w+", (sec.get("text", "") + " " + sec.get("title", "")).lower())
            tokenized_docs.append(tokens)
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        avg_dl = sum(len(d) for d in tokenized_docs) / max(n, 1)

        results = []
        for idx, (sec, tokens) in enumerate(zip(sections, tokenized_docs)):
            dl = len(tokens)
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            score = 0.0
            for term in query_terms:
                if term not in df:
                    continue
                idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1)
                tf = tf_map.get(term, 0)
                tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))
                score += idf * tf_norm

            results.append({**sec, "score": round(score, 4)})

        return results


# ── PythonSandboxTool ──────────────────────────────────────────────────────────

class PythonSandboxTool:
    """
    Execute Python code for standardization calculations.

    Security model: uses ast.parse to block imports, exec, eval, open,
    and other dangerous builtins. Only math, statistics, and basic numeric
    operations are allowed.
    """

    tool_id = "python_sandbox"

    _ALLOWED_MODULES = {"math", "statistics"}
    _BLOCKED_NAMES = {"__import__", "exec", "eval", "open", "compile",
                       "globals", "locals", "vars", "getattr", "setattr",
                       "delattr", "hasattr", "__builtins__", "subprocess",
                       "os", "sys", "importlib"}

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        code: str = params.get("code", "")
        input_vars: Dict[str, Any] = params.get("input_vars", {})

        if not code:
            return ToolResult(tool_id=self.tool_id, success=False,
                               data={}, error="code is required")

        # Safety check
        safety_error = self._check_safety(code)
        if safety_error:
            return ToolResult(tool_id=self.tool_id, success=False,
                               data={}, error=f"Safety check failed: {safety_error}")

        # Execute in restricted namespace
        namespace: Dict[str, Any] = {
            "__builtins__": {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "len": len, "int": int, "float": float,
                "str": str, "bool": bool, "list": list, "dict": dict,
                "range": range, "enumerate": enumerate, "zip": zip,
                "isinstance": isinstance, "print": print,
                "None": None, "True": True, "False": False,
            },
        }

        # Inject math
        import math as _math
        namespace["math"] = _math
        namespace["sqrt"] = _math.sqrt
        namespace["log"] = _math.log
        namespace["exp"] = _math.exp

        # Inject input variables
        namespace.update(input_vars)

        try:
            exec(compile(code, "<sandbox>", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            return ToolResult(tool_id=self.tool_id, success=False,
                               data={}, error=f"Execution error: {exc}")

        # Collect output (anything in namespace that isn't a builtin or input)
        output = {
            k: v for k, v in namespace.items()
            if not k.startswith("_") and k not in input_vars
            and k not in ("math", "sqrt", "log", "exp")
            and isinstance(v, (int, float, str, bool, list, dict, type(None)))
        }
        return ToolResult(tool_id=self.tool_id, success=True, data=output)

    def _check_safety(self, code: str) -> Optional[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"Syntax error: {exc}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in self._ALLOWED_MODULES:
                        return f"Import not allowed: {alias.name}"
            if isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] not in self._ALLOWED_MODULES:
                    return f"Import not allowed: {node.module}"
            if isinstance(node, ast.Name) and node.id in self._BLOCKED_NAMES:
                return f"Blocked name: {node.id}"
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in self._BLOCKED_NAMES:
                    return f"Blocked function: {func.id}"

        return None


# ── SourceTraceabilityVerifier ────────────────────────────────────────────────

class SourceTraceabilityVerifier:
    """
    Verify that an extracted numeric value appears at the stated location_id
    in the parsed document sections.

    Returns match/partial_match/mismatch for each verification request.
    """

    tool_id = "source_traceability_verifier"

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        value = params.get("value")          # numeric value to verify
        location_id: str = params.get("location_id", "")
        sections: List[Dict] = params.get("sections", [])
        tolerance: float = float(params.get("tolerance", 0.5))

        if value is None or not location_id or not sections:
            return ToolResult(tool_id=self.tool_id, success=False,
                               data={}, error="value, location_id, and sections required")

        # Find target section
        target = next((s for s in sections if s.get("location_id") == location_id), None)

        if target is None:
            return ToolResult(
                tool_id=self.tool_id, success=True,
                data={
                    "status": "mismatch",
                    "found_at": None,
                    "reason": f"Location '{location_id}' not found in document sections",
                },
            )

        status, found_at = self._verify_in_section(value, target, sections, tolerance)
        return ToolResult(
            tool_id=self.tool_id, success=True,
            data={"status": status, "found_at": found_at},
        )

    def _verify_in_section(
        self, value: Any, target: Dict, all_sections: List[Dict], tolerance: float
    ) -> Tuple[str, Optional[str]]:
        """Check if numeric value string appears in target section text."""
        value_str = str(value)
        target_text = target.get("text", "")

        # Exact string match
        if value_str in target_text:
            return "match", target["location_id"]

        # Near-numeric match (rounding tolerance)
        try:
            v = float(value)
            # Find all numbers in section text
            numbers = [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", target_text)]
            for n in numbers:
                if abs(n - v) <= tolerance:
                    return "match", target["location_id"]
        except (ValueError, TypeError):
            pass

        # Search other sections
        for sec in all_sections:
            if sec["location_id"] == target["location_id"]:
                continue
            sec_text = sec.get("text", "")
            if value_str in sec_text:
                return "partial_match", sec["location_id"]
            try:
                v = float(value)
                numbers = [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", sec_text)]
                if any(abs(n - v) <= tolerance for n in numbers):
                    return "partial_match", sec["location_id"]
            except (ValueError, TypeError):
                pass

        return "mismatch", None


# ── Factory ───────────────────────────────────────────────────────────────────

def get_all_extraction_tools() -> List[Tool]:
    return [
        FullTextFetcherTool(),
        DocumentStructureParser(),
        SemanticLocatorTool(),
        PythonSandboxTool(),
        SourceTraceabilityVerifier(),
    ]
