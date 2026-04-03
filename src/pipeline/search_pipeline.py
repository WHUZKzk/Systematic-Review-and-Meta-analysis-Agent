"""
Search Pipeline — Stage 1: Literature Retrieval

Implements the 8-step search flow defined in the PRISMA_2020_therapeutic Meta-Skill:

  search_step_1  pico_term_generation      LLM  → raw PICO term table
  search_step_2  mesh_validation_protocol  Code → annotated term table (API)
  search_step_3  boolean_query_construction Code → PubMed boolean query (rule-based)
  search_step_4  (no skill)                Code → PubMed search + abstract fetch
  search_step_5  search_augmentation       LLM  → augmented terms
  search_step_6  search_self_check         Code → self-check report + final query
  search_step_7  reference_chaining        Code → citation pool (backward chaining)
  search_step_8  (no skill)                Code → merge + deduplicate → candidate pool

Output (SearchResult):
    candidate_pool   — list of {pmid, title, abstract, ...} dicts
    search_report    — PRISMA-S compatible search summary
    execution_log    — full audit log
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_layer import ExecutorAgent, StepSpec, TaskInstruction
from llm_backend import get_registry
from skill_layer import get_loader
from tool_layer.pubmed_tools import (
    PubMedSearchTool,
    MeSHValidatorTool,
    AbstractFetcherTool,
    CitationFetcherTool,
)

logger = logging.getLogger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    candidate_pool: List[Dict[str, Any]]         # [{pmid, title, abstract, ...}, ...]
    search_report: Dict[str, Any]                # PRISMA-S data
    pico_terms: Dict[str, Any]                   # Final validated PICO terms
    main_query: str                              # Final PubMed query used
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class SearchPipeline:
    """
    Orchestrates the full search stage.

    Args:
        review_question: dict with keys P, I, C, O (PICO natural language description)
        llm_backend_name: name of LLM backend to use (default: "default")
        top_k_for_augmentation: how many top abstracts to send to search_augmentation LLM
        max_pubmed_results: cap on PubMed search results
        save_dir: if set, intermediate results are saved here for inspection

    Example:
        pipeline = SearchPipeline(review_question={"P": "...", "I": "...", "O": "..."})
        result = pipeline.run()
    """

    def __init__(
        self,
        review_question: Dict[str, str],
        llm_backend_name: str = "default",
        top_k_for_augmentation: int = 10,
        max_pubmed_results: int = 5000,
        save_dir: Optional[Path] = None,
    ):
        self.review_question = review_question
        self.top_k = top_k_for_augmentation
        self.max_pubmed_results = max_pubmed_results
        self.save_dir = save_dir

        # Build agent with tools
        llm = get_registry().get(llm_backend_name)
        self._agent = ExecutorAgent(llm=llm, skill_loader=get_loader())

        tools = [
            PubMedSearchTool(),
            MeSHValidatorTool(),
            AbstractFetcherTool(),
            CitationFetcherTool(),
        ]
        self._agent.register_tools(tools)

        # Register code-step handlers
        self._agent.register_step_handlers({
            "search_step_2": self._handle_mesh_validation,
            "search_step_3": self._handle_boolean_query_construction,
            "search_step_4": self._handle_pubmed_search,
            "search_step_6": self._handle_self_check,
            "search_step_7": self._handle_reference_chaining,
            "search_step_8": self._handle_merge_and_report,
        })

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> SearchResult:
        """Execute the full 8-step search pipeline."""
        logger.info("SearchPipeline: starting for PICO question: %s", str(self.review_question)[:120])

        step_sequence = [
            StepSpec("search_step_1", "search.pico_term_generation",       True,  [],                    "生成PICO四维度检索术语"),
            StepSpec("search_step_2", "search.mesh_validation_protocol",   False, ["mesh_validator"],     "MeSH术语API批量验证"),
            StepSpec("search_step_3", "search.boolean_query_construction",  False, [],                    "组装PubMed布尔检索式"),
            StepSpec("search_step_4", None,                                 False, ["pubmed_search", "abstract_fetcher"], "执行PubMed检索+获取摘要"),
            StepSpec("search_step_5", "search.search_augmentation",         True,  [],                    "PICO维度感知定向增强"),
            StepSpec("search_step_6", "search.search_self_check",           False, ["mesh_validator"],    "合并增强词+自检"),
            StepSpec("search_step_7", "search.reference_chaining",          False, ["citation_fetcher"],  "后向引文追溯"),
            StepSpec("search_step_8", None,                                 False, [],                    "最终合并去重+报告"),
        ]

        task = TaskInstruction(
            stage="search",
            step_id="search_full",
            skill_ids=[s.skill_id for s in step_sequence if s.skill_id],
            input_data={
                "step_sequence": step_sequence,
                "shared_context": {
                    "review_question": self.review_question,
                    "_top_k": self.top_k,
                    "_max_results": self.max_pubmed_results,
                },
            },
        )

        stage_output = self._agent.execute(task)

        if not stage_output.success:
            raise RuntimeError(f"SearchPipeline failed: {stage_output.error}")

        ctx = stage_output.data
        candidate_pool = ctx.get("search_step_8", {}).get("candidate_pool", [])
        search_report = ctx.get("search_step_8", {}).get("search_report", {})
        pico_terms = ctx.get("search_step_1", {}).get("pico_terms", {})
        main_query = ctx.get("search_step_6", {}).get("final_query", "") or \
                     ctx.get("search_step_3", {}).get("main_query", "")

        result = SearchResult(
            candidate_pool=candidate_pool,
            search_report=search_report,
            pico_terms=pico_terms,
            main_query=main_query,
            execution_log=self._agent.dump_log(),
            warnings=stage_output.warnings,
        )

        if self.save_dir:
            self._save_result(result)

        logger.info(
            "SearchPipeline: done. Candidate pool size=%d",
            len(candidate_pool),
        )
        return result

    # ── Step 2: MeSH Validation (code) ───────────────────────────────────────

    def _handle_mesh_validation(
        self, step_context: Dict, agent: ExecutorAgent
    ) -> Dict[str, Any]:
        shared = step_context["shared_context"]
        pico_output = shared.get("search_step_1", {})
        pico_terms = pico_output.get("pico_terms", {})

        # Collect all mesh_candidates across all PICO dimensions
        all_candidates: List[str] = []
        candidate_to_dim: Dict[str, str] = {}
        for dim, dim_data in pico_terms.items():
            if isinstance(dim_data, dict):
                for candidate in dim_data.get("mesh_candidates", []):
                    if candidate and candidate not in all_candidates:
                        all_candidates.append(candidate)
                        candidate_to_dim[candidate] = dim

        if not all_candidates:
            logger.warning("No MeSH candidates to validate.")
            return {"annotated_terms": {dim: {"all_terms": []} for dim in pico_terms}}

        # Run MeSH validator tool
        validator_result = agent.call_tool(
            "mesh_validator", {"terms": all_candidates}
        )

        validated: Dict[str, Dict] = {dim: {"all_terms": []} for dim in pico_terms}

        if validator_result.success:
            for r in validator_result.data.get("results", []):
                dim = candidate_to_dim.get(r["term"], "population")
                validated[dim]["all_terms"].append(r)

        # Also add non-mesh terms (core_terms + synonyms) as free-text
        for dim, dim_data in pico_terms.items():
            if not isinstance(dim_data, dict):
                continue
            existing_terms = {t["term"] for t in validated[dim]["all_terms"]}
            for term in dim_data.get("core_terms", []) + dim_data.get("synonyms", []):
                if term and term not in existing_terms:
                    validated[dim]["all_terms"].append({
                        "term": term,
                        "mesh_status": "free_text",
                        "mapped_to": None,
                        "search_field": "[tiab]",
                        "entry_terms": [],
                        "mesh_uid": None,
                    })
                    existing_terms.add(term)

        return {"annotated_terms": validated}

    # ── Step 3: Boolean Query Construction (code) ─────────────────────────────

    def _handle_boolean_query_construction(
        self, step_context: Dict, agent: ExecutorAgent
    ) -> Dict[str, Any]:
        shared = step_context["shared_context"]
        annotated = shared.get("search_step_2", {}).get("annotated_terms", {})

        blocks: Dict[str, str] = {}
        for dim, dim_data in annotated.items():
            terms = dim_data.get("all_terms", [])
            parts = []
            for t in terms:
                term = t["term"]
                field = t.get("search_field", "[tiab]")
                if field == "[MeSH Terms]":
                    parts.append(f'"{term}"[MeSH Terms]')
                else:
                    parts.append(f'"{term}"[tiab]')
            if parts:
                blocks[f"{dim}_block"] = " OR ".join(parts)

        # Build main query: (P) AND (I) — O is optional to maximize recall
        required = []
        for dim in ["population", "intervention"]:
            blk = blocks.get(f"{dim}_block", "")
            if blk:
                required.append(f"({blk})")

        outcome_block = blocks.get("outcome_block", "")
        if outcome_block:
            required.append(f"({outcome_block})")

        main_query = " AND\n".join(required) if required else ""

        return {
            "main_query": main_query,
            "dimension_blocks": blocks,
            "supplementary_query": "",
        }

    # ── Step 4: PubMed Search + Abstract Fetch (code) ─────────────────────────

    def _handle_pubmed_search(
        self, step_context: Dict, agent: ExecutorAgent
    ) -> Dict[str, Any]:
        shared = step_context["shared_context"]
        main_query = shared.get("search_step_3", {}).get("main_query", "")
        max_results = shared.get("_max_results", 5000)
        top_k = shared.get("_top_k", 10)

        if not main_query:
            raise ValueError("main_query is empty — cannot execute PubMed search")

        # Execute search
        search_result = agent.call_tool(
            "pubmed_search",
            {"query": main_query, "max_results": max_results}
        )
        if not search_result.success:
            raise RuntimeError(f"PubMed search failed: {search_result.error}")

        pmids: List[str] = search_result.data["pmids"]
        total_count: int = search_result.data["total_count"]

        # Fetch abstracts for Top-K (for augmentation) + all for pool
        # Fetch all in batches for the full pool
        all_fetch = agent.call_tool("abstract_fetcher", {"pmids": pmids})
        if not all_fetch.success:
            logger.warning("Abstract fetch partially failed: %s", all_fetch.error)

        records: List[Dict] = all_fetch.data.get("records", []) if all_fetch.success else []

        # Top-K by index (first k are most relevant in default PubMed ranking)
        top_k_abstracts = records[:top_k]

        return {
            "initial_pmids": pmids,
            "total_count": total_count,
            "initial_pool": records,
            "top_k_abstracts": top_k_abstracts,
            "search_date": date.today().isoformat(),
        }

    # ── Step 6: Self-Check + Augmented Query (code) ────────────────────────────

    def _handle_self_check(
        self, step_context: Dict, agent: ExecutorAgent
    ) -> Dict[str, Any]:
        shared = step_context["shared_context"]
        step3 = shared.get("search_step_3", {})
        step4 = shared.get("search_step_4", {})
        step5 = shared.get("search_step_5", {})
        step2 = shared.get("search_step_2", {})
        annotated = step2.get("annotated_terms", {})

        checks = []
        warnings: List[str] = []

        # Check 1: Term balance
        term_counts = {}
        for dim, dim_data in annotated.items():
            term_counts[dim] = len(dim_data.get("all_terms", []))

        p_count = term_counts.get("population", 0)
        i_count = term_counts.get("intervention", 0)
        o_count = term_counts.get("outcome", 0)

        checks.append({
            "check_id": "term_balance",
            "passed": p_count >= 3 and i_count >= 3,
            "severity": "hard",
            "message": f"P={p_count}词, I={i_count}词, O={o_count}词",
        })
        if p_count < 3 or i_count < 3:
            warnings.append(f"术语平衡性不足: P={p_count}, I={i_count}")

        # Check 2: MeSH coverage
        def has_valid_mesh(dim: str) -> bool:
            return any(
                t.get("mesh_status") == "valid_mesh"
                for t in annotated.get(dim, {}).get("all_terms", [])
            )
        p_mesh = has_valid_mesh("population")
        i_mesh = has_valid_mesh("intervention")
        checks.append({
            "check_id": "mesh_coverage",
            "passed": p_mesh and i_mesh,
            "severity": "soft",
            "message": f"P MeSH覆盖={'是' if p_mesh else '否'}, I MeSH覆盖={'是' if i_mesh else '否'}",
        })
        if not (p_mesh and i_mesh):
            warnings.append("P或I维度缺少有效MeSH词")

        # Check 3: Result count sanity
        total_count = step4.get("total_count", 0)
        count_ok = 50 <= total_count <= 50000
        checks.append({
            "check_id": "result_count_sanity",
            "passed": count_ok,
            "severity": "soft",
            "message": f"初步检索结果数={total_count}",
        })
        if total_count < 50:
            warnings.append("检索结果过少（<50），可能存在漏检")
        elif total_count > 50000:
            warnings.append("检索结果过多（>50000），检索式可能过于宽泛")

        # Build augmented final query by incorporating step 5 additions
        augmented_terms = step5.get("augmented_terms", {})
        final_query = step3.get("main_query", "")
        dimension_blocks = dict(step3.get("dimension_blocks", {}))

        if augmented_terms:
            for dim_key in ["population_additions", "intervention_additions", "outcome_additions"]:
                additions = augmented_terms.get(dim_key, [])
                if not additions:
                    continue
                dim = dim_key.replace("_additions", "")
                blk_key = f"{dim}_block"
                existing_block = dimension_blocks.get(blk_key, "")
                extra = " OR ".join(f'"{t}"[tiab]' for t in additions)
                if existing_block and extra:
                    dimension_blocks[blk_key] = f"{existing_block} OR {extra}"

            # Rebuild main query with augmented blocks
            required = []
            for dim in ["population", "intervention", "outcome"]:
                blk = dimension_blocks.get(f"{dim}_block", "")
                if blk:
                    required.append(f"({blk})")
            final_query = " AND\n".join(required) if required else final_query

        self_check_passed = all(c["passed"] for c in checks if c["severity"] == "hard")

        return {
            "self_check_passed": self_check_passed,
            "checks": checks,
            "final_query": final_query,
            "augmented_dimension_blocks": dimension_blocks,
            "_warnings": warnings,
        }

    # ── Step 7: Reference Chaining (code) ─────────────────────────────────────

    def _handle_reference_chaining(
        self, step_context: Dict, agent: ExecutorAgent
    ) -> Dict[str, Any]:
        from config import CITATION_CO_CITE_THRESHOLD
        shared = step_context["shared_context"]
        step4 = shared.get("search_step_4", {})

        initial_pmids: List[str] = step4.get("initial_pmids", [])
        initial_pool_records: List[Dict] = step4.get("initial_pool", [])
        initial_pool_set = set(initial_pmids)

        if not initial_pmids:
            return {
                "chaining_pool": [],
                "chaining_stats": {
                    "seed_count": 0, "references_found": 0,
                    "total_refs": 0, "above_threshold": 0, "new_candidates": 0,
                },
            }

        # Select seed studies: Top-20 by index (PubMed relevance ranking)
        seed_pmids = initial_pmids[:20]

        # Fetch references for seeds
        chain_result = agent.call_tool("citation_fetcher", {"pmids": seed_pmids})
        if not chain_result.success:
            logger.warning("Citation fetch failed: %s", chain_result.error)
            return {
                "chaining_pool": [],
                "chaining_stats": {
                    "seed_count": len(seed_pmids), "references_found": 0,
                    "total_refs": 0, "above_threshold": 0, "new_candidates": 0,
                },
            }

        citation_map: Dict[str, List[str]] = chain_result.data.get("citation_map", {})

        # Count co-citations
        co_cite: Dict[str, int] = {}
        total_refs = 0
        references_found = 0
        for seed_pmid, refs in citation_map.items():
            if refs:
                references_found += 1
                total_refs += len(refs)
                for ref_pmid in refs:
                    co_cite[ref_pmid] = co_cite.get(ref_pmid, 0) + 1

        # Filter by threshold and remove already known PMIDs
        above_threshold = [
            pmid for pmid, count in co_cite.items()
            if count >= CITATION_CO_CITE_THRESHOLD
        ]
        chaining_pool = [p for p in above_threshold if p not in initial_pool_set]

        return {
            "chaining_pool": chaining_pool,
            "chaining_stats": {
                "seed_count": len(seed_pmids),
                "references_found": references_found,
                "total_refs": total_refs,
                "above_threshold": len(above_threshold),
                "new_candidates": len(chaining_pool),
            },
        }

    # ── Step 8: Merge + Report (code) ─────────────────────────────────────────

    def _handle_merge_and_report(
        self, step_context: Dict, agent: ExecutorAgent
    ) -> Dict[str, Any]:
        shared = step_context["shared_context"]
        step4 = shared.get("search_step_4", {})
        step6 = shared.get("search_step_6", {})
        step7 = shared.get("search_step_7", {})
        step1 = shared.get("search_step_1", {})

        initial_pool: List[Dict] = step4.get("initial_pool", [])
        chaining_pool_pmids: List[str] = step7.get("chaining_pool", [])
        chaining_stats: Dict = step7.get("chaining_stats", {})
        search_date: str = step4.get("search_date", date.today().isoformat())

        # Fetch abstracts for chaining pool items not already in initial pool
        initial_pmid_set = {r["pmid"] for r in initial_pool}
        new_chain_pmids = [p for p in chaining_pool_pmids if p not in initial_pmid_set]

        chaining_records: List[Dict] = []
        if new_chain_pmids:
            fetch_result = agent.call_tool("abstract_fetcher", {"pmids": new_chain_pmids})
            if fetch_result.success:
                chaining_records = fetch_result.data.get("records", [])
            for rec in chaining_records:
                rec["source_tag"] = "citation_chaining"

        # Tag initial pool records
        for rec in initial_pool:
            if "source_tag" not in rec:
                rec["source_tag"] = "pubmed_search"

        # Merge + deduplicate by PMID
        candidate_pool_map: Dict[str, Dict] = {}
        for rec in initial_pool + chaining_records:
            pmid = rec.get("pmid", "")
            if pmid and pmid not in candidate_pool_map:
                candidate_pool_map[pmid] = rec

        candidate_pool = list(candidate_pool_map.values())

        # Build search report (PRISMA-S compatible)
        search_report = {
            "databases_searched": ["PubMed"],
            "search_date": search_date,
            "initial_count": step4.get("total_count", len(initial_pool)),
            "initial_fetched": len(initial_pool),
            "chaining_count": chaining_stats.get("new_candidates", 0),
            "total_before_dedup": len(initial_pool) + len(chaining_records),
            "dedup_removed": (len(initial_pool) + len(chaining_records)) - len(candidate_pool),
            "final_candidate_count": len(candidate_pool),
            "self_check_passed": step6.get("self_check_passed", False),
            "self_check_warnings": step6.get("_warnings", []),
            "chaining_stats": chaining_stats,
            "final_query": step6.get("final_query", ""),
            "pico_terms_summary": {
                dim: len(dim_data.get("all_terms", []))
                for dim, dim_data in shared.get("search_step_2", {}).get("annotated_terms", {}).items()
            },
        }

        return {
            "candidate_pool": candidate_pool,
            "search_report": search_report,
        }

    # ── Save helper ───────────────────────────────────────────────────────────

    def _save_result(self, result: SearchResult) -> None:
        if not self.save_dir:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)

        with open(self.save_dir / "candidate_pool.json", "w", encoding="utf-8") as f:
            json.dump(result.candidate_pool, f, ensure_ascii=False, indent=2)
        with open(self.save_dir / "search_report.json", "w", encoding="utf-8") as f:
            json.dump(result.search_report, f, ensure_ascii=False, indent=2)
        with open(self.save_dir / "pico_terms.json", "w", encoding="utf-8") as f:
            json.dump(result.pico_terms, f, ensure_ascii=False, indent=2)
        with open(self.save_dir / "execution_log.json", "w", encoding="utf-8") as f:
            json.dump(result.execution_log, f, ensure_ascii=False, indent=2)

        logger.info("SearchPipeline: results saved to %s", self.save_dir)


# ── Convenience function ──────────────────────────────────────────────────────

def run_search(
    pico: Dict[str, str],
    save_dir: Optional[Path] = None,
    llm_backend: str = "default",
    max_results: int = 5000,
) -> SearchResult:
    """
    Convenience wrapper to run the search pipeline from a PICO dict.

    Args:
        pico: {"P": "...", "I": "...", "C": "...", "O": "..."}
        save_dir: optional directory to persist results
        llm_backend: LLM backend name (from BackendRegistry)
        max_results: max PubMed results to retrieve

    Returns:
        SearchResult with candidate_pool, search_report, pico_terms, main_query
    """
    pipeline = SearchPipeline(
        review_question=pico,
        llm_backend_name=llm_backend,
        max_pubmed_results=max_results,
        save_dir=save_dir,
    )
    return pipeline.run()
