"""
Tool Layer — PubMed / NCBI E-utilities Tools  (D1.1 – D1.4)

D1.1  PubMedSearchTool     — execute a boolean query, return PMID list
D1.2  MeSHValidatorTool    — validate a term against the NCBI MeSH database
D1.3  AbstractFetcherTool  — batch-fetch title + abstract + metadata by PMID
D1.4  CitationFetcherTool  — fetch references cited by a paper (backward chaining)

All tools inherit the BaseAgent.Tool ABC (tool_id + run()).
Rate limiting: NCBI allows 10 req/s with API key, 3 req/s without.
"""

from __future__ import annotations
import logging
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)


# ── Shared HTTP session ────────────────────────────────────────────────────────

def _get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "SRMASystem/1.0 (automated research tool)"})
    return session

_session = _get_session()


def _eutils_get(endpoint: str, params: Dict[str, Any], retries: int = 3) -> requests.Response:
    """GET request to an NCBI E-utilities endpoint with retry logic."""
    from config import PUBMED_BASE_URL, NCBI_API_KEY, NCBI_EMAIL
    params = {**params, "email": NCBI_EMAIL}
    if NCBI_API_KEY and NCBI_API_KEY != "YOUR_NCBI_API_KEY_HERE":
        params["api_key"] = NCBI_API_KEY
    url = f"{PUBMED_BASE_URL}/{endpoint}"
    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            logger.warning("E-utilities request failed (attempt %d): %s. Retrying in %ds", attempt, exc, wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable")


def _sleep_ncbi() -> None:
    """Respect NCBI rate limits (0.34s → ~3 req/s without key, 0.11s with key)."""
    from config import NCBI_API_KEY
    has_key = bool(NCBI_API_KEY and NCBI_API_KEY != "YOUR_NCBI_API_KEY_HERE")
    time.sleep(0.12 if has_key else 0.37)


# ── D1.1  PubMedSearchTool ─────────────────────────────────────────────────────

class PubMedSearchTool:
    """
    Execute a PubMed boolean query and return a list of PMIDs.

    Input:
        query        (str)  — PubMed boolean query string
        max_results  (int)  — max PMIDs to return (default: PUBMED_MAX_RESULTS from config)
        date_range   (str)  — optional date filter, e.g. "2000:2024"

    Output:
        {
            "pmids":        [str, ...],
            "total_count":  int,
            "query_used":   str,
        }
    """

    tool_id = "pubmed_search"

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        from config import PUBMED_MAX_RESULTS, PUBMED_RETMAX
        query = input_data["query"]
        max_results = input_data.get("max_results", PUBMED_MAX_RESULTS)
        date_range = input_data.get("date_range")

        if date_range:
            query = f"({query}) AND {date_range}[dp]"

        # Step 1: esearch to get total count and first batch of IDs
        params: Dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, PUBMED_RETMAX),
            "retmode": "json",
            "usehistory": "y",
        }
        _sleep_ncbi()
        resp = _eutils_get("esearch.fcgi", params)
        data = resp.json()
        esearch = data.get("esearchresult", {})
        total_count = int(esearch.get("count", 0))
        webenv = esearch.get("webenv", "")
        query_key = esearch.get("querykey", "")
        pmids: List[str] = esearch.get("idlist", [])

        # Step 2: if total > first batch, fetch remaining in pages
        retmax = min(max_results, PUBMED_RETMAX)
        retstart = retmax
        while len(pmids) < min(total_count, max_results):
            _sleep_ncbi()
            page_params: Dict[str, Any] = {
                "db": "pubmed",
                "query_key": query_key,
                "WebEnv": webenv,
                "retstart": retstart,
                "retmax": retmax,
                "retmode": "json",
            }
            page_resp = _eutils_get("esearch.fcgi", page_params)
            page_data = page_resp.json()
            batch = page_data.get("esearchresult", {}).get("idlist", [])
            if not batch:
                break
            pmids.extend(batch)
            retstart += retmax

        pmids = pmids[:max_results]
        logger.info(
            "PubMed search: total=%d, fetched=%d, query=%s",
            total_count, len(pmids), query[:80],
        )
        return {
            "pmids": pmids,
            "total_count": total_count,
            "query_used": query,
        }


# ── D1.2  MeSHValidatorTool ────────────────────────────────────────────────────

class MeSHValidatorTool:
    """
    Validate a list of candidate MeSH terms via the NCBI MeSH database.

    Input:
        terms   (list[str])  — candidate MeSH term strings to validate
        pico_map (dict)      — {term: dimension} mapping for context

    Output:
        {
            "results": [
                {
                    "term":         str,
                    "mesh_status":  "valid_mesh" | "mapped" | "not_found",
                    "mapped_to":    str | null,      # standard MeSH heading if mapped
                    "search_field": "[MeSH Terms]" | "[tiab]",
                    "entry_terms":  [str, ...],      # Entry Terms from MeSH record
                    "mesh_uid":     str | null,
                },
                ...
            ]
        }
    """

    tool_id = "mesh_validator"

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        terms: List[str] = input_data.get("terms", [])
        results = []
        for term in terms:
            result = self._validate_single(term)
            results.append(result)
            _sleep_ncbi()
        return {"results": results}

    def _validate_single(self, term: str) -> Dict[str, Any]:
        """Check if `term` is a valid MeSH term. Returns status dict."""
        # Search the MeSH database for the exact term
        params = {
            "db": "mesh",
            "term": f'"{term}"[MeSH Terms]',
            "retmax": 5,
            "retmode": "json",
        }
        try:
            resp = _eutils_get("esearch.fcgi", params)
            data = resp.json().get("esearchresult", {})
            count = int(data.get("count", 0))
            ids = data.get("idlist", [])

            if count > 0 and ids:
                # Fetch the MeSH record to get Entry Terms
                entry_terms, mesh_heading = self._fetch_mesh_record(ids[0])
                # Check if the term matches the heading exactly (case-insensitive)
                if mesh_heading and mesh_heading.lower() == term.lower():
                    return {
                        "term": term,
                        "mesh_status": "valid_mesh",
                        "mapped_to": None,
                        "search_field": "[MeSH Terms]",
                        "entry_terms": entry_terms,
                        "mesh_uid": ids[0],
                    }
                elif mesh_heading:
                    # term is an Entry Term → maps to a standard heading
                    return {
                        "term": term,
                        "mesh_status": "mapped",
                        "mapped_to": mesh_heading,
                        "search_field": "[MeSH Terms]",
                        "entry_terms": entry_terms,
                        "mesh_uid": ids[0],
                    }

            # Try searching by term as Entry Term (broader search)
            params2 = {
                "db": "mesh",
                "term": f'"{term}"',
                "retmax": 5,
                "retmode": "json",
            }
            _sleep_ncbi()
            resp2 = _eutils_get("esearch.fcgi", params2)
            data2 = resp2.json().get("esearchresult", {})
            ids2 = data2.get("idlist", [])
            if ids2:
                entry_terms2, mesh_heading2 = self._fetch_mesh_record(ids2[0])
                return {
                    "term": term,
                    "mesh_status": "mapped",
                    "mapped_to": mesh_heading2 or term,
                    "search_field": "[MeSH Terms]",
                    "entry_terms": entry_terms2,
                    "mesh_uid": ids2[0],
                }

            # Not found in MeSH
            return {
                "term": term,
                "mesh_status": "not_found",
                "mapped_to": None,
                "search_field": "[tiab]",
                "entry_terms": [],
                "mesh_uid": None,
            }
        except Exception as exc:
            logger.error("MeSH validation error for '%s': %s", term, exc)
            return {
                "term": term,
                "mesh_status": "not_found",
                "mapped_to": None,
                "search_field": "[tiab]",
                "entry_terms": [],
                "mesh_uid": None,
            }

    def _fetch_mesh_record(self, mesh_uid: str) -> tuple[List[str], Optional[str]]:
        """Fetch a MeSH record and extract the Heading Name and Entry Terms."""
        params = {
            "db": "mesh",
            "id": mesh_uid,
            "retmode": "xml",
        }
        try:
            _sleep_ncbi()
            resp = _eutils_get("efetch.fcgi", params)
            root = ET.fromstring(resp.text)
            # MeSH XML: DescriptorRecord > DescriptorName > String
            heading = None
            entry_terms: List[str] = []

            heading_el = root.find(".//DescriptorName/String")
            if heading_el is not None:
                heading = heading_el.text

            # Entry Terms are in ConceptList > Concept > TermList > Term > String
            for term_el in root.findall(".//Term/String"):
                t = term_el.text
                if t and t != heading:
                    entry_terms.append(t)

            return entry_terms[:10], heading  # cap at 10 entry terms
        except Exception as exc:
            logger.warning("Failed to fetch MeSH record %s: %s", mesh_uid, exc)
            return [], None


# ── D1.3  AbstractFetcherTool ──────────────────────────────────────────────────

class AbstractFetcherTool:
    """
    Batch-fetch title, abstract, and metadata for a list of PMIDs.

    Input:
        pmids       (list[str])  — PMIDs to fetch
        batch_size  (int)        — fetch batch size (default: 200)

    Output:
        {
            "records": [
                {
                    "pmid":          str,
                    "title":         str,
                    "abstract":      str,
                    "authors":       [str, ...],
                    "journal":       str,
                    "year":          str,
                    "pub_types":     [str, ...],
                    "mesh_headings": [str, ...],
                    "pmc_id":        str | null,
                    "doi":           str | null,
                },
                ...
            ],
            "failed_pmids": [str, ...]
        }
    """

    tool_id = "abstract_fetcher"

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pmids: List[str] = input_data.get("pmids", [])
        batch_size: int = input_data.get("batch_size", 200)

        records = []
        failed: List[str] = []

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            try:
                batch_records, batch_failed = self._fetch_batch(batch)
                records.extend(batch_records)
                failed.extend(batch_failed)
            except Exception as exc:
                logger.error("Batch fetch failed for PMIDs %s-%s: %s", batch[0], batch[-1], exc)
                failed.extend(batch)
            _sleep_ncbi()

        logger.info(
            "AbstractFetcher: fetched=%d, failed=%d from %d PMIDs",
            len(records), len(failed), len(pmids),
        )
        return {"records": records, "failed_pmids": failed}

    def _fetch_batch(self, pmids: List[str]) -> tuple[List[Dict], List[str]]:
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        resp = _eutils_get("efetch.fcgi", params)
        return self._parse_pubmed_xml(resp.text, pmids)

    def _parse_pubmed_xml(
        self, xml_text: str, requested_pmids: List[str]
    ) -> tuple[List[Dict], List[str]]:
        records = []
        found_pmids: set = set()

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.error("XML parse error: %s", exc)
            return [], requested_pmids

        for article in root.findall(".//PubmedArticle"):
            try:
                record = self._parse_article(article)
                if record:
                    records.append(record)
                    found_pmids.add(record["pmid"])
            except Exception as exc:
                logger.warning("Failed to parse article: %s", exc)

        failed = [p for p in requested_pmids if p not in found_pmids]
        return records, failed

    def _parse_article(self, article: ET.Element) -> Optional[Dict]:
        # PMID
        pmid_el = article.find(".//PMID")
        if pmid_el is None:
            return None
        pmid = pmid_el.text or ""

        # Title
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""

        # Abstract — handle structured abstracts with multiple AbstractText elements
        abstract_parts = []
        for ab_el in article.findall(".//AbstractText"):
            label = ab_el.get("Label", "")
            text = "".join(ab_el.itertext())
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName", "")
            first = author.findtext("ForeName", "")
            if last:
                authors.append(f"{last} {first}".strip())

        # Journal + Year
        journal = article.findtext(".//Journal/Title", "") or article.findtext(".//MedlineJournalInfo/MedlineTA", "")
        year = article.findtext(".//PubDate/Year", "") or article.findtext(".//PubDate/MedlineDate", "")[:4] if article.findtext(".//PubDate/MedlineDate") else ""

        # Publication types
        pub_types = [pt.text for pt in article.findall(".//PublicationType") if pt.text]

        # MeSH headings
        mesh_headings = [
            mh.findtext("DescriptorName", "")
            for mh in article.findall(".//MeshHeading")
        ]

        # PMC ID and DOI
        pmc_id = None
        doi = None
        for id_el in article.findall(".//ArticleId"):
            id_type = id_el.get("IdType", "")
            if id_type == "pmc":
                pmc_id = id_el.text
            elif id_type == "doi":
                doi = id_el.text

        return {
            "pmid": pmid,
            "title": title.strip(),
            "abstract": abstract.strip(),
            "authors": authors[:6],  # Cap at 6 authors
            "journal": journal,
            "year": year,
            "pub_types": pub_types,
            "mesh_headings": [mh for mh in mesh_headings if mh],
            "pmc_id": pmc_id,
            "doi": doi,
        }


# ── D1.4  CitationFetcherTool ──────────────────────────────────────────────────

class CitationFetcherTool:
    """
    Fetch the references cited BY a list of papers (backward citation chaining).
    Uses NCBI elink with linkname=pubmed_pubmed_refs.

    Input:
        pmids       (list[str])  — PMIDs of seed papers
        batch_size  (int)        — elink batch size (default: 50)

    Output:
        {
            "citation_map": {pmid: [cited_pmid, ...], ...},
            "all_cited_pmids": [str, ...],  # union, deduplicated
            "failed_pmids":    [str, ...],
        }
    """

    tool_id = "citation_fetcher"

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pmids: List[str] = input_data.get("pmids", [])
        batch_size: int = input_data.get("batch_size", 50)

        citation_map: Dict[str, List[str]] = {}
        failed: List[str] = []
        all_cited: set = set()

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            try:
                batch_map, batch_failed = self._fetch_refs_batch(batch)
                citation_map.update(batch_map)
                failed.extend(batch_failed)
                for refs in batch_map.values():
                    all_cited.update(refs)
            except Exception as exc:
                logger.error("Citation fetch error for batch %s: %s", batch[:3], exc)
                failed.extend(batch)
            _sleep_ncbi()

        logger.info(
            "CitationFetcher: processed=%d seeds, total_cited=%d unique refs, failed=%d",
            len(pmids), len(all_cited), len(failed),
        )
        return {
            "citation_map": citation_map,
            "all_cited_pmids": sorted(all_cited),
            "failed_pmids": failed,
        }

    def _fetch_refs_batch(
        self, pmids: List[str]
    ) -> tuple[Dict[str, List[str]], List[str]]:
        """Fetch references for a batch via elink."""
        params: Dict[str, Any] = {
            "dbfrom": "pubmed",
            "db":     "pubmed",
            "id":     pmids,  # requests will repeat the param
            "linkname": "pubmed_pubmed_refs",
            "retmode": "xml",
        }
        resp = _eutils_get_multi("elink.fcgi", params)
        return self._parse_elink_xml(resp.text, pmids)

    def _parse_elink_xml(
        self, xml_text: str, requested_pmids: List[str]
    ) -> tuple[Dict[str, List[str]], List[str]]:
        citation_map: Dict[str, List[str]] = {}
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.error("Elink XML parse error: %s", exc)
            return {}, requested_pmids

        # Each <LinkSet> corresponds to one source PMID
        for linkset in root.findall(".//LinkSet"):
            id_el = linkset.find(".//IdList/Id")
            if id_el is None:
                continue
            source_pmid = id_el.text or ""
            ref_pmids = [
                el.text
                for el in linkset.findall(".//LinkSetDb/Link/Id")
                if el.text
            ]
            citation_map[source_pmid] = ref_pmids

        found = set(citation_map.keys())
        failed = [p for p in requested_pmids if p not in found]
        return citation_map, failed


def _eutils_get_multi(endpoint: str, params: Dict[str, Any], retries: int = 3) -> requests.Response:
    """
    E-utilities GET that supports multi-value parameters (e.g., multiple id= values).
    Uses requests with a list value for the 'id' param.
    """
    from config import PUBMED_BASE_URL, NCBI_API_KEY, NCBI_EMAIL
    p = {k: v for k, v in params.items() if k != "id"}
    p["email"] = NCBI_EMAIL
    if NCBI_API_KEY and NCBI_API_KEY != "YOUR_NCBI_API_KEY_HERE":
        p["api_key"] = NCBI_API_KEY

    # Build URL manually to handle multiple 'id' values
    base = f"{PUBMED_BASE_URL}/{endpoint}"
    id_list = params.get("id", [])
    query_string = urlencode(p) + "".join(f"&id={pid}" for pid in id_list)
    url = f"{base}?{query_string}"

    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(url, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


# ── Tool registry helper ───────────────────────────────────────────────────────

def get_all_pubmed_tools() -> List[Any]:
    """Return instances of all PubMed tools, ready to register with an Agent."""
    return [
        PubMedSearchTool(),
        MeSHValidatorTool(),
        AbstractFetcherTool(),
        CitationFetcherTool(),
    ]
