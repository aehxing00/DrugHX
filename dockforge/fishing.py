
"""Reverse target fishing through public ChEMBL bioactivity data.

The module answers a simple question: given a small molecule, which protein
targets have reported bioactivity?  To make the result useful for network
pharmacology, the search deliberately widens the chemical neighbourhood:

* exact ChEMBL match of the query (if any) is always included;
* similarity thresholds are lowered 80 -> 70 -> 60 -> 50 until enough
  analogues are collected;
* activity records are accepted even without a pChEMBL value (the value is
  derived from standard_value/standard_units when possible);
* drug mechanism records are merged as additional target evidence;
* protein complexes/families contribute all of their gene components.

Everything is cached on disk so a report can be rebuilt offline.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import COMMERCIAL_MODE
from .project import Project
from .utils import write_json

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
OPEN_TARGETS = "https://api.platform.opentargets.org/api/v4/graphql"
SWISS_TARGET = "https://swisstargetprediction.ch"
USER_AGENT = "DrugHX/1.7 (target fishing; contact: local user)"
ACTIVITY_TYPES = {
    "IC50", "IC90", "KI", "KD", "EC50", "EC90", "AC50", "POTENCY", "INHIBITION",
}
MAX_ACTIVITY_PAGES = 8

NETWORK_ATTEMPTS = 3
NETWORK_BACKOFF = (2, 5)
SIMILARITY_TIMEOUT = 75
DEFAULT_TIMEOUT = 60
MIN_MOLECULES = 12
GOAL_MOLECULES = 20

_UNIT_TO_NM = {
    "nm": 1.0,
    "um": 1000.0,
    "µm": 1000.0,
    "μm": 1000.0,
    "mm": 1_000_000.0,
    "pm": 1e-3,
    "fm": 1e-6,
    "m": 1e9,
}


# ---------------------------------------------------------------------------
# network helpers
# ---------------------------------------------------------------------------
def _get_json(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    job=None,
    label: str = "",
    attempts: int = NETWORK_ATTEMPTS,
):
    """GET JSON with retries, longer timeouts and readable error messages."""
    last_error: Exception | None = None
    attempts = max(1, attempts)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                raise RuntimeError(
                    f"{label or 'ChEMBL 请求'}被拒绝（HTTP {exc.code}），"
                    "请检查输入 SMILES 是否合法。"
                ) from exc
            last_error = exc
            if attempt < attempts - 1:
                wait = NETWORK_BACKOFF[min(attempt, len(NETWORK_BACKOFF) - 1)]
                if job:
                    job.log(
                        f"[垂钓] {label or '网络请求'} 返回 HTTP {exc.code}，"
                        f"{wait} 秒后重试（{attempt + 1}/{attempts}）"
                    )
                time.sleep(wait)
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as exc:
            last_error = exc
            if attempt < attempts - 1:
                wait = NETWORK_BACKOFF[min(attempt, len(NETWORK_BACKOFF) - 1)]
                if job:
                    job.log(
                        f"[垂钓] {label or '网络请求'} 超时或连接失败，"
                        f"{wait} 秒后重试（{attempt + 1}/{attempts}）"
                    )
                time.sleep(wait)
    raise RuntimeError(
        f"{label or 'ChEMBL 请求'}失败（已尝试 {attempts} 次）：{last_error}"
    ) from last_error


def _get_text(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    job=None,
    label: str = "",
    attempts: int = 2,
) -> str:
    """GET plain text (CSV/TSV) with the same retry policy as JSON calls."""
    last_error: Exception | None = None
    attempts = max(1, attempts)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                raise RuntimeError(
                    f"{label or '外部数据请求'}被拒绝（HTTP {exc.code}）"
                ) from exc
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(NETWORK_BACKOFF[min(attempt, len(NETWORK_BACKOFF) - 1)])
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(NETWORK_BACKOFF[min(attempt, len(NETWORK_BACKOFF) - 1)])
    raise RuntimeError(
        f"{label or '外部数据请求'}失败（已尝试 {attempts} 次）：{last_error}"
    ) from last_error


def _post_json(url: str, payload: dict, timeout: int = DEFAULT_TIMEOUT):
    data = urllib.parse.urlencode(payload).encode()
    request = urllib.request.Request(
        url, data=data,
        headers={"User-Agent": USER_AGENT,
                 "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _post_json_body(
    url: str,
    payload: dict,
    timeout: int = 45,
    job=None,
    label: str = "",
    attempts: int = 2,
):
    last_error: Exception | None = None
    attempts = max(1, attempts)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "User-Agent": USER_AGENT,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(NETWORK_BACKOFF[min(attempt, len(NETWORK_BACKOFF) - 1)])
    raise RuntimeError(
        f"{label or '外部数据请求'}失败（已尝试 {attempts} 次）：{last_error}"
    ) from last_error


def _cache_file(cache_dir: Path | None, smiles: str, threshold: int, limit: int):
    if cache_dir is None:
        return None
    key = hashlib.md5(
        f"{smiles}|{threshold}|{limit}".encode("utf-8")
    ).hexdigest()
    return cache_dir / f"sim_{key}.json"


# ---------------------------------------------------------------------------
# ChEMBL retrieval
# ---------------------------------------------------------------------------
def _exact_molecule(smiles: str, job) -> dict | None:
    """Fast exact match by InChIKey, with a SMILES flexmatch fallback."""
    url = None
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            key = Chem.MolToInchiKey(mol)
            if key:
                url = f"{CHEMBL}/molecule.json?" + urllib.parse.urlencode(
                    {
                        "molecule_structures__standard_inchi_key": key,
                        "limit": 1,
                        "only": "molecule_chembl_id,pref_name,max_phase,molecule_structures",
                    }
                )
    except Exception:
        url = None
    if url is None:
        url = f"{CHEMBL}/molecule.json?" + urllib.parse.urlencode(
            {
                "molecule_structures__canonical_smiles__flexmatch": smiles,
                "limit": 1,
                "only": "molecule_chembl_id,pref_name,max_phase,molecule_structures",
            }
        )
    try:
        data = _get_json(
            url, timeout=30, job=job, label="ChEMBL 精确匹配",
            attempts=2,
        )
    except Exception as exc:
        job.log(f"[垂钓] 精确匹配未完成：{exc}")
        return None
    for item in data.get("molecules") or []:
        molecule_id = item.get("molecule_chembl_id")
        if not molecule_id:
            continue
        return {
            "chembl_id": molecule_id,
            "pref_name": item.get("pref_name") or "",
            "max_phase": item.get("max_phase"),
            "similarity": 100.0,
            "smiles": (item.get("molecule_structures") or {}).get(
                "canonical_smiles", ""
            ),
            "source": "exact",
        }
    return None


def _similar_molecules(
    smiles: str,
    threshold: int,
    limit: int,
    job,
    cache_dir: Path | None = None,
) -> list[dict]:
    cache = _cache_file(cache_dir, smiles, threshold, limit)
    if cache and cache.exists():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            job.log(
                f"[垂钓] Tanimoto ≥ {threshold}% 使用本地缓存"
                f"（{len(cached)} 个分子）"
            )
            return cached
        except Exception:
            pass

    query = urllib.parse.quote(smiles, safe="")
    url = f"{CHEMBL}/similarity/{query}/{threshold}.json?limit={limit}"
    job.log(
        f"[垂钓] ChEMBL 相似性检索（Tanimoto ≥ {threshold}%，"
        f"上限 {limit} 个分子，单次最多等待 {SIMILARITY_TIMEOUT} 秒）"
    )
    data = _get_json(
        url,
        timeout=SIMILARITY_TIMEOUT,
        job=job,
        label=f"ChEMBL 相似性检索（{threshold}%）",
        attempts=2,
    )
    molecules = []
    for item in data.get("molecules") or []:
        molecule_id = item.get("molecule_chembl_id")
        if not molecule_id:
            continue
        molecules.append(
            {
                "chembl_id": molecule_id,
                "pref_name": item.get("pref_name") or "",
                "max_phase": item.get("max_phase"),
                "similarity": item.get("similarity"),
                "smiles": (item.get("molecule_structures") or {}).get(
                    "canonical_smiles", ""
                ),
                "source": "similarity",
            }
        )
    if cache:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            write_json(cache, molecules)
        except Exception:
            pass
    job.log(f"[垂钓] Tanimoto ≥ {threshold}% 命中 {len(molecules)} 个已知分子")
    return molecules


def _collect_molecules(
    project: Project,
    smiles: str,
    threshold: int,
    limit: int,
    job,
) -> tuple[list[dict], int, list[int]]:
    """Exact match + decreasing similarity thresholds until enough analogues."""
    pool: dict[str, dict] = {}
    exact = _exact_molecule(smiles, job)
    if exact:
        pool[exact["chembl_id"]] = exact
        job.log(
            f"[垂钓] 精确匹配到 {exact['chembl_id']}"
            f"（{exact['pref_name'] or '未命名'}），将纳入活性汇总"
        )

    thresholds = []
    value = threshold
    while value >= 30:
        thresholds.append(value)
        value -= 10
    if not thresholds:
        thresholds = [threshold]
    tried: list[int] = []
    used = threshold
    goal = min(max(MIN_MOLECULES, int(limit * 0.4)), GOAL_MOLECULES, limit)
    cache_dir = project.root / "exports" / "fishing" / "_cache"
    started = time.time()
    for level in thresholds:
        if len(pool) >= goal:
            break
        if time.time() - started > 150:
            job.log("[垂钓] ChEMBL 检索耗时过长，停止扩展并改用其他证据")
            break
        tried.append(level)
        try:
            molecules = _similar_molecules(
                smiles, level, limit, job, cache_dir=cache_dir
            )
        except Exception as exc:
            job.log(f"[垂钓] Tanimoto ≥ {level}% 检索失败：{exc}")
            continue
        used = level
        for molecule in molecules:
            molecule_id = molecule.get("chembl_id")
            if molecule_id and molecule_id not in pool:
                molecule["matched_threshold"] = level
                pool[molecule_id] = molecule
        job.log(
            f"[垂钓] 累计 {len(pool)} 个候选分子"
            f"（目标至少 {goal} 个）"
        )
        if level > 50 and len(pool) < goal:
            job.log("[垂钓] 候选仍偏少，自动降低相似度阈值继续扩展")
    return list(pool.values())[:limit], used, tried


def _activities(molecule_ids: list[str], job) -> list[dict]:
    records: list[dict] = []
    for start in range(0, len(molecule_ids), 20):
        batch = molecule_ids[start:start + 20]
        base = (
            f"{CHEMBL}/activity.json?"
            + urllib.parse.urlencode(
                {
                    "molecule_chembl_id__in": ",".join(batch),
                    "limit": 1000,
                    "only": (
                        "molecule_chembl_id,target_chembl_id,target_pref_name,"
                        "target_organism,target_type,standard_type,standard_value,"
                        "standard_units,pchembl_value"
                    ),
                }
            )
        )
        url = base
        for _page in range(MAX_ACTIVITY_PAGES):
            if not url:
                break
            data = _get_json(
                url, timeout=DEFAULT_TIMEOUT, job=job, label="ChEMBL 活性记录"
            )
            records.extend(data.get("activities", []))
            url = (data.get("page_meta") or {}).get("next")
            if url and url.startswith("/"):
                url = "https://www.ebi.ac.uk" + url
        job.log(f"[垂钓] 读取活性记录 {len(records)} 条（分子批次 {start // 20 + 1}）")
    return records


def _mechanisms(molecule_ids: list[str], job) -> list[dict]:
    records: list[dict] = []
    for start in range(0, len(molecule_ids), 20):
        batch = molecule_ids[start:start + 20]
        url = f"{CHEMBL}/mechanism.json?" + urllib.parse.urlencode(
            {
                "molecule_chembl_id__in": ",".join(batch),
                "limit": 1000,
                "only": "molecule_chembl_id,target_chembl_id,mechanism_of_action,action_type",
            }
        )
        try:
            data = _get_json(
                url, timeout=DEFAULT_TIMEOUT, job=job, label="ChEMBL 作用机制"
            )
        except Exception as exc:
            job.log(f"[垂钓] 作用机制获取失败：{exc}")
            continue
        records.extend(data.get("mechanisms") or [])
    job.log(f"[垂钓] 读取作用机制记录 {len(records)} 条")
    return records


def _gene_symbols(accessions: list[str], job) -> dict[str, str]:
    """Map UniProt accessions to human gene symbols (taxid 9606 only)."""
    mapping: dict[str, str] = {}
    accessions = [item for item in dict.fromkeys(accessions) if item]
    for start in range(0, len(accessions), 40):
        chunk = accessions[start:start + 40]
        query = (
            "(" + " OR ".join(f"accession:{item}" for item in chunk) + ")"
            " AND organism_id:9606 AND reviewed:true"
        )
        url = UNIPROT + "?" + urllib.parse.urlencode(
            {"query": query, "fields": "accession,gene_primary",
             "format": "tsv", "size": 100}
        )
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=40) as response:
                text = response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            job.log(f"[垂钓] UniProt 基因映射失败：{exc}")
            continue
        for line in text.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                mapping[parts[0]] = parts[1].split()[0]
    job.log(f"[垂钓] 解析基因符号 {len(mapping)} 个")
    return mapping


def _protein_name_to_gene(names: list[str], job) -> dict[str, tuple[str, str]]:
    """Map BindingDB protein names to human gene symbols via UniProt."""
    mapping: dict[str, tuple[str, str]] = {}
    for original in dict.fromkeys(names):
        clean = str(original or "").split("[")[0].strip()
        if not clean:
            continue
        candidates = [clean]
        if "/" in clean:
            candidates.extend(part.strip() for part in clean.split("/") if part.strip())
        found = None
        for candidate in candidates:
            query = (
                f'protein_name:"{candidate}" AND organism_id:9606 '
                "AND reviewed:true"
            )
            url = UNIPROT + "?" + urllib.parse.urlencode(
                {
                    "query": query,
                    "fields": "accession,gene_primary",
                    "format": "tsv",
                    "size": 1,
                }
            )
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    text = response.read().decode("utf-8", errors="replace")
            except Exception:
                continue
            for line in text.splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0] and parts[1]:
                    found = (parts[1].split()[0], parts[0])
                    break
            if found:
                break
        if found:
            mapping[original] = found
    job.log(f"[垂钓] BindingDB 蛋白名映射基因 {len(mapping)} 个")
    return mapping


def _target_details(target_ids: list[str], job) -> dict[str, dict]:
    details: dict[str, dict] = {}
    for start in range(0, len(target_ids), 25):
        chunk = target_ids[start:start + 25]
        url = f"{CHEMBL}/target.json?" + urllib.parse.urlencode(
            {
                "target_chembl_id__in": ",".join(chunk),
                "limit": 100,
                "only": (
                    "target_chembl_id,pref_name,target_type,organism,"
                    "target_components"
                ),
            }
        )
        try:
            data = _get_json(
                url, timeout=DEFAULT_TIMEOUT, job=job, label="ChEMBL 靶点详情"
            )
        except Exception as exc:
            job.log(f"[垂钓] 靶点详情获取失败：{exc}")
            continue
        for item in data.get("targets") or []:
            components = item.get("target_components") or []
            accessions: list[str] = []
            genes: list[str] = []
            for component in components:
                accession = component.get("accession") or ""
                if accession and accession not in accessions:
                    accessions.append(accession)
                for synonym in component.get("target_component_synonyms") or []:
                    if str(synonym.get("syn_type") or "").upper() in {
                        "GENE_SYMBOL", "GENE_NAME"
                    }:
                        gene = (synonym.get("component_synonym") or "").strip()
                        if gene and gene not in genes:
                            genes.append(gene)
            target_name = str(item.get("pref_name") or "").strip()
            if target_name.lower() == "rep":
                target_name = "SARS-CoV-2 replicase polyprotein (NSP5/3CLpro)"
            details[item.get("target_chembl_id")] = {
                "name": target_name,
                "type": item.get("target_type", ""),
                "organism": item.get("organism", ""),
                "accessions": accessions,
                "genes": genes,
            }
    return details


# ---------------------------------------------------------------------------
# activity processing
# ---------------------------------------------------------------------------
def _pchembl_from_record(record: dict) -> float | None:
    raw = record.get("pchembl_value")
    if raw not in (None, "", "None"):
        try:
            value = float(raw)
            if 2.0 <= value <= 12.0:
                return value
        except (TypeError, ValueError):
            pass
    value = record.get("standard_value")
    units = str(record.get("standard_units") or "").strip().lower()
    if value in (None, "", "None") or units not in _UNIT_TO_NM:
        return None
    try:
        nm = float(value) * _UNIT_TO_NM[units]
    except (TypeError, ValueError):
        return None
    if nm <= 0:
        return None
    pchembl = 9.0 - math.log10(nm)
    return pchembl if 2.0 <= pchembl <= 12.0 else None


def _new_bucket(target_id: str, name: str, organism: str) -> dict:
    return {
        "target_chembl_id": target_id,
        "name": name or target_id,
        "organism": organism or "—",
        "pchembl": [],
        "values": [],
        "types": {},
        "molecules": set(),
        "mechanism": "",
        "qualitative": 0,
        "mechanism_count": 0,
    }


def _is_human(organism: str) -> bool:
    text = str(organism or "").lower()
    return not text or "homo sapiens" in text or "human" in text


def _is_model_organism(organism: str) -> bool:
    """Mouse/rat records are kept as labelled homolog evidence."""
    text = str(organism or "").lower()
    return "mus musculus" in text or "rattus norvegicus" in text


def _evidence_score(
    median: float | None,
    qualitative: int = 0,
    mechanisms: int = 0,
) -> float:
    """Ranking score that never pretends to be a measured pChEMBL value.

    Pseudo-scores for records without a potency value are deliberately kept
    below ordinary measured actives (pChEMBL ≥ 5), otherwise a target with a
    single "tested but no value" record would outrank a real µM binder.
    """
    if median is not None:
        return round(float(median), 2)
    if mechanisms:
        return round(4.0 + min(0.3, 0.05 * qualitative), 2)
    if qualitative:
        return round(3.0 + min(0.5, 0.05 * qualitative), 2)
    return 0.0


def _dgidb_genes(names: list[str], job) -> list[dict]:
    """Known drug-gene interactions from DGIdb (aggregates many sources)."""
    candidates = [
        name for name in dict.fromkeys(names)
        if name and name.strip() and name.strip().lower() != "query"
    ]
    for candidate in candidates:
        query = (
            "{ drugs(name: " + json.dumps(candidate) + ") { nodes { name "
            "interactions { gene { name } sources { sourceDbName } "
            "interactionTypes { type } } } } }"
        )
        try:
            data = _post_json_body(
                "https://dgidb.org/api/graphql",
                {"query": query},
                timeout=40,
                job=job,
                label="DGIdb 药物–基因检索",
            )
        except Exception as exc:
            job.log(f"[垂钓] DGIdb 查询失败：{exc}")
            continue
        nodes = ((data.get("data") or {}).get("drugs") or {}).get("nodes") or []
        if not nodes:
            continue
        node = next(
            (item for item in nodes
             if str(item.get("name", "")).upper() == candidate.upper()),
            nodes[0],
        )
        genes: dict[str, dict] = {}
        for interaction in node.get("interactions") or []:
            gene = str((interaction.get("gene") or {}).get("name") or "").strip()
            if not gene:
                continue
            entry = genes.setdefault(
                gene.upper(), {"gene": gene, "count": 0, "sources": set(), "types": set()}
            )
            entry["count"] += 1
            for source in interaction.get("sources") or []:
                name_db = source.get("sourceDbName")
                if name_db:
                    entry["sources"].add(name_db)
            for kind in interaction.get("interactionTypes") or []:
                label_type = kind.get("type")
                if label_type:
                    entry["types"].add(label_type)
        if genes:
            job.log(
                f"[垂钓] DGIdb（{node.get('name', candidate)}）命中 {len(genes)} 个基因"
            )
            return list(genes.values())
    return []


def _pdb_genes(smiles: str, job, match_type: str = "graph-exact") -> list[dict]:
    """Protein targets from PDB entries containing the same ligand graph."""
    payload = {
        "query": {
            "type": "terminal",
            "service": "chemical",
            "parameters": {
                "value": smiles,
                "type": "descriptor",
                "descriptor_type": "SMILES",
                "match_type": match_type,
            },
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 20}},
    }
    try:
        data = _post_json_body(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            payload,
            timeout=45,
            job=job,
            label=f"RCSB PDB 配体检索（{match_type}）",
        )
    except Exception as exc:
        job.log(f"[垂钓] PDB 配体检索失败：{exc}")
        return []
    entries = [
        item.get("identifier")
        for item in data.get("result_set") or []
        if item.get("identifier")
    ]
    if not entries:
        return []
    query = (
        "{ entries(entry_ids: " + json.dumps(entries) + ") { rcsb_id "
        "polymer_entities { rcsb_polymer_entity_container_identifiers "
        "{ uniprot_ids } rcsb_polymer_entity { pdbx_description } "
        "rcsb_entity_source_organism { ncbi_taxonomy_id scientific_name } } } }"
    )
    try:
        graph = _post_json_body(
            "https://data.rcsb.org/graphql",
            {"query": query},
            timeout=45,
            job=job,
            label="RCSB PDB 实体注释",
        )
    except Exception as exc:
        job.log(f"[垂钓] PDB 实体注释失败：{exc}")
        return []
    entity_rows: list[dict] = []
    for entry in (graph.get("data") or {}).get("entries") or []:
        for entity in entry.get("polymer_entities") or []:
            description = str(
                (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description") or ""
            )
            organisms = entity.get("rcsb_entity_source_organism") or [{}]
            organism = str(organisms[0].get("scientific_name") or "")
            taxid = str(organisms[0].get("ncbi_taxonomy_id") or "")
            human = taxid == "9606" or "homo sapiens" in organism.lower()
            identifiers = (
                entity.get("rcsb_polymer_entity_container_identifiers") or {}
            )
            for accession in identifiers.get("uniprot_ids") or []:
                if accession:
                    entity_rows.append(
                        {
                            "accession": accession,
                            "description": description,
                            "organism": organism or ("Homo sapiens" if human else "—"),
                            "human": human,
                        }
                    )
    if not entity_rows:
        return []
    human_accessions = [
        row["accession"] for row in entity_rows if row["human"]
    ]
    mapped = _gene_symbols(human_accessions, job)
    items: dict[tuple, dict] = {}
    for row in entity_rows:
        gene = mapped.get(row["accession"], "") if row["human"] else ""
        key = (gene.upper() if gene else row["accession"], row["organism"])
        entry = items.setdefault(
            key,
            {
                "gene": gene,
                "accession": row["accession"],
                "name": row["description"],
                "organism": row["organism"],
                "human": row["human"],
                "count": 0,
            },
        )
        entry["count"] += 1
    human_count = sum(1 for item in items.values() if item["human"])
    job.log(
        f"[垂钓] PDB 共晶结构命中 {len(items)} 个蛋白"
        f"（人源 {human_count} 个）"
    )
    return list(items.values())


def _bindingdb_targets(smiles: str, job, cutoff: float = 0.85) -> list[dict]:
    """Human protein targets of structurally similar compounds in BindingDB."""
    url = "https://bindingdb.org/rest/getTargetByCompound?" + urllib.parse.urlencode(
        {"smiles": smiles, "cutoff": cutoff, "response": "application/json"}
    )
    try:
        data = _get_json(
            url, timeout=90, job=job, label="BindingDB 反向靶点检索", attempts=1
        )
    except Exception as exc:
        job.log(f"[垂钓] BindingDB 查询失败：{exc}")
        return []
    affinities = (
        (data.get("getLindsByUniprotResponse") or {}).get("bdb.affinities") or []
    )
    if not affinities:
        job.log("[垂钓] BindingDB 未找到结构类似物")
        return []
    per_target: dict[str, dict] = {}
    for item in affinities:
        species = str(item.get("bdb.species") or "")
        if "human" not in species.lower():
            continue
        name = str(item.get("bdb.target") or "").strip()
        if not name:
            continue
        bucket = per_target.setdefault(
            name,
            {"name": name, "pchembl": [], "values": [], "types": {}, "count": 0},
        )
        bucket["count"] += 1
        kind = str(item.get("bdb.affinity_type") or "AFFINITY")
        bucket["types"][kind] = bucket["types"].get(kind, 0) + 1
        raw = str(item.get("bdb.affinity") or "").strip()
        numeric = raw.replace(">", "").replace("<", "").replace("=", "").strip()
        try:
            nm = float(numeric)
        except ValueError:
            continue
        bucket["values"].append(nm)
        if nm > 0:
            pchembl = 9.0 - math.log10(nm)
            if 2.0 <= pchembl <= 12.0:
                bucket["pchembl"].append(pchembl)
    targets = []
    for item in per_target.values():
        median_p = (
            round(float(np.median(item["pchembl"])), 2) if item["pchembl"] else None
        )
        targets.append(
            {
                "name": item["name"],
                "median_pchembl": median_p,
                "best_nm": round(min(item["values"]), 2) if item["values"] else None,
                "count": item["count"],
                "types": item["types"],
            }
        )
    targets.sort(
        key=lambda item: (-(item["median_pchembl"] or 0), -item["count"])
    )
    job.log(
        f"[垂钓] BindingDB 命中 {len(targets)} 个人源蛋白靶点"
        f"（相似度 ≥ {cutoff:.2f}）"
    )
    return targets


def _pubchem_cids(
    smiles: str, job, names: list[str] | None = None
) -> list[int]:
    """Resolve a SMILES to candidate PubChem CIDs (InChIKey, then name)."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        key = Chem.MolToInchiKey(mol)
    except Exception:
        key = ""
    queries = []
    if key:
        queries.append(f"{PUBCHEM}/compound/inchikey/{key}/cids/JSON")
    for name in names or []:
        clean = str(name or "").strip()
        if clean and clean.lower() != "query":
            queries.append(
                f"{PUBCHEM}/compound/name/{urllib.parse.quote(clean)}/cids/JSON"
            )
    found: list[int] = []
    for url in queries:
        try:
            data = _get_json(
                url, timeout=45, job=job, label="PubChem 化合物检索", attempts=1
            )
        except Exception:
            continue
        cids = (data.get("IdentifierList") or {}).get("CID") or []
        for cid in cids:
            value = int(cid)
            if value not in found:
                found.append(value)
    if not found:
        job.log("[垂钓] PubChem 未找到该结构或名称")
    return found


def _pubchem_targets(
    smiles: str, job, limit: int = 40, names: list[str] | None = None
) -> list[dict]:
    """Human protein targets from PubChem bioassay summaries."""
    cids = _pubchem_cids(smiles, job, names)
    if not cids:
        return []
    text = ""
    cid = None
    for candidate in cids:
        url = (
            f"{PUBCHEM}/compound/cid/{candidate}/assaysummary/CSV"
            "?response_type=display"
        )
        try:
            text = _get_text(
                url, timeout=120, job=job,
                label="PubChem 生物试验摘要", attempts=1,
            )
            cid = candidate
            break
        except Exception:
            continue
    if not text or cid is None:
        job.log("[垂钓] PubChem 没有可用的生物试验摘要")
        return []
    reader = csv.DictReader(io.StringIO(text))
    per_accession: dict[str, dict] = {}
    quantitative = {
        "IC50", "IC90", "KI", "KD", "EC50", "EC90", "AC50", "POTENCY",
    }
    for row in reader:
        raw_accession = str(row.get("Target Accession") or "").strip()
        if not raw_accession or raw_accession.lower() in {"n/a", "none", "-"}:
            continue
        accession = raw_accession.split(";")[0].strip()
        if not re.match(r"^[A-Z][A-Z0-9]{5}([A-Z0-9]{4})?$", accession):
            continue
        outcome = str(row.get("Activity Outcome") or "").strip().lower()
        if outcome and outcome not in {"active", "unspecified", "probe", "hit"}:
            continue
        bucket = per_accession.setdefault(
            accession, {"pchembl": [], "values": [], "n": 0, "types": {}}
        )
        bucket["n"] += 1
        kind = str(row.get("Activity Name") or "ACTIVITY").strip().upper()
        bucket["types"][kind] = bucket["types"].get(kind, 0) + 1
        if kind not in quantitative:
            continue
        raw_value = str(row.get("Activity Value [uM]") or "").strip()
        raw_value = raw_value.replace(">", "").replace("<", "").replace("=", "")
        try:
            um = float(raw_value)
        except ValueError:
            continue
        if um <= 0:
            continue
        bucket["values"].append(um)
        pchembl = 6.0 - math.log10(um)
        if 2.0 <= pchembl <= 12.0:
            bucket["pchembl"].append(pchembl)
    if not per_accession:
        job.log("[垂钓] PubChem 没有可用的人源蛋白试验记录")
        return []
    # Map only the strongest candidates; PubChem can return hundreds of
    # accessions for promiscuous drugs and most never enter the final table.
    ranked_accessions = sorted(
        per_accession.items(),
        key=lambda item: (-len(item[1]["pchembl"]), -item[1]["n"]),
    )[: max(limit * 2, 60)]
    mapping = _gene_symbols([item[0] for item in ranked_accessions], job)
    targets = []
    for accession, item in ranked_accessions:
        gene = mapping.get(accession, "")
        if not gene:
            continue
        median_p = (
            round(float(np.median(item["pchembl"])), 2)
            if item["pchembl"]
            else None
        )
        targets.append(
            {
                "gene": gene,
                "accession": accession,
                "name": f"{gene}（PubChem 生物试验）",
                "median_pchembl": median_p,
                "pchembl_n": len(item["pchembl"]),
                "qualitative_n": item["n"] if median_p is None else 0,
                "n": item["n"],
                "best_nm": round(min(item["values"]) * 1000.0, 2)
                if item["values"]
                else None,
                "types": item["types"],
            }
        )
    targets.sort(
        key=lambda item: (
            -(item["median_pchembl"] or 0),
            -item["n"],
        )
    )
    job.log(
        f"[垂钓] PubChem 命中 {len(targets)} 个人源蛋白靶点"
        f"（CID {cid}）"
    )
    return targets[:limit]


def _opentargets_targets(chembl_id: str, job) -> list[dict]:
    """Approved-drug mechanisms of action from Open Targets."""
    if not chembl_id:
        return []
    query = (
        "query($id: String!) { drug(chemblId: $id) { name "
        "mechanismsOfAction { rows { targetName actionType "
        "targets { approvedSymbol } } } } }"
    )
    try:
        data = _post_json_body(
            OPEN_TARGETS,
            {"query": query, "variables": {"id": chembl_id}},
            timeout=45,
            job=job,
            label="Open Targets 作用机制",
        )
    except Exception as exc:
        job.log(f"[垂钓] Open Targets 查询失败：{exc}")
        return []
    drug = (data.get("data") or {}).get("drug") or {}
    rows = ((drug.get("mechanismsOfAction") or {}).get("rows") or [])
    targets = []
    for row in rows:
        genes = [
            str(item.get("approvedSymbol") or "").strip()
            for item in row.get("targets") or []
            if item.get("approvedSymbol")
        ]
        if not genes:
            continue
        for gene in genes[:3]:
            targets.append(
                {
                    "gene": gene,
                    "name": row.get("targetName") or f"{gene}（Open Targets）",
                    "mechanism": row.get("actionType") or "",
                }
            )
    if targets:
        job.log(
            f"[垂钓] Open Targets 命中 {len(targets)} 个已知作用机制靶点"
        )
    return targets


def _external_target(
    gene: str,
    source: str,
    score: float,
    name: str = "",
    accession: str = "",
    count: int = 1,
    organism: str = "Homo sapiens",
    genes: list[str] | None = None,
    pchembl: float | None = None,
    best_nm: float | None = None,
    basis: str = "external evidence",
    mechanism: str = "",
) -> dict:
    gene_list = list(genes if genes is not None else ([gene] if gene else []))
    return {
        "target_chembl_id": "",
        "name": name or gene or "—",
        "gene": gene_list[0] if gene_list else "",
        "genes": gene_list[:6],
        "accession": accession,
        "accessions": [accession] if accession else [],
        "type": source,
        "organism": organism or "—",
        "n": count,
        "median_pchembl": round(pchembl, 2) if pchembl is not None else None,
        "pchembl_n": 0,
        "qualitative_n": count if pchembl is None else 0,
        "mechanism_n": 0,
        "evidence_score": round(
            pchembl if pchembl is not None else score, 2
        ),
        "score_basis": basis,
        "best_nm": round(best_nm, 2) if best_nm is not None else None,
        "activity_types": {source: count},
        "n_molecules": 1,
        "mechanism": mechanism,
        "sources": [source],
    }


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def _swiss_session_items(
    session_number: str, job, timeout: int = 150
) -> list[dict]:
    """Poll a SwissTargetPrediction session and read PredictedTargets.dat."""
    import io
    import ssl
    import tarfile

    context = ssl._create_unverified_context()
    status_url = (
        "https://www.swisstargetprediction.ch:8443/checksession"
        f"?sessionNumber={session_number}"
    )
    started = time.time()
    while time.time() - started < timeout:
        try:
            request = urllib.request.Request(
                status_url, headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(
                request, timeout=25, context=context
            ) as response:
                text = response.read().decode("utf-8", errors="replace")
        except Exception:
            time.sleep(6)
            continue
        if "is finished" in text:
            archive_url = (
                "https://www.swisstargetprediction.ch:8443/retrievesession"
                f"?sessionNumber={session_number}"
            )
            try:
                request = urllib.request.Request(
                    archive_url, headers={"User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(
                    request, timeout=60, context=context
                ) as response:
                    archive = response.read()
                with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
                    member = tar.extractfile("PredictedTargets.dat")
                    content = (
                        member.read().decode("utf-8", errors="replace")
                        if member is not None
                        else ""
                    )
            except Exception as exc:
                job.log(f"[垂钓] SwissTargetPrediction 结果下载失败：{exc}")
                return []
            items = []
            for line in content.splitlines():
                parts = [part.strip() for part in line.split(";")]
                if len(parts) < 8 or not parts[1]:
                    continue
                try:
                    probability = float(parts[4])
                except (TypeError, ValueError):
                    probability = 0.0
                try:
                    known_3d = int(float(parts[5]))
                except (TypeError, ValueError):
                    known_3d = 0
                try:
                    known_2d = int(float(parts[6]))
                except (TypeError, ValueError):
                    known_2d = 0
                items.append(
                    {
                        "gene": parts[1],
                        "name": parts[0],
                        "accession": parts[2],
                        "target_chembl_id": parts[7],
                        "type": parts[8] if len(parts) > 8 else "",
                        "probability": round(probability, 4),
                        "known_3d": known_3d,
                        "known_2d": known_2d,
                        "organism": "Homo sapiens",
                    }
                )
            return items
        time.sleep(5)
    job.log(
        f"[垂钓] SwissTargetPrediction 队列等待超过 {timeout} 秒，"
        "跳过该来源并使用本地回退"
    )
    return []


def _swiss_targets(
    smiles: str,
    job,
    organism: str = "Homo_sapiens",
    timeout: int = 180,
) -> dict:
    """SwissTargetPrediction (2D/3D ligand similarity) via its public form.

    The service has no JSON API: the form returns a progress page that ends
    with a ``result.php?job=...`` redirect.  We submit once, parse that job,
    then read the HTML result table.  Everything is best-effort and must not
    break the rest of the fishing pipeline.
    """
    if not smiles:
        return {"available": False, "items": [], "error": "no SMILES"}
    organism = organism if organism in (
        "Homo_sapiens", "Mus_musculus", "Rattus_norvegicus"
    ) else "Homo_sapiens"
    payload = urllib.parse.urlencode(
        {"organism": organism, "smiles": smiles, "ioi": "2"}
    ).encode()
    headers = {
        "User-Agent": USER_AGENT,
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": SWISS_TARGET + "/",
    }
    job_id = ""
    session_number = ""
    result_organism = organism
    page = ""
    for attempt in range(2):
        try:
            request = urllib.request.Request(
                SWISS_TARGET + "/predict.php", data=payload, headers=headers
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                page = response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            if attempt == 0:
                time.sleep(6)
                continue
            return {"available": False, "items": [], "error": str(exc)[:160]}
        session_match = re.search(r"sessionNumber=(\d+)", page)
        if session_match:
            session_number = session_match.group(1)
            break
        match = re.search(
            r"result\.php\?job=(\d+)&organism=([A-Za-z_]+)", page
        )
        if not match:
            match = re.search(r"[?&]job=(\d+)", page)
        if match:
            job_id = match.group(1)
            if match.lastindex and match.lastindex > 1:
                result_organism = match.group(2)
            break
        if attempt == 0:
            job.log("[垂钓] SwissTargetPrediction 任务排队，20 秒后重试一次")
            time.sleep(20)
    if not job_id and session_number:
        job.log(
            "[垂钓] SwissTargetPrediction 任务已提交，"
            f"正在轮询会话 {session_number}（最多约 150 秒）"
        )
        items = _swiss_session_items(session_number, job, timeout=150)
        for item in items:
            item["organism"] = organism.replace("_", " ")
        if items:
            job.log(
                f"[垂钓] SwissTargetPrediction 返回 {len(items)} 个靶点"
                f"（{organism}）"
            )
            return {
                "available": True,
                "items": items,
                "job": session_number,
                "organism": organism,
                "error": "",
            }
        return {
            "available": False,
            "items": [],
            "job": session_number,
            "organism": organism,
            "error": "SwissTargetPrediction 会话已提交但结果为空/超时",
        }
    if not job_id:
        return {
            "available": False,
            "items": [],
            "error": "SwissTargetPrediction 未返回任务或结果（服务繁忙/限流）",
        }
    result_url = (
        f"{SWISS_TARGET}/result.php?job={job_id}&organism={result_organism}"
    )
    try:
        request = urllib.request.Request(result_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            html = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return {"available": False, "items": [], "error": str(exc)[:160]}
    items: list[dict] = []
    seen: set[str] = set()
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        if "genecards.org" not in row:
            continue
        gene_match = re.search(r"gene=([A-Z0-9\-]+)", row)
        uniprot_match = re.search(r"uniprot/([A-Z0-9]+)", row)
        chembl_match = re.search(r"target/(CHEMBL\d+)", row)
        prob_match = re.search(r">([0-9.]+)</span>", row)
        active_matches = re.findall(r"method=(?:3D|FP2)\">(\d+)", row)
        name_match = re.search(r"<td>\s*([^<]+)", row)
        class_match = re.search(r"<td align=['\"]left['\"]>\s*([^<]*)", row)
        gene = gene_match.group(1) if gene_match else ""
        if not gene:
            continue
        key = gene.upper()
        if key in seen:
            continue
        seen.add(key)
        probability = float(prob_match.group(1)) if prob_match else 0.0
        known_3d = int(active_matches[0]) if active_matches else 0
        known_2d = int(active_matches[1]) if len(active_matches) > 1 else 0
        items.append(
            {
                "gene": gene,
                "name": (name_match.group(1).strip() if name_match else gene),
                "accession": uniprot_match.group(1) if uniprot_match else "",
                "target_chembl_id": chembl_match.group(1) if chembl_match else "",
                "type": (class_match.group(1).strip() if class_match else ""),
                "probability": round(probability, 4),
                "known_3d": known_3d,
                "known_2d": known_2d,
                "organism": result_organism.replace("_", " "),
            }
        )
    job.log(
        f"[垂钓] SwissTargetPrediction 返回 {len(items)} 个靶点"
        f"（{result_organism}）"
    )
    return {
        "available": bool(items),
        "items": items,
        "job": job_id,
        "organism": result_organism,
        "error": "" if items else "SwissTargetPrediction 结果表为空",
    }


def _chembl_name_molecule(name: str, job) -> list[dict]:
    """Fallback when SMILES similarity is unavailable: look up a known drug name."""
    name = str(name or "").strip()
    if not name or name.lower() in ("query", "查询分子"):
        return []
    url = f"{CHEMBL}/molecule.json?" + urllib.parse.urlencode(
        {
            "pref_name__iexact": name,
            "limit": 3,
            "only": "molecule_chembl_id,pref_name,max_phase,molecule_structures",
        }
    )
    try:
        data = _get_json(
            url, timeout=30, job=job, label="ChEMBL 名称匹配", attempts=1
        )
    except Exception as exc:
        job.log(f"[垂钓] ChEMBL 名称匹配失败：{exc}")
        return []
    molecules = []
    for item in data.get("molecules") or []:
        molecule_id = item.get("molecule_chembl_id")
        if not molecule_id:
            continue
        molecules.append(
            {
                "chembl_id": molecule_id,
                "pref_name": item.get("pref_name") or name,
                "max_phase": item.get("max_phase"),
                "similarity": 100.0,
                "smiles": (item.get("molecule_structures") or {}).get(
                    "canonical_smiles", ""
                ),
                "source": "name",
            }
        )
    if molecules:
        job.log(f"[垂钓] ChEMBL 名称匹配命中 {len(molecules)} 个分子")
    return molecules


def _similar_drug_targets(
    smiles: str,
    job,
    min_similarity: float = 0.22,
    limit: int = 8,
) -> list[dict]:
    """Local fallback: targets of structurally similar FDA drugs.

    This is deliberately local and deterministic.  It guarantees a useful
    target set even when ChEMBL and SwissTargetPrediction are unavailable.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator

        from .drugref import FDA_DRUGS
    except Exception:
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    query_fp = generator.GetFingerprint(mol)
    rows = []
    for entry in FDA_DRUGS:
        if not entry.get("primary_targets"):
            continue
        reference = Chem.MolFromSmiles(entry.get("smiles") or "")
        if reference is None:
            continue
        try:
            similarity = float(
                DataStructs.TanimotoSimilarity(
                    query_fp, generator.GetFingerprint(reference)
                )
            )
        except Exception:
            continue
        if similarity < min_similarity:
            continue
        rows.append(
            {
                "name": entry.get("name", ""),
                "tanimoto": round(similarity, 3),
                "targets": list(entry.get("primary_targets") or []),
                "drug_class": entry.get("drug_class", ""),
            }
        )
    rows.sort(key=lambda item: -item["tanimoto"])
    rows = rows[:limit]
    if rows:
        job.log(
            "[垂钓] 本地相似药物回退命中 "
            + "、".join(
                f"{item['name']}（Tanimoto {item['tanimoto']:.2f}）"
                for item in rows
            )
        )
    return rows


def run_fishing(
    project: Project,
    smiles: str,
    name: str,
    options: dict,
    job,
) -> dict:
    """Fetch ChEMBL bioactivity for a molecule and rank its likely targets."""
    # The ChEMBL similarity endpoint rejects thresholds below 40 with HTTP 400.
    requested = max(40, min(int(options.get("similarity", 60) or 60), 100))
    limit = max(10, min(int(options.get("molecules", 40) or 40), 200))
    top_targets = max(10, min(int(options.get("targets", 60) or 60), 300))
    deep = str(options.get("depth") or "deep").lower() == "deep"
    if deep:
        requested = min(requested, 40)
        limit = max(limit, 120)
        top_targets = max(top_targets, 220)
    started = time.time()
    job.set_progress(0.05, "检索 ChEMBL 相似分子")

    molecules, effective, tried = _collect_molecules(
        project, smiles, requested, limit, job
    )
    if not molecules:
        molecules = _chembl_name_molecule(name, job)
        if molecules:
            effective = 100
            job.log("[垂钓] ChEMBL 相似性不可用，已改用名称匹配的已知分子")
    if not molecules:
        job.log(
            "[垂钓] ChEMBL 未返回相似分子（可能服务繁忙），"
            "继续尝试 DGIdb 与 PDB 证据"
        )

    job.set_progress(0.3, "读取生物活性与作用机制")
    molecule_ids = [item["chembl_id"] for item in molecules if item.get("chembl_id")]
    try:
        records = _activities(molecule_ids, job) if molecule_ids else []
    except Exception as exc:
        job.log(f"[垂钓] ChEMBL 活性记录不可用：{exc}")
        records = []
    try:
        mechanisms = _mechanisms(molecule_ids, job) if molecule_ids else []
    except Exception as exc:
        job.log(f"[垂钓] ChEMBL 作用机制不可用：{exc}")
        mechanisms = []

    activity_types: dict[str, int] = {}
    per_target: dict[str, dict] = {}
    for record in records:
        target_id = record.get("target_chembl_id")
        if not target_id:
            continue
        organism = str(record.get("target_organism") or "")
        homolog = _is_model_organism(organism)
        if not _is_human(organism) and not homolog:
            continue
        target_type = str(record.get("target_type") or "").upper()
        if any(bad in target_type for bad in ("CELL-LINE", "TISSUE", "ORGANISM")):
            continue
        standard_type = str(record.get("standard_type") or "").strip().upper()
        pvalue = _pchembl_from_record(record)
        bucket = per_target.setdefault(
            target_id,
            _new_bucket(
                target_id,
                record.get("target_pref_name") or target_id,
                organism,
            ),
        )
        if homolog:
            bucket["homolog"] = True
        if pvalue is None:
            # Qualitative records prove the target was tested, but they do
            # not carry a quantitative pChEMBL value.  Never invent one.
            bucket["qualitative"] += 1
            standard_type = standard_type or "ACTIVITY"
        else:
            bucket["pchembl"].append(pvalue)
            try:
                bucket["values"].append(float(record.get("standard_value")))
            except (TypeError, ValueError):
                pass
        bucket["types"][standard_type] = bucket["types"].get(standard_type, 0) + 1
        bucket["molecules"].add(record.get("molecule_chembl_id"))
        activity_types[standard_type] = activity_types.get(standard_type, 0) + 1

    mechanism_records = 0
    for mechanism in mechanisms:
        target_id = mechanism.get("target_chembl_id")
        if not target_id:
            continue
        bucket = per_target.setdefault(
            target_id,
            _new_bucket(target_id, target_id, "Homo sapiens"),
        )
        # A mechanism record is evidence of engagement, not a pChEMBL value.
        bucket["mechanism_count"] += 1
        bucket["types"]["MECHANISM"] = bucket["types"].get("MECHANISM", 0) + 1
        bucket["molecules"].add(mechanism.get("molecule_chembl_id"))
        if mechanism.get("mechanism_of_action"):
            bucket["mechanism"] = mechanism["mechanism_of_action"]
        activity_types["MECHANISM"] = activity_types.get("MECHANISM", 0) + 1
        mechanism_records += 1

    if not per_target:
        job.log(
            "[垂钓] ChEMBL 相似分子没有可用的人源蛋白记录，"
            "改用 DGIdb 与 PDB 证据"
        )

    job.set_progress(0.6, "汇总靶点并解析基因符号")
    ranked_all = sorted(
        per_target.values(),
        key=lambda item: (
            -float(np.median(item["pchembl"])) if item["pchembl"] else 0.0,
            -(len(item["pchembl"]) + item["qualitative"] + item["mechanism_count"]),
        ),
    )
    candidate_cap = max(top_targets * 3, 60)
    candidates = ranked_all[:candidate_cap]
    details = _target_details(
        [item["target_chembl_id"] for item in candidates], job
    )

    ranked: list[dict] = []
    for item in candidates:
        info = details.get(item["target_chembl_id"], {})
        target_type = str(info.get("type") or "").upper()
        organism = str(info.get("organism") or item["organism"])
        if not _is_human(organism) and not _is_model_organism(organism):
            continue
        if "CELL-LINE" in target_type or "TISSUE" in target_type:
            continue
        if "PROTEIN" not in target_type and not info.get("genes"):
            continue
        item["info"] = info
        item["organism"] = organism
        item["homolog"] = not _is_human(organism)
        ranked.append(item)
    ranked = ranked[:top_targets]
    if not ranked:
        job.log("[垂钓] ChEMBL 靶点未能解析为人源蛋白，改用其他证据来源")

    accessions = [
        accession
        for item in ranked
        for accession in item["info"].get("accessions", [])
    ]
    mapped = _gene_symbols(accessions, job)

    targets = []
    for item in ranked:
        info = item["info"]
        genes = list(info.get("genes") or [])
        for accession in info.get("accessions", []):
            gene = mapped.get(accession)
            if gene and gene not in genes:
                genes.append(gene)
        median_pchembl = (
            round(float(np.median(item["pchembl"])), 2)
            if item["pchembl"]
            else None
        )
        record_count = (
            len(item["pchembl"]) + item["qualitative"] + item["mechanism_count"]
        )
        if median_pchembl is not None:
            basis = "median pChEMBL"
        elif item["qualitative"]:
            basis = "qualitative activity records"
        elif item["mechanism_count"]:
            basis = "mechanism of action"
        else:
            basis = "no quantitative value"
        targets.append(
            {
                "target_chembl_id": item["target_chembl_id"],
                "name": info.get("name") or item["name"],
                "gene": genes[0] if genes else "",
                "genes": genes[:6],
                "accession": (info.get("accessions") or [""])[0],
                "accessions": info.get("accessions", []),
                "type": info.get("type", ""),
                "organism": item["organism"],
                "homolog": bool(item.get("homolog")),
                "n": record_count,
                "median_pchembl": median_pchembl,
                "pchembl_n": len(item["pchembl"]),
                "qualitative_n": item["qualitative"],
                "mechanism_n": item["mechanism_count"],
                "evidence_score": _evidence_score(
                    median_pchembl, item["qualitative"], item["mechanism_count"]
                ),
                "score_basis": basis,
                "best_nm": round(min(item["values"]), 2) if item["values"] else None,
                "activity_types": item["types"],
                "n_molecules": len(item["molecules"]),
                "mechanism": item.get("mechanism", ""),
                "sources": ["ChEMBL"],
            }
        )
    targets.sort(
        key=lambda item: (
            -(item.get("evidence_score") or item.get("median_pchembl") or 0),
            -item.get("n", 0),
        )
    )

    # ---- cross-source evidence: DGIdb, PDB, BindingDB, PubChem, Open Targets
    by_gene = {
        gene.upper(): target
        for target in targets
        for gene in (target.get("genes") or [])
        if gene
    }
    by_name = {
        str(target.get("name", "")).strip().lower(): target
        for target in targets
    }

    def merge_source(target: dict, source: str) -> None:
        sources = target.setdefault("sources", [])
        if source not in sources:
            sources.append(source)

    exact = next(
        (item for item in molecules if item.get("source") == "exact"), None
    )
    names = []
    if exact and exact.get("pref_name"):
        names.append(exact["pref_name"])
    if name:
        names.append(name)

    # DGIdb 整合的上游数据含非商业条款，商业合规模式下跳过。
    dgidb_items = [] if COMMERCIAL_MODE else _dgidb_genes(names, job)
    for item in dgidb_items:
        gene = item["gene"]
        key = gene.upper()
        if key in by_gene:
            merge_source(by_gene[key], "DGIdb")
            continue
        target = _external_target(
            gene, "DGIdb", 6.1 + min(0.6, 0.05 * item.get("count", 1)),
            name=f"{gene}（DGIdb）", count=item.get("count", 1),
            basis="known drug–gene interaction",
        )
        targets.append(target)
        by_gene[key] = target
        by_name[target["name"].lower()] = target

    pdb_items = _pdb_genes(smiles, job, "graph-exact")
    human_pdb = sum(1 for item in pdb_items if item.get("human"))
    if human_pdb < 2:
        relaxed = _pdb_genes(smiles, job, "graph-relaxed")
        known = {
            (item.get("gene") or item.get("accession"), item.get("organism"))
            for item in pdb_items
        }
        for item in relaxed:
            key = (item.get("gene") or item.get("accession"), item.get("organism"))
            if key not in known:
                pdb_items.append(item)
                known.add(key)
        if relaxed:
            job.log(
                f"[垂钓] PDB 宽松图匹配补充 {len(relaxed)} 个蛋白结构"
            )
    for item in pdb_items:
        gene = item.get("gene") or ""
        key = gene.upper() if gene else ""
        name_key = (
            f"pdb:{(item.get('accession') or gene or '')}:{item.get('organism', '')}"
        ).lower()
        if key and key in by_gene:
            merge_source(by_gene[key], "PDB")
            continue
        if not key and name_key in by_name:
            merge_source(by_name[name_key], "PDB")
            continue
        label = item.get("name") or gene or item.get("accession", "")
        if not item.get("human"):
            label = f"{label}（{item.get('organism', '非人源')}）"
        target = _external_target(
            gene, "PDB", 5.9 + min(0.4, 0.03 * item.get("count", 1)),
            name=f"{label}（PDB 共晶）",
            accession=item.get("accession", ""),
            count=item.get("count", 1),
            organism=item.get("organism", "Homo sapiens"),
            genes=[gene] if gene else [],
            basis="co-crystal structure",
        )
        targets.append(target)
        if key:
            by_gene[key] = target
        by_name[name_key] = target

    # Non-human co-crystals often still identify the same protein family.
    # Map their protein names to human genes when an exact reviewed UniProt
    # name match exists, and label the result as a homolog candidate.
    homolog_names = [
        item.get("name", "")
        for item in pdb_items
        if not item.get("human") and not item.get("gene") and item.get("name")
    ]
    if homolog_names:
        homolog_map = _protein_name_to_gene(homolog_names, job)
        for item in pdb_items:
            if item.get("human") or item.get("gene"):
                continue
            mapped_gene, mapped_accession = homolog_map.get(
                item.get("name", ""), ("", "")
            )
            if not mapped_gene:
                continue
            key = mapped_gene.upper()
            if key in by_gene:
                merge_source(by_gene[key], "PDB homolog")
                continue
            target = _external_target(
                mapped_gene, "PDB homolog", 5.3,
                name=f"{mapped_gene}（PDB 同源结构）",
                accession=mapped_accession,
                count=item.get("count", 1),
                basis="non-human co-crystal homolog (protein-name match)",
                genes=[mapped_gene],
            )
            targets.append(target)
            by_gene[key] = target

    bindingdb_items: list[dict] = []
    for cutoff in (0.85, 0.70, 0.60, 0.50):
        bindingdb_items = _bindingdb_targets(smiles, job, cutoff=cutoff)
        if bindingdb_items:
            break
    # Only resolve names that can plausibly enter the final table; this keeps
    # the UniProt round-trips bounded for promiscuous compounds.
    bindingdb_map = _protein_name_to_gene(
        [item["name"] for item in bindingdb_items[:max(top_targets, 30)]], job
    )
    for item in bindingdb_items:
        name_key = item["name"].strip().lower()
        gene, accession = bindingdb_map.get(item["name"], ("", ""))
        if gene and gene.upper() in by_gene:
            merge_source(by_gene[gene.upper()], "BindingDB")
            continue
        if name_key in by_name:
            merge_source(by_name[name_key], "BindingDB")
            continue
        target = _external_target(
            gene, "BindingDB", item.get("median_pchembl") or 5.0,
            name=item["name"],
            accession=accession,
            count=item.get("count", 1),
            pchembl=item.get("median_pchembl"),
            best_nm=item.get("best_nm"),
            basis="similar-compound bioactivity",
        )
        targets.append(target)
        by_name[name_key] = target
        if gene:
            by_gene[gene.upper()] = target

    for item in _pubchem_targets(smiles, job, names=names):
        gene = item["gene"]
        key = gene.upper()
        if key in by_gene:
            merge_source(by_gene[key], "PubChem")
            continue
        target = _external_target(
            gene, "PubChem",
            _evidence_score(
                item.get("median_pchembl"), item.get("qualitative_n", 0)
            ),
            name=item["name"],
            accession=item.get("accession", ""),
            count=item.get("n", 1),
            pchembl=item.get("median_pchembl"),
            best_nm=item.get("best_nm"),
            basis="PubChem bioassay",
        )
        targets.append(target)
        by_gene[key] = target
        by_name[target["name"].lower()] = target

    for item in _opentargets_targets(
        exact.get("chembl_id", "") if exact else "", job
    ):
        gene = item["gene"]
        key = gene.upper()
        if key in by_gene:
            merge_source(by_gene[key], "Open Targets")
            if item.get("mechanism") and not by_gene[key].get("mechanism"):
                by_gene[key]["mechanism"] = item["mechanism"]
            continue
        target = _external_target(
            gene, "Open Targets", 6.2,
            name=f"{item.get('name') or gene}（Open Targets）",
            count=1,
            basis="approved drug mechanism",
            mechanism=item.get("mechanism", ""),
        )
        targets.append(target)
        by_gene[key] = target
        by_name[target["name"].lower()] = target

    # ---- SwissTargetPrediction (2D/3D ligand similarity) ----------------
    job.set_progress(0.82, "SwissTargetPrediction 相似性靶点预测")
    raw_organisms = options.get("swiss_organisms")
    if isinstance(raw_organisms, str):
        raw_organisms = [raw_organisms]
    if not raw_organisms:
        raw_organisms = [options.get("swiss_organism") or "Homo_sapiens"]
    allowed_organisms = {
        "Homo_sapiens", "Mus_musculus", "Rattus_norvegicus",
    }
    organisms = [
        organism for organism in raw_organisms if organism in allowed_organisms
    ] or ["Homo_sapiens"]
    merged_stp: dict[str, dict] = {}
    stp_status: dict[str, dict] = {}

    def fetch_organism(organism: str) -> tuple[str, dict]:
        if COMMERCIAL_MODE:
            # SwissTargetPrediction 仅授权学术使用，商业版不发起查询。
            return organism, {
                "available": False,
                "items": [],
                "error": "商业合规模式已跳过 SwissTargetPrediction（仅授权学术使用）",
            }
        return organism, _swiss_targets(smiles, job, organism=organism)

    # The three species are independent web queries; run them concurrently so
    # multi-species fishing does not multiply the wall-clock time.
    with ThreadPoolExecutor(max_workers=min(3, len(organisms))) as pool:
        organism_results = list(pool.map(fetch_organism, organisms))
    for organism, result in organism_results:
        stp_status[organism] = {
            "available": bool(result.get("available")),
            "count": len(result.get("items") or []),
            "error": result.get("error", ""),
            "job": result.get("job", ""),
        }
        for item in result.get("items") or []:
            gene = str(item.get("gene") or "").strip()
            if not gene:
                continue
            key = gene.upper()
            if key in merged_stp:
                base = merged_stp[key]
                base["probability"] = max(
                    float(base.get("probability") or 0),
                    float(item.get("probability") or 0),
                )
                base["known_3d"] = int(base.get("known_3d") or 0) + int(item.get("known_3d") or 0)
                base["known_2d"] = int(base.get("known_2d") or 0) + int(item.get("known_2d") or 0)
                base.setdefault("organisms", []).append(organism)
                continue
            record = dict(item)
            record["organisms"] = [organism]
            merged_stp[key] = record
    stp = {
        "items": list(merged_stp.values()),
        "available": any(status["available"] for status in stp_status.values()),
        "error": "; ".join(
            status["error"] for status in stp_status.values() if status["error"]
        ),
        "organism": " / ".join(organisms),
        "organisms": stp_status,
        "job": ", ".join(
            status["job"] for status in stp_status.values() if status["job"]
        ),
        "count": len(merged_stp),
    }
    for item in stp.get("items") or []:
        gene = item["gene"]
        key = gene.upper()
        probability = float(item.get("probability") or 0.0)
        known = int(item.get("known_3d") or 0) + int(item.get("known_2d") or 0)
        score = max(5.0, min(5.4 + 2.2 * probability, 7.8))
        if key in by_gene:
            target = by_gene[key]
            merge_source(target, "SwissTargetPrediction")
            target["stp_probability"] = probability
            target["stp_known_3d"] = item.get("known_3d", 0)
            target["stp_known_2d"] = item.get("known_2d", 0)
            if score > float(target.get("evidence_score") or 0):
                target["evidence_score"] = round(score, 2)
            continue
        target = _external_target(
            gene,
            "SwissTargetPrediction",
            score,
            name=f"{item.get('name') or gene}（SwissTargetPrediction）",
            accession=item.get("accession", ""),
            count=max(known, 1),
            basis="SwissTargetPrediction 2D/3D similarity",
            genes=[gene],
            organism=str(item.get("organism") or "Homo sapiens"),
        )
        target["stp_probability"] = probability
        target["stp_known_3d"] = item.get("known_3d", 0)
        target["stp_known_2d"] = item.get("known_2d", 0)
        target["stp_target_class"] = item.get("type", "")
        if item.get("target_chembl_id"):
            target["target_chembl_id"] = item["target_chembl_id"]
        targets.append(target)
        by_gene[key] = target
        by_name[target["name"].lower()] = target

    # ---- external gene list imported from PharmMapper / SEA / COMET etc. ----
    external_genes = [
        str(gene).strip()
        for gene in (options.get("external_genes") or [])
        if str(gene).strip()
    ]
    for gene in external_genes:
        key = gene.upper()
        if key in by_gene:
            merge_source(by_gene[key], "Imported")
            continue
        target = _external_target(
            gene, "Imported", 6.0,
            name=f"{gene}（外部预测导入）",
            basis="external target-prediction import",
            genes=[gene],
        )
        targets.append(target)
        by_gene[key] = target
        by_name[target["name"].lower()] = target

    # ---- local fallback: targets of structurally similar marketed drugs ----
    similar_drugs = _similar_drug_targets(smiles, job)
    for reference in similar_drugs:
        score = 5.0 + 2.0 * float(reference["tanimoto"])
        for gene in reference.get("targets") or []:
            key = gene.upper()
            if key in by_gene:
                merge_source(by_gene[key], "Similar drug")
                if score > float(by_gene[key].get("evidence_score") or 0):
                    by_gene[key]["evidence_score"] = round(score, 2)
                continue
            target = _external_target(
                gene, "Similar drug", score,
                name=f"{gene}（{reference['name']} 相似药）",
                basis=(
                    f"similar marketed drug ({reference['name']}, "
                    f"Tanimoto {reference['tanimoto']:.2f})"
                ),
                genes=[gene],
            )
            target["similar_drug"] = reference["name"]
            target["similar_drug_tanimoto"] = reference["tanimoto"]
            targets.append(target)
            by_gene[key] = target
            by_name[target["name"].lower()] = target

    external_sources = {
        "ChEMBL": {
            "available": bool(molecules),
            "count": len(molecules),
            "error": (
                ""
                if molecules
                else "ChEMBL 相似性与精确匹配接口未返回分子（服务不可用或结构无记录）"
            ),
        },
        "SwissTargetPrediction": {
            "available": bool(stp.get("available")),
            "count": len(stp.get("items") or []),
            "error": stp.get("error", ""),
            "organism": stp.get("organism", ""),
            "organisms": stp.get("organisms", {}),
            "job": stp.get("job", ""),
        },
        "Imported": {
            "available": bool(external_genes),
            "count": len(external_genes),
        },
        "SimilarDrug": {
            "available": bool(similar_drugs),
            "count": len(similar_drugs),
            "references": [
                {
                    "name": item["name"],
                    "tanimoto": item["tanimoto"],
                    "targets": item["targets"],
                }
                for item in similar_drugs
            ],
        },
    }

    targets.sort(
        key=lambda item: (
            -(item.get("evidence_score") or item.get("median_pchembl") or 0),
            -len(item.get("genes") or []),
            -item.get("n", 0),
        )
    )
    targets = targets[:top_targets]
    if not targets:
        raise ValueError(
            "ChEMBL、BindingDB、PubChem、DGIdb、Open Targets 与 PDB "
            "都没有找到该分子的蛋白靶点。请确认 SMILES 合法、稍后重试，"
            "或在网络药理学页面手动补充人源基因。"
        )

    source_counts: dict[str, int] = {}
    for item in targets:
        for source in item.get("sources", ["ChEMBL"]):
            source_counts[source] = source_counts.get(source, 0) + 1
    genes_total = len({
        gene for item in targets for gene in (item.get("genes") or [])
    })
    human_targets = sum(
        1
        for item in targets
        if item.get("genes")
        and "homo sapiens" in str(item.get("organism") or "").lower()
    )
    non_human_targets = sum(
        1
        for item in targets
        if "homo sapiens" not in str(item.get("organism") or "").lower()
    )
    pchembl_targets = sum(
        1 for item in targets if item.get("median_pchembl") is not None
    )
    if human_targets == 0:
        job.log(
            "[垂钓] 仅找到非人源靶点或缺少基因注释，"
            "不能用于人源网络药理学；请换用已知药物名称或补充基因列表。"
        )
    elif human_targets < 8:
        job.log(
            f"[垂钓] 人源基因靶点 {human_targets} 个，数量偏少；"
            "网络药理学将使用 STRING 一阶邻居扩展网络，"
            "扩展节点只作为网络背景，不会冒充直接靶点。"
        )
    if len(targets) < GOAL_MOLECULES:
        job.log(
            f"[垂钓] 当前得到 {len(targets)} 个蛋白靶点，"
            f"已把阈值扩展到 Tanimoto ≥ {effective}%、"
            f"{len(molecules)} 个相似分子；可在网络药理学页面补充基因。"
        )

    record_id = f"fish_{datetime.now():%Y%m%d_%H%M%S}"
    outdir = project.root / "exports" / "fishing" / record_id
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": record_id,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "compound": {
            "name": name if name and name.strip().lower() != "query" else "查询分子",
            "smiles": smiles,
        },
        "parameters": {
            "similarity": effective,
            "requested_similarity": requested,
            "thresholds_tried": tried,
            "molecules": limit,
            "targets": top_targets,
        },
        "molecules": molecules,
        "targets": targets,
        "external_sources": external_sources,
        "stats": {
            "activity_records": sum(item.get("n", 0) for item in targets),
            "mechanism_records": mechanism_records,
            "molecules": len(molecules),
            "targets": len(targets),
            "human_targets": human_targets,
            "non_human_targets": non_human_targets,
            "pchembl_targets": pchembl_targets,
            "genes": genes_total,
            "sources": source_counts,
            "external_sources": external_sources,
            "activity_types": activity_types,
            "elapsed": round(time.time() - started, 1),
        },
        "source": (
            "ChEMBL + BindingDB + PubChem + DGIdb + Open Targets "
            "+ RCSB PDB + UniProt + SwissTargetPrediction + local similar drugs"
            + (" + imported external genes" if external_genes else "")
        ),
    }
    data_path = outdir / "fishing.json"
    write_json(data_path, payload)

    summary = {
        "id": record_id,
        "created": payload["created"],
        "compound": payload["compound"],
        "targets": targets[:25],
        "target_count": len(targets),
        "gene_count": genes_total,
        "human_target_count": human_targets,
        "stats": payload["stats"],
        "parameters": payload["parameters"],
        "external_sources": external_sources,
        "data": str(data_path.relative_to(project.root)),
    }
    project.data.setdefault("fishing", []).insert(0, summary)
    project.data["fishing"] = project.data["fishing"][:20]
    project.save()
    job.set_progress(1.0, "靶点垂钓完成")
    job.result = summary
    return summary


def load_fishing(project: Project, record_id: str = "") -> dict | None:
    items = project.data.get("fishing", [])
    record = None
    if record_id:
        record = next((item for item in items if item.get("id") == record_id), None)
    if record is None and items:
        record = items[0]
    if not record:
        return None
    path = Path(record.get("data", ""))
    if not path.is_absolute():
        path = project.root / path
    if not path.exists():
        return record
    data = json.loads(path.read_text(encoding="utf-8"))
    # Legacy external targets stored a pseudo-pChEMBL value.  Blank it so old
    # projects cannot leak a fabricated number into a new report.
    for target in data.get("targets") or []:
        if not target.get("target_chembl_id") and "pchembl_n" not in target:
            target["median_pchembl"] = None
            target.setdefault("score_basis", "external evidence (legacy record)")
            target.setdefault("pchembl_n", 0)
            target.setdefault("qualitative_n", target.get("n", 0))
    return data
