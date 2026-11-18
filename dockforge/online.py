"""Best-effort online evidence for compounds and targets.

All calls are optional: the report must still render when the machine is
offline.  Each adapter returns ``available`` plus a short error message instead
of raising, and results are cached in memory for one hour.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = "DrugHX/2.0 (academic docking workbench)"
_CACHE: dict[str, tuple[float, object]] = {}
_TTL = 3600
SPECIES_ORGANISM = {
    "human": 9606,
    "pig": 9823,
    "cattle": 9913,
    "sheep": 9940,
    "dog": 9615,
    "cat": 9685,
    "rat": 10116,
}


def _cached(key: str):
    hit = _CACHE.get(key)
    if not hit:
        return None
    stamp, value = hit
    if time.time() - stamp > _TTL:
        _CACHE.pop(key, None)
        return None
    return value


def _store(key: str, value):
    _CACHE[key] = (time.time(), value)
    return value


def _get_json(url: str, timeout: int = 18):
    cached = _cached(url)
    if cached is not None:
        return cached
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8", errors="replace"))
    return _store(url, data)


def _fail(error: Exception | str) -> dict:
    return {"available": False, "error": str(error)[:180]}


def _smiles_variants(smiles: str) -> list[str]:
    """Original SMILES plus RDKit-normalised variants for fragile web APIs."""
    variants = [str(smiles).strip()]
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            reduced = Chem.RemoveHs(mol)
            variants.append(Chem.MolToSmiles(reduced, isomericSmiles=True))
            variants.append(Chem.MolToSmiles(reduced, isomericSmiles=False))
    except Exception:
        pass
    unique = []
    for item in variants:
        if item and item not in unique:
            unique.append(item)
    return unique


def pubchem_compound(smiles: str) -> dict:
    """Experimental identity and computed descriptors from PubChem."""
    if not smiles:
        return _fail("no SMILES")
    payload = None
    last_error: Exception | None = None
    for variant in _smiles_variants(smiles):
        encoded = urllib.parse.quote(variant, safe="")
        prop_url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
            f"{encoded}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,"
            "IsomericSMILES,XLogP,ExactMass,TPSA,HBondDonorCount,HBondAcceptorCount,"
            "RotatableBondCount/JSON"
        )
        try:
            payload = _get_json(prop_url)
            break
        except Exception as exc:
            last_error = exc
    if payload is None:
        return _fail(last_error or "PubChem request failed")
    try:
        rows = payload.get("PropertyTable", {}).get("Properties") or []
        if not rows:
            return _fail("PubChem returned no record")
        row = rows[0]
        cid = row.get("CID")
        name = ""
        if cid:
            try:
                syn = _get_json(
                    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                    f"{cid}/synonyms/JSON"
                )
                names = (
                    syn.get("InformationList", {}).get("Information", [{}])[0]
                    .get("Synonym") or []
                )
                name = names[0] if names else ""
            except Exception:
                name = ""
        return {
            "available": True,
            "cid": cid,
            "name": name,
            "formula": row.get("MolecularFormula"),
            "mw": row.get("MolecularWeight"),
            "xlogp": row.get("XLogP"),
            "tpsa": row.get("TPSA"),
            "hbd": row.get("HBondDonorCount"),
            "hba": row.get("HBondAcceptorCount"),
            "rotb": row.get("RotatableBondCount"),
            "source": "PubChem PUG REST",
        }
    except Exception as exc:
        return _fail(exc)


def chembl_compound(smiles: str) -> dict:
    """ChEMBL molecule record matched by canonical SMILES."""
    if not smiles:
        return _fail("no SMILES")
    payload = None
    last_error: Exception | None = None
    for variant in _smiles_variants(smiles):
        encoded = urllib.parse.quote(variant, safe="")
        url = (
            "https://www.ebi.ac.uk/chembl/api/data/molecule.json?"
            f"molecule_structures__canonical_smiles__flexmatch={encoded}&limit=1"
        )
        try:
            payload = _get_json(url)
            break
        except Exception as exc:
            last_error = exc
    if payload is None:
        return _fail(last_error or "ChEMBL request failed")
    try:
        rows = payload.get("molecules") or []
        if not rows:
            return _fail("no ChEMBL match")
        row = rows[0]
        return {
            "available": True,
            "chembl_id": row.get("molecule_chembl_id"),
            "pref_name": row.get("pref_name") or "",
            "max_phase": row.get("max_phase"),
            "first_approval": row.get("first_approval"),
            "molecule_type": row.get("molecule_type"),
            "indication_class": row.get("indication_class"),
            "source": "ChEMBL REST",
        }
    except Exception as exc:
        return _fail(exc)


def uniprot_target(gene: str, species: str = "human") -> dict:
    """Reviewed UniProt entry: name, function, location, tissue specificity."""
    gene = str(gene or "").strip()
    if not gene:
        return _fail("no gene")
    organism = SPECIES_ORGANISM.get(species, 9606)
    query = f"gene_exact:{gene} AND organism_id:{organism} AND reviewed:true"
    url = (
        "https://rest.uniprot.org/uniprotkb/search?"
        + urllib.parse.urlencode(
            {
                "query": query,
                "format": "json",
                "size": "1",
                "fields": (
                    "accession,gene_names,xref_ensembl,protein_name,"
                    "cc_function,cc_subcellular_location,cc_tissue_specificity"
                ),
            }
        )
    )
    try:
        payload = _get_json(url)
        rows = payload.get("results") or []
        if not rows:
            return _fail(f"UniProt has no reviewed {gene} entry for {species}")
        row = rows[0]
        protein_name = (
            ((row.get("proteinDescription") or {}).get("recommendedName") or {})
            .get("fullName", {})
            .get("value", "")
        )
        function = ""
        tissue = ""
        locations = []
        for comment in row.get("comments") or []:
            kind = comment.get("commentType")
            if kind == "FUNCTION" and not function:
                texts = comment.get("texts") or []
                if texts:
                    function = str(texts[0].get("value") or "")
            elif kind == "TISSUE SPECIFICITY" and not tissue:
                texts = comment.get("texts") or []
                if texts:
                    tissue = str(texts[0].get("value") or "")
            elif kind == "SUBCELLULAR LOCATION":
                for item in comment.get("subcellularLocations") or []:
                    value = (item.get("location") or {}).get("value")
                    if value and value not in locations:
                        locations.append(value)
        ensembl = ""
        for xref in row.get("uniProtKBCrossReferences") or []:
            if xref.get("database") != "Ensembl":
                continue
            for prop in xref.get("properties") or []:
                if prop.get("key") == "GeneId" and prop.get("value"):
                    ensembl = prop["value"].split(".")[0]
                    break
            if ensembl:
                break
        return {
            "available": True,
            "gene": gene,
            "species": species,
            "accession": row.get("primaryAccession"),
            "protein_name": protein_name,
            "function": function,
            "subcellular": locations,
            "tissue_specificity": tissue,
            "ensembl": ensembl,
            "source": "UniProt REST",
        }
    except Exception as exc:
        return _fail(exc)


def hpa_target(ensembl: str) -> dict:
    """Human Protein Atlas gene record (human only)."""
    if not ensembl or not str(ensembl).startswith("ENSG"):
        return _fail("no Ensembl gene id")
    url = f"https://www.proteinatlas.org/{ensembl}.json"
    try:
        payload = _get_json(url)
        top = payload.get("RNA tissue specific nTPM")
        if not isinstance(top, dict):
            top = {}
        return {
            "available": True,
            "gene": payload.get("Gene"),
            "rna_specificity": payload.get("RNA tissue specificity"),
            "rna_distribution": payload.get("RNA tissue distribution"),
            "rna_top_tissue": {
                str(key): value for key, value in list(top.items())[:6]
            },
            "protein_tissue_specificity": payload.get("Protein tissue specificity"),
            "tissue_cluster": payload.get("Tissue expression cluster"),
            "subcellular": payload.get("Subcellular main location") or [],
            "source": "Human Protein Atlas",
        }
    except Exception as exc:
        return _fail(exc)


def target_profile(gene: str, species: str = "human") -> dict:
    data = uniprot_target(gene, species)
    if data.get("available") and species == "human" and data.get("ensembl"):
        data["hpa"] = hpa_target(data["ensembl"])
    else:
        data["hpa"] = {
            "available": False,
            "error": "HPA 仅提供人源表达" if species == "human" else "非人源物种无 HPA 数据",
        }
    data["gene"] = gene
    data["species"] = species
    return data


def target_profiles(genes: list[str], species: str = "human", limit: int = 12) -> dict:
    genes = [str(gene).strip() for gene in genes if str(gene).strip()][:limit]
    if not genes:
        return {"available": False, "error": "没有靶点基因", "targets": []}
    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(target_profile, gene, species): gene for gene in genes
        }
        for future in as_completed(futures):
            gene = futures[future]
            try:
                results[gene] = future.result()
            except Exception as exc:
                results[gene] = {"available": False, "error": str(exc), "gene": gene}
    ordered = [results.get(gene, {"gene": gene, "available": False}) for gene in genes]
    return {
        "available": any(item.get("available") for item in ordered),
        "species": species,
        "targets": ordered,
    }


def online_evidence(
    smiles: str,
    genes: list[str] | None = None,
    species: str = "human",
) -> dict:
    """Aggregate optional online evidence; never raises."""
    genes = [str(gene).strip() for gene in (genes or []) if str(gene).strip()]
    compound = {"pubchem": pubchem_compound(smiles), "chembl": chembl_compound(smiles)}
    targets = target_profiles(genes, species, limit=10) if genes else {
        "available": False,
        "error": "未提供靶点基因",
        "targets": [],
    }
    available = any(
        block.get("available")
        for block in (compound.get("pubchem"), compound.get("chembl"))
    ) or targets.get("available")
    return {
        "available": bool(available),
        "queried_at": time.strftime("%Y-%m-%d %H:%M"),
        "species": species,
        "compound": compound,
        "targets": targets,
        "sources": [
            "PubChem PUG REST",
            "ChEMBL REST",
            "UniProt REST",
            "Human Protein Atlas",
        ],
        "caveat": (
            "Online records are database evidence, not predictions; availability "
            "depends on the network and on whether the molecule/target is curated."
        ),
    }


# ---------------------------------------------------------------------------
# Online intelligence adapters
#
# Everything below is free and key-less.  Each adapter returns ``available``
# plus a short error string instead of raising, so the UI can degrade
# gracefully when a service is unreachable.
# ---------------------------------------------------------------------------
ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)

ENRICHR_LIBRARIES = {
    "KEGG_2021_Human": "KEGG 2021 Human",
    "GO_Biological_Process_2023": "GO Biological Process 2023",
    "Reactome_2022": "Reactome 2022",
    "WikiPathway_2023_Human": "WikiPathway 2023 Human",
}


def _hash_key(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _post_json(url: str, payload: dict, timeout: int = 45):
    key = _hash_key("POSTJSON", url, json.dumps(payload, sort_keys=True))
    cached = _cached(key)
    if cached is not None:
        return cached
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"User-Agent": UA, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8", errors="replace"))
    return _store(key, data)


def _get_text(url: str, timeout: int = 90) -> str:
    cached = _cached(url)
    if cached is not None:
        return str(cached)
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    return _store(url, text)


def _post_text(url: str, body: str, timeout: int = 300) -> str:
    key = _hash_key("POSTTEXT", url, body)
    cached = _cached(key)
    if cached is not None:
        return str(cached)
    request = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        headers={"User-Agent": UA, "Content-Type": "text/plain"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    return _store(key, text)


def _post_multipart(url: str, fields: dict[str, str], timeout: int = 60):
    boundary = "----DrugHX" + uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        chunks.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        )
    body = ("".join(chunks) + f"--{boundary}--\r\n").encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "User-Agent": UA,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _clean_sequence(text: str) -> str:
    return re.sub(r"[^A-Za-z]", "", str(text or "")).upper()


def plddt_stats(pdb_text: str) -> dict:
    """Per-residue pLDDT stored in the B-factor column of CA atoms."""
    values = []
    for line in str(pdb_text or "").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            values.append(float(line[60:66]))
        except ValueError:
            continue
    if not values:
        return {"mean": None, "min": None, "max": None, "n": 0,
                "very_high": None, "confident": None, "low": None}
    # AlphaFold writes pLDDT on a 0-100 scale, ESMFold on a 0-1 scale.
    if max(values) <= 1.5:
        values = [value * 100.0 for value in values]
    n = len(values)
    return {
        "mean": round(sum(values) / n, 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "n": n,
        "very_high": round(100.0 * sum(v >= 90 for v in values) / n, 1),
        "confident": round(100.0 * sum(70 <= v < 90 for v in values) / n, 1),
        "low": round(100.0 * sum(v < 50 for v in values) / n, 1),
    }


def uniprot_lookup(query: str, species: str = "human") -> dict:
    """Reviewed UniProt entry with sequence, Ensembl id and annotation."""
    query = str(query or "").strip()
    if not query:
        return _fail("no query")
    organism = SPECIES_ORGANISM.get(species, 9606)
    if ACCESSION_RE.match(query.upper()):
        search = f"accession:{query.upper()}"
    else:
        search = f"gene_exact:{query} AND organism_id:{organism} AND reviewed:true"
    fields = (
        "accession,id,gene_names,protein_name,organism_name,length,sequence,"
        "xref_ensembl,cc_function,cc_subcellular_location,cc_tissue_specificity"
    )

    def _request(term: str):
        url = (
            "https://rest.uniprot.org/uniprotkb/search?"
            + urllib.parse.urlencode(
                {"query": term, "format": "json", "size": "1", "fields": fields}
            )
        )
        return _get_json(url, timeout=35)

    payload = None
    last_error: Exception | None = None
    for term in (search, f"gene:{query} AND organism_id:{organism}"):
        try:
            payload = _request(term)
            if (payload.get("results") or []):
                break
        except Exception as exc:
            last_error = exc
    if payload is None:
        return _fail(last_error or "UniProt request failed")
    rows = payload.get("results") or []
    if not rows:
        return _fail(f"UniProt 未找到 {query}（物种 {species}）的已审阅条目")
    row = rows[0]
    protein_name = (
        ((row.get("proteinDescription") or {}).get("recommendedName") or {})
        .get("fullName", {})
        .get("value", "")
    )
    function = ""
    tissue = ""
    locations = []
    for comment in row.get("comments") or []:
        kind = comment.get("commentType")
        texts = comment.get("texts") or []
        if kind == "FUNCTION" and not function and texts:
            function = str(texts[0].get("value") or "")
        elif kind == "TISSUE SPECIFICITY" and not tissue and texts:
            tissue = str(texts[0].get("value") or "")
        elif kind == "SUBCELLULAR LOCATION":
            for item in comment.get("subcellularLocations") or []:
                value = (item.get("location") or {}).get("value")
                if value and value not in locations:
                    locations.append(value)
    ensembl = ""
    for xref in row.get("uniProtKBCrossReferences") or []:
        if xref.get("database") != "Ensembl":
            continue
        for prop in xref.get("properties") or []:
            if prop.get("key") == "GeneId" and prop.get("value"):
                ensembl = prop["value"].split(".")[0]
                break
        if ensembl:
            break
    genes = []
    for gene in row.get("genes") or []:
        name = (gene.get("geneName") or {}).get("value")
        if name and name not in genes:
            genes.append(name)
    sequence = (row.get("sequence") or {}).get("value") or ""
    return {
        "available": True,
        "query": query,
        "species": species,
        "accession": row.get("primaryAccession"),
        "entry_name": row.get("uniProtkbId"),
        "gene": genes[0] if genes else query,
        "genes": genes,
        "protein_name": protein_name,
        "organism": (row.get("organism") or {}).get("scientificName", ""),
        "length": (row.get("sequence") or {}).get("length") or len(sequence),
        "sequence": sequence,
        "ensembl": ensembl,
        "function": function,
        "subcellular": locations,
        "tissue_specificity": tissue,
        "source": "UniProt REST",
    }


def alphafold_model(accession: str) -> dict:
    """Download the latest AlphaFold DB model for a UniProt accession."""
    accession = str(accession or "").strip().upper()
    if not ACCESSION_RE.match(accession):
        return _fail("需要有效的 UniProt 登录号")
    try:
        rows = _get_json(
            f"https://alphafold.ebi.ac.uk/api/prediction/{accession}", timeout=35
        )
    except Exception as exc:
        return _fail(exc)
    if not isinstance(rows, list) or not rows:
        return _fail(f"AlphaFold DB 没有 {accession} 的模型")
    row = next((item for item in rows if item.get("isUniProt")), rows[0])
    pdb_url = row.get("pdbUrl")
    if not pdb_url:
        return _fail("AlphaFold DB 记录缺少 PDB 下载地址")
    try:
        pdb_text = _get_text(pdb_url, timeout=120)
    except Exception as exc:
        return _fail(exc)
    return {
        "available": True,
        "source": "AlphaFold DB",
        "accession": accession,
        "gene": row.get("gene") or "",
        "protein_name": row.get("uniprotDescription") or "",
        "organism": row.get("organismScientificName") or "",
        "length": (row.get("sequenceEnd") or 0) - (row.get("sequenceStart") or 0) + 1,
        "sequence": row.get("sequence") or "",
        "model_version": row.get("latestVersion"),
        "global_metric": row.get("globalMetricValue"),
        "pdb": pdb_text,
        "plddt": plddt_stats(pdb_text),
        "caveat": "AlphaFold 模型为预测结构，柔性区与配体结合构象需实验验证。",
    }


def esmfold_model(sequence: str) -> dict:
    """Predict a single-chain structure with the ESMFold public endpoint."""
    seq = _clean_sequence(sequence)
    if len(seq) < 20:
        return _fail("序列过短：至少需要 20 个氨基酸")
    if len(seq) > 600:
        return _fail("ESMFold 在线服务建议 ≤600 aa；请只提交目标结构域，或改用 AlphaFold DB")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            pdb_text = _post_text(
                "https://api.esmatlas.com/foldSequence/v1/pdb/", seq, timeout=300
            )
            if "ATOM" not in pdb_text:
                raise RuntimeError("ESMFold 返回内容不是 PDB")
            stats = plddt_stats(pdb_text)
            return {
                "available": True,
                "source": "ESMFold (ESM Atlas)",
                "accession": "",
                "gene": "",
                "protein_name": "de novo prediction",
                "length": len(seq),
                "sequence": seq,
                "model_version": "ESMFold v1",
                "global_metric": stats["mean"],
                "pdb": pdb_text,
                "plddt": stats,
                "caveat": "ESMFold 为单链预测，不含配体、辅因子与多聚体界面。",
            }
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    return _fail(last_error or "ESMFold request failed")


def rcsb_structures(accession: str, limit: int = 8) -> dict:
    """Experimental PDB entries mapped to a UniProt accession."""
    accession = str(accession or "").strip().upper()
    if not accession:
        return _fail("no accession")
    search = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": (
                    "rcsb_polymer_entity_container_identifiers."
                    "reference_sequence_identifiers.database_accession"
                ),
                "operator": "exact_match",
                "value": accession,
            },
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max(1, min(limit, 20))},
            "results_verbosity": "compact",
        },
    }
    try:
        found = _post_json("https://search.rcsb.org/rcsbsearch/v2/query", search)
    except Exception as exc:
        return _fail(exc)
    ids = found.get("result_set") or []
    if not ids:
        return {"available": True, "entries": [], "total": 0, "source": "RCSB PDB"}
    query = (
        "{ entries(entry_ids: %s) { rcsb_id struct { title } "
        "rcsb_entry_info { resolution_combined } nonpolymer_entities { "
        "nonpolymer_comp { chem_comp { id name } } } } }" % json.dumps(ids)
    )
    try:
        data = _post_json("https://data.rcsb.org/graphql", {"query": query})
    except Exception as exc:
        return _fail(exc)
    rows = (data.get("data") or {}).get("entries") or []
    by_id = {row.get("rcsb_id"): row for row in rows}
    entries = []
    for pdb_id in ids:
        row = by_id.get(pdb_id) or {}
        ligands = []
        for entity in row.get("nonpolymer_entities") or []:
            comp = ((entity.get("nonpolymer_comp") or {}).get("chem_comp") or {})
            if comp.get("id"):
                ligands.append({"id": comp["id"], "name": comp.get("name") or ""})
        resolution = (row.get("rcsb_entry_info") or {}).get("resolution_combined") or []
        entries.append(
            {
                "pdb_id": pdb_id,
                "title": (row.get("struct") or {}).get("title") or "",
                "resolution": resolution[0] if resolution else None,
                "ligands": ligands,
            }
        )
    return {
        "available": True,
        "entries": entries,
        "total": found.get("total_count", len(entries)),
        "source": "RCSB PDB",
    }


def opentargets_target(ensembl: str, limit: int = 8) -> dict:
    """Disease associations and tractability evidence from Open Targets."""
    ensembl = str(ensembl or "").strip()
    if not ensembl.startswith("ENSG"):
        return _fail("需要 Ensembl 基因 ID")
    query = (
        "query TargetIntel($id: String!) { target(ensemblId: $id) { "
        "approvedSymbol approvedName biotype functionDescriptions "
        "tractability { modality label value } "
        "associatedDiseases(page: {index: 0, size: %d}) { "
        "count rows { score disease { id name } } } } }" % max(1, min(limit, 20))
    )
    try:
        data = _post_json(
            "https://api.platform.opentargets.org/api/v4/graphql",
            {"query": query, "variables": {"id": ensembl}},
        )
    except Exception as exc:
        return _fail(exc)
    target = (data.get("data") or {}).get("target")
    if not target:
        return _fail("Open Targets 没有该靶点")
    diseases = []
    for row in ((target.get("associatedDiseases") or {}).get("rows") or []):
        disease = row.get("disease") or {}
        diseases.append(
            {
                "id": disease.get("id", ""),
                "name": disease.get("name", ""),
                "score": round(float(row.get("score") or 0), 3),
            }
        )
    tractability = []
    for row in target.get("tractability") or []:
        if row.get("value") is True:
            tractability.append(
                f"{row.get('modality', '')}: {row.get('label', '')}"
            )
    return {
        "available": True,
        "ensembl": ensembl,
        "symbol": target.get("approvedSymbol") or "",
        "name": target.get("approvedName") or "",
        "biotype": target.get("biotype") or "",
        "function": (target.get("functionDescriptions") or [""])[0],
        "diseases": diseases,
        "disease_count": (target.get("associatedDiseases") or {}).get("count", 0),
        "tractability": tractability[:12],
        "source": "Open Targets Platform",
    }


def target_intel(query: str, species: str = "human") -> dict:
    """UniProt + Open Targets + experimental PDB structures for one target."""
    uniprot = uniprot_lookup(query, species)
    if not uniprot.get("available"):
        return {"available": False, "error": uniprot.get("error"), "uniprot": uniprot}
    accession = uniprot.get("accession") or ""
    pdb_block = {"available": False, "entries": [], "error": "缺少 UniProt 登录号"}
    ot_block = {"available": False, "error": "缺少 Ensembl ID（Open Targets 仅支持人源基因）"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        if accession:
            futures[pool.submit(rcsb_structures, accession)] = "pdb"
        if uniprot.get("ensembl") and species == "human":
            futures[pool.submit(opentargets_target, uniprot["ensembl"])] = "ot"
        for future in as_completed(futures):
            kind = futures[future]
            try:
                value = future.result()
            except Exception as exc:
                value = _fail(exc)
            if kind == "pdb":
                pdb_block = value
            else:
                ot_block = value
    return {
        "available": True,
        "query": query,
        "species": species,
        "uniprot": uniprot,
        "pdb": pdb_block,
        "opentargets": ot_block,
        "sources": ["UniProt REST", "RCSB PDB", "Open Targets Platform"],
    }


def _pubchem_properties(cids: list[int]) -> list[dict]:
    ids = [str(int(cid)) for cid in cids if str(cid).isdigit()]
    if not ids:
        return []
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        + ",".join(ids)
        + "/property/Title,MolecularFormula,MolecularWeight,CanonicalSMILES,"
        "IsomericSMILES,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,"
        "RotatableBondCount/JSON"
    )
    payload = _get_json(url, timeout=40)
    rows = (payload.get("PropertyTable") or {}).get("Properties") or []
    by_id = {int(row.get("CID")): row for row in rows if row.get("CID")}
    synonyms: dict[int, list[str]] = {}
    try:
        syn = _get_json(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            + ",".join(ids)
            + "/synonyms/JSON",
            timeout=40,
        )
        for info in (syn.get("InformationList") or {}).get("Information") or []:
            cid = int(info.get("CID") or 0)
            if cid:
                synonyms[cid] = info.get("Synonym") or []
    except Exception:
        synonyms = {}
    output = []
    for cid in [int(item) for item in ids]:
        row = by_id.get(cid)
        if not row:
            continue
        names = synonyms.get(cid) or []
        output.append(
            {
                "cid": cid,
                "title": row.get("Title") or (names[0] if names else ""),
                "synonyms": names[:6],
                "formula": row.get("MolecularFormula"),
                "mw": row.get("MolecularWeight"),
                "smiles": row.get("IsomericSMILES") or row.get("CanonicalSMILES") or row.get("ConnectivitySMILES"),
                "canonical_smiles": row.get("CanonicalSMILES") or row.get("ConnectivitySMILES"),
                "xlogp": row.get("XLogP"),
                "tpsa": row.get("TPSA"),
                "hbd": row.get("HBondDonorCount"),
                "hba": row.get("HBondAcceptorCount"),
                "rotb": row.get("RotatableBondCount"),
            }
        )
    return output


def pubchem_compound(query: str, kind: str = "name") -> dict:
    """PubChem identity/descriptors by name or SMILES."""
    query = str(query or "").strip()
    if not query:
        return _fail("no query")
    encoded = urllib.parse.quote(query, safe="")
    endpoint = "smiles" if str(kind).lower() == "smiles" else "name"
    try:
        payload = _get_json(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{endpoint}/"
            f"{encoded}/cids/JSON",
            timeout=35,
        )
    except Exception as exc:
        return _fail(exc)
    cids = (payload.get("IdentifierList") or {}).get("CID") or []
    if not cids:
        return _fail("PubChem 没有匹配记录")
    rows = _pubchem_properties(cids[:1])
    if not rows:
        return _fail("PubChem 记录缺少属性")
    data = rows[0]
    data.update({"available": True, "source": "PubChem PUG REST", "query": query})
    return data


def pubchem_similar(smiles: str, threshold: float = 0.9, limit: int = 10) -> dict:
    """2D Tanimoto analogues from PubChem fast similarity search."""
    smiles = str(smiles or "").strip()
    if not smiles:
        return _fail("no SMILES")
    try:
        pct = int(round(float(threshold) * 100))
    except (TypeError, ValueError):
        pct = 90
    pct = max(40, min(99, pct))
    limit = max(1, min(int(limit or 10), 25))
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/"
        f"smiles/{urllib.parse.quote(smiles, safe='')}/cids/JSON"
        f"?Threshold={pct}&MaxRecords={limit}"
    )
    try:
        payload = _get_json(url, timeout=50)
    except Exception as exc:
        return _fail(exc)
    cids = (payload.get("IdentifierList") or {}).get("CID") or []
    if not cids:
        return {"available": True, "rows": [], "threshold": pct / 100, "source": "PubChem"}
    rows = _pubchem_properties(cids)
    return {
        "available": True,
        "rows": rows,
        "threshold": pct / 100,
        "count": len(rows),
        "source": "PubChem fast similarity (2D Tanimoto)",
    }


def enrichr_enrich(genes: list[str], library: str = "KEGG_2021_Human",
                   limit: int = 15) -> dict:
    """Pathway/GO enrichment through the public Enrichr service."""
    cleaned = []
    for gene in genes or []:
        value = str(gene).strip()
        if value and value not in cleaned:
            cleaned.append(value)
    cleaned = cleaned[:500]
    if len(cleaned) < 3:
        return _fail("至少需要 3 个基因")
    library = library if library in ENRICHR_LIBRARIES else "KEGG_2021_Human"
    try:
        added = _post_multipart(
            "https://maayanlab.cloud/Enrichr/addList",
            {"list": "\n".join(cleaned), "description": "DrugHX enrichment"},
        )
        user_list = added.get("userListId")
        if not user_list:
            return _fail("Enrichr 未返回 userListId")
        payload = _get_json(
            "https://maayanlab.cloud/Enrichr/enrich?"
            + urllib.parse.urlencode(
                {"userListId": user_list, "backgroundType": library}
            ),
            timeout=60,
        )
    except Exception as exc:
        return _fail(exc)
    raw = payload.get(library) or []
    rows = []
    for item in raw[: max(1, min(int(limit or 15), 50))]:
        try:
            rows.append(
                {
                    "rank": item[0],
                    "term": item[1],
                    "p_value": item[2],
                    "combined_score": item[4],
                    "overlap_genes": item[5] or [],
                    "adjusted_p": item[6],
                }
            )
        except (IndexError, TypeError):
            continue
    return {
        "available": True,
        "library": library,
        "library_label": ENRICHR_LIBRARIES[library],
        "input_count": len(cleaned),
        "rows": rows,
        "source": "Enrichr",
    }
