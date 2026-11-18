"""Network pharmacology: STRING PPI, multi-library Enrichr enrichment,
compound–target–disease (C-T-D) network and target–pathway Sankey data.

Input is a list of gene symbols (usually the top targets from target fishing).
The module returns a PPI network, enrichment tables for eight Enrichr
libraries, the C-T-D / Sankey structures and three CNS-style A4 figure pages
used by the final PDF report.
"""
from __future__ import annotations

import json
import math
import re
import textwrap
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from . import figures, i18n
from .project import Project
from .utils import write_json

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124 Safari/537.36"
)
STRING_API = "https://string-db.org/api/json/network"
ENRICHR_ADD = "https://maayanlab.cloud/Enrichr/addList"
ENRICHR_QUERY = "https://maayanlab.cloud/Enrichr/enrich"
LIBRARIES = {
    "kegg": "KEGG_2021_Human",
    "go_bp": "GO_Biological_Process_2023",
    "go_cc": "GO_Cellular_Component_2023",
    "go_mf": "GO_Molecular_Function_2023",
    "reactome": "Reactome_2022",
    "wikipathway": "WikiPathway_2023_Human",
    "disgenet": "DisGeNET",
    "hallmark": "MSigDB_Hallmark_2020",
}
LIB_TITLES = {
    "kegg": "KEGG 通路",
    "go_bp": "GO 生物学过程（BP）",
    "go_cc": "GO 细胞组分（CC）",
    "go_mf": "GO 分子功能（MF）",
    "reactome": "Reactome 通路",
    "wikipathway": "WikiPathway 通路",
    "disgenet": "DisGeNET 疾病关联",
    "hallmark": "MSigDB Hallmark 基因集",
}
FDR_CUTOFF = 0.05
_REQUEST_PAUSE = 0.25
_PATHWAY_KEYS = ("kegg", "reactome")
_PALETTE = [
    "#0f766e", "#0369a1", "#b45309", "#9f1239", "#4d7c0f", "#6d28d9",
    "#0e7490", "#a16207", "#be123c", "#15803d", "#7c3aed", "#475569",
]


def _string_network(
    genes: list[str], score: int, job, add_nodes: int = 0
) -> dict:
    params = {
        "identifiers": "\r".join(genes),
        "species": 9606,
        "required_score": score,
        "caller_identity": "DrugHX",
    }
    if add_nodes:
        params["add_nodes"] = add_nodes
    url = STRING_API + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        edges_raw = json.loads(response.read().decode("utf-8", errors="replace"))
    edges = []
    degrees: dict[str, int] = {gene: 0 for gene in genes}
    for item in edges_raw:
        first = item.get("preferredName_A")
        second = item.get("preferredName_B")
        if not first or not second:
            continue
        edges.append(
            {
                "a": first,
                "b": second,
                "score": int(round(float(item.get("score", 0)) * 1000)),
            }
        )
        degrees[first] = degrees.get(first, 0) + 1
        degrees[second] = degrees.get(second, 0) + 1
    seed_set = set(genes)
    nodes = [
        {
            "name": name,
            "degree": degree,
            "seed": name in seed_set,
        }
        for name, degree in sorted(degrees.items(), key=lambda kv: -kv[1])
    ]
    expanded = sorted(
        name for name in degrees if name not in seed_set and degrees[name] > 0
    )
    mapped = [gene for gene in genes if gene in degrees]
    missing = [gene for gene in genes if gene not in degrees]
    if missing:
        job.log(
            f"[网络] STRING 未映射 {len(missing)} 个基因："
            + "、".join(missing[:8])
        )
    if expanded:
        job.log(
            f"[网络] STRING 一阶邻居扩展 {len(expanded)} 个基因"
            "（仅作为网络背景，不计为直接靶点）"
        )
    job.log(
        f"[网络] STRING 返回 {len(edges)} 条边、{len(nodes)} 个节点"
        f"（种子 {len(mapped)} 个）"
    )
    return {
        "nodes": nodes,
        "edges": edges,
        "seed_genes": genes,
        "mapped_genes": mapped,
        "expanded_genes": expanded,
    }


def _enrichr(genes: list[str], library: str, job) -> list[dict]:
    boundary = "----DrugHXBoundary"
    parts = []
    for name, value in (("list", "\n".join(genes)), ("description", "DrugHX")):
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n"
        )
    body = ("".join(parts) + f"--{boundary}--\r\n").encode()
    request = urllib.request.Request(
        ENRICHR_ADD, data=body,
        headers={
            "User-Agent": UA,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode())
    user_list = payload.get("userListId")
    if not user_list:
        raise ValueError("Enrichr 未返回 userListId")
    url = ENRICHR_QUERY + "?" + urllib.parse.urlencode(
        {"userListId": user_list, "backgroundType": library}
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode())
    terms = data.get(library) or []
    rows = []
    for term in terms[:30]:
        try:
            rows.append(
                {
                    "term": term[1],
                    "p": float(term[2]),
                    "combined": float(term[4]),
                    "genes": term[5] or [],
                    "fdr": float(term[6]) if term[6] is not None else None,
                }
            )
        except (IndexError, TypeError, ValueError):
            continue
    rows.sort(key=lambda item: item["p"])
    job.log(f"[网络] {library} 富集条目 {len(rows)} 个")
    return rows


def _enrichment_all(genes: list[str], job) -> dict[str, dict]:
    """Query every Enrichr library; a single failing library is retried once."""
    enrichment: dict[str, dict] = {}
    for index, (key, library) in enumerate(LIBRARIES.items()):
        if index:
            time.sleep(_REQUEST_PAUSE)
        rows = None
        last_error = None
        for attempt in range(2):
            try:
                rows = _enrichr(genes, library, job)
                break
            except Exception as exc:  # 单个库失败不影响整体
                last_error = exc
                if attempt == 0:
                    job.log(f"[网络] {library} 请求失败，2 秒后重试：{exc}")
                    time.sleep(2.0)
        if rows is None:
            job.log(f"[网络] {library} 富集失败，已跳过：{last_error}")
            enrichment[key] = {
                "library": library,
                "terms": [],
                "significant": 0,
                "error": str(last_error)[:200],
            }
            continue
        significant = sum(
            1
            for item in rows
            if (item.get("fdr") if item.get("fdr") is not None else 1.0) < FDR_CUTOFF
        )
        enrichment[key] = {
            "library": library,
            "terms": rows[:30],
            "significant": significant,
        }
        if significant:
            job.log(f"[网络] {library} 中 FDR < 0.05 的条目 {significant} 个")
    return enrichment


def _clean_term(term) -> str:
    """Strip database identifiers that Enrichr appends to a term."""
    text = " ".join(str(term or "").split())
    text = re.sub(
        r"\s*\((?:GO:|hsa|WP|R-HSA)[-:]?\d+\)\s*$", "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\s+(?:GO:|hsa|WP|R-HSA)[-:]?\d+\s*$", "", text, flags=re.IGNORECASE
    )
    return text or str(term or "")


def _ellipsis(text: str, limit: int = 42) -> str:
    text = " ".join(i18n.tr(str(text or "")).split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 1)].rstrip(" ,;:-") + "…"


def _term_score(item: dict) -> float:
    return -math.log10(max(float(item.get("p") or 1.0), 1e-300))


def _fdr_score(item: dict) -> float:
    fdr = item.get("fdr")
    value = float(fdr) if fdr is not None else float(item.get("p") or 1.0)
    return -math.log10(max(value, 1e-300))


def _ctd_network(
    genes: list[str], compound: dict, enrichment: dict[str, dict], job
) -> dict | None:
    """Compound–target–disease network; diseases come from DisGeNET only."""
    entries = (enrichment.get("disgenet") or {}).get("terms") or []
    lookup = {gene.upper(): gene for gene in genes}
    diseases: list[dict] = []
    seen: set[str] = set()
    for item in entries:
        label = _clean_term(item.get("term"))
        if not label or label.lower() in seen:
            continue
        hits: list[str] = []
        for raw in item.get("genes") or []:
            gene = lookup.get(str(raw).strip().upper())
            if gene and gene not in hits:
                hits.append(gene)
        if not hits:
            continue
        seen.add(label.lower())
        diseases.append(
            {
                "term": _ellipsis(label, 46),
                "p": float(item.get("p") or 1.0),
                "fdr": item.get("fdr"),
                "genes": hits,
            }
        )
    diseases.sort(key=lambda item: item["p"])
    diseases = diseases[:12]
    if not diseases:
        job.log("[网络] DisGeNET 无与输入基因重叠的疾病条目，跳过 C-T-D 网络")
        return None

    scores = [-math.log10(max(item["p"], 1e-300)) for item in diseases]
    top_score = max(scores) or 1.0
    compound_name = str((compound or {}).get("name") or "查询分子").strip() or "查询分子"
    nodes = [
        {
            "id": "compound",
            "label": compound_name,
            "type": "compound",
            "degree": len(genes),
        }
    ]
    edges = [
        {"source": "compound", "target": f"gene:{gene}", "weight": 1.0,
         "evidence": "target"}
        for gene in genes
    ]
    disease_hits: dict[str, int] = {gene: 0 for gene in genes}
    for index, disease in enumerate(diseases):
        node_id = f"disease:{index + 1}"
        nodes.append(
            {
                "id": node_id,
                "label": disease["term"],
                "type": "disease",
                "degree": len(disease["genes"]),
                "p": disease["p"],
                "fdr": disease.get("fdr"),
                "genes": disease["genes"],
            }
        )
        for gene in disease["genes"]:
            disease_hits[gene] = disease_hits.get(gene, 0) + 1
            edges.append(
                {
                    "source": f"gene:{gene}",
                    "target": node_id,
                    "weight": round(max(scores[index] / top_score, 0.05), 3),
                    "overlap": len(disease["genes"]),
                    "evidence": "DisGeNET",
                }
            )
    for gene in genes:
        nodes.append(
            {
                "id": f"gene:{gene}",
                "label": gene,
                "type": "gene",
                "degree": 1 + disease_hits.get(gene, 0),
            }
        )
    job.log(
        f"[网络] C-T-D 网络：1 个化合物、{len(genes)} 个基因、"
        f"{len(diseases)} 个疾病条目、{len(edges)} 条边"
    )
    return {
        "nodes": nodes,
        "edges": edges,
        "compound": compound_name,
        "diseases": len(diseases),
        "source": "DisGeNET",
    }


def _sankey_data(
    genes: list[str], enrichment: dict[str, dict], job, limit: int = 12
) -> dict | None:
    """Target→pathway Sankey structure from significant KEGG/Reactome terms."""
    pathways: list[dict] = []
    seen: set[str] = set()
    for key in _PATHWAY_KEYS:
        for item in (enrichment.get(key) or {}).get("terms") or []:
            fdr = item.get("fdr")
            if fdr is None or float(fdr) >= FDR_CUTOFF:
                continue
            members = {str(gene).strip().upper() for gene in item.get("genes") or []}
            hits = [gene for gene in genes if gene.upper() in members]
            if len(hits) < 2:
                continue
            name = _clean_term(item.get("term"))
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            pathways.append(
                {
                    "name": name,
                    "library": key,
                    "genes": hits,
                    "p": float(item.get("p") or 1.0),
                    "score": _term_score(item),
                }
            )
    pathways.sort(key=lambda item: (-len(item["genes"]), -item["score"]))
    pathways = pathways[:limit]
    if not pathways:
        job.log("[网络] 无 ≥2 个输入基因命中的显著 KEGG/Reactome 通路，跳过桑基图")
        return None

    counts: dict[str, int] = {}
    for pathway in pathways:
        for gene in pathway["genes"]:
            counts[gene] = counts.get(gene, 0) + 1
    gene_order = sorted(counts, key=lambda gene: (-counts[gene], gene))
    nodes = [
        {"id": f"gene:{gene}", "label": gene, "type": "gene",
         "value": counts[gene]}
        for gene in gene_order
    ]
    links = []
    for index, pathway in enumerate(pathways):
        node_id = f"pathway:{index + 1}"
        nodes.append(
            {
                "id": node_id,
                "label": pathway["name"],
                "type": "pathway",
                "library": pathway["library"],
                "value": len(pathway["genes"]),
                "p": pathway["p"],
                "score": round(pathway["score"], 3),
                "genes": pathway["genes"],
            }
        )
        for gene in pathway["genes"]:
            links.append(
                {
                    "source": f"gene:{gene}",
                    "target": node_id,
                    "value": 1,
                    "weight": round(pathway["score"], 3),
                }
            )
    job.log(
        f"[网络] 桑基图：{len(pathways)} 条通路、{len(links)} 条基因–通路连接"
        f"（仅保留至少 2 个输入基因命中的显著通路）"
    )
    return {
        "nodes": nodes,
        "links": links,
        "genes": gene_order,
        "pathways": [
            {
                "name": item["name"],
                "library": item["library"],
                "genes": item["genes"],
                "p": item["p"],
            }
            for item in pathways
        ],
    }


def _enrichment_panel(ax, key: str, block: dict, norm, cmap: str):
    terms = [item for item in (block.get("terms") or []) if item.get("p")]
    terms.sort(key=lambda item: (item.get("fdr") if item.get("fdr") is not None
                                 else item["p"]))
    terms = terms[:5]
    title = figures._wrap_cjk(
        f"{LIB_TITLES.get(key, key)}"
        f"（显著 {int(block.get('significant') or 0)} 条）",
        40,
    )
    if not terms:
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=12.0, pad=8, color=figures.MUTED)
        error = str(block.get("error") or "").strip()
        if error:
            message = (
                "Request failed (skipped)"
                if i18n.get_lang() == "en"
                else "请求失败（已跳过）"
            )
            detail = _ellipsis(error, 64) if i18n.get_lang() != "en" else ""
        else:
            message = "No usable terms" if i18n.get_lang() == "en" else "未获取到可用条目"
            detail = ""
        ax.text(
            0.5, 0.5, message + ("\n" + detail if detail else ""),
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10.5, color=figures.MUTED, linespacing=1.5,
        )
        return None
    labels = [_ellipsis(_clean_term(item["term"]), 42) for item in terms]
    values = np.array([_term_score(item) for item in terms])
    colors = np.array([_fdr_score(item) for item in terms])
    sizes = [max(40.0, 30.0 + 22.0 * len(item.get("genes") or [])) for item in terms]
    y = np.arange(len(terms))
    scatter = ax.scatter(
        values, y, s=sizes, c=colors, cmap=cmap, norm=norm,
        edgecolor="white", linewidth=1.0, zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=figures.INK)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=12.0, pad=8)
    ax.set_xlabel("−log$_{10}$ P", fontsize=11.5)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_xlim(0, max(float(values.max()) * 1.20, 1.8))
    ax.axvline(
        -math.log10(0.05), color=figures.MUTED, linewidth=1.0,
        linestyle=(0, (3, 2)), zorder=1,
    )
    figures._compact_frame(ax, "x")
    return scatter


def enrichment_pdf(data: dict, outpath: Path) -> Path:
    """Multi-library enrichment dot plots (CNS style, A4 portrait)."""
    figures._style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    enrichment = data.get("enrichment") or {}
    order = [key for key in LIBRARIES if key in enrichment]
    order += [key for key in enrichment if key not in LIBRARIES]
    groups = [order[index:index + 3] for index in range(0, len(order), 3)]

    with PdfPages(outpath) as pdf:
        if not groups:
            fig = figures._a4_page(
                "图 21　多库富集分析", "Enrichr 未返回可用富集结果",
                "Fig. 21　Multi-library enrichment",
            )
            ax = fig.add_axes([0.12, 0.32, 0.76, 0.42])
            ax.axis("off")
            ax.text(
                0.5, 0.5, "Enrichr 未返回可用富集条目", transform=ax.transAxes,
                ha="center", va="center", fontsize=15, color=figures.MUTED,
            )
            figures._caption(fig, "Enrichr 请求均失败或未返回条目，本页仅作说明。")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            return outpath

        for page_index, group in enumerate(groups):
            suffix = "" if page_index == 0 else "（续）"
            fig = figures._a4_page(
                f"图 21　多库富集分析{suffix}",
                "Enrichr 多库富集：点大小为命中基因数，颜色为 −log10 FDR",
                "Fig. 21　Multi-library enrichment",
            )
            grid = fig.add_gridspec(
                len(group), 1, left=0.47, right=0.88, top=0.872,
                bottom=0.248, hspace=0.72 if len(group) >= 3 else 1.05,
            )
            scores = []
            for key in group:
                for item in ((enrichment.get(key) or {}).get("terms") or [])[:5]:
                    if item.get("p"):
                        scores.append(_fdr_score(item))
            norm = Normalize(
                vmin=0.0, vmax=max(max(scores) if scores else 1.0, 1.3)
            )
            scatters = []
            for row, key in enumerate(group):
                ax = fig.add_subplot(grid[row])
                scatter = _enrichment_panel(
                    ax, key, enrichment.get(key) or {}, norm, "viridis"
                )
                if scatter is not None:
                    scatters.append(scatter)
                tag_y = 1.0 + 0.030 / max(ax.get_position().height, 0.01)
                figures._panel_tag(ax, "abc"[row % 3], x=-0.32, y=tag_y)
            if scatters:
                cax = fig.add_axes([0.900, 0.36, 0.012, 0.34])
                colorbar = fig.colorbar(scatters[0], cax=cax)
                colorbar.set_label("−log$_{10}$ FDR", fontsize=10.5)
                colorbar.ax.tick_params(labelsize=9.5)
            if i18n.get_lang() == "en":
                caption = (
                    "Figure 21. Multi-library enrichment dot plot: x-axis, "
                    "−log10 P; dot size, number of input genes in the term; "
                    "colour, −log10 FDR (brighter is more significant); dashed "
                    "line, P = 0.05 reference (−log10 0.05 ≈ 1.30). Only the five "
                    "lowest-FDR terms per library are shown; term names are "
                    "truncated to 42 characters. Libraries: KEGG 2021 Human, "
                    "GO BP/CC/MF 2023, Reactome 2022, WikiPathway 2023 Human, "
                    "DisGeNET and MSigDB Hallmark 2020; enrichment depends on "
                    "existing database annotations and is only for hypothesis "
                    "generation."
                )
            else:
                caption = (
                    "Enrichr 多库富集分析点图：横轴为 −log10 P，点大小为条目命中的输入基因数，"
                    "颜色为 −log10 FDR（越亮越显著），虚线为 P = 0.05 参考线"
                    "（−log10 0.05 ≈ 1.30）。每个库仅展示 FDR 最小的前 5 条条目，"
                    "条目名称截断至 42 字符。Enrichr 库包括 KEGG 2021 Human、"
                    "GO BP/CC/MF 2023、Reactome 2022、WikiPathway 2023 Human、"
                    "DisGeNET 与 MSigDB Hallmark 2020；富集结果依赖数据库既有注释，"
                    "仅用于机制假设生成。"
                )
            failed = [
                LIB_TITLES.get(key, key)
                for key in group
                if str((enrichment.get(key) or {}).get("error") or "").strip()
            ]
            if failed:
                caption += (
                    " Libraries that failed after one retry and were skipped: "
                    + ", ".join(failed)
                    + "."
                    if i18n.get_lang() == "en"
                    else "以下库请求失败并跳过："
                    + "、".join(failed)
                    + "；对应面板仅说明缺失原因，不代表未检出显著条目。"
                )
            figures._caption(fig, caption)
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    return outpath


def ctd_pdf(data: dict, outpath: Path) -> Path:
    """Compound–target–disease network on an A4 portrait page."""
    figures._style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    ctd = data.get("ctd") or {}
    nodes = ctd.get("nodes") or []
    edges = ctd.get("edges") or []
    compound_name = str((data.get("compound") or {}).get("name") or "查询分子")

    with PdfPages(outpath) as pdf:
        english = i18n.get_lang() == "en"
        subtitle = (
            "C-T-D network: compound at the centre, genes in the middle ring, diseases outside"
            if english
            else "C-T-D 网络：候选分子居中，靶点基因中环，疾病条目外环"
        )
        fig = figures._a4_page(
            "图 22　化合物–靶点–疾病网络",
            subtitle,
            "Fig. 22　Compound–target–disease network",
        )
        if not nodes:
            ax = fig.add_axes([0.12, 0.32, 0.76, 0.42])
            ax.axis("off")
            ax.text(
                0.5, 0.5, "DisGeNET 富集未返回可用疾病条目\n未构建 C-T-D 网络",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=15, color=figures.MUTED, linespacing=1.6,
            )
            figures._caption(
                fig,
                "DisGeNET 富集结果中没有与输入基因重叠的疾病条目，为避免伪造疾病关联，"
                "本图未绘制。",
            )
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            return outpath

        gene_nodes = [node for node in nodes if node.get("type") == "gene"]
        disease_nodes = [node for node in nodes if node.get("type") == "disease"]
        compound_node = next(
            (node for node in nodes if node.get("type") == "compound"), None
        )
        positions: dict[str, tuple[float, float]] = {}
        if compound_node:
            positions[compound_node["id"]] = (0.0, 0.0)
        gene_radius = 0.62
        for index, node in enumerate(gene_nodes):
            angle = math.pi / 2 - 2 * math.pi * index / max(len(gene_nodes), 1)
            positions[node["id"]] = (
                gene_radius * math.cos(angle), gene_radius * math.sin(angle)
            )
        disease_radius = 1.16
        for index, node in enumerate(disease_nodes):
            angle = math.pi / 2 - 2 * math.pi * (index + 0.5) / max(
                len(disease_nodes), 1
            )
            positions[node["id"]] = (
                disease_radius * math.cos(angle), disease_radius * math.sin(angle)
            )

        ax = fig.add_axes([0.13, 0.232, 0.74, 0.626])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)

        for edge in edges:
            start = positions.get(edge.get("source"))
            end = positions.get(edge.get("target"))
            if not start or not end:
                continue
            if edge.get("evidence") == "target":
                ax.plot(
                    [start[0], end[0]], [start[1], end[1]],
                    color="#cbd5e1", linewidth=1.0, zorder=1,
                    solid_capstyle="round",
                )
            else:
                weight = min(float(edge.get("weight") or 0.0), 1.0)
                ax.plot(
                    [start[0], end[0]], [start[1], end[1]],
                    color="#94a3b8", linewidth=0.6 + 1.6 * weight,
                    alpha=0.5, zorder=1,
                )

        for node in nodes:
            x, y = positions.get(node["id"], (0.0, 0.0))
            kind = node.get("type")
            degree = int(node.get("degree") or 0)
            if kind == "compound":
                size, color = 1250.0, "#b45309"
            elif kind == "gene":
                size, color = 110.0 + 95.0 * degree, "#0f766e"
            else:
                size, color = 95.0 + 62.0 * degree, "#9f1239"
            ax.scatter(
                [x], [y], s=size, color=color, edgecolor="white",
                linewidth=1.4, zorder=4,
            )

        if compound_node:
            ax.text(
                *positions[compound_node["id"]],
                "Cpd" if english else "化合物",
                ha="center", va="center", fontsize=8.6,
                fontweight="bold", color="white", zorder=6,
            )

        for node in gene_nodes:
            x, y = positions[node["id"]]
            angle = math.atan2(y, x)
            flip = math.cos(angle) < 0
            rotation = math.degrees(angle) + (180 if flip else 0)
            ax.text(
                x * 1.18, y * 1.18, node.get("label", ""),
                rotation=rotation, rotation_mode="anchor",
                ha="right" if flip else "left", va="center",
                fontsize=11.5, color=figures.INK, zorder=5,
                bbox={"boxstyle": "round,pad=0.10", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.70},
            )

        for index, node in enumerate(disease_nodes):
            x, y = positions[node["id"]]
            angle = math.atan2(y, x)
            flip = math.cos(angle) < 0
            rotation = math.degrees(angle) + (180 if flip else 0)
            label = textwrap.fill(_ellipsis(node.get("label", ""), 42), 24)
            # 交错外移，降低相邻疾病标签以及与基因标签的重叠
            radius = 1.14 if index % 2 == 0 else 1.38
            ax.text(
                x * radius, y * radius, label,
                rotation=rotation, rotation_mode="anchor",
                ha="right" if flip else "left", va="center",
                fontsize=9.8, color="#7f1d1d", linespacing=1.20, zorder=6,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.72},
            )

        handles = [
            Line2D([], [], marker="o", color="none", markerfacecolor="#b45309",
                   markeredgecolor="white", markersize=15, label="化合物"),
            Line2D([], [], marker="o", color="none", markerfacecolor="#0f766e",
                   markeredgecolor="white", markersize=11, label="靶点基因"),
            Line2D([], [], marker="o", color="none", markerfacecolor="#9f1239",
                   markeredgecolor="white", markersize=11,
                   label="疾病条目（DisGeNET）"),
        ]
        fig.legend(
            handles=handles, loc="lower left", bbox_to_anchor=(0.065, 0.866),
            ncol=3, frameon=False, fontsize=11, handletextpad=0.4,
            columnspacing=1.8,
        )
        if english:
            caption = (
                f"The compound ({compound_name}) is at the centre, "
                f"{len(gene_nodes)} target genes form the middle ring and "
                f"{len(disease_nodes)} disease terms the outer ring. Node size "
                "scales with connectivity; light grey lines are compound–target "
                "(C–T) links and darker lines are gene–disease (T–D) links, "
                "with width proportional to normalised −log10 P. Disease terms "
                "are the top 12 DisGeNET enrichment entries overlapping the input "
                "genes (ranked by P); they represent existing database records "
                "and are hypotheses requiring validation, not causal evidence."
            )
        else:
            caption = (
                f"化合物（{compound_name}）位于中心，{len(gene_nodes)} 个靶点基因构成中间环，"
                f"{len(disease_nodes)} 个疾病条目位于外环；节点大小与连接度成正比，"
                "浅灰色细线为化合物–靶点关系（C–T），较深的细线为基因–疾病关系（T–D，"
                "线宽与 −log10 P 归一化权重成正比）。疾病条目取自 DisGeNET 富集结果中"
                "与输入基因有重叠的前 12 条（按 P 值排序），仅代表数据库中已有的关联记录，"
                "属于待验证的假设，不表示因果关系。"
            )
        figures._caption(fig, caption, y=0.160)
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)
    return outpath


def _sankey_ribbon(
    ax, x0: float, x1: float, top0: float, bottom0: float,
    top1: float, bottom1: float, color: str,
):
    t = np.linspace(0.0, 1.0, 80)
    xs = x0 + (x1 - x0) * t

    def curve(start: float, end: float) -> np.ndarray:
        return (
            ((1 - t) ** 3) * start
            + 3 * ((1 - t) ** 2) * t * start
            + 3 * (1 - t) * (t ** 2) * end
            + (t ** 3) * end
        )

    top = curve(top0, top1)
    bottom = curve(bottom0, bottom1)
    ax.fill(
        np.concatenate([xs, xs[::-1]]),
        np.concatenate([top, bottom[::-1]]),
        color=color, alpha=0.58, linewidth=0, zorder=2,
    )


def sankey_pdf(data: dict, outpath: Path) -> Path:
    """Target→pathway Sankey diagram on an A4 portrait page."""
    figures._style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    sankey = data.get("sankey") or {}
    nodes = sankey.get("nodes") or []
    links = sankey.get("links") or []

    with PdfPages(outpath) as pdf:
        fig = figures._a4_page(
            "图 23　靶点–通路桑基图",
            "左列输入基因、右列 KEGG / Reactome 显著通路，流宽度为命中基因数",
            "Fig. 23　Target–pathway Sankey diagram",
        )
        if not links:
            ax = fig.add_axes([0.12, 0.32, 0.76, 0.42])
            ax.axis("off")
            ax.text(
                0.5, 0.5, "无 ≥2 个输入基因命中的显著通路\n未绘制桑基图",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=15, color=figures.MUTED, linespacing=1.6,
            )
            figures._caption(
                fig,
                "KEGG 与 Reactome 富集中没有同时满足 FDR < 0.05 且至少 2 个输入基因命中的通路，"
                "为避免单基因噪声，本图未绘制。",
            )
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            return outpath

        gene_nodes = [node for node in nodes if node.get("type") == "gene"]
        path_nodes = [node for node in nodes if node.get("type") == "pathway"]
        gene_value = {node["id"]: 0 for node in gene_nodes}
        path_value = {node["id"]: 0 for node in path_nodes}
        for link in links:
            gene_value[link["source"]] = gene_value.get(link["source"], 0) + 1
            path_value[link["target"]] = path_value.get(link["target"], 0) + 1
        total = max(len(links), 1)
        gap = 0.016 if len(gene_nodes) + len(path_nodes) < 18 else 0.010

        def stack(ids: list[str], values: dict[str, int]):
            span = 1.0 - gap * max(len(ids) - 1, 0)
            cursor = 1.0
            spans: dict[str, tuple[float, float]] = {}
            for node_id in ids:
                height = values.get(node_id, 0) / total * span
                spans[node_id] = (cursor, cursor - height)
                cursor -= height + gap
            return spans, span

        left, left_span = stack([node["id"] for node in gene_nodes], gene_value)
        right, right_span = stack([node["id"] for node in path_nodes], path_value)

        ax = fig.add_axes([0.12, 0.245, 0.76, 0.612])
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.04, 1.04)
        ax.axis("off")

        left_x, right_x, bar_w = 0.20, 0.70, 0.014
        palette = {
            node["id"]: _PALETTE[index % len(_PALETTE)]
            for index, node in enumerate(path_nodes)
        }
        left_cursor = {node_id: spans[0] for node_id, spans in left.items()}
        right_cursor = {node_id: spans[0] for node_id, spans in right.items()}
        left_step = left_span / total
        right_step = right_span / total
        for link in links:
            source, target = link["source"], link["target"]
            top0 = left_cursor.get(source)
            top1 = right_cursor.get(target)
            if top0 is None or top1 is None:
                continue
            bottom0 = top0 - left_step
            bottom1 = top1 - right_step
            left_cursor[source] = bottom0
            right_cursor[target] = bottom1
            _sankey_ribbon(
                ax, left_x + bar_w, right_x, top0, bottom0, top1, bottom1,
                palette.get(target, "#94a3b8"),
            )

        for node in gene_nodes:
            top, bottom = left[node["id"]]
            ax.add_patch(
                Rectangle(
                    (left_x, bottom), bar_w, max(top - bottom, 0.004),
                    facecolor="#0f766e", edgecolor="none", zorder=4,
                )
            )
            ax.text(
                left_x - 0.014, (top + bottom) / 2, node.get("label", ""),
                ha="right", va="center", fontsize=12, color=figures.INK,
                zorder=5,
            )
        label_items = []
        for node in path_nodes:
            top, bottom = right[node["id"]]
            ax.add_patch(
                Rectangle(
                    (right_x, bottom), bar_w, max(top - bottom, 0.004),
                    facecolor=palette[node["id"]], edgecolor="none", zorder=4,
                )
            )
            label = textwrap.fill(_ellipsis(node.get("label", ""), 40), 26)
            center = (top + bottom) / 2
            label_items.append([center, label, label.count("\n") + 1, center])
        # Thin pathway bars sit close together, so centred labels collide.
        # Push them apart while keeping a leader line to the original bar.
        line_h = 0.0245
        items = sorted(label_items, key=lambda item: -item[0])
        for index in range(1, len(items)):
            prev_y, prev_lines = items[index - 1][0], items[index - 1][2]
            y, lines = items[index][0], items[index][2]
            min_y = prev_y - (prev_lines + lines) * line_h / 2 - 0.004
            if y > min_y:
                items[index][0] = min_y
        deficit = -0.032 - (items[-1][0] - items[-1][2] * line_h / 2)
        if deficit > 0:
            for item in items:
                item[0] += deficit
        for y, label, _lines, anchor in items:
            if abs(y - anchor) > 0.006:
                ax.plot(
                    [right_x + bar_w, right_x + bar_w + 0.008],
                    [anchor, y],
                    color="#94a3b8", linewidth=0.6, zorder=5,
                )
            ax.text(
                right_x + bar_w + 0.014, y, label,
                ha="left", va="center", fontsize=9.8, color=figures.INK,
                linespacing=1.25, zorder=5,
            )

        ax.text(
            left_x - 0.014, 1.035, f"基因（{len(gene_nodes)}）",
            ha="right", va="bottom", fontsize=11.5, fontweight="bold",
            color="#0f766e",
        )
        used_libraries = []
        for item in ((data.get("sankey") or {}).get("pathways") or []):
            library = str(item.get("library") or "").lower()
            label = {"kegg": "KEGG", "reactome": "Reactome"}.get(library, library)
            if label and label not in used_libraries:
                used_libraries.append(label)
        library_text = " / ".join(used_libraries) or "Enrichr"
        ax.text(
            right_x + bar_w + 0.010, 1.035,
            f"通路（{len(path_nodes)}，{library_text}）",
            ha="left", va="bottom", fontsize=10.8, fontweight="bold",
            color=figures.INK,
        )
        kegg_failed = bool(
            ((data.get("enrichment") or {}).get("kegg") or {}).get("error")
        )
        if i18n.get_lang() == "en":
            caption = (
                f"The Sankey diagram maps {len(gene_nodes)} input genes to "
                f"{len(path_nodes)} significant pathways: genes on the left, "
                "pathways on the right; each gene–pathway link counts once, so ribbon "
                "width is proportional to the number of input genes in that pathway "
                "and colours only distinguish pathways. Only KEGG 2021 Human and "
                "Reactome 2022 pathways with FDR < 0.05 and at least two input genes "
                "are kept (top 12 by gene count) to avoid single-gene noise; pathway "
                "names are truncated to 40 characters."
            )
            if kegg_failed:
                caption += (
                    " The KEGG library failed in this run and was skipped; only "
                    "Reactome pathways are shown."
                )
            caption += (
                " The mapping reflects existing database annotations and does not "
                "constitute functional validation."
            )
        else:
            caption = (
                f"桑基图展示 {len(gene_nodes)} 个输入基因与 {len(path_nodes)} 条显著通路之间的映射："
                "左列为基因、右列为通路，每条基因–通路连接计 1，因此流带宽度正比于该通路命中的"
                "输入基因数，流带颜色仅用于区分不同通路。仅保留 KEGG 2021 Human 与 Reactome 2022 "
                "中 FDR < 0.05 且至少 2 个输入基因命中的通路（按命中基因数取前 12 条），"
                "以避免单基因噪声；通路名称截断至 40 字符。"
            )
            if kegg_failed:
                caption += "本次 KEGG 库请求失败并已跳过，因此仅展示 Reactome 通路。"
            caption += "该映射反映数据库中已有注释，不构成功能验证。"
        figures._caption(fig, caption)
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)
    return outpath


def run_netpharm(
    project: Project,
    genes: list[str],
    compound: dict,
    options: dict,
    job,
) -> dict:
    genes = [str(gene).strip() for gene in genes if str(gene).strip()]
    genes = list(dict.fromkeys(genes))
    if len(genes) < 2:
        raise ValueError(
            "至少需要 2 个带基因符号的靶点才能构建网络；"
            "若靶点不足，请在页面补充人源基因。"
        )
    limit = max(5, min(int(options.get("genes", 60) or 60), 150))
    genes = genes[:limit]
    score = max(150, min(int(options.get("string_score", 400) or 400), 900))
    expand = max(0, min(int(options.get("expand", 15) or 0), 50))
    job.set_progress(0.15, "构建 STRING 蛋白互作网络")
    ppi = _string_network(genes, score, job, add_nodes=expand)
    if len(ppi["edges"]) < 1:
        raise ValueError(
            f"STRING 未返回任何互作边（输入 {len(genes)} 个基因，0 条边）。"
            "这些基因可能不是人源符号或置信度过高；"
            "请补充人源基因、降低 STRING 阈值，或增大网络扩展节点数。"
        )

    job.set_progress(0.45, "多库富集分析（Enrichr，8 个库）")
    enrichment = _enrichment_all(genes, job)
    significant_total = sum(
        int(block.get("significant") or 0) for block in enrichment.values()
    )
    if significant_total == 0:
        raise ValueError(
            "八个 Enrichr 库（KEGG、GO BP/CC/MF、Reactome、WikiPathway、"
            "DisGeNET、Hallmark）都没有 FDR < 0.05 的条目，数据不足，"
            "未生成网络药理学结果。请增加或更换基因列表。"
        )
    job.log(
        f"[网络] 八个库共 {significant_total} 个 FDR < 0.05 的条目"
    )
    kegg = (enrichment.get("kegg") or {}).get("terms") or []
    go = (enrichment.get("go_bp") or {}).get("terms") or []
    kegg_significant = int((enrichment.get("kegg") or {}).get("significant") or 0)
    go_significant = int((enrichment.get("go_bp") or {}).get("significant") or 0)

    job.set_progress(0.70, "构建化合物–靶点–疾病网络")
    ctd = _ctd_network(genes, compound, enrichment, job)
    job.set_progress(0.78, "构建靶点–通路桑基图数据")
    sankey = _sankey_data(genes, enrichment, job)

    record_id = f"net_{datetime.now():%Y%m%d_%H%M%S}"
    outdir = project.root / "exports" / "netpharm" / record_id
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": record_id,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "compound": compound,
        "genes": genes,
        "parameters": {
            "genes": limit,
            "string_score": score,
            "expand": expand,
        },
        "ppi": ppi,
        "supplied_genes": list(options.get("supplied_genes") or []),
        "enrichment": enrichment,
        "kegg": kegg,
        "go": go,
        "ctd": ctd,
        "sankey": sankey,
        "stats": {
            "genes": len(genes),
            "mapped_genes": len(ppi.get("mapped_genes") or []),
            "expanded_genes": len(ppi.get("expanded_genes") or []),
            "edges": len(ppi["edges"]),
            "kegg_significant": kegg_significant,
            "go_significant": go_significant,
            "libraries": len(LIBRARIES),
            "significant": {
                key: int(block.get("significant") or 0)
                for key, block in enrichment.items()
            },
            "ctd_nodes": len((ctd or {}).get("nodes") or []),
            "ctd_edges": len((ctd or {}).get("edges") or []),
            "sankey_pathways": len((sankey or {}).get("pathways") or []),
        },
        "source": (
            "STRING v12 + Enrichr (KEGG 2021 Human; GO BP/CC/MF 2023; "
            "Reactome 2022; WikiPathway 2023 Human; DisGeNET; "
            "MSigDB Hallmark 2020)"
        ),
    }

    job.set_progress(0.88, "绘制多库富集、C-T-D 与桑基图")
    pdf_paths: dict[str, str] = {}
    jobs = [("enrichment", enrichment_pdf)]
    if ctd:
        jobs.append(("ctd", ctd_pdf))
    if sankey:
        jobs.append(("sankey", sankey_pdf))
    for name, func in jobs:
        target = outdir / f"{name}.pdf"
        try:
            func(payload, target)
            pdf_paths[name] = str(target.relative_to(project.root))
            job.log(f"[网络] 已生成图件 {target.name}")
        except Exception as exc:  # 图件失败不影响数据结果
            job.log(f"[网络] {name} 图件生成失败：{exc}")
    payload["pdf"] = pdf_paths

    data_path = outdir / "netpharm.json"
    write_json(data_path, payload)

    summary = {
        "id": record_id,
        "created": payload["created"],
        "compound": compound,
        "genes": genes,
        "supplied_genes": list(options.get("supplied_genes") or []),
        "expanded_genes": ppi.get("expanded_genes") or [],
        "edges": len(ppi["edges"]),
        "kegg": kegg[:10],
        "go": go[:10],
        "enrichment": {
            key: {
                "library": block.get("library"),
                "terms": (block.get("terms") or [])[:5],
                "significant": int(block.get("significant") or 0),
            }
            for key, block in enrichment.items()
        },
        "ctd": (
            {
                "nodes": len(ctd.get("nodes") or []),
                "edges": len(ctd.get("edges") or []),
                "diseases": ctd.get("diseases", 0),
            }
            if ctd
            else None
        ),
        "sankey": (
            {
                "genes": len(sankey.get("genes") or []),
                "pathways": len(sankey.get("pathways") or []),
                "links": len(sankey.get("links") or []),
            }
            if sankey
            else None
        ),
        "pdf": pdf_paths,
        "stats": payload["stats"],
        "source": payload["source"],
        "data": str(data_path.relative_to(project.root)),
    }
    project.data.setdefault("netpharm", []).insert(0, summary)
    project.data["netpharm"] = project.data["netpharm"][:20]
    project.save()
    job.set_progress(1.0, "网络药理学分析完成")
    job.result = summary
    return summary


def load_netpharm(project: Project, record_id: str = "") -> dict | None:
    items = project.data.get("netpharm", [])
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
    return json.loads(path.read_text(encoding="utf-8"))
