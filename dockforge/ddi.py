"""Drug–drug interaction hypotheses for a candidate molecule.

The analysis is deliberately rule based and transparent.  It combines:

* pharmacodynamic overlap through shared protein targets (from target fishing);
* pharmacokinetic competition when the candidate inhibits a CYP enzyme that
  metabolises the co-administered drug;
* additive cardiotoxicity risk when both molecules have a high predicted hERG
  liability;
* structural / class similarity as a weak prior for overlapping pharmacology.

Nothing here is a clinical DDI prediction.  Every record is a hypothesis that
must be checked against labels, guidelines and experimental data.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, rdFingerprintGenerator

from . import admet, figures, i18n

RDLogger.DisableLog("rdApp.*")


def _profile(smiles: str, name: str) -> dict:
    return admet.profile_smiles(smiles, name=name)


def _value(profile: dict, key: str, default: float = 0.0) -> float:
    for item in profile.get("admet") or []:
        if item.get("key") == key:
            try:
                return float(item.get("value") or 0)
            except (TypeError, ValueError):
                return default
    return default


_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fingerprint(mol: Chem.Mol):
    return _FP_GEN.GetFingerprint(mol)


def _tanimoto(first, second) -> float:
    if first is None or second is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(first, second))


def _library() -> dict[str, dict]:
    try:
        from .drugref import FDA_DRUGS

        return {
            str(item.get("name") or "").strip().lower(): item
            for item in FDA_DRUGS
            if item.get("name")
        }
    except Exception:
        return {}


def _resolve_drug(item, library: dict[str, dict]) -> dict:
    if isinstance(item, str):
        name = item.strip()
        record = library.get(name.lower()) or {}
        return {**record, "name": record.get("name") or name}
    record = dict(item or {})
    name = str(record.get("name") or "").strip()
    base = library.get(name.lower()) or {}
    merged = {**base, **record}
    merged["name"] = merged.get("name") or name or "未命名药物"
    return merged


def _normalise_targets(values) -> list[str]:
    targets = []
    for item in values or []:
        text = str(item or "").strip().upper()
        if text and text not in targets:
            targets.append(text)
    return targets


def _score_components(candidate: dict, drug: dict, shared: list[str]) -> dict:
    cyp3a4 = _value(candidate["profile"], "cyp3a4")
    cyp2d6 = _value(candidate["profile"], "cyp2d6")
    herg_c = _value(candidate["profile"], "herg")
    herg_d = _value(drug.get("profile") or {}, "herg")
    metabolised = {
        str(item or "").upper()
        for item in (drug.get("metabolized_by") or [])
    }

    target_component = min(1.0, len(shared) / 3.0)
    cyp_component = 0.0
    if "CYP3A4" in metabolised:
        cyp_component = max(cyp_component, cyp3a4 / 100.0)
    if "CYP2D6" in metabolised:
        cyp_component = max(cyp_component, cyp2d6 / 100.0)
    herg_component = min(1.0, (herg_c + herg_d) / 120.0) if drug.get("profile") else 0.0
    class_component = 0.0
    if candidate.get("drug_class") and drug.get("drug_class"):
        if candidate["drug_class"].strip().lower() == drug["drug_class"].strip().lower():
            class_component = 1.0
    class_component = max(class_component, min(1.0, drug.get("tanimoto", 0.0) / 0.7))
    score = 100 * (
        0.38 * target_component
        + 0.27 * cyp_component
        + 0.20 * herg_component
        + 0.15 * class_component
    )
    return {
        "score": round(float(score), 1),
        "target_component": round(target_component * 100, 1),
        "cyp_component": round(cyp_component * 100, 1),
        "herg_component": round(herg_component * 100, 1),
        "class_component": round(class_component * 100, 1),
        "cyp3a4": round(cyp3a4, 1),
        "cyp2d6": round(cyp2d6, 1),
        "herg_candidate": round(herg_c, 1),
        "herg_codrug": round(herg_d, 1),
    }


def analyse(
    candidate_smiles: str,
    candidate_name: str = "候选分子",
    co_drugs=None,
    candidate_targets=None,
    candidate_class: str = "",
) -> dict:
    """Build rule-based DDI hypotheses for a candidate and a co-drug list."""
    mol = Chem.MolFromSmiles(str(candidate_smiles or ""))
    if mol is None:
        raise ValueError("候选分子 SMILES 无法解析")
    profile = _profile(candidate_smiles, candidate_name or "候选分子")
    fingerprint = _fingerprint(mol)
    library = _library()
    drugs = [_resolve_drug(item, library) for item in (co_drugs or [])]
    if not drugs:
        raise ValueError("请至少选择一个同时使用的药物")

    candidate_targets = _normalise_targets(candidate_targets)
    records = []
    for drug in drugs:
        targets = _normalise_targets(
            drug.get("primary_targets") or drug.get("targets") or []
        )
        shared = sorted(set(candidate_targets) & set(targets))
        drug_profile = None
        if drug.get("smiles"):
            try:
                drug_profile = _profile(str(drug["smiles"]), str(drug.get("name") or "药物"))
            except Exception:
                drug_profile = None
        drug_mol = Chem.MolFromSmiles(str(drug.get("smiles") or "")) if drug.get("smiles") else None
        tanimoto = _tanimoto(fingerprint, _fingerprint(drug_mol)) if drug_mol else 0.0
        record = {
            "name": drug.get("name") or "未命名药物",
            "drug_class": drug.get("drug_class") or drug.get("class") or "",
            "indication": drug.get("indication") or "",
            "targets": targets,
            "shared_targets": shared,
            "metabolized_by": list(drug.get("metabolized_by") or []),
            "tanimoto": round(tanimoto, 3),
            "profile": drug_profile,
        }
        if not targets and not drug_profile:
            record.update(
                {
                    "score": None,
                    "category": "信息不足",
                    "components": {},
                    "mechanism": "缺少靶点与结构信息，无法判断",
                }
            )
            records.append(record)
            continue
        components = _score_components(
            {"profile": profile, "drug_class": candidate_class},
            record,
            shared,
        )
        record["components"] = components
        score = components["score"]
        if score >= 55:
            category = "高"
        elif score >= 30:
            category = "中"
        else:
            category = "低"
        mechanisms = []
        if shared:
            mechanisms.append("共享靶点：" + "、".join(shared[:5]))
        if components["cyp_component"] >= 25:
            mechanisms.append("候选分子可能抑制共同代谢酶，抬高联用药暴露")
        if components["herg_component"] >= 45:
            mechanisms.append("两者 hERG 风险叠加，需关注 QT 间期")
        if tanimoto >= 0.65:
            mechanisms.append("结构/类别相近，药理作用可能重叠")
        if not mechanisms:
            mechanisms.append("现有规则未发现明确互作机制")
        record["score"] = score
        record["category"] = category
        record["mechanism"] = "；".join(mechanisms)
        records.append(record)

    records.sort(
        key=lambda item: (item.get("score") is None, -(item.get("score") or 0))
    )
    summary = {
        "high": sum(1 for item in records if item.get("category") == "高"),
        "medium": sum(1 for item in records if item.get("category") == "中"),
        "low": sum(1 for item in records if item.get("category") == "低"),
        "unknown": sum(1 for item in records if item.get("category") == "信息不足"),
    }
    return {
        "candidate": {
            "name": candidate_name or "候选分子",
            "smiles": profile["smiles"],
            "targets": candidate_targets,
            "drug_class": candidate_class,
            "profile": profile,
        },
        "co_drugs": records,
        "summary": summary,
        "method": (
            "规则式 DDI 假设：靶点重叠（靶点垂钓/参考库）、CYP3A4/2D6 竞争、"
            "hERG 风险叠加与结构类别相似度加权"
        ),
        "limitations": (
            "该分析不是临床 DDI 预测。共享靶点不代表同向或拮抗，CYP 与 hERG 为计算风险，"
            "必须结合说明书、相互作用数据库与实验数据判断。"
        ),
    }


def _panel_tag(ax, letter: str, x: float = -0.16, y: float = 1.06):
    figures._panel_tag(ax, letter, x=x, y=y)


def ddi_pdf(data: dict, outpath: Path) -> Path:
    """A4 CNS-style DDI summary figure."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    figures._style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    records = [item for item in data.get("co_drugs") or [] if item.get("score") is not None]
    if not records:
        raise ValueError("没有可用于绘图的药物互作记录")

    candidate = data.get("candidate") or {}
    fig = figures._a4_page(
        "图 20　候选分子–已上市药物相互作用假设",
        f"{candidate.get('name', '候选分子')} 与 {len(data.get('co_drugs') or [])} 个联用药物的规则式 DDI 画像",
        "Fig. 20　Drug–drug interaction hypotheses",
    )
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0],
        wspace=0.55, hspace=0.42,
        left=0.21, right=0.90, top=0.845, bottom=0.235,
    )

    ax1 = fig.add_subplot(grid[0, 0])
    shown = records[:12]
    names = [item["name"] for item in shown][::-1]
    scores = [item["score"] for item in shown][::-1]
    colors = [
        figures.RED if item["category"] == "高" else figures.AMBER
        if item["category"] == "中" else figures.TEAL
        for item in shown
    ][::-1]
    y = np.arange(len(shown))
    ax1.barh(y, scores, color=colors, height=0.58, edgecolor="none")
    for index, value in enumerate(scores):
        ax1.text(value + 1.2, index, f"{value:.0f}", va="center", fontsize=9.2, color=figures.INK)
    ax1.set_yticks(y)
    ax1.set_yticklabels([figures._trunc(name, 17) for name in names], fontsize=8.6)
    ax1.set_xlim(0, max(20.0, max(scores) * 1.25))
    ax1.set_xlabel("综合互作风险（0–100）", fontsize=10.5)
    ax1.set_title("联用药物风险排序", loc="left", fontsize=13.5, pad=10)
    figures._compact_frame(ax1)
    _panel_tag(ax1, "a", x=-0.34)

    ax2 = fig.add_subplot(grid[0, 1])
    columns = ["相似", "靶点", "3A4", "2D6", "hERG", "综合"]
    if i18n.get_lang() == "en":
        columns = ["Sim", "Target", "3A4", "2D6", "hERG", "Total"]
    matrix = []
    for item in shown[:10]:
        components = item.get("components") or {}
        matrix.append(
            [
                item.get("tanimoto", 0.0) * 100,
                components.get("target_component", 0.0),
                components.get("cyp3a4", 0.0),
                components.get("cyp2d6", 0.0),
                components.get("herg_component", 0.0),
                item.get("score", 0.0),
            ]
        )
    image = ax2.imshow(np.asarray(matrix, dtype=float), cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax2.set_xticks(np.arange(len(columns)))
    ax2.set_xticklabels(columns, rotation=0, ha="center", fontsize=7.8)
    ax2.set_yticks(np.arange(len(matrix)))
    ax2.set_yticklabels(
        [figures._trunc(item["name"], 15) for item in shown[:10]], fontsize=8.4
    )
    ax2.set_title("互作证据矩阵", loc="left", fontsize=13.5, pad=10)
    for i in range(len(matrix)):
        for j in range(len(columns)):
            value = matrix[i][j]
            ax2.text(
                j, i, f"{value:.0f}", ha="center", va="center", fontsize=7.8,
                color="white" if value > 58 else figures.INK,
            )
    cbar = fig.colorbar(image, ax=ax2, fraction=0.035, pad=0.03)
    cbar.set_label("分值（0–100）", fontsize=8.0, labelpad=4)
    cbar.ax.tick_params(labelsize=7.6)
    _panel_tag(ax2, "b", x=-0.30)

    ax3 = fig.add_subplot(grid[1, 0])
    ax3.set_aspect("equal")
    ax3.axis("off")
    ax3.set_xlim(-1.95, 1.95)
    ax3.set_ylim(-1.80, 1.80)
    ax3.scatter([0], [0], s=560, color=figures.BLUE, edgecolor="white", linewidth=2.0, zorder=5)
    center_label = "Cand." if i18n.get_lang() == "en" else "候选"
    ax3.text(0, 0, center_label, ha="center", va="center", fontsize=8.6,
             color="white", fontweight="bold", zorder=6)
    count = max(1, len(shown[:10]))
    for index, item in enumerate(shown[:10]):
        angle = math.pi / 2 - 2 * math.pi * index / count
        x = 1.12 * math.cos(angle)
        y = 1.12 * math.sin(angle)
        color = (
            figures.RED if item["category"] == "高" else figures.AMBER
            if item["category"] == "中" else figures.TEAL
        )
        ax3.plot([0, x], [0, y], color=color, linewidth=1.0 + item["score"] / 45, alpha=0.75, zorder=2)
        ax3.scatter([x], [y], s=120, color=color, edgecolor="white", linewidth=1.2, zorder=4)
        horizontal = math.cos(angle)
        if horizontal > 0.25:
            alignment = "left"
        elif horizontal < -0.25:
            alignment = "right"
        else:
            alignment = "center"
        ax3.text(
            1.30 * horizontal, 1.36 * math.sin(angle),
            figures._trunc(item["name"], 16),
            ha=alignment, va="center", fontsize=7.4, color=figures.INK,
        )
    ax3.set_title("候选–联用药物互作网络", loc="left", fontsize=13.5, pad=10)
    _panel_tag(ax3, "c", x=-0.12, y=1.06)

    ax4 = fig.add_subplot(grid[1, 1])
    ax4.axis("off")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title("优先核查清单", loc="left", fontsize=13.5, pad=10)
    lines = []
    for item in records[:8]:
        lines.append(
            f"• {item['name']}  {item['score']:.0f}（{item['category']}）"
        )
        detail = []
        if item.get("shared_targets"):
            detail.append("共享 " + "、".join(item["shared_targets"][:3]))
        if item.get("drug_class"):
            detail.append(item["drug_class"])
        if detail:
            lines.append("   " + " · ".join(detail))
    ax4.text(
        0.0, 0.96, "\n".join(lines), ha="left", va="top",
        fontsize=8.6, color=figures.INK, linespacing=1.55,
    )
    _panel_tag(ax4, "d", x=-0.12, y=1.06)

    figures._caption(
        fig,
        "按共享靶点、CYP3A4/2D6 竞争、hERG 风险叠加与结构/类别相似度加权的综合风险；"
        "各项证据的分值矩阵；候选–联用药物互作网络，边宽表示综合风险；"
        "（d）优先核查清单。共享靶点不代表同向或拮抗，CYP 与 hERG 为计算风险；"
        "本图只生成 DDI 假设，必须结合说明书、相互作用数据库与实验数据判断。",
    )
    with PdfPages(outpath) as pdf:
        pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    return outpath
