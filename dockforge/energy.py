"""Alternative binding-energy estimates and consensus rescoring.

Docking scores are empirical and depend on the scoring function.  This module
collects the values already available in a run (Vina, Vinardo, MM-GBSA) and
adds an X-Score-like empirical term, then reports a transparent consensus
estimate.  It never replaces a rigorous free-energy calculation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import figures, i18n


def T(zh: str, en: str) -> str:
    return en if i18n.get_lang() == "en" else zh


def empirical_score(
    interaction: dict,
    affinity: float | None = None,
    heavy: int | None = None,
    rotb: int | None = None,
) -> dict:
    counts = (interaction or {}).get("counts") or {}
    hbonds = int(counts.get("hbonds") or 0)
    hydrophobic = int(counts.get("hydrophobic") or 0)
    salt = int(counts.get("salt_bridges") or 0)
    pi = int(counts.get("pi_stacking") or 0)
    halogen = int(counts.get("halogens") or 0)
    metal = int(counts.get("metals") or 0)
    rotb = int(rotb or 0)
    heavy = int(heavy or 0)
    value = (
        -0.70 * hbonds - 0.12 * hydrophobic - 1.20 * salt
        - 0.45 * pi - 0.50 * halogen - 0.30 * metal
        + 0.22 * rotb + 0.012 * heavy + 1.0
    )
    return {
        "value": round(float(value), 2),
        "terms": {
            "hbonds": hbonds,
            "hydrophobic": hydrophobic,
            "salt_bridges": salt,
            "pi_stacking": pi,
            "halogens": halogen,
            "metals": metal,
            "rotb": rotb,
            "heavy": heavy,
        },
        "method": T(
            "X-Score 类经验项（DrugHX）",
            "X-Score-like empirical term (DrugHX)",
        ),
        "note": T(
            "经验打分，只用于同一体系内的相对比较，不是自由能。",
            "Empirical score for relative comparison within one system, not a free energy.",
        ),
    }


def _complex_block(interaction: dict, mmgbsa: dict | None) -> dict:
    counts = (interaction or {}).get("counts") or {}
    residues = (interaction or {}).get("residues") or []
    buried = (mmgbsa or {}).get("buried_fraction")
    return {
        "hbonds": int(counts.get("hbonds") or 0),
        "hydrophobic": int(counts.get("hydrophobic") or 0),
        "salt_bridges": int(counts.get("salt_bridges") or 0),
        "pi_stacking": int(counts.get("pi_stacking") or 0),
        "halogens": int(counts.get("halogens") or 0),
        "metals": int(counts.get("metals") or 0),
        "interface_residues": len(residues),
        "buried_fraction": float(buried) if buried is not None else None,
        "confidence": T("中", "Medium") if counts else T("低", "Low"),
    }


def consensus(
    result: dict,
    interaction: dict,
    props: dict | None = None,
    mmgbsa: dict | None = None,
) -> dict:
    props = props or {}
    vina = result.get("affinity")
    vinardo = result.get("vinardo")
    empirical = empirical_score(
        interaction,
        vina,
        heavy=props.get("heavy"),
        rotb=props.get("rotb"),
    )
    methods = [
        {
            "name": "Vina",
            "value": float(vina) if vina is not None else None,
            "weight": 0.35,
            "note": T("AutoDock Vina 1.2.5 经验打分", "AutoDock Vina 1.2.5 empirical score"),
        },
        {
            "name": "Vinardo",
            "value": float(vinardo) if vinardo is not None else None,
            "weight": 0.25,
            "note": T("Vinardo 重打分", "Vinardo rescoring"),
        },
        {
            "name": "Empirical",
            "value": empirical["value"],
            "weight": 0.20,
            "note": empirical["method"],
        },
    ]
    if mmgbsa and mmgbsa.get("total") is not None:
        methods.append(
            {
                "name": "MM-GBSA",
                "value": float(mmgbsa["total"]),
                "weight": 0.0,
                "note": T(
                    "MM-GBSA 近似（无熵校正），单独比较、不参与对接尺度共识",
                    "MM-GBSA approximation (no entropy correction), compared separately and excluded from the docking-scale consensus",
                ),
            }
        )
    available = [
        item for item in methods
        if item["name"] in ("Vina", "Vinardo", "Empirical")
        and item["value"] is not None and np.isfinite(item["value"])
    ]
    if not available:
        return {
            "methods": methods,
            "consensus": None,
            "confidence": T("低", "Low"),
            "complex": _complex_block(interaction, mmgbsa),
            "note": T(
                "没有可比较的对接尺度方法；MM-GBSA 单独列出，不与对接打分平均。",
                "No docking-scale methods were available; MM-GBSA is listed separately and is not averaged with docking scores.",
            ),
        }
    weight_sum = sum(item["weight"] for item in available)
    estimate = sum(item["value"] * item["weight"] for item in available) / weight_sum
    spread = max(item["value"] for item in available) - min(item["value"] for item in available)
    if len(available) >= 3 and spread <= 2.0:
        confidence = T("中", "Medium")
    else:
        confidence = T("低", "Low")
    return {
        "methods": methods,
        "consensus": round(float(estimate), 2),
        "spread": round(float(spread), 2),
        "confidence": confidence,
        "complex": _complex_block(interaction, mmgbsa),
        "note": T(
            "共识值只对 Vina、Vinardo 与经验项等对接尺度方法加权；MM-GBSA 单独列出、不参与平均。"
            "方法间差异反映打分函数系统偏差，不能视为实验自由能。",
            "The consensus only averages docking-scale methods (Vina, Vinardo and the empirical term); MM-GBSA is listed separately and is not averaged. "
            "The spread reflects scoring-function bias and is not an experimental free energy.",
        ),
    }


def energy_pdf(data: dict, name: str, receptor: str, outpath: Path) -> Path:
    figures._style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    methods = data.get("methods") or []
    with PdfPages(outpath) as pdf:
        fig = figures._a4_page(
            T("附图 A4　结合能多方法比较", "Fig. A4　Multi-method binding-energy comparison"),
            T(
                f"{name} × {receptor} · Vina、Vinardo、经验项与 MM-GBSA 共识",
                f"{name} × {receptor} · Vina, Vinardo, empirical term and MM-GBSA consensus",
            ),
            footer=T("DrugHX · 结合能比较", "DrugHX · Binding-energy comparison"),
        )
        # MM-GBSA is on a different energy scale (tens of kcal/mol) and carries
        # weight 0 in the consensus; plotting it on the docking-score axis would
        # compress every docking method into an unreadable band, so it is shown
        # in the table only.
        available = [
            item for item in methods
            if item.get("value") is not None and float(item.get("weight") or 0) > 0
        ]
        separate = [
            item for item in methods
            if item.get("value") is not None and float(item.get("weight") or 0) <= 0
        ]
        labels = [
            T("经验项", "Empirical") if item["name"] == "Empirical" else item["name"]
            for item in available
        ]
        values = [float(item["value"]) for item in available]
        colors = [figures.TEAL, figures.BLUE, figures.AMBER, figures.RED][: len(values)]
        ax = fig.add_axes([0.18, 0.50, 0.74, 0.30])
        y = np.arange(len(labels))
        ax.barh(y, values, color=colors, height=0.60)
        for index, value in enumerate(values):
            ax.text(
                value - 0.1 if value < 0 else value + 0.1,
                index, f"{value:.2f}",
                va="center",
                ha="right" if value < 0 else "left",
                fontsize=9.0, color=figures.INK,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel(T("结合能估计（kcal/mol）", "Binding-energy estimate (kcal/mol)"), fontsize=9.2)
        ax.set_title(T("不同打分/能量方法的比较", "Comparison of scoring and energy methods"), loc="left", fontsize=12.5, pad=8)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=figures.GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        if separate:
            ax.text(
                0.99, -0.24,
                T(
                    "MM-GBSA（无熵校正）量纲不同，仅在表中单独列出，不并入对接打分坐标轴与共识平均。",
                    "MM-GBSA (no entropy correction) has a different scale; it is listed in the table only and is neither plotted on the docking-score axis nor averaged into the consensus.",
                ),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.4, color=figures.MUTED, wrap=True,
            )
        rows = [[T("方法", "Method"), T("数值", "Value"), T("权重", "Weight"), T("说明", "Note")]]
        for item in methods:
            value = item.get("value")
            rows.append([
                T("经验项", "Empirical") if item.get("name") == "Empirical" else item.get("name", ""),
                f"{float(value):.2f}" if value is not None else "—",
                f"{float(item.get('weight') or 0):.2f}",
                str(item.get("note", ""))[:70],
            ])
        figures._table(fig, 0.455, rows, [0.16, 0.12, 0.10, 0.48], fontsize=8.2, row_height=0.028)
        consensus_value = data.get("consensus")
        summary = [
            T(
                f"共识估计 {consensus_value:.2f} kcal/mol；方法间跨度 {float(data.get('spread') or 0):.2f} kcal/mol；置信度 {data.get('confidence', '—')}。"
                if consensus_value is not None
                else "没有可用的结合能方法。",
                f"Consensus estimate {consensus_value:.2f} kcal/mol; spread {float(data.get('spread') or 0):.2f} kcal/mol; confidence {data.get('confidence', '—')}."
                if consensus_value is not None
                else "No binding-energy method was available.",
            ),
            T(
                "复合物与氢键评估：{hb} 个氢键、{hy} 处疏水、{sb} 个盐桥、{pi} 处 π-堆积、"
                "界面残基 {res} 个；埋藏分数 {buried}；氢键几何与作用力判定见附图 A2 与图 14。".format(
                    hb=(data.get("complex") or {}).get("hbonds", 0),
                    hy=(data.get("complex") or {}).get("hydrophobic", 0),
                    sb=(data.get("complex") or {}).get("salt_bridges", 0),
                    pi=(data.get("complex") or {}).get("pi_stacking", 0),
                    res=(data.get("complex") or {}).get("interface_residues", 0),
                    buried=(
                        f"{float((data.get('complex') or {}).get('buried_fraction')):.0%}"
                        if (data.get("complex") or {}).get("buried_fraction") is not None
                        else "—"
                    ),
                ),
                "Complex and H-bond assessment: {hb} H-bonds, {hy} hydrophobic contacts, {sb} salt bridges, {pi} π-stacking contacts, {res} interface residues; buried fraction {buried}; H-bond geometry and interaction criteria are shown in Fig. A2 and Fig. 14.".format(
                    hb=(data.get("complex") or {}).get("hbonds", 0),
                    hy=(data.get("complex") or {}).get("hydrophobic", 0),
                    sb=(data.get("complex") or {}).get("salt_bridges", 0),
                    pi=(data.get("complex") or {}).get("pi_stacking", 0),
                    res=(data.get("complex") or {}).get("interface_residues", 0),
                    buried=(
                        f"{float((data.get('complex') or {}).get('buried_fraction')):.0%}"
                        if (data.get("complex") or {}).get("buried_fraction") is not None
                        else "—"
                    ),
                ),
            ),
            data.get("note", ""),
        ]
        figures._text_blocks(
            fig, 0.295,
            [(T("判读", "Interpretation"), summary)],
            width=52, size=9.4, stop=0.155,
        )
        figures._caption(
            fig,
            T(
                "工具与数据：AutoDock Vina 1.2.5、Vinardo、DrugHX X-Score-like 经验项、DrugHX MM-GBSA 近似。"
                "所有数值仅用于同一体系内的相对比较，不是实验自由能。",
                "Tools and data: AutoDock Vina 1.2.5, Vinardo, DrugHX X-Score-like empirical term, DrugHX MM-GBSA approximation. "
                "All values are for relative comparison within one system and are not experimental free energies.",
            ),
            y=0.150,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)
    return outpath


def run_mode(run: dict) -> dict:
    """Classify a docking run as refined or large-scale screening."""
    params = (run or {}).get("params") or {}
    kind = str((run or {}).get("kind") or "")
    results = (run or {}).get("results") or []
    count = len([item for item in results if item.get("status") == "ok"])
    exhaustiveness = int(params.get("exhaustiveness") or 8)
    modes = int(params.get("num_modes") or 9)
    engine = str(params.get("engine") or "cpu")
    if kind == "design":
        label = T("设计分子对接", "Design-candidate docking")
        key = "design"
    elif kind == "optimization":
        label = T("优化候选对接", "Optimization-candidate docking")
        key = "optimization"
    elif exhaustiveness >= 16 and modes >= 9 and count <= 40:
        label = T("精细对接（高 exhaustiveness）", "Refined docking (high exhaustiveness)")
        key = "refined"
    elif count >= 40 or exhaustiveness <= 8:
        label = T("大规模筛选", "Large-scale screening")
        key = "screening"
    else:
        label = T("标准对接", "Standard docking")
        key = "standard"
    return {
        "key": key,
        "label": label,
        "count": count,
        "exhaustiveness": exhaustiveness,
        "num_modes": modes,
        "engine": engine,
    }
