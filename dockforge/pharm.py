"""Species-aware pharmacology, formulation, safety and tissue-distribution
prediction for a single compound.

The models are deliberately transparent descriptor rules.  They are intended
for triage and for structuring a discussion, not as a substitute for
measured ADME, organ toxicity or a validated PBPK model.  Every block carries a
confidence note and the report states the tools and data sources used.
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from . import admet, figures, i18n, online
from .config import APP_VERSION

SPECIES = {
    "human": {
        "zh": "人", "en": "Human", "organism_id": 9606, "rumen": False,
        "food": False, "body_weight": 70.0,
        "temperature": 37.0, "plasma_ph": 7.4, "gfr": 120.0,
        "fat_fraction": 0.21, "muscle_fraction": 0.40,
    },
    "pig": {
        "zh": "猪", "en": "Pig", "organism_id": 9823, "rumen": False,
        "food": True, "body_weight": 70.0,
        "temperature": 39.0, "plasma_ph": 7.4, "gfr": 110.0,
        "fat_fraction": 0.18, "muscle_fraction": 0.42,
    },
    "cattle": {
        "zh": "牛", "en": "Cattle", "organism_id": 9913, "rumen": True,
        "food": True, "body_weight": 500.0,
        "temperature": 38.5, "plasma_ph": 7.4, "gfr": 100.0,
        "fat_fraction": 0.15, "muscle_fraction": 0.45,
    },
    "sheep": {
        "zh": "羊", "en": "Sheep", "organism_id": 9940, "rumen": True,
        "food": True, "body_weight": 50.0,
        "temperature": 39.0, "plasma_ph": 7.4, "gfr": 105.0,
        "fat_fraction": 0.18, "muscle_fraction": 0.43,
    },
    "rat": {
        "zh": "大鼠", "en": "Rat", "organism_id": 10116, "rumen": False,
        "food": False, "body_weight": 0.25,
        "temperature": 37.5, "plasma_ph": 7.4, "gfr": 130.0,
        "fat_fraction": 0.07, "muscle_fraction": 0.45,
    },
    "dog": {
        "zh": "犬", "en": "Dog", "organism_id": 9615, "rumen": False,
        "food": False, "body_weight": 15.0,
        "temperature": 38.5, "plasma_ph": 7.4, "gfr": 95.0,
        "fat_fraction": 0.16, "muscle_fraction": 0.45,
    },
    "cat": {
        "zh": "猫", "en": "Cat", "organism_id": 9685, "rumen": False,
        "food": False, "body_weight": 4.0,
        "temperature": 38.5, "plasma_ph": 7.4, "gfr": 85.0,
        "fat_fraction": 0.22, "muscle_fraction": 0.40,
    },
}

TISSUE_FACTORS = {
    "liver": 1.25,
    "kidney": 1.30,
    "lung": 1.00,
    "heart": 0.90,
    "brain": 0.55,
    "muscle": 0.75,
    "fat": 1.60,
    "skin": 0.80,
    "intestine": 1.10,
}
TISSUE_LABELS = {
    "liver": ("肝", "Liver"),
    "kidney": ("肾", "Kidney"),
    "lung": ("肺", "Lung"),
    "heart": ("心", "Heart"),
    "brain": ("脑", "Brain"),
    "muscle": ("肌肉", "Muscle"),
    "fat": ("脂肪", "Fat"),
    "skin": ("皮肤", "Skin"),
    "intestine": ("肠道", "Intestine"),
}
TISSUE_VOLUME = {
    "liver": 0.026, "kidney": 0.004, "lung": 0.010, "heart": 0.005,
    "brain": 0.020, "muscle": 0.40, "fat": 0.21, "skin": 0.07,
    "intestine": 0.017,
}


def T(zh: str, en: str) -> str:
    return en if i18n.get_lang() == "en" else zh


_HPA_ZH = {
    "Tissue enriched": "组织富集",
    "Tissue enhanced": "组织增强",
    "Low tissue specificity": "低组织特异性",
    "Group enriched": "组富集",
    "Expressed in all": "广泛表达",
    "Not detected": "未检出",
    "Mixed": "混合",
    "Detected in all": "广泛检出",
    "Detected in some": "部分检出",
    "Detected in single": "单一组织检出",
}


def _hpa_label(value) -> str:
    text = str(value or "")
    return _HPA_ZH.get(text, text) if i18n.get_lang() != "en" else text


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _confidence(score: float, note: str) -> dict:
    score = round(_clamp(score, 0, 100), 0)
    if score >= 70:
        level = T("高", "High")
    elif score >= 45:
        level = T("中", "Medium")
    else:
        level = T("低", "Low")
    return {"score": int(score), "level": level, "note": note}


def _smarts_count(mol: Chem.Mol, pattern: str) -> int:
    query = Chem.MolFromSmarts(pattern)
    if query is None:
        return 0
    return len(mol.GetSubstructMatches(query))


def _species_confidence(species: str) -> float:
    if species == "human":
        return 82.0
    if species in ("dog", "cat"):
        return 60.0
    if species in ("pig", "cattle", "sheep"):
        return 55.0
    return 50.0


def _admet_value(profile: dict, key: str, default: float = 0.0) -> float:
    for item in profile.get("admet") or []:
        if item.get("key") == key:
            try:
                return float(item.get("value"))
            except (TypeError, ValueError):
                return default
    return default


def oral_block(profile: dict, species: str) -> dict:
    phys = profile.get("physchem") or {}
    logs = float(phys.get("logs") or -4.0)
    mw = float(phys.get("mw") or 300.0)
    logp = float(phys.get("logp") or 2.0)
    hia = _admet_value(profile, "hia", 60.0)
    caco = _admet_value(profile, "caco2", -4.8)
    cyp3a4 = _admet_value(profile, "cyp3a4", 20.0)
    pgp = _admet_value(profile, "pgp", 25.0)
    soluble = logs >= -4.0
    permeable = hia >= 70.0 or caco >= -4.7
    if soluble and permeable:
        bcs = "I"
    elif (not soluble) and permeable:
        bcs = "II"
    elif soluble and (not permeable):
        bcs = "III"
    else:
        bcs = "IV"
    solubility_mg_ml = max(1e-6, 10 ** logs * mw)
    dose_number = (100.0 / 250.0) / solubility_mg_ml
    f_percent = _clamp(
        0.86 * hia - 7.0 * max(0.0, cyp3a4 / 100.0) - 5.0 * max(0.0, pgp / 100.0),
        3.0, 95.0,
    )
    if species != "human":
        # Veterinary species carry extra first-pass and rumen uncertainty.
        f_percent = _clamp(f_percent * 0.88, 2.0, 95.0)
    food_effect = (
        T("高脂餐可能提高吸收并增加暴露波动", "A high-fat meal may increase absorption and exposure variability")
        if bcs in ("II", "IV") and logp >= 2.5
        else T("食物效应预计较小", "Food effect is expected to be small")
    )
    return {
        "bcs": bcs,
        "hia": round(hia, 1),
        "caco2": round(caco, 2),
        "solubility_mg_ml": round(solubility_mg_ml, 4),
        "dose_number": round(dose_number, 2),
        "f_percent": round(f_percent, 1),
        "food_effect": food_effect,
        "soluble": soluble,
        "permeable": permeable,
        "confidence": _confidence(
            _species_confidence(species) - (8 if dose_number > 10 else 0),
            T(
                "基于 ESOL 溶解度、TPSA/LogP 吸收规则与种间外推（剂量数按 100 mg/250 mL 假设）；不是实测生物利用度。",
                "Based on ESOL solubility, TPSA/LogP absorption rules and species extrapolation (dose number assumes 100 mg/250 mL); not a measured bioavailability.",
            ),
        ),
    }


def rumen_block(profile: dict, species: str) -> dict:
    info = SPECIES.get(species, SPECIES["human"])
    if not info.get("rumen"):
        return {
            "applicable": False,
            "note": T("该物种不是反刍动物，不计算过瘤胃指标。", "Not a ruminant species; rumen bypass is not calculated."),
        }
    phys = profile.get("physchem") or {}
    mw = float(phys.get("mw") or 300.0)
    logp = float(phys.get("logp") or 2.0)
    logs = float(phys.get("logs") or -4.0)
    hbd = int(phys.get("hbd") or 0)
    mol = Chem.MolFromSmiles(profile.get("smiles") or "") if profile.get("smiles") else None
    esters = _smarts_count(mol, "[CX3](=O)[OX2][#6]") if mol is not None else 0
    base_bypass = (
        45.0 + 6.0 * (logp - 2.0) + 0.03 * (mw - 300.0)
        - 5.0 * (logs + 3.0) - 2.0 * hbd
    )
    bypass = _clamp(base_bypass, 5.0, 95.0)
    degradation = _clamp(18.0 + 18.0 * esters - 0.8 * (logs + 3.0) * 10.0, 5.0, 95.0)
    if bypass < 40:
        strategy = T(
            "天然过瘤胃不足，建议脂肪包被、瘤胃保护蛋白包被或 pH 敏感包衣。",
            "Native rumen bypass is insufficient; use fat coating, rumen-protected protein coating or pH-sensitive enteric coating.",
        )
    elif bypass < 70:
        strategy = T(
            "部分过瘤胃，可通过包被或与瘤胃保护载体复配提高到达皱胃的比例。",
            "Partial bypass; coating or a rumen-protected carrier can increase the fraction reaching the abomasum.",
        )
    else:
        strategy = T(
            "天然过瘤胃较好，仍需验证瘤胃液中的稳定性与采食后的释放。",
            "Native bypass is favourable, but stability in rumen fluid and release after feeding still require verification.",
        )
    return {
        "applicable": True,
        "bypass_percent": round(bypass, 1),
        "degradation_risk": round(degradation, 1),
        "protected_bypass_percent": round(_clamp(bypass + 22.0, 0.0, 98.0), 1),
        "strategy": strategy,
        "confidence": _confidence(
            45.0,
            T(
                "基于 logP、MW、溶解度与氢键数的经验规则；必须用瘤胃液/尼龙袋试验验证。",
                "Empirical rule from logP, MW, solubility and H-bond count; must be verified with rumen-fluid or nylon-bag tests.",
            ),
        ),
    }


def microbiota_block(profile: dict) -> dict:
    smiles = profile.get("smiles") or ""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    alert_keys = []
    alerts = []
    patterns = [
        ("azo", T("偶氮键（可被菌群还原）", "Azo bond (microbial reduction)"), "[N;X2]=[N;X2]"),
        ("nitro", T("硝基/硝基芳环", "Nitro/nitroaromatic"), "[$([N+](=O)[O-])]"),
        ("quat", T("季铵盐（抗菌活性）", "Quaternary ammonium (antimicrobial)"), "[N+;X4]"),
        ("polyphenol", T("多酚/邻二酚", "Polyphenol/catechol"), "c1ccc(O)c(O)c1"),
        ("sulfonamide", T("磺胺类结构", "Sulfonamide"), "S(=O)(=O)N"),
        ("michael", T("Michael 受体（菌群酶反应）", "Michael acceptor (microbial enzymes)"), "[CX3]=[CX3][CX3]=[OX1]"),
        ("quinone", T("醌类氧化还原结构", "Quinone redox motif"), "O=C1C=CC(=O)C=C1"),
        ("glycoside", T("糖苷/缩醛（可被糖苷酶水解）", "Glycoside/acetal (glycosidase substrate)"), "[CX4H1](O)(O)"),
    ]
    if mol is not None:
        for key, label, smarts in patterns:
            if _smarts_count(mol, smarts):
                alert_keys.append(key)
                alerts.append(label)
    score = _clamp(12.0 + 15.0 * len(alerts) + (8.0 if (profile.get("physchem") or {}).get("mw", 0) > 600 else 0.0), 0, 95)
    if any(key in ("quat", "nitro") for key in alert_keys):
        direction = T("潜在抑制/扰动肠道菌群", "Potential inhibition or perturbation of gut microbiota")
    elif alerts:
        direction = T("可能改变菌群代谢或屏障相关过程", "May alter microbial metabolism or barrier-related processes")
    else:
        direction = T("未见明确菌群结构警示，风险较低", "No explicit microbiota structural alert; low risk expected")
    return {
        "score": round(score, 1),
        "category": T("高", "High") if score >= 60 else T("中", "Medium") if score >= 30 else T("低", "Low"),
        "direction": direction,
        "alerts": alerts,
        "alert_keys": alert_keys,
        "confidence": _confidence(
            38.0,
            T(
                "结构警示规则，不是菌群生长或宏基因组预测；阳性不等于临床菌群失调。",
                "Structural-alert rules, not a growth or metagenomic prediction; a positive flag is not clinical dysbiosis.",
            ),
        ),
    }


def nucleic_acid_block(profile: dict) -> dict:
    smiles = profile.get("smiles") or ""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    phys = profile.get("physchem") or {}
    logp = float(phys.get("logp") or 2.0)
    rings = int(phys.get("aromatic_rings") or 0)
    charge = int(phys.get("charge") or 0)
    rotb = int(phys.get("rotb") or 0)
    intercalation = _clamp(8.0 + 13.0 * max(0, rings - 2) + 4.5 * logp + 10.0 * max(0, charge), 0, 95)
    groove = _clamp(6.0 + 9.0 * max(0, charge) + 3.0 * rotb + 4.0 * max(0.0, logp - 2.0), 0, 95)
    covalent = []
    covalent_keys = []
    if mol is not None:
        for key, label, smarts in (
            ("epoxide", T("环氧化物", "Epoxide"), "C1OC1"),
            ("michael", T("Michael 受体", "Michael acceptor"), "[CX3]=[CX3][CX3]=[OX1]"),
            ("alkyl_halide", T("烷基卤", "Alkyl halide"), "[CX4][Cl,Br,I]"),
            ("nitroso", T("亚硝基", "Nitroso"), "[NX2]=O"),
            ("quinone", T("醌", "Quinone"), "O=C1C=CC(=O)C=C1"),
            ("nitro", T("芳香硝基", "Aromatic nitro"), "[$([N+](=O)[O-])][c]"),
        ):
            if _smarts_count(mol, smarts):
                covalent_keys.append(key)
                covalent.append(label)
    overall = _clamp(max(intercalation, groove) + 10.0 * len(covalent), 0, 98)
    if overall >= 60:
        category = T("高", "High")
        note = T(
            "提示可能存在嵌入或沟槽结合倾向，并伴随共价/DNA 损伤警示；需用 DNA 结合实验或核酸对接验证。",
            "Suggests possible intercalation or groove binding with covalent/DNA-damage alerts; verify with DNA-binding assays or nucleic-acid docking.",
        )
    elif overall >= 35:
        category = T("中", "Medium")
        note = T(
            "存在一定的平面芳香/正电特征，可能发生弱核酸结合，需实验确认。",
            "Some planar aromatic/cationic character suggests weak nucleic-acid binding; experimental confirmation is needed.",
        )
    else:
        category = T("低", "Low")
        note = T(
            "没有明显的核酸嵌入或共价结合警示，但不排除弱非特异结合。",
            "No explicit intercalation or covalent-binding alert, although weak nonspecific binding cannot be excluded.",
        )
    return {
        "intercalation": round(intercalation, 1),
        "groove": round(groove, 1),
        "covalent_alerts": covalent,
        "covalent_keys": covalent_keys,
        "overall": round(overall, 1),
        "category": category,
        "note": note,
        "confidence": _confidence(
            40.0,
            T(
                "基于平面芳香性、电荷与结构警示的启发式评分；不是核酸对接或实验结合常数。",
                "Heuristic score from planarity, charge and structural alerts; not nucleic-acid docking or a measured binding constant.",
            ),
        ),
    }


def stability_block(profile: dict) -> dict:
    smiles = profile.get("smiles") or ""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    counts = {
        "ester": _smarts_count(mol, "[CX3](=O)[OX2][#6]") if mol else 0,
        "amide": _smarts_count(mol, "[CX3](=O)[NX3]") if mol else 0,
        "lactone": _smarts_count(mol, "[CX3](=O)[OX2][#6;R]") if mol else 0,
        "sulfonamide": _smarts_count(mol, "S(=O)(=O)N") if mol else 0,
        "aldehyde": _smarts_count(mol, "[CX3H1](=O)") if mol else 0,
        "phenol": _smarts_count(mol, "c[OH]") if mol else 0,
        "aniline": _smarts_count(mol, "[NX3;H2,H1][c]") if mol else 0,
        "epoxide": _smarts_count(mol, "C1OC1") if mol else 0,
    }
    acid = _clamp(8 + 22 * counts["ester"] + 16 * counts["lactone"] + 5 * counts["amide"], 0, 95)
    neutral = _clamp(6 + 18 * counts["ester"] + 14 * counts["lactone"] + 4 * counts["amide"], 0, 95)
    plasma = _clamp(5 + 20 * counts["ester"] + 12 * counts["lactone"], 0, 95)
    oxidation = _clamp(5 + 12 * counts["phenol"] + 10 * counts["aniline"] + 4 * max(0, int((profile.get("physchem") or {}).get("aromatic_rings") or 0) - 2), 0, 95)
    photo = _clamp(5 + 6 * int((profile.get("physchem") or {}).get("aromatic_rings") or 0) + 4 * max(0.0, float((profile.get("physchem") or {}).get("logp") or 0) - 3), 0, 95)
    risks = [acid, neutral, plasma, oxidation, photo]
    overall = _clamp(max(risks) * 0.7 + sum(risks) / len(risks) * 0.3, 0, 95)
    return {
        "counts": counts,
        "ph_1_2": round(acid, 1),
        "ph_6_8": round(neutral, 1),
        "plasma": round(plasma, 1),
        "oxidation": round(oxidation, 1),
        "phototoxicity": round(photo, 1),
        "overall": round(overall, 1),
        "category": T("高", "High") if overall >= 60 else T("中", "Medium") if overall >= 30 else T("低", "Low"),
        "recommendation": (
            T("避免长期水溶液放置，优先固体剂型并做强制降解试验。", "Avoid long-term aqueous storage; prefer a solid dosage form and run forced-degradation studies.")
            if overall >= 55
            else T("常规稳定性风险可控，仍建议做 pH、氧化与光照强制降解。", "Routine stability risk is manageable; pH, oxidative and photolytic forced degradation are still recommended.")
        ),
        "confidence": _confidence(
            55.0,
            T("基于官能团水解/氧化规则的筛查，不替代强制降解与血浆稳定性实验。", "Functional-group hydrolysis/oxidation rules; not a substitute for forced-degradation or plasma-stability assays."),
        ),
    }


def formulation_block(profile: dict, oral: dict, stability: dict) -> dict:
    phys = profile.get("physchem") or {}
    drugs = profile.get("druglike") or {}
    logs = float(phys.get("logs") or -4.0)
    bcs = oral.get("bcs")
    basic = bool(drugs.get("basic_center"))
    acidic = bool(drugs.get("acidic_center"))
    dose_number = float(oral.get("dose_number") or 0)
    if bcs == "I":
        form = T("普通片剂或胶囊（速释）", "Conventional immediate-release tablet or capsule")
        risk = T("低", "Low")
    elif bcs == "II":
        form = T("无定形固体分散体、脂质制剂或纳米晶", "Amorphous solid dispersion, lipid formulation or nanocrystal")
        risk = T("中", "Medium")
    elif bcs == "III":
        form = T("口服溶液、分散片或含渗透促进剂的制剂", "Oral solution, dispersible tablet or permeation-enhancer formulation")
        risk = T("中", "Medium")
    else:
        form = T("脂质纳米载体或固体分散体 + 渗透促进剂", "Lipid nanocarrier or solid dispersion plus permeation enhancer")
        risk = T("高", "High")
    if basic or acidic:
        salt = T("存在可电离中心，成盐可行性较高；需筛选反离子与晶型。", "Ionizable centre present; salt formation is feasible but counter-ion and polymorph screening are needed.")
    elif logs < -4.0:
        salt = T("中性且溶解度低，优先考虑共晶、共无定形或固体分散体。", "Neutral and poorly soluble; consider co-crystal, co-amorphous or solid dispersion.")
    else:
        salt = T("无需成盐，常规制剂即可。", "No salt formation needed; conventional formulation is appropriate.")
    excipient = []
    if (stability.get("counts") or {}).get("aldehyde"):
        excipient.append(T("避免伯胺/酰肼类辅料（醛反应）", "Avoid primary-amine/hydrazide excipients (aldehyde reactivity)"))
    if (stability.get("counts") or {}).get("ester"):
        excipient.append(T("注意酯类水解，控制制剂水分与 pH", "Watch ester hydrolysis; control moisture and pH"))
    if (stability.get("counts") or {}).get("phenol"):
        excipient.append(T("需抗氧化剂与避光包装", "Antioxidant and light-protective packaging recommended"))
    if not excipient:
        excipient.append(T("未发现特异性辅料不相容警示", "No specific excipient-incompatibility alert"))
    developability = _clamp(
        100 - 18 * (bcs in ("II", "IV")) - 12 * (dose_number > 10) - 0.25 * float(stability.get("overall") or 0),
        0, 100,
    )
    return {
        "bcs": bcs,
        "dose_number": round(dose_number, 2),
        "form": form,
        "risk": risk,
        "salt_or_cocrystal": salt,
        "excipient": excipient,
        "developability": round(developability, 1),
        "confidence": _confidence(
            52.0,
            T("基于 BCS、剂量数与官能团的制剂可行性规则，不替代处方前研究。", "Formulation feasibility rules from BCS, dose number and functional groups; not a substitute for preformulation work."),
        ),
    }


def risk_block(profile: dict, nucleic: dict, stability: dict) -> dict:
    phys = profile.get("physchem") or {}
    logp = float(phys.get("logp") or 2.0)
    rings = int(phys.get("aromatic_rings") or 0)
    ames = _admet_value(profile, "ames", 15.0)
    herg = _admet_value(profile, "herg", 20.0)
    cyp3a4 = _admet_value(profile, "cyp3a4", 20.0)
    cyp2d6 = _admet_value(profile, "cyp2d6", 20.0)
    pgp = _admet_value(profile, "pgp", 25.0)
    counts = stability.get("counts") or {}
    aniline = counts.get("aniline", 0)
    epoxide = counts.get("epoxide", 0)
    covalent_keys = set(nucleic.get("covalent_keys") or [])
    michael = 1 if "michael" in covalent_keys else 0
    nitro = 1 if "nitro" in covalent_keys else 0
    dili = _clamp(12 + 14 * max(0.0, logp - 3.0) + 8 * max(0, rings - 2) + 12 * aniline + 8 * nitro, 0, 95)
    carcinogenicity = _clamp(0.55 * ames + 12 * (aniline + epoxide + michael + nitro), 0, 95)
    reprotox = _clamp(10 + 10 * max(0.0, logp - 4.0) + 10 * epoxide + 8 * michael, 0, 95)
    sensitization = _clamp(8 + 22 * michael + 16 * aniline + 12 * epoxide, 0, 95)
    phototoxicity = _clamp(stability.get("phototoxicity") or 10, 0, 95)
    environmental = _clamp(8 + 12 * max(0.0, logp - 4.5) + 6 * rings, 0, 95)
    endpoints = [
        ("hERG", herg, T("亲脂性与碱性中心驱动的经验风险", "Empirical risk from lipophilicity and basic centres")),
        ("DILI", dili, T("亲脂性、芳环与苯胺/硝基警示", "Lipophilicity, aromatic rings and aniline/nitro alerts")),
        ("Ames", ames, T("结构警示子汇总", "Sum of structural alerts")),
        (T("致癌性", "Carcinogenicity"), carcinogenicity, T("Ames 与共价/DNA 反应警示的组合", "Combination of Ames and covalent/DNA-reactive alerts")),
        (T("生殖毒性", "Reproductive toxicity"), reprotox, T("高亲脂性与反应性基团的启发式风险", "Heuristic risk from high lipophilicity and reactive groups")),
        (T("皮肤致敏", "Skin sensitization"), sensitization, T("Michael 受体、苯胺与环氧化物警示", "Michael acceptor, aniline and epoxide alerts")),
        (T("光毒性", "Phototoxicity"), phototoxicity, T("芳环数量与亲脂性", "Aromatic ring count and lipophilicity")),
        (T("环境风险", "Environmental risk"), environmental, T("高 logP 与持久性倾向", "High logP and persistence tendency")),
        (T("CYP3A4 抑制", "CYP3A4 inhibition"), cyp3a4, T("大体积亲脂芳环体系", "Large lipophilic aromatic systems")),
        (T("CYP2D6 抑制", "CYP2D6 inhibition"), cyp2d6, T("碱性氮与亲脂性", "Basic nitrogen and lipophilicity")),
        (T("P-gp 外排", "P-gp efflux"), pgp, T("高 TPSA 与碱性氮", "High TPSA and basic nitrogen")),
    ]
    for name, value, note in endpoints:
        value = _clamp(float(value), 0, 95)
    overall = _clamp(
        0.22 * herg + 0.22 * dili + 0.18 * ames + 0.12 * carcinogenicity
        + 0.10 * reprotox + 0.08 * sensitization + 0.08 * environmental,
        0, 95,
    )
    level = T("高", "High") if overall >= 60 else T("中", "Medium") if overall >= 30 else T("低", "Low")
    return {
        "endpoints": [
            {
                "name": name,
                "value": round(float(value), 1),
                "level": T("高", "High") if value >= 60 else T("中", "Medium") if value >= 30 else T("低", "Low"),
                "note": note,
            }
            for name, value, note in endpoints
        ],
        "overall": round(overall, 1),
        "level": level,
        "confidence": _confidence(
            46.0,
            T("规则式风险筛查，不能替代 GLP 毒理与临床安全数据。", "Rule-based risk screening; not a substitute for GLP toxicology or clinical safety data."),
        ),
    }


def tissue_block(profile: dict, species: str, target_data: dict | None = None) -> dict:
    phys = profile.get("physchem") or {}
    logp = float(phys.get("logp") or 2.0)
    logbb = _admet_value(profile, "logbb", 0.0)
    info = SPECIES.get(species, SPECIES["human"])
    rows = []
    for tissue, factor in TISSUE_FACTORS.items():
        if tissue == "brain":
            kp = 10 ** _clamp(logbb, -1.0, 1.0) * factor
        elif tissue == "fat":
            kp = 10 ** _clamp(0.55 * logp - 0.45, -0.8, 1.5)
        elif tissue in ("liver", "kidney"):
            kp = factor * 10 ** _clamp(0.22 * logp - 0.10, -0.5, 1.2)
        else:
            kp = factor * 10 ** _clamp(0.26 * logp - 0.20, -0.7, 1.0)
        kp = _clamp(kp, 0.02, 30.0)
        volume = TISSUE_VOLUME.get(tissue, 0.01)
        if tissue == "fat":
            volume = info.get("fat_fraction", volume)
        elif tissue == "muscle":
            volume = info.get("muscle_fraction", volume)
        rows.append(
            {
                "tissue": tissue,
                "label": T(*TISSUE_LABELS.get(tissue, (tissue, tissue))),
                "kp": round(kp, 2),
                "volume": volume,
                "load": kp * volume,
            }
        )
    total = sum(row["load"] for row in rows) or 1.0
    for row in rows:
        row["share"] = round(100.0 * row["load"] / total, 1)
    rows.sort(key=lambda row: -row["share"])
    target_rows = []
    overlap = 0.0
    if target_data:
        for item in (target_data.get("targets") or []):
            hpa = item.get("hpa") or {}
            top = hpa.get("rna_top_tissue") or {}
            tissue = next(iter(top.keys()), "") if top else ""
            value = next(iter(top.values()), "") if top else ""
            target_rows.append(
                {
                    "gene": item.get("gene", ""),
                    "protein": item.get("protein_name", ""),
                    "tissue": tissue,
                    "ntpm": value,
                    "distribution": _hpa_label(hpa.get("rna_distribution", "")),
                    "specificity": _hpa_label(hpa.get("rna_specificity", "")),
                    "tissue_text": (item.get("tissue_specificity") or "")[:160],
                }
            )
        top_tissues = {
            row["tissue"].lower() for row in target_rows if row.get("tissue")
        }
        matched = [row["kp"] for row in rows if row["tissue"].lower() in top_tissues]
        if matched:
            overlap = _clamp(100.0 * sum(matched) / max(max(row["kp"] for row in rows), 1e-6) / max(len(matched), 1), 0, 100)
    return {
        "species": species,
        "species_label": T(info.get("zh", ""), info.get("en", "")),
        "kp_rows": rows,
        "target_rows": target_rows,
        "overlap_score": round(overlap, 1),
        "confidence": _confidence(
            35.0 if (target_data and target_data.get("available")) else 30.0,
            T(
                "化合物 Kp 为 logP 与组织因子的经验分配模型（不含血浆蛋白结合项），不是 PBPK；靶点组织表达来自 HPA/UniProt（人源）。",
                "Compound Kp is an empirical logP/tissue-factor partition model (no plasma-protein-binding term), not PBPK; target tissue expression comes from HPA/UniProt (human).",
            ),
        ),
    }


# ---------------------------------------------------------------------------
# Veterinary extension: MRL, withdrawal period, microbial metabolism
# ---------------------------------------------------------------------------
# Example maximum residue limits (µg/kg).  Real limits are drug- and
# market-specific (EU 37/2010, US 21 CFR 556, China GB 31650); these values are
# only a screening default and are labelled as such everywhere they appear.
_VET_CLASSES = [
    (
        "sulfonamide",
        "磺胺类",
        "S(=O)(=O)[N;X3]",
        {"muscle": 100, "liver": 100, "kidney": 100, "fat": 100},
    ),
    (
        "beta_lactam",
        "β-内酰胺类",
        "[NX3;R1]1C(=O)[CX4][CX4]1",
        {"muscle": 50, "liver": 50, "kidney": 50, "fat": 50},
    ),
    (
        "fluoroquinolone",
        "氟喹诺酮类",
        "O=C(O)c1cn(C)c2cc(F)c(N)cc12",
        {"muscle": 100, "liver": 200, "kidney": 300, "fat": 100},
    ),
    (
        "aminoglycoside",
        "氨基糖苷类",
        "[NX3;H2,H1,H0;!R][CX4;R]",
        {"muscle": 500, "liver": 500, "kidney": 1000, "fat": 500},
    ),
]

_MICROBIAL_RULES = [
    (
        "azo",
        "偶氮还原",
        "[#6:1][N;X2:2]=[N;X2:3][#6:4]>>[#6:1][N:2].[N:3][#6:4]",
    ),
    ("nitro", "硝基还原", "[#6:1][N+:2](=[O:3])[O-:4]>>[#6:1][N:2]"),
    (
        "sulfoxide",
        "亚砜还原",
        "[#6:1][S;X3:2](=[O:3])[#6:4]>>[#6:1][S:2][#6:4]",
    ),
    (
        "ester",
        "酯/葡萄糖醛酸水解",
        "[C:1](=[O:2])[O;X2:3][#6:4]>>[C:1](=[O:2])[OH].[#6:4][OH]",
    ),
    (
        "amide",
        "酰胺/乙酰化水解",
        "[C:1](=[O:2])[N;X3:3]>>[C:1](=[O:2])[OH].[N:3]",
    ),
    ("n_oxide", "N-氧化物还原", "[#6:1][N+:2][O-:3]>>[#6:1][N:2]"),
]


def _vet_class(mol: Chem.Mol | None) -> dict:
    if mol is None:
        return {"applicable": False}
    for key, label, smarts, limits in _VET_CLASSES:
        query = Chem.MolFromSmarts(smarts)
        if query is not None and mol.HasSubstructMatch(query):
            return {
                "applicable": True,
                "key": key,
                "label": label,
                "limits": limits,
                "confidence": 55.0,
            }
    if (
        mol.GetRingInfo().NumRings() >= 2
        and _smarts_count(mol, "[NX3;H2,H1]") >= 2
        and _smarts_count(mol, "[OX2H]") >= 3
        and _smarts_count(mol, "c") == 0
    ):
        return {
            "applicable": True,
            "key": "aminoglycoside_like",
            "label": "氨基糖苷样（多氨基多羟基）",
            "limits": {"muscle": 500, "liver": 500, "kidney": 1000, "fat": 500},
            "confidence": 40.0,
        }
    return {"applicable": False}


def mrl_block(profile: dict, species: str) -> dict:
    """Class-level maximum residue limits, explicitly marked as examples."""
    info = SPECIES.get(species, SPECIES["human"])
    if not info.get("food"):
        return {
            "applicable": False,
            "note": T(
                "非食用动物（人或伴侣动物），不适用兽药 MRL。",
                "Not a food-producing species (human or companion animal); MRLs do not apply.",
            ),
        }
    mol = Chem.MolFromSmiles(profile.get("smiles") or "") if profile.get("smiles") else None
    info = _vet_class(mol)
    if not info.get("applicable"):
        return {
            "applicable": False,
            "note": T(
                "未识别到常见兽药类别；请按目标药物与目标市场法规手动核对 MRL。",
                "No common veterinary class detected; check the MRL manually for the drug and market.",
            ),
        }
    return {
        "applicable": True,
        "class_key": info["key"],
        "class_label": info["label"],
        "limits_ug_kg": info["limits"],
        "confidence": _confidence(
            info.get("confidence", 40.0),
            T(
                "示例 MRL（欧盟 37/2010 / 美国 21 CFR 556 / 中国 GB 31650 的常见量级），"
                "不是法规原文，注册或残留试验前必须以目标市场现行法规为准。",
                "Example MRLs (order of magnitude from EU 37/2010, US 21 CFR 556 and China GB 31650); "
                "not the legal text. Always confirm against the current regulation before registration or residue studies.",
            ),
        ),
    }


def withdrawal_block(profile: dict, species: str, mrl: dict) -> dict:
    """Heuristic tissue depletion / withdrawal-period estimate."""
    info = SPECIES.get(species, SPECIES["human"])
    if not info.get("food"):
        return {
            "applicable": False,
            "note": T(
                "非食用动物（人或伴侣动物），不计算休药期。",
                "Not a food-producing species (human or companion animal); no withdrawal period is calculated.",
            ),
        }
    phys = profile.get("physchem") or {}
    logp = float(phys.get("logp") or 2.0)
    mw = float(phys.get("mw") or 300.0)
    # Terminal half-life proxy: lipophilic, larger molecules deplete more slowly.
    half_life_h = _clamp(4.0 + 3.2 * max(0.0, logp) + 0.012 * (mw - 300.0), 2.0, 160.0)
    limit = 100.0
    if mrl.get("applicable"):
        limit = float((mrl.get("limits_ug_kg") or {}).get("muscle") or 100.0)
    # 95% depletion needs ~4.32 half-lives; add one half-life as safety margin.
    days = int(math.ceil((5.32 * half_life_h) / 24.0))
    days = max(1, min(days, 120))
    strategy = T(
        "按脂肪/肌肉残留风险保守估计，建议以目标组织的实测消除曲线复核。",
        "Conservative estimate from fat/muscle depletion; confirm with a measured tissue elimination curve.",
    )
    return {
        "applicable": True,
        "half_life_h": round(half_life_h, 1),
        "mrl_ug_kg": limit,
        "withdrawal_days": days,
        "strategy": strategy,
        "confidence": _confidence(
            35.0,
            T(
                "仅由 logP、分子量与类别 MRL 推导的经验值，不能替代残留消除试验；"
                "不同给药途径、剂量与剂型的休药期差异很大。",
                "Empirical value derived from logP, MW and a class-level MRL; it cannot replace a residue depletion study. "
                "Route, dose and formulation change the withdrawal period substantially.",
            ),
        ),
    }


def microbial_metabolism(profile: dict) -> dict:
    """Predict gut-microbial transformations and, where possible, products."""
    smiles = profile.get("smiles") or ""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {"available": False, "items": []}
    items = []
    for key, label, smarts in _MICROBIAL_RULES:
        query = Chem.MolFromSmarts(smarts.split(">>")[0])
        count = len(mol.GetSubstructMatches(query)) if query is not None else 0
        if not count:
            continue
        products: list[str] = []
        try:
            reaction = AllChem.ReactionFromSmarts(smarts)
            for product_set in reaction.RunReactants((mol,)):
                for product in product_set:
                    try:
                        Chem.SanitizeMol(product)
                    except Exception:
                        continue
                    if any(atom.GetNumRadicalElectrons() for atom in product.GetAtoms()):
                        continue
                    value = Chem.MolToSmiles(product)
                    if value and value != smiles and value not in products:
                        products.append(value)
                    if len(products) >= 3:
                        break
                if len(products) >= 3:
                    break
        except Exception:
            products = []
        items.append(
            {
                "key": key,
                "label": label,
                "sites": int(count),
                "products": products,
                "predicted": bool(products),
            }
        )
    return {
        "available": bool(items),
        "items": items,
        "note": T(
            "按常见菌群还原/水解规则枚举，产物仅作假设；需用粪便菌群孵育或高分辨质谱确认。",
            "Enumerated from common microbial reduction/hydrolysis rules; products are hypotheses and require faecal-microbiota incubation or high-resolution MS confirmation.",
        ),
    }


_REFERENCE_FPS = None


def _reference_fingerprints():
    global _REFERENCE_FPS
    if _REFERENCE_FPS is not None:
        return _REFERENCE_FPS
    from .drugref import FDA_DRUGS

    entries = []
    for item in FDA_DRUGS:
        mol = Chem.MolFromSmiles(item.get("smiles") or "")
        if mol is None:
            continue
        entries.append(
            (
                item.get("name") or "drug",
                AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
            )
        )
    _REFERENCE_FPS = entries
    return entries


def applicability_block(smiles: str) -> dict:
    """Applicability domain against a reference set of marketed drugs.

    A rule-based model is only meaningful near chemistries it was designed
    for.  The nearest-neighbour Tanimoto similarity to the built-in FDA drug
    set is a transparent proxy for "is this molecule in familiar space".
    """
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return {"available": False, "note": T("无法解析结构。", "SMILES could not be parsed.")}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    scored = []
    for name, ref in _reference_fingerprints():
        try:
            scored.append((DataStructs.TanimotoSimilarity(fp, ref), name))
        except Exception:
            continue
    if not scored:
        return {"available": False, "note": T("参考集不可用。", "Reference set unavailable.")}
    scored.sort(reverse=True)
    top = scored[:5]
    max_sim = float(top[0][0])
    mean_top = sum(value for value, _ in top) / len(top)
    if max_sim >= 0.55:
        level = T("域内", "In domain")
    elif max_sim >= 0.35:
        level = T("边缘", "Borderline")
    else:
        level = T("域外", "Out of domain")
    return {
        "available": True,
        "max_similarity": round(max_sim, 3),
        "mean_top5": round(mean_top, 3),
        "nearest": top[0][1],
        "level": level,
        "in_domain": max_sim >= 0.35,
        "note": T(
            "以 45 个上市药为参考集，用 Morgan/Tanimoto 最近邻判断适用域；"
            "域外分子的规则预测不确定性更高。",
            "Nearest-neighbour Tanimoto to 45 marketed drugs defines the applicability domain; "
            "out-of-domain predictions carry higher uncertainty.",
        ),
    }


def allometry_block(profile: dict, species: str) -> dict:
    """Allometric cross-species PK extrapolation from a human reference."""
    info = SPECIES.get(species, SPECIES["human"])
    phys = profile.get("physchem") or {}
    logp = float(phys.get("logp") or 2.0)
    tpsa = float(phys.get("tpsa") or 60.0)
    # Human reference clearance / volume: deliberately simple and monotonic.
    cl_human = _clamp(2.0 + 1.5 * logp - 0.01 * tpsa, 0.3, 25.0)
    vd_human = _clamp(0.5 + 0.25 * logp, 0.2, 8.0)
    ratio = max(0.01, float(info.get("body_weight") or 70.0) / 70.0)
    # Per-kg clearance scales with BW^-0.25; volume is nearly isometric.
    clearance = cl_human * ratio ** -0.25
    volume = vd_human * ratio ** -0.05
    half_life = 0.693 * volume / max(0.01, clearance * 0.06)
    return {
        "applicable": True,
        "body_weight_kg": round(float(info.get("body_weight") or 70.0), 2),
        "cl_ml_min_kg": round(clearance, 2),
        "vd_l_kg": round(volume, 2),
        "half_life_h": round(half_life, 2),
        "confidence": _confidence(
            40.0,
            T(
                "以人源经验清除率/分布容积为锚点、按 0.75 指数做异速外推；"
                "未考虑种属特异性代谢酶与血浆蛋白差异，必须用药代试验校正。",
                "Human empirical clearance/volume anchored allometric extrapolation (0.75 exponent); "
                "species-specific enzymes and plasma-protein binding are not modelled and must be calibrated experimentally.",
            ),
        ),
    }


def svdi_block(
    oral: dict,
    rumen: dict,
    stability: dict,
    formulation: dict,
    risk: dict,
    withdrawal: dict,
    species: str,
    applicability: dict,
) -> dict:
    """Species-aware Veterinary Developability Index (0-100).

    The index reweights the same transparent blocks for food-producing,
    ruminant and companion-animal use, so a molecule that looks good in a
    human rule set is not automatically called "developable" for cattle or a
    cat.  It is a screening heuristic, not a regulatory score.
    """
    info = SPECIES.get(species, SPECIES["human"])
    components: dict[str, float] = {
        "oral": _clamp(float(oral.get("f_percent") or 0) / 100.0, 0.0, 1.0),
        "stability": _clamp((100.0 - float(stability.get("overall") or 0)) / 100.0, 0.0, 1.0),
        "formulation": _clamp(float(formulation.get("developability") or 0) / 100.0, 0.0, 1.0),
        "safety": _clamp((100.0 - float(risk.get("overall") or 0)) / 100.0, 0.0, 1.0),
    }
    weights = {
        "oral": 0.34,
        "stability": 0.18,
        "formulation": 0.18,
        "safety": 0.16,
    }
    if info.get("rumen"):
        components["rumen"] = _clamp(float(rumen.get("bypass_percent") or 0) / 100.0, 0.0, 1.0)
        weights["rumen"] = 0.14
    if info.get("food"):
        days = float(withdrawal.get("withdrawal_days") or 90)
        components["residue"] = _clamp(1.0 - days / 90.0, 0.0, 1.0)
        weights["residue"] = 0.14
    else:
        components["applicability"] = _clamp(
            float(applicability.get("max_similarity") or 0), 0.0, 1.0
        )
        weights["applicability"] = 0.14
    total_weight = sum(weights.values())
    score = 100.0 * sum(
        weights[key] * components[key] for key in weights
    ) / max(total_weight, 1e-6)
    level = (
        T("高", "High") if score >= 70
        else T("中", "Medium") if score >= 45
        else T("低", "Low")
    )
    return {
        "score": round(score, 1),
        "level": level,
        "components": {key: round(value, 3) for key, value in components.items()},
        "weights": weights,
        "note": T(
            "按种属用途重加权的可开发性指数：食用动物计入休药期，反刍动物计入过瘤胃，"
            "伴侣动物计入适用域；仅用于排序与讨论。",
            "Species-aware developability index: withdrawal for food animals, rumen bypass for ruminants, "
            "applicability domain for companion animals; for ranking and discussion only.",
        ),
    }


def covalent_warheads(smiles: str) -> list[str]:
    """Detect common covalent warheads that Vina cannot model explicitly."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return []
    patterns = [
        ("腈/亚胺型弹头", "nitrile/imine warhead", "[CX2]#N"),
        ("腈/亚胺型弹头", "nitrile/imine warhead", "[CX3H1]=[NX2]"),
        ("醛基", "aldehyde", "[CX3H1](=O)"),
        ("Michael 受体", "Michael acceptor", "[CX3]=[CX3][CX3]=[OX1]"),
        ("环氧化物", "epoxide", "C1OC1"),
        ("烷基卤", "alkyl halide", "[CX4][Cl,Br,I]"),
        ("硼酸/硼酸酯", "boronic acid/ester", "[#5]([OX2H1,OX1-])[OX2H1,OX1-]"),
        ("α-酮酰胺", "α-ketoamide", "[CX3](=O)[CX3](=O)"),
        ("磺酰氟", "sulfonyl fluoride", "S(=O)(=O)F"),
    ]
    found = []
    for zh, en, smarts in patterns:
        query = Chem.MolFromSmarts(smarts)
        if query is not None and mol.HasSubstructMatch(query):
            label = T(zh, en)
            if label not in found:
                found.append(label)
    return found


def predict(
    smiles: str,
    name: str = "",
    species: str = "human",
    genes: list[str] | None = None,
    affinity: float | None = None,
    include_online: bool = True,
) -> dict:
    species = species if species in SPECIES else "human"
    profile = admet.profile_smiles(smiles, name=name, affinity=affinity)
    genes = [str(gene).strip() for gene in (genes or []) if str(gene).strip()]
    online_data = None
    if include_online:
        try:
            online_data = online.online_evidence(smiles, genes, species)
        except Exception as exc:
            online_data = {"available": False, "error": str(exc)}
    oral = oral_block(profile, species)
    rumen = rumen_block(profile, species)
    microbiota = microbiota_block(profile)
    nucleic = nucleic_acid_block(profile)
    stability = stability_block(profile)
    formulation = formulation_block(profile, oral, stability)
    risk = risk_block(profile, nucleic, stability)
    tissue = tissue_block(profile, species, (online_data or {}).get("targets"))
    mrl = mrl_block(profile, species)
    withdrawal = withdrawal_block(profile, species, mrl)
    microbial = microbial_metabolism(profile)
    applicability = applicability_block(profile.get("smiles") or smiles)
    allometry = allometry_block(profile, species)
    svdi = svdi_block(
        oral, rumen, stability, formulation, risk, withdrawal, species, applicability
    )
    confidences = [
        oral["confidence"], rumen.get("confidence"), microbiota["confidence"],
        nucleic["confidence"], stability["confidence"], formulation["confidence"],
        risk["confidence"], tissue["confidence"],
        withdrawal.get("confidence"),
    ]
    confidences = [item for item in confidences if item]
    overall_conf = sum(item["score"] for item in confidences) / max(len(confidences), 1)
    return {
        "name": name or profile.get("name") or "compound",
        "smiles": profile.get("smiles") or smiles,
        "species": species,
        "species_label": T(SPECIES[species]["zh"], SPECIES[species]["en"]),
        "generated": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
        "profile": profile,
        "physchem": profile.get("physchem") or {},
        "oral": oral,
        "rumen": rumen,
        "microbiota": microbiota,
        "nucleic_acid": nucleic,
        "stability": stability,
        "formulation": formulation,
        "risk": risk,
        "tissue": tissue,
        "mrl": mrl,
        "withdrawal": withdrawal,
        "microbial_metabolism": microbial,
        "applicability": applicability,
        "allometry": allometry,
        "svdi": svdi,
        "online": online_data,
        "covalent_warheads": covalent_warheads(profile.get("smiles") or smiles),
        "confidence": {
            "score": round(overall_conf, 0),
            "level": T("高", "High") if overall_conf >= 70 else T("中", "Medium") if overall_conf >= 45 else T("低", "Low"),
            "note": T(
                "综合置信度按各模块输入的完整度、模型类型与种间外推程度加权；规则模型不应当作实测值。",
                "Overall confidence weights input completeness, model type and species extrapolation; rule-based values are not measurements.",
            ),
        },
        "methods": [
            "RDKit descriptors",
            "ESOL / Clark correlations",
            "DrugHX species rules",
            "UniProt REST",
            "Human Protein Atlas",
            "PubChem PUG REST",
            "ChEMBL REST",
        ],
    }


# ---------------------------------------------------------------------------
# PDF pages
# ---------------------------------------------------------------------------
def _kpi_row(fig, items: list[tuple[str, str, str]], top: float = 0.805):
    width = 0.205
    gap = 0.015
    left = 0.07
    for index, (label, value, color) in enumerate(items[:4]):
        x = left + index * (width + gap)
        fig.add_artist(
            Rectangle(
                (x, top - 0.085), width, 0.085, transform=fig.transFigure,
                facecolor="#f8fafc", edgecolor=figures.GRID, linewidth=0.8,
            )
        )
        fig.text(x + 0.012, top - 0.018, label, fontsize=8.6, color=figures.MUTED, va="top")
        fig.text(
            x + 0.012, top - 0.046, value, fontsize=15.5, color=color,
            fontweight="bold", va="top",
        )


def _risk_color(value: float) -> str:
    if value >= 60:
        return figures.RED
    if value >= 30:
        return figures.AMBER
    return figures.TEAL


def _bar_panel(ax, labels, values, colors, title):
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.62, edgecolor="none")
    for index, value in enumerate(values):
        ax.text(
            max(value, 0) + 1.5, index, f"{value:.1f}",
            va="center", fontsize=9.4, color=figures.INK,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.8, color=figures.INK)
    ax.set_ylim(len(labels) - 0.45, -0.85)
    ax.set_xlim(0, 110)
    ax.set_xlabel("0–100", fontsize=9.0, color=figures.MUTED)
    ax.set_title(title, loc="left", fontsize=12.5, pad=8, color=figures.INK)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=figures.GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def pharm_pdf(data: dict, outpath: Path) -> Path:
    """Six-page A4 pharmacology report."""
    figures._style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    name = data.get("name") or "compound"
    species = data.get("species_label") or data.get("species") or ""
    oral = data.get("oral") or {}
    rumen = data.get("rumen") or {}
    microbiota = data.get("microbiota") or {}
    nucleic = data.get("nucleic_acid") or {}
    stability = data.get("stability") or {}
    formulation = data.get("formulation") or {}
    risk = data.get("risk") or {}
    tissue = data.get("tissue") or {}
    confidence = data.get("confidence") or {}

    with PdfPages(outpath) as pdf:
        # ------------------------------------------------------------ page 1
        fig = figures._a4_page(
            T("图 C1　综合药理与风险预测", "Fig. C1　Integrated pharmacology and risk assessment"),
            T(
                f"{name} · {species} · 口服、稳定性、剂型、核酸结合、组织分布与综合风险",
                f"{name} · {species} · oral exposure, stability, formulation, nucleic-acid binding, tissue distribution and integrated risk",
            ),
            footer=T("DrugHX · 综合药理报告", "DrugHX · Pharmacology report"),
        )
        _kpi_row(
            fig,
            [
                (
                    T("预测口服生物利用度", "Predicted oral F"),
                    f"{float(oral.get('f_percent') or 0):.0f}%",
                    figures.TEAL,
                ),
                (T("BCS 分类", "BCS class"), str(oral.get("bcs", "—")), figures.BLUE),
                (
                    T("肠道菌群风险", "Microbiota risk"),
                    str(microbiota.get("category", "—")),
                    _risk_color(float(microbiota.get("score") or 0)),
                ),
                (
                    T("综合风险", "Integrated risk"),
                    str(risk.get("level", "—")),
                    _risk_color(float(risk.get("overall") or 0)),
                ),
            ],
        )
        ax = fig.add_axes([0.19, 0.43, 0.74, 0.23])
        labels = [
            T("口服暴露", "Oral exposure"),
            T("瘤胃过瘤胃", "Rumen bypass"),
            T("菌群扰动", "Microbiota"),
            T("核酸结合", "Nucleic-acid binding"),
            T("稳定性风险", "Stability risk"),
            T("制剂可开发性", "Formulation"),
            T("综合风险", "Integrated risk"),
        ]
        values = [
            float(oral.get("f_percent") or 0),
            float(rumen.get("bypass_percent") or 0) if rumen.get("applicable") else 0.0,
            float(microbiota.get("score") or 0),
            float(nucleic.get("overall") or 0),
            float(stability.get("overall") or 0),
            float(formulation.get("developability") or 0),
            float(risk.get("overall") or 0),
        ]
        colors = [
            figures.TEAL, figures.BLUE, _risk_color(values[2]), _risk_color(values[3]),
            _risk_color(values[4]), figures.TEAL, _risk_color(values[6]),
        ]
        _bar_panel(ax, labels, values, colors, T("关键模块评分", "Key module scores"))
        bullets = [
            T(
                f"口服：BCS {oral.get('bcs', '—')}，预测 F ≈ {float(oral.get('f_percent') or 0):.0f}%；{oral.get('food_effect', '')}",
                f"Oral: BCS {oral.get('bcs', '—')}, predicted F ≈ {float(oral.get('f_percent') or 0):.0f}%; {oral.get('food_effect', '')}",
            ),
            T(
                f"稳定性：{stability.get('category', '—')}风险；建议：{stability.get('recommendation', '')}",
                f"Stability: {stability.get('category', '—')} risk; {stability.get('recommendation', '')}",
            ),
            T(
                f"剂型：{formulation.get('form', '—')}；{formulation.get('salt_or_cocrystal', '')}",
                f"Formulation: {formulation.get('form', '—')}; {formulation.get('salt_or_cocrystal', '')}",
            ),
            T(
                f"综合风险 {risk.get('level', '—')}（{float(risk.get('overall') or 0):.0f}/100）；总体置信度 {confidence.get('level', '—')}。",
                f"Integrated risk {risk.get('level', '—')} ({float(risk.get('overall') or 0):.0f}/100); overall confidence {confidence.get('level', '—')}.",
            ),
        ]
        figures._text_blocks(
            fig, 0.385,
            [(T("主要结论", "Key findings"), bullets)],
            width=52, size=9.8, stop=0.16,
        )
        figures._caption(
            fig,
            T(
                "综合药理页汇总口服、稳定性、剂型、菌群、核酸结合与风险规则模型；所有数值为筛查级估计。"
                "工具与数据：RDKit 描述符、ESOL、Clark 方程、DrugHX 物种规则；靶点与在线证据见 C6 页。",
                "This page aggregates oral, stability, formulation, microbiota, nucleic-acid and risk rule models; all values are screening-level estimates. "
                "Tools and data: RDKit descriptors, ESOL, Clark equation, DrugHX species rules; target and online evidence are on page C6.",
            ),
            y=0.150,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # ------------------------------------------------------------ page 2
        fig = figures._a4_page(
            T("图 C2　口服性能、过瘤胃与肠道菌群", "Fig. C2　Oral performance, rumen bypass and gut microbiota"),
            T(
                f"{name} · 物种：{species}",
                f"{name} · species: {species}",
            ),
            footer=T("DrugHX · 综合药理报告", "DrugHX · Pharmacology report"),
        )
        oral_rows = [
            [T("指标", "Endpoint"), T("数值", "Value"), T("解释", "Interpretation")],
            [T("BCS 分类", "BCS class"), str(oral.get("bcs", "—")), T("溶解度 × 通透性", "Solubility × permeability")],
            [T("HIA", "HIA"), f"{float(oral.get('hia') or 0):.1f}%", T("人肠吸收经验值", "Empirical human intestinal absorption")],
            [T("Caco-2", "Caco-2"), f"{float(oral.get('caco2') or 0):.2f}", T("log Papp", "log Papp")],
            [T("溶解度", "Solubility"), f"{float(oral.get('solubility_mg_ml') or 0):.4f} mg/mL", T("ESOL 预测", "ESOL prediction")],
            [T("剂量数", "Dose number"), f"{float(oral.get('dose_number') or 0):.2f}", T("100 mg / 250 mL", "100 mg / 250 mL")],
            [T("预测 F", "Predicted F"), f"{float(oral.get('f_percent') or 0):.1f}%", T("种间外推，非实测", "Species extrapolation, not measured")],
        ]
        figures._table(fig, 0.855, oral_rows, [0.22, 0.20, 0.44], fontsize=8.4, row_height=0.027)
        rumen_text = []
        if rumen.get("applicable"):
            rumen_text = [
                T(
                    f"天然过瘤胃约 {float(rumen.get('bypass_percent') or 0):.0f}%，瘤胃降解风险 {float(rumen.get('degradation_risk') or 0):.0f}/100；保护后可望达到约 {float(rumen.get('protected_bypass_percent') or 0):.0f}%。",
                    f"Native rumen bypass ≈ {float(rumen.get('bypass_percent') or 0):.0f}%, degradation risk {float(rumen.get('degradation_risk') or 0):.0f}/100; protection may reach ≈ {float(rumen.get('protected_bypass_percent') or 0):.0f}%.",
                ),
                rumen.get("strategy", ""),
                T("过瘤胃规则仅适用于牛/羊，必须用瘤胃液或尼龙袋试验验证。", "Rumen rules apply to cattle/sheep only and must be verified with rumen-fluid or nylon-bag tests."),
            ]
        else:
            rumen_text = [rumen.get("note", T("该物种不适用。", "Not applicable for this species."))]
        microbiota_text = [
            T(
                f"菌群扰动评分 {float(microbiota.get('score') or 0):.0f}/100（{microbiota.get('category', '—')}）：{microbiota.get('direction', '')}",
                f"Microbiota perturbation score {float(microbiota.get('score') or 0):.0f}/100 ({microbiota.get('category', '—')}): {microbiota.get('direction', '')}",
            ),
            T("结构警示：" + ("、".join(microbiota.get("alerts") or []) or "无"),
              "Structural alerts: " + (", ".join(microbiota.get("alerts") or []) or "none")),
            T("该模块是结构规则筛查，不是菌群生长或宏基因组预测。", "This module is a structural-rule screen, not a growth or metagenomic prediction."),
        ]
        figures._text_blocks(
            fig, 0.575,
            [
                (T("口服性能", "Oral performance"), [
                    T(
                        f"预测 F ≈ {float(oral.get('f_percent') or 0):.1f}%；剂量数 {float(oral.get('dose_number') or 0):.2f}；{oral.get('food_effect', '')}",
                        f"Predicted F ≈ {float(oral.get('f_percent') or 0):.1f}%; dose number {float(oral.get('dose_number') or 0):.2f}; {oral.get('food_effect', '')}",
                    )
                ]),
                (T("过瘤胃（牛/羊）", "Rumen bypass (cattle/sheep)"), rumen_text),
                (T("肠道菌群", "Gut microbiota"), microbiota_text),
            ],
            width=58, size=9.4, stop=0.155,
        )
        figures._caption(
            fig,
            T(
                "口服与瘤胃为物种感知的经验规则；菌群为结构警示。工具与数据：RDKit、ESOL、Clark、DrugHX 物种规则；"
                "不替代药代动力学、瘤胃袋试验或菌群测序。",
                "Oral and rumen blocks are species-aware empirical rules; microbiota is a structural-alert screen. Tools and data: RDKit, ESOL, Clark, DrugHX species rules; "
                "not a substitute for PK studies, nylon-bag tests or microbiome sequencing.",
            ),
            y=0.150,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # ------------------------------------------------------------ page 3
        fig = figures._a4_page(
            T("图 C3　代谢稳定性与剂型研究", "Fig. C3　Metabolic stability and formulation"),
            T(f"{name} · pH、血浆、氧化、光稳定性与制剂策略", f"{name} · pH, plasma, oxidative and photolytic stability plus formulation strategy"),
            footer=T("DrugHX · 综合药理报告", "DrugHX · Pharmacology report"),
        )
        ax = fig.add_axes([0.19, 0.50, 0.74, 0.27])
        labels = [
            T("pH 1.2 水解", "pH 1.2 hydrolysis"),
            T("pH 6.8 水解", "pH 6.8 hydrolysis"),
            T("血浆酯酶", "Plasma esterase"),
            T("氧化", "Oxidation"),
            T("光毒性/光解", "Phototoxicity"),
        ]
        values = [
            float(stability.get("ph_1_2") or 0),
            float(stability.get("ph_6_8") or 0),
            float(stability.get("plasma") or 0),
            float(stability.get("oxidation") or 0),
            float(stability.get("phototoxicity") or 0),
        ]
        _bar_panel(ax, labels, values, [_risk_color(v) for v in values], T("稳定性风险", "Stability risks"))
        counts = stability.get("counts") or {}
        detail = T(
            "官能团计数：酯 " + str(counts.get("ester", 0)) + "、酰胺 " + str(counts.get("amide", 0))
            + "、内酯 " + str(counts.get("lactone", 0)) + "、磺酰胺 " + str(counts.get("sulfonamide", 0))
            + "、醛 " + str(counts.get("aldehyde", 0)) + "、酚 " + str(counts.get("phenol", 0))
            + "、苯胺 " + str(counts.get("aniline", 0)) + "。",
            "Functional-group counts: ester " + str(counts.get("ester", 0)) + ", amide " + str(counts.get("amide", 0))
            + ", lactone " + str(counts.get("lactone", 0)) + ", sulfonamide " + str(counts.get("sulfonamide", 0))
            + ", aldehyde " + str(counts.get("aldehyde", 0)) + ", phenol " + str(counts.get("phenol", 0))
            + ", aniline " + str(counts.get("aniline", 0)) + ".",
        )
        figures._text_blocks(
            fig, 0.455,
            [
                (T("稳定性判读", "Stability interpretation"), [
                    T(
                        f"总体 {stability.get('category', '—')}风险（{float(stability.get('overall') or 0):.0f}/100）。{stability.get('recommendation', '')}",
                        f"Overall {stability.get('category', '—')} risk ({float(stability.get('overall') or 0):.0f}/100). {stability.get('recommendation', '')}",
                    ),
                    detail,
                ]),
                (T("剂型建议", "Formulation recommendation"), [
                    T(
                        f"BCS {formulation.get('bcs', '—')}；建议 {formulation.get('form', '—')}；制剂风险 {formulation.get('risk', '—')}。",
                        f"BCS {formulation.get('bcs', '—')}; suggested {formulation.get('form', '—')}; formulation risk {formulation.get('risk', '—')}.",
                    ),
                    formulation.get("salt_or_cocrystal", ""),
                    T("辅料提示：" + "；".join(formulation.get("excipient") or []),
                      "Excipient notes: " + "; ".join(formulation.get("excipient") or [])),
                    T(
                        f"制剂可开发性 {float(formulation.get('developability') or 0):.0f}/100。",
                        f"Formulation developability {float(formulation.get('developability') or 0):.0f}/100.",
                    ),
                ]),
            ],
            width=58, size=9.4, stop=0.155,
        )
        figures._caption(
            fig,
            T(
                "稳定性与剂型为官能团规则；工具与数据：RDKit SMARTS、ESOL、BCS 规则、DrugHX 制剂规则。"
                "强制降解、血浆稳定性和处方前研究仍是必需步骤。",
                "Stability and formulation are functional-group rules. Tools and data: RDKit SMARTS, ESOL, BCS rules, DrugHX formulation rules. "
                "Forced degradation, plasma stability and preformulation studies remain mandatory.",
            ),
            y=0.150,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # ------------------------------------------------------------ page 4
        fig = figures._a4_page(
            T("图 C4　核酸结合与综合风险", "Fig. C4　Nucleic-acid binding and integrated risk"),
            T(f"{name} · 嵌入/沟槽结合倾向、共价警示与毒理风险谱", f"{name} · intercalation/groove-binding tendency, covalent alerts and toxicology profile"),
            footer=T("DrugHX · 综合药理报告", "DrugHX · Pharmacology report"),
        )
        ax = fig.add_axes([0.21, 0.48, 0.72, 0.31])
        endpoints = (risk.get("endpoints") or [])[:11]
        labels = [item.get("name", "") for item in endpoints]
        values = [float(item.get("value") or 0) for item in endpoints]
        _bar_panel(ax, labels, values, [_risk_color(v) for v in values], T("毒理与安全风险谱", "Toxicology and safety profile"))
        figures._text_blocks(
            fig, 0.435,
            [
                (T("核酸结合", "Nucleic-acid binding"), [
                    T(
                        f"嵌入倾向 {float(nucleic.get('intercalation') or 0):.0f}/100，沟槽结合倾向 {float(nucleic.get('groove') or 0):.0f}/100，综合 {float(nucleic.get('overall') or 0):.0f}/100（{nucleic.get('category', '—')}）。",
                        f"Intercalation {float(nucleic.get('intercalation') or 0):.0f}/100, groove binding {float(nucleic.get('groove') or 0):.0f}/100, overall {float(nucleic.get('overall') or 0):.0f}/100 ({nucleic.get('category', '—')}).",
                    ),
                    nucleic.get("note", ""),
                    T("共价/DNA 反应警示：" + ("、".join(nucleic.get("covalent_alerts") or []) or "无"),
                      "Covalent/DNA-reactive alerts: " + (", ".join(nucleic.get("covalent_alerts") or []) or "none")),
                ]),
                (T("综合风险", "Integrated risk"), [
                    T(
                        f"总体 {risk.get('level', '—')}（{float(risk.get('overall') or 0):.0f}/100）。",
                        f"Overall {risk.get('level', '—')} ({float(risk.get('overall') or 0):.0f}/100).",
                    ),
                    T("高风险项：" + ("、".join(item["name"] for item in endpoints if float(item.get("value") or 0) >= 60) or "无"),
                      "High-risk endpoints: " + (", ".join(item["name"] for item in endpoints if float(item.get("value") or 0) >= 60) or "none")),
                    T("规则式筛查不能替代 GLP 毒理、遗传毒性与临床安全数据。", "Rule-based screening does not replace GLP toxicology, genetic toxicology or clinical safety data."),
                ]),
            ],
            width=58, size=9.4, stop=0.155,
        )
        figures._caption(
            fig,
            T(
                "核酸结合为平面芳香性/电荷/结构警示的启发式评分，不是核酸对接或结合常数；"
                "风险谱整合 RDKit 警示、hERG/CYP 经验式与 DrugHX 规则。工具：RDKit、DrugHX 风险规则。",
                "Nucleic-acid binding is a heuristic score from planarity, charge and structural alerts, not nucleic-acid docking or a binding constant; "
                "the risk profile integrates RDKit alerts, hERG/CYP correlations and DrugHX rules. Tools: RDKit, DrugHX risk rules.",
            ),
            y=0.150,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # ------------------------------------------------------------ page 5
        fig = figures._a4_page(
            T("图 C5　化合物与靶点的组织分布", "Fig. C5　Compound and target tissue distribution"),
            T(f"{name} · 物种：{species} · 经验 Kp 与靶点组织表达", f"{name} · species: {species} · empirical Kp and target tissue expression"),
            footer=T("DrugHX · 综合药理报告", "DrugHX · Pharmacology report"),
        )
        rows = tissue.get("kp_rows") or []
        ax = fig.add_axes([0.19, 0.52, 0.74, 0.26])
        labels = [str(row.get("label") or row.get("tissue")) for row in rows]
        values = [float(row.get("kp") or 0) for row in rows]
        limit = max(max(values, default=1.0) * 1.28, 1.0)
        ax.barh(np.arange(len(labels)), values, color=figures.TEAL, height=0.62)
        for index, value in enumerate(values):
            ax.text(value + limit * 0.015, index, f"{value:.2f}", va="center", fontsize=8.4, color=figures.INK)
        ax.set_xlim(0, limit)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8.8)
        ax.invert_yaxis()
        ax.set_xlabel("Kp (tissue/plasma)", fontsize=8.4, color=figures.MUTED)
        ax.set_title(T("预测组织–血浆分配系数", "Predicted tissue–plasma partition"), loc="left", fontsize=11.2, pad=6)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=figures.GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        share_rows = [[T("组织", "Tissue"), "Kp", T("分布占比", "Distribution share")]]
        for row in rows[:6]:
            share_rows.append([
                str(row.get("label") or row.get("tissue")),
                f"{float(row.get('kp') or 0):.2f}",
                f"{float(row.get('share') or 0):.1f}%",
            ])
        figures._table(fig, 0.485, share_rows, [0.25, 0.20, 0.30], fontsize=8.2, row_height=0.024)
        target_text = [
            T(
                f"{row.get('gene')}: {row.get('tissue')}（{row.get('ntpm') or '—'} nTPM）；{row.get('specificity') or ''}",
                f"{row.get('gene')}: {row.get('tissue')} ({row.get('ntpm') or '—'} nTPM); {row.get('specificity') or ''}",
            )
            for row in (tissue.get("target_rows") or [])[:4]
            if row.get("tissue")
        ]
        if not target_text:
            target_text.append(
                T(
                    "非人源物种无 HPA 表达数据；靶点组织表达仅在人源报告中展示。",
                    "HPA expression data are human-only; target tissue expression is shown in human reports.",
                )
                if data.get("species") != "human"
                else T(
                    "未提供靶点基因，或在线表达数据不可用。",
                    "No target genes supplied, or online expression data unavailable.",
                )
            )
        figures._text_blocks(
            fig, 0.300,
            [
                (T("靶点组织表达（HPA/UniProt）", "Target tissue expression (HPA/UniProt)"), target_text),
                (T("联合判读", "Combined interpretation"), [
                    T(
                        f"化合物–靶点组织重叠评分 {float(tissue.get('overlap_score') or 0):.0f}/100。该评分把预测 Kp 与靶点高表达组织对齐，仅用于提出组织暴露假设。",
                        f"Compound–target tissue overlap score {float(tissue.get('overlap_score') or 0):.0f}/100. The score aligns predicted Kp with tissues showing high target expression and is only a hypothesis-generating aid.",
                    ),
                    T("Kp 为经验分配模型，不是 PBPK；人源 HPA 数据不能直接外推到猪/牛/羊/鼠。", "Kp is an empirical partition model, not PBPK; human HPA data cannot be directly extrapolated to pig/cattle/sheep/rat."),
                ]),
            ],
            width=58, size=9.2, stop=0.155,
        )
        figures._caption(
            fig,
            T(
                "组织分布工具与数据：RDKit 描述符、ESOL、Clark logBB、DrugHX 物种 Kp 规则、Human Protein Atlas、UniProt。"
                "所有 Kp 与重叠评分均为筛查级假设。",
                "Tissue-distribution tools and data: RDKit descriptors, ESOL, Clark logBB, DrugHX species Kp rules, Human Protein Atlas, UniProt. "
                "All Kp values and overlap scores are screening-level hypotheses.",
            ),
            y=0.150,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # ------------------------------------------------------------ page 6
        fig = figures._a4_page(
            T("图 C6　在线证据、工具与置信度", "Fig. C6　Online evidence, tools and confidence"),
            T(f"{name} · PubChem、ChEMBL、UniProt、HPA 查询状态与方法边界", f"{name} · PubChem, ChEMBL, UniProt and HPA query status plus model boundaries"),
            footer=T("DrugHX · 综合药理报告", "DrugHX · Pharmacology report"),
        )
        online = data.get("online") or {}
        compound = online.get("compound") or {}
        rows = [[T("来源", "Source"), T("状态", "Status"), T("关键结果", "Key result")]]
        pubchem = compound.get("pubchem") or {}
        chembl = compound.get("chembl") or {}
        pubchem_cid = pubchem.get("cid")
        if pubchem.get("available") and pubchem_cid:
            pubchem_status = T("可用", "available")
            pubchem_result = f"CID {pubchem_cid} · {pubchem.get('formula') or '—'}"
        elif pubchem.get("available"):
            pubchem_status = T("可用（未命中）", "available (no match)")
            pubchem_result = T(
                "PubChem 未收录该结构", "structure not found in PubChem"
            )
        else:
            pubchem_status = T("不可用", "unavailable")
            pubchem_result = str(pubchem.get("error", ""))[:50]
        rows.append([
            "PubChem",
            pubchem_status,
            pubchem_result,
        ])
        rows.append([
            "ChEMBL",
            T("可用", "available") if chembl.get("available") else T("不可用", "unavailable"),
            f"{chembl.get('chembl_id') or '—'} · {chembl.get('pref_name') or ''}" if chembl.get("available") else str(chembl.get("error", ""))[:50],
        ])
        for item in ((online.get("targets") or {}).get("targets") or [])[:4]:
            hpa = item.get("hpa") or {}
            rows.append([
                str(item.get("gene", "target")),
                T("可用", "available") if item.get("available") else T("不可用", "unavailable"),
                (_hpa_label(hpa.get("rna_specificity")) or item.get("protein_name") or item.get("error") or "")[:46],
            ])
        figures._table(fig, 0.855, rows, [0.15, 0.14, 0.55], fontsize=8.2, row_height=0.026)
        conf_rows = [[T("模块", "Module"), T("置信度", "Confidence"), T("说明", "Note")]]
        for label, block in (
            (T("口服", "Oral"), oral),
            (T("瘤胃", "Rumen"), rumen if rumen.get("applicable") else None),
            (T("菌群", "Microbiota"), microbiota),
            (T("稳定性", "Stability"), stability),
            (T("剂型", "Formulation"), formulation),
            (T("核酸结合", "Nucleic acid"), nucleic),
            (T("风险", "Risk"), risk),
            (T("组织分布", "Tissue"), tissue),
        ):
            if not block:
                continue
            conf = block.get("confidence") or {}
            conf_rows.append([
                label,
                str(conf.get("level", "—")),
                textwrap.fill(str(conf.get("note", "")), 54)[:160],
            ])
        figures._table(fig, 0.535, conf_rows[:9], [0.16, 0.10, 0.58], fontsize=7.6, row_height=0.0215)
        figures._text_blocks(
            fig, 0.325,
            [
                (T("方法边界", "Method boundaries"), [
                    T(
                        "在线结果来自数据库记录，不是预测；查询失败时对应模块降级为本地规则，不影响报告生成。",
                        "Online results are database records, not predictions; if a query fails, the module falls back to local rules and the report still renders.",
                    ),
                    T(
                        "所有规则模型均需实验验证：口服/瘤胃需药代与瘤胃袋试验，菌群需测序，核酸结合需结合实验，组织分布需 PBPK 或成像数据。",
                        "All rule models require experimental validation: PK and rumen-bag studies for oral/rumen, sequencing for microbiota, binding assays for nucleic acids, and PBPK or imaging for tissue distribution.",
                    ),
                    T(
                        "综合置信度：" + str(confidence.get("level", "—")) + "（" + str(confidence.get("score", "—")) + "/100）。" + str(confidence.get("note", "")),
                        "Overall confidence: " + str(confidence.get("level", "—")) + " (" + str(confidence.get("score", "—")) + "/100). " + str(confidence.get("note", "")),
                    ),
                ]),
                (T("工具与数据来源", "Tools and data sources"), [
                    T(
                        "RDKit；ESOL（Delaney 2004）；Clark logBB；DrugHX 物种/制剂/风险规则；",
                        "RDKit; ESOL (Delaney 2004); Clark logBB; DrugHX species/formulation/risk rules;",
                    ),
                    T(
                        "PubChem PUG REST；ChEMBL REST；UniProt REST；Human Protein Atlas；Matplotlib。",
                        "PubChem PUG REST; ChEMBL REST; UniProt REST; Human Protein Atlas; Matplotlib.",
                    ),
                ]),
            ],
            width=58, size=8.8, stop=0.155,
        )
        figures._caption(
            fig,
            T(
                "在线证据仅反映数据库收录情况，可能随版本变化；所有查询均带超时与失败降级。"
                "本页为实时查询状态；靶点垂钓的 ChEMBL/BindingDB 命中可能来自缓存或独立查询，两者口径不同。"
                "工具：PubChem、ChEMBL、UniProt、Human Protein Atlas、RDKit、Matplotlib。",
                "Online evidence reflects database coverage and may change between releases; every query has a timeout and a graceful fallback. "
                "This page reports live query status; ChEMBL/BindingDB hits in target fishing may come from cache or independent queries, so the two statuses are not directly comparable. "
                "Tools: PubChem, ChEMBL, UniProt, Human Protein Atlas, RDKit, Matplotlib.",
            ),
            y=0.148,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        if data.get("species") != "human":
            _page_species(pdf, data)
            if (data.get("withdrawal") or {}).get("applicable"):
                _page_vet(pdf, data)

    return outpath


def _page_species(pdf, data: dict):
    """Species extrapolation and the Veterinary Developability Index."""
    name = data.get("name") or "compound"
    species = data.get("species_label") or data.get("species") or ""
    svdi = data.get("svdi") or {}
    allometry = data.get("allometry") or {}
    applicability = data.get("applicability") or {}
    components = svdi.get("components") or {}
    weights = svdi.get("weights") or {}

    fig = figures._a4_page(
        T("图 C8　种间外推与兽药可开发性指数", "Fig. C8　Cross-species scaling and veterinary developability"),
        T(
            f"{name} · {species} · 异速药代外推与种属加权可开发性",
            f"{name} · {species} · allometric PK and species-weighted developability",
        ),
        footer=T("DrugHX · 种属扩展页", "DrugHX · Species extension"),
    )
    _kpi_row(
        fig,
        [
            (
                T("兽药可开发性指数", "Veterinary developability"),
                f"{svdi.get('score', '—')} / 100",
                _risk_color(100.0 - float(svdi.get("score") or 0)),
            ),
            (
                T("异速外推半衰期", "Allometric t½"),
                f"{allometry.get('half_life_h', '—')} h",
                figures.BLUE,
            ),
            (
                T("清除率", "Clearance"),
                f"{allometry.get('cl_ml_min_kg', '—')} mL/min/kg",
                figures.TEAL,
            ),
            (
                T("适用域", "Applicability"),
                str(applicability.get("level") or "—"),
                figures.TEAL if applicability.get("in_domain") else figures.AMBER,
            ),
        ],
    )
    component_rows = [
        T(
            f"{key}：贡献值 {value:.2f}（权重 {weights.get(key, 0):.2f}）",
            f"{key}: value {value:.2f} (weight {weights.get(key, 0):.2f})",
        )
        for key, value in components.items()
    ]
    allometry_rows = [
        T(
            f"体重 {allometry.get('body_weight_kg', '—')} kg；CL {allometry.get('cl_ml_min_kg', '—')} mL/min/kg；"
            f"Vd {allometry.get('vd_l_kg', '—')} L/kg；t½ {allometry.get('half_life_h', '—')} h。",
            f"BW {allometry.get('body_weight_kg', '—')} kg; CL {allometry.get('cl_ml_min_kg', '—')} mL/min/kg; "
            f"Vd {allometry.get('vd_l_kg', '—')} L/kg; t½ {allometry.get('half_life_h', '—')} h.",
        ),
        T(
            str((allometry.get("confidence") or {}).get("note") or ""),
            str((allometry.get("confidence") or {}).get("note") or ""),
        ),
        T(
            f"适用域：最近邻 {applicability.get('nearest', '—')}，"
            f"Tanimoto {applicability.get('max_similarity', '—')}。",
            f"Applicability: nearest {applicability.get('nearest', '—')}, "
            f"Tanimoto {applicability.get('max_similarity', '—')}.",
        ),
    ]
    figures._text_blocks(
        fig,
        0.66,
        [
            (T("可开发性分量", "Developability components"), component_rows),
            (T("异速外推", "Allometric scaling"), allometry_rows),
        ],
        width=66, size=9.2, stop=0.155,
    )
    figures._caption(
        fig,
        T(
            "异速外推以人源经验清除率/分布容积为锚点（CL ∝ BW^-0.25，Vd ∝ BW^-0.05），"
            "未建模种属特异代谢酶与血浆蛋白结合；可开发性指数为种属加权启发式排序，"
            "不能替代药代、残留或临床前试验。",
            "Allometric scaling anchors on human empirical clearance/volume (CL ∝ BW^-0.25, Vd ∝ BW^-0.05) "
            "and does not model species-specific enzymes or plasma-protein binding. The developability index is a "
            "species-weighted heuristic for ranking, not a substitute for PK, residue or preclinical studies.",
        ),
        y=0.148,
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_vet(pdf, data: dict):
    """Extra page for food-producing animals: MRL, withdrawal, microbial fate."""
    name = data.get("name") or "compound"
    species = data.get("species_label") or data.get("species") or ""
    mrl = data.get("mrl") or {}
    withdrawal = data.get("withdrawal") or {}
    microbial = data.get("microbial_metabolism") or {}
    items = microbial.get("items") or []
    sites = sum(int(item.get("sites") or 0) for item in items)
    limits = mrl.get("limits_ug_kg") or {}

    fig = figures._a4_page(
        T("图 C7　兽药残留与菌群代谢扩展", "Fig. C7　Veterinary residue and microbial metabolism"),
        T(
            f"{name} · {species} · MRL、休药期与肠道菌群转化",
            f"{name} · {species} · MRL, withdrawal period and microbial transformation",
        ),
        footer=T("DrugHX · 兽药扩展页", "DrugHX · Veterinary extension"),
    )
    _kpi_row(
        fig,
        [
            (T("兽药类别", "Veterinary class"), str(mrl.get("class_label") or "—"), figures.TEAL),
            (
                T("示例 MRL（肌肉）", "Example MRL (muscle)"),
                f"{limits.get('muscle', '—')} µg/kg",
                figures.BLUE,
            ),
            (
                T("预估休药期", "Estimated withdrawal"),
                f"{withdrawal.get('withdrawal_days', '—')} d",
                _risk_color(min(100.0, float(withdrawal.get("withdrawal_days") or 0) * 4)),
            ),
            (T("菌群转化位点", "Microbial sites"), str(sites), figures.AMBER),
        ],
    )

    micro_bullets = []
    for item in items[:6]:
        products = item.get("products") or []
        product_text = "；".join(value[:42] for value in products[:2]) or "仅提示，未生成结构"
        micro_bullets.append(
            T(
                f"{item.get('label')}：{item.get('sites')} 个位点；产物 {product_text}",
                f"{item.get('label')}: {item.get('sites')} site(s); products {product_text}",
            )
        )
    if not micro_bullets:
        micro_bullets = [T("未命中常见菌群还原/水解规则。", "No common microbial reduction/hydrolysis motif found.")]

    residue_bullets = [
        T(
            f"类别判定：{mrl.get('class_label') or '未识别'}；MRL 为示例值，注册前必须以目标市场法规核对。",
            f"Class: {mrl.get('class_label') or 'not detected'}; MRLs are examples and must be checked against the target market regulation.",
        ),
        T(
            f"休药期估计：{withdrawal.get('withdrawal_days', '—')} 天（半衰期代理 {withdrawal.get('half_life_h', '—')} h，目标 MRL {withdrawal.get('mrl_ug_kg', '—')} µg/kg）。",
            f"Withdrawal estimate: {withdrawal.get('withdrawal_days', '—')} d (half-life proxy {withdrawal.get('half_life_h', '—')} h, target MRL {withdrawal.get('mrl_ug_kg', '—')} µg/kg).",
        ),
        T(
            str(withdrawal.get("strategy") or ""),
            str(withdrawal.get("strategy") or ""),
        ),
        T(
            "本页所有数值均为筛查级规则估计，不能替代残留消除试验、瘤胃袋试验或粪便菌群孵育。",
            "Every value on this page is a screening-level rule estimate and cannot replace residue depletion, rumen-bag or faecal-microbiota studies.",
        ),
    ]

    figures._text_blocks(
        fig,
        0.72,
        [
            (T("残留与休药期", "Residues and withdrawal"), residue_bullets),
            (T("菌群代谢假设", "Microbial metabolism hypotheses"), micro_bullets),
        ],
        width=64, size=9.4, stop=0.155,
    )
    figures._caption(
        fig,
        T(
            f"置信度：MRL {mrl.get('confidence', {}).get('level', '—')}；休药期 {withdrawal.get('confidence', {}).get('level', '—')}。"
            "MRL 量级参考欧盟 37/2010、美国 21 CFR 556 与中国 GB 31650 的常见值，不是法规原文。"
            "菌群产物由 SMARTS 规则枚举，仅作为分析假设。",
            "Confidence: MRL " + str(mrl.get("confidence", {}).get("level", "—")) + "; withdrawal " + str(withdrawal.get("confidence", {}).get("level", "—")) + ". "
            "MRL magnitudes follow common values in EU 37/2010, US 21 CFR 556 and China GB 31650, not the legal text. "
            "Microbial products are enumerated by SMARTS rules and are hypotheses only.",
        ),
        y=0.148,
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _cover_page(outpath: Path, data: dict):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.add_artist(
        Rectangle(
            (0.0, 0.0), 0.032, 1.0, transform=fig.transFigure,
            facecolor=figures.TEAL, edgecolor="none",
        )
    )
    fig.text(0.07, 0.88, "DrugHX", fontsize=34, fontweight="bold", color=figures.TEAL)
    fig.text(
        0.07, 0.82,
        T("化合物综合药理报告", "Comprehensive Compound Pharmacology Report"),
        fontsize=27, fontweight="bold", color=figures.INK,
    )
    fig.add_artist(
        Line2D(
            [0.07, 0.60], [0.795, 0.795], transform=fig.transFigure,
            color=figures.TEAL, linewidth=2.0,
        )
    )
    fig.text(0.07, 0.735, T("化合物：", "Compound: ") + str(data.get("name", "—")), fontsize=15, color=figures.INK)
    fig.text(0.07, 0.695, T("物种：", "Species: ") + str(data.get("species_label", data.get("species", "—"))), fontsize=15, color=figures.INK)
    fig.text(0.07, 0.655, T("生成时间：", "Generated: ") + str(data.get("generated", "—")), fontsize=12, color=figures.MUTED)
    fig.text(0.07, 0.625, T(f"软件作者：HX　·　版本 {APP_VERSION}", f"Author: HX　·　Version {APP_VERSION}"), fontsize=12, color=figures.MUTED)
    sections = [
        T("理化性质与 ADMET", "Physicochemical properties and ADMET"),
        T("口服性能、过瘤胃与肠道菌群", "Oral performance, rumen bypass and gut microbiota"),
        T("代谢稳定性与剂型研究", "Metabolic stability and formulation"),
        T("核酸结合与综合风险", "Nucleic-acid binding and integrated risk"),
        T("化合物与靶点的组织分布", "Compound and target tissue distribution"),
        T("在线证据、工具与置信度", "Online evidence, tools and confidence"),
    ]
    fig.text(0.07, 0.55, T("报告内容", "Contents"), fontsize=14, fontweight="bold", color=figures.TEAL)
    for index, line in enumerate(sections):
        fig.text(0.09, 0.515 - index * 0.033, f"{index + 1}. {line}", fontsize=11.5, color=figures.INK)
    fig.add_artist(
        Rectangle(
            (0.065, 0.16), 0.86, 0.11, transform=fig.transFigure,
            facecolor="#f0fdfa", edgecolor="#99f6e4", linewidth=0.8,
        )
    )
    fig.text(
        0.085, 0.235,
        textwrap.fill(
            T(
                "本报告不依赖分子对接：没有对接结果时仍可输出化合物性质、ADMET、稳定性、剂型、风险与组织分布。",
                "This report does not require docking: compound properties, ADMET, stability, formulation, risk and tissue distribution are available without any docking run.",
            ),
            62,
        ),
        fontsize=10.5, color=figures.INK, va="top",
    )
    fig.text(
        0.085, 0.185,
        T(
            "所有数值为筛查级估计，不能替代实测 ADME、GLP 毒理、PBPK 或临床试验。",
            "All values are screening-level estimates and cannot replace measured ADME, GLP toxicology, PBPK or clinical trials.",
        ),
        fontsize=9.5, color=figures.MUTED, va="top",
    )
    fig.text(
        0.07, 0.075,
        T(
            "工具与数据：RDKit、ESOL、Clark logBB、DrugHX 规则、PubChem、ChEMBL、UniProt、Human Protein Atlas、Matplotlib。",
            "Tools and data: RDKit, ESOL, Clark logBB, DrugHX rules, PubChem, ChEMBL, UniProt, Human Protein Atlas, Matplotlib.",
        ),
        fontsize=8.0, color=figures.MUTED,
    )
    fig.savefig(outpath, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def compound_report_pdf(data: dict, admet_path: Path, outpath: Path) -> Path:
    """Merge cover, ADMET page and pharmacology pages without any docking data."""
    from pypdf import PdfWriter

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    workdir = outpath.parent
    cover = workdir / "_pharm_cover.pdf"
    body = workdir / "_pharm_body.pdf"
    _cover_page(cover, data)
    pharm_pdf(data, body)
    writer = PdfWriter()
    for path in (cover, admet_path, body):
        if path and Path(path).is_file():
            writer.append(str(path))
    with open(outpath, "wb") as handle:
        writer.write(handle)
    writer.close()
    for path in (cover, body):
        try:
            path.unlink()
        except OSError:
            pass
    return outpath
