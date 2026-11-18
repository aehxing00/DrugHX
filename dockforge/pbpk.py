"""Minimal physiologically based pharmacokinetics and withdrawal intervals.

The existing pharmacology module extrapolates clearance allometrically and
estimates a withdrawal period heuristically.  This module adds a small
perfusion-limited PBPK model for seven species (human, pig, cattle, sheep, dog,
cat, rat) with plasma, liver, kidney, muscle, fat and a rest compartment.

It is deliberately a screening model: tissue partition coefficients follow a
simplified Poulin–Theil water/lipid description, absorption is a single depot,
and elimination is hepatic/renal clearance from plasma.  It is still enough to
generate concentration–time curves, compare tissue residues with example MRLs
and estimate a withdrawal interval with Monte-Carlo uncertainty.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

# Body weight (kg), tissue volume fractions and blood-flow fractions of cardiac
# output.  Volumes/flows are rounded literature values for screening.
SPECIES = {
    "human": {
        "bw": 70.0,
        "vols": {"plasma": 0.045, "liver": 0.026, "kidney": 0.004, "muscle": 0.40, "fat": 0.20},
        "flows": {"liver": 0.25, "kidney": 0.19, "muscle": 0.17, "fat": 0.05},
    },
    "pig": {
        "bw": 75.0,
        "vols": {"plasma": 0.050, "liver": 0.025, "kidney": 0.004, "muscle": 0.40, "fat": 0.18},
        "flows": {"liver": 0.25, "kidney": 0.15, "muscle": 0.20, "fat": 0.05},
    },
    "cattle": {
        "bw": 500.0,
        "vols": {"plasma": 0.050, "liver": 0.015, "kidney": 0.003, "muscle": 0.35, "fat": 0.15},
        "flows": {"liver": 0.25, "kidney": 0.15, "muscle": 0.20, "fat": 0.04},
    },
    "sheep": {
        "bw": 60.0,
        "vols": {"plasma": 0.050, "liver": 0.020, "kidney": 0.003, "muscle": 0.35, "fat": 0.12},
        "flows": {"liver": 0.25, "kidney": 0.16, "muscle": 0.20, "fat": 0.04},
    },
    "dog": {
        "bw": 15.0,
        "vols": {"plasma": 0.050, "liver": 0.025, "kidney": 0.005, "muscle": 0.45, "fat": 0.15},
        "flows": {"liver": 0.25, "kidney": 0.15, "muscle": 0.20, "fat": 0.05},
    },
    "cat": {
        "bw": 4.0,
        "vols": {"plasma": 0.050, "liver": 0.025, "kidney": 0.005, "muscle": 0.45, "fat": 0.22},
        "flows": {"liver": 0.25, "kidney": 0.15, "muscle": 0.20, "fat": 0.05},
    },
    "rat": {
        "bw": 0.3,
        "vols": {"plasma": 0.040, "liver": 0.040, "kidney": 0.007, "muscle": 0.40, "fat": 0.07},
        "flows": {"liver": 0.25, "kidney": 0.14, "muscle": 0.20, "fat": 0.07},
    },
}

# Tissue water / lipid fractions for the simplified Poulin–Theil Kp.
_TISSUE_COMP = {
    "liver": (0.73, 0.05),
    "kidney": (0.76, 0.04),
    "muscle": (0.76, 0.02),
    "fat": (0.15, 0.85),
    "rest": (0.75, 0.04),
}
_PLASMA_COMP = (0.93, 0.02)
_MONITORED = ("muscle", "liver", "kidney", "fat")


def _partition(logp: float) -> dict:
    kow = 10 ** max(-2.0, min(float(logp or 0.0), 6.0))
    plasma = _PLASMA_COMP[0] + _PLASMA_COMP[1] * kow
    result = {}
    for tissue, (water, lipid) in _TISSUE_COMP.items():
        value = (water + lipid * kow) / plasma
        result[tissue] = float(max(0.05, min(value, 50.0)))
    return result


def _volumes(species: str, bw: float) -> dict:
    info = SPECIES.get(species, SPECIES["human"])
    vols = {key: info["vols"][key] * bw for key in info["vols"]}
    rest = 1.0 - sum(info["vols"].values())
    vols["rest"] = max(rest, 0.10) * bw
    return vols


def _flows(species: str, bw: float) -> dict:
    info = SPECIES.get(species, SPECIES["human"])
    cardiac = 0.235 * (bw ** 0.75) * 60.0  # L/h
    flows = {key: info["flows"][key] * cardiac for key in info["flows"]}
    flows["rest"] = max(cardiac - sum(flows.values()), 0.05 * cardiac)
    return flows


def _rhs(t, y, params):
    gut, plasma, liver, kidney, muscle, fat, rest = y
    vols = params["vols"]
    flows = params["flows"]
    kp = params["kp"]
    c_plasma = max(plasma, 0.0) / vols["plasma"]
    c_liver = max(liver, 0.0) / vols["liver"]
    c_kidney = max(kidney, 0.0) / vols["kidney"]
    c_muscle = max(muscle, 0.0) / vols["muscle"]
    c_fat = max(fat, 0.0) / vols["fat"]
    c_rest = max(rest, 0.0) / vols["rest"]
    absorbed = params["ka"] * max(gut, 0.0) * params["f"]
    eliminated = params["cl_l_h"] * params["fu"] * c_plasma
    return [
        -params["ka"] * max(gut, 0.0),
        absorbed
        + sum(
            flows[key] * (tissue / kp[key])
            for key, tissue in (
                ("liver", c_liver),
                ("kidney", c_kidney),
                ("muscle", c_muscle),
                ("fat", c_fat),
                ("rest", c_rest),
            )
        )
        - sum(flows.values()) * c_plasma
        - eliminated,
        flows["liver"] * (c_plasma - c_liver / kp["liver"]),
        flows["kidney"] * (c_plasma - c_kidney / kp["kidney"]),
        flows["muscle"] * (c_plasma - c_muscle / kp["muscle"]),
        flows["fat"] * (c_plasma - c_fat / kp["fat"]),
        flows["rest"] * (c_plasma - c_rest / kp["rest"]),
    ]


def simulate(
    dose_mg_kg: float,
    mw: float,
    logp: float,
    fu: float = 0.2,
    cl_ml_min_kg: float = 5.0,
    species: str = "human",
    route: str = "oral",
    f_percent: float = 50.0,
    ka: float = 1.2,
    duration_days: float = 14.0,
    mrl_ug_kg: dict | None = None,
    points: int = 400,
) -> dict:
    """Simulate plasma/tissue concentrations and the withdrawal interval."""
    species = species if species in SPECIES else "human"
    info = SPECIES[species]
    bw = info["bw"]
    dose = max(float(dose_mg_kg), 1e-4) * bw
    f_fraction = min(max(float(f_percent) / 100.0, 0.01), 1.0)
    cl_l_h = max(float(cl_ml_min_kg), 0.01) * bw * 60.0 / 1000.0
    params = {
        "vols": _volumes(species, bw),
        "flows": _flows(species, bw),
        "kp": _partition(logp),
        "fu": min(max(float(fu), 0.001), 1.0),
        "cl_l_h": cl_l_h,
        "ka": max(float(ka), 0.02),
        "f": f_fraction if route == "oral" else min(max(f_fraction, 0.05), 1.0),
    }
    if route == "oral":
        y0 = [dose, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif route == "im":
        y0 = [dose, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        params["ka"] = min(params["ka"], 0.35)
        params["f"] = min(max(params["f"], 0.30), 1.0)
    else:  # iv bolus
        y0 = [0.0, dose, 0.0, 0.0, 0.0, 0.0, 0.0]
    duration_h = max(float(duration_days), 1.0) * 24.0
    t_eval = np.linspace(0.0, duration_h, max(int(points), 80))
    solution = solve_ivp(
        _rhs, (0.0, duration_h), y0, t_eval=t_eval, args=(params,),
        method="LSODA", rtol=1e-6, atol=1e-9,
    )
    if not solution.success or solution.y.size == 0:
        raise RuntimeError(f"PBPK 求解失败：{solution.message}")
    vols = params["vols"]
    curves = {
        "plasma": solution.y[1] / vols["plasma"],
        "liver": solution.y[2] / vols["liver"],
        "kidney": solution.y[3] / vols["kidney"],
        "muscle": solution.y[4] / vols["muscle"],
        "fat": solution.y[5] / vols["fat"],
    }
    time_h = solution.t
    plasma = np.maximum(curves["plasma"], 0.0)
    cmax = float(np.max(plasma))
    tmax = float(time_h[int(np.argmax(plasma))])
    auc = float(np.trapezoid(plasma, time_h))
    half_life = _terminal_half_life(time_h, plasma)

    limits = mrl_ug_kg or _default_mrl(species)
    mrl_rows = []
    withdrawal = 0.0
    for tissue in _MONITORED:
        limit_ug = limits.get(tissue)
        if limit_ug is None:
            continue
        limit_mg = float(limit_ug) / 1000.0
        curve = np.maximum(curves[tissue], 0.0)
        above = np.where(curve > limit_mg)[0]
        last_h = float(time_h[above[-1]]) if above.size else 0.0
        days = last_h / 24.0
        withdrawal = max(withdrawal, days)
        mrl_rows.append(
            {
                "tissue": tissue,
                "mrl_ug_kg": float(limit_ug),
                "cmax_mg_kg": round(float(np.max(curve)), 4),
                "days_above_mrl": round(days, 2),
                "exceeded": bool(above.size),
            }
        )
    output = {
        "species": species,
        "body_weight_kg": bw,
        "dose_mg_kg": round(dose / bw, 3),
        "dose_mg": round(dose, 3),
        "route": route,
        "mw": mw,
        "logp": logp,
        "fu": params["fu"],
        "f_percent": round(f_fraction * 100, 1),
        "cl_ml_min_kg": round(float(cl_ml_min_kg), 2),
        "cl_l_h": round(cl_l_h, 3),
        "ka": round(params["ka"], 3),
        "kp": {key: round(value, 2) for key, value in params["kp"].items()},
        "time_h": [round(float(value), 2) for value in time_h],
        "curves": {
            key: [round(float(max(value, 0.0)), 5) for value in curve]
            for key, curve in curves.items()
        },
        "cmax_mg_l": round(cmax, 4),
        "tmax_h": round(tmax, 2),
        "auc_mg_h_l": round(auc, 3),
        "half_life_h": None if half_life is None else round(half_life, 2),
        "mrl_rows": mrl_rows,
        "withdrawal_days": int(math.ceil(withdrawal)) if withdrawal else 0,
        "censored": bool(withdrawal >= duration_h / 24.0 - 0.5),
        "duration_days": round(duration_h / 24.0, 1),
        "mrl_source": "示例 MRL（需按目标市场法规核对）",
        "note": (
            "简化灌注限制 PBPK：血浆、肝、肾、肌、脂肪与其余组织；Kp 用 Poulin–Theil 水/脂近似，"
            "清除从血浆一阶消除。用于立项前残留与休药期筛查，不能替代注册残留试验。"
        ),
    }
    if output["censored"]:
        output["note"] += " 估算休药期达到模拟时长上限，请延长模拟天数后再判断。"
    return output


def _terminal_half_life(time_h: np.ndarray, plasma: np.ndarray) -> float | None:
    mask = plasma > max(float(np.max(plasma)) * 0.02, 1e-9)
    if mask.sum() < 4:
        return None
    times = time_h[mask]
    values = np.log(plasma[mask])
    start = int(len(times) * 0.55)
    if len(times) - start < 3:
        return None
    slope = float(np.polyfit(times[start:], values[start:], 1)[0])
    if slope >= -1e-6:
        return None
    return 0.693 / abs(slope)


def _default_mrl(species: str) -> dict:
    info = SPECIES.get(species, SPECIES["human"])
    # Only food-producing species get default example MRLs.
    if species in ("pig", "cattle", "sheep"):
        return {"muscle": 100.0, "liver": 300.0, "kidney": 500.0, "fat": 100.0}
    return {}


def withdrawal_monte_carlo(
    dose_mg_kg: float,
    mw: float,
    logp: float,
    species: str = "pig",
    route: str = "oral",
    f_percent: float = 50.0,
    fu: float = 0.2,
    cl_ml_min_kg: float = 5.0,
    ka: float = 1.2,
    duration_days: float = 30.0,
    mrl_ug_kg: dict | None = None,
    samples: int = 80,
    seed: int = 42,
) -> dict:
    """Monte-Carlo withdrawal interval from parameter uncertainty."""
    rng = np.random.default_rng(seed)
    intervals = []
    for _ in range(max(int(samples), 10)):
        scale_cl = float(np.clip(rng.lognormal(0.0, 0.35), 0.4, 2.5))
        scale_f = float(np.clip(rng.normal(1.0, 0.15), 0.6, 1.4))
        scale_ka = float(np.clip(rng.lognormal(0.0, 0.30), 0.5, 2.0))
        result = simulate(
            dose_mg_kg=dose_mg_kg,
            mw=mw,
            logp=logp,
            fu=fu,
            cl_ml_min_kg=cl_ml_min_kg * scale_cl,
            species=species,
            route=route,
            f_percent=min(f_percent * scale_f, 100.0),
            ka=ka * scale_ka,
            duration_days=duration_days,
            mrl_ug_kg=mrl_ug_kg,
            points=160,
        )
        intervals.append(float(result["withdrawal_days"]))
    values = np.asarray(intervals, dtype=float)
    return {
        "samples": int(values.size),
        "median_days": float(np.median(values)),
        "p95_days": float(np.percentile(values, 95)),
        "max_days": float(np.max(values)),
        "note": "参数扰动：CL 对数正态 0.35、F 15%、ka 0.30；取 95 分位作为保守休药期建议。",
    }


def depot_parameters(smiles: str, mw: float, logp: float) -> dict:
    """Convenience defaults for the API/UI."""
    return {
        "mw": round(float(mw), 2),
        "logp": round(float(logp), 2),
        "ka_oral": round(max(0.2, min(3.0, 2.0 - 0.15 * float(logp))), 3),
        "fu_default": 0.2,
    }
