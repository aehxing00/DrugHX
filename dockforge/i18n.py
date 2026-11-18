"""Report language context and English rendering layer.

The whole application is written with Chinese labels.  Rather than maintain a
second copy of every plotting function, the report generator switches the
language context and the cache below translates every text string that
matplotlib draws.  Chinese remains the default; English is used only while an
English report is being generated.

The translation cache is a JSON dictionary ``{Chinese: English}``.  Exact hits
are preferred; for composed strings (f-strings, numbers appended to labels) a
longest-first phrase replacement is used as a fallback.
"""
from __future__ import annotations

import json
import re
from contextvars import ContextVar
from pathlib import Path

from matplotlib.text import Text

_LANG: ContextVar[str] = ContextVar("drughx_report_lang", default="zh")
_CACHE: dict[str, str] = {}
_PATTERN: re.Pattern | None = None
_MEMORY: dict[tuple[str, str], str] = {}
_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")
_HAS_CJK_ALL = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")
# Chemistry radicals that only occur in generated molecule names; they are
# safe to translate even as single characters because Chinese output never
# passes through ``tr``.
_SINGLE_OK = set("苯萘蒽菲芘氟氯溴碘酯肟硫氧氮")

# Fallback for composed strings whose numbers prevent an exact dictionary
# hit: after phrase replacement, any residual Chinese is translated word by
# word and Chinese punctuation is converted.  This keeps figure labels and
# captions fully English even when a sentence was assembled from fragments.
_RESIDUAL = {
    "代表性": "representative ",
    "代表性化合物": "representative compound",
    "环境": "environment",
    "芳香环": "aromatic ring",
    "杂环": "heterocycle",
    "环丙基": "cyclopropyl",
    "环系": "ring system",
    "环数": "rings",
    "环": "ring",
    "立体中心": "stereocentre",
    "砌块": "building block",
    "数据": "data",
    "数值": "value",
    "阈值": "threshold",
    "分布": "distribution",
    "分析": "analysis",
    "分钟": "min",
    "小时": "h",
    "百分比": "percentage",
    "命中": "hit",
    "摩尔活性值": "molar activity",
    "共晶": "co-crystal",
    "显示": "shown",
    "供受体": "donor–acceptor",
    "绿色虚线": "green dashed line",
    "虚线": "dashed line",
    "距离": "distance",
    "编号": "index",
    "标注": "annotated",
    "能量分解": "energy decomposition",
    "几何判据": "geometric criteria",
    "不等同于": "is not equivalent to",
    "显著": "significant",
    "条目": "entry",
    "点大小": "point size",
    "横轴": "x-axis",
    "纵轴": "y-axis",
    "越亮": "the brighter",
    "参考线": "reference line",
    "正电": "positively charged",
    "负电": "negatively charged",
    "二维": "2D",
    "构成": "composition",
    "原子": "atom",
    "最近": "nearest",
    "覆盖": "coverage",
    "感知": "perceived",
    "可电离": "ionisable",
    "电离": "ionisation",
    "邻域": "neighbourhood",
    "判定": "determine",
    "估算": "estimated",
    "埋藏比例": "buried fraction",
    "接触": "contact",
    "能量贡献": "energy contribution",
    "能量项总计": "total energy terms",
    "贡献": "contribution",
    "模板电荷": "template charge",
    "电荷": "charge",
    "截断": "cutoff",
    "非绝对": "not absolute",
    "分解": "decomposition",
    "近似": "approximate",
    "结果": "result",
    "方法": "Methods",
    "结合": "binding",
    "部分": "partial",
    "一致": "consistent",
    "含": "contains ",
    "项": " item",
    "级": "-level",
    "型": "-type",
    "部": " part",
    "解": "decomposition",
    "排序": "ranking",
    "映射": "mapping",
    "对应关系": "correspondence",
    "供体": "donor",
    "采用": "adopted",
    "采": "",
    "位": " ",
    "分辨率": "resolution",
    "中位数": "median",
    "共晶配体": "co-crystal ligand",
    "药效团": "pharmacophore",
    "类药性": "drug-likeness",
    "可旋转键": "rotatable bonds",
    "分子量": "molecular weight",
    "结合自由能": "binding free energy",
    "自由能": "free energy",
    "氢键供体": "hydrogen-bond donor",
    "氢键受体": "hydrogen-bond acceptor",
    "疏水接触": "hydrophobic contact",
    "范德华": "van der Waals",
    "去溶剂化": "desolvation",
    "静电": "electrostatic",
    "作用力": "interaction",
    "覆盖率": "coverage",
    "互补性": "complementarity",
    "互补": "complementary",
    "特征": "feature",
    "中心": "centre",
    "残基": "residue",
    "疏水": "hydrophobic",
    "氢键": "hydrogen bond",
    "靶点": "target",
    "配体": "ligand",
    "受体": "receptor",
    "蛋白": "protein",
    "口袋": "pocket",
    "构象": "conformation",
    "打分": "score",
    "对接": "docking",
    "设计分子": "designed molecule",
    "设计": "design",
    "候选": "candidate",
    "母体": "parent",
    "改造": "modification",
    "取代基": "substituent",
    "枚举": "enumeration",
    "最近邻": "nearest neighbour",
    "数据库": "database",
    "结构": "structure",
    "活性": "activity",
    "相似度": "similarity",
    "指纹": "fingerprint",
    "通路": "pathway",
    "基因": "gene",
    "富集": "enrichment",
    "化合物": "compound",
    "疾病": "disease",
    "剂量": "dose",
    "毒性": "toxicity",
    "代谢": "metabolism",
    "清除率": "clearance",
    "半衰期": "half-life",
    "稳定性": "stability",
    "选择性": "selectivity",
    "抑制剂": "inhibitor",
    "非甾体抗炎药": "NSAID",
    "适应症": "indication",
    "机制": "mechanism",
    "代表性": "representative ",
    "以上": "above",
    "以下": "below",
    "首位": "top rank",
    "本次": "this run ",
    "当前": "current ",
    "共": "in total ",
    "约": "approximately ",
    "至": " to ",
    "与": " and ",
    "和": " and ",
    "或": " or ",
    "及": " and ",
    "的": " ",
    "以": " with ",
    "在": " in ",
    "为": " is ",
    "按": " by ",
    "由": " by ",
    "从": " from ",
    "对": " for ",
    "将": " ",
    "其": " its ",
    "该": "this ",
    "本": "this ",
    "此": "this ",
    "并": " and ",
    "且": " and ",
    "但": " but ",
    "若": "if ",
    "可": " can ",
    "用": " use ",
    "需": " need ",
    "使": " make ",
    "已": "already ",
    "未": "not ",
    "不": "not ",
    "无": "no ",
    "有": "has ",
    "是": "is ",
    "个": " ",
    "条": " ",
    "种": " ",
    "次": " ",
    "分": " ",
    "秒": " s ",
    "中": " ",
    "内": " within ",
    "外": " outside ",
    "上": " ",
    "下": " ",
    "前": " ",
    "后": " ",
    "第": "",
    "页": "Page ",
    "图": "Figure ",
    "表": "Table ",
    "列": "column ",
    "行": "row ",
    "值": "value ",
    "数": "number ",
    "优": "excellent",
    "良": "good",
    "差": "poor",
    "低": "low",
    "高": "high",
}

_PUNCT = str.maketrans(
    {
        "、": ", ",
        "，": ", ",
        "。": ". ",
        "；": "; ",
        "：": ": ",
        "（": " (",
        "）": ") ",
        "！": "! ",
        "？": "? ",
        "　": " ",
        "「": ' "',
        "」": '" ',
        "『": ' "',
        "』": '" ',
        "【": " [",
        "】": "] ",
        "《": " <",
        "》": "> ",
        "…": "...",
        "＋": "+",
        "－": "-",
        "×": " × ",
    }
)
_RESIDUAL_KEYS = sorted(_RESIDUAL, key=len, reverse=True)
_RESIDUAL_RULES = [
    (
        key,
        _RESIDUAL[key],
        re.compile(r"(?<=[A-Za-z0-9])(?=" + re.escape(key) + r")"),
        (
            re.compile(
                r"(?<=" + re.escape(_RESIDUAL[key]) + r")(?=[A-Za-z0-9])"
            )
            if _RESIDUAL[key]
            else None
        ),
    )
    for key in _RESIDUAL_KEYS
]


def _cleanup(text: str) -> str:
    for key, value, before, after in _RESIDUAL_RULES:
        if key in text:
            # Keep spaces between Latin words and the substituted term.
            text = before.sub(" ", text)
            text = text.replace(key, value)
            if after is not None:
                text = after.sub(" ", text)
    text = text.translate(_PUNCT)
    text = re.sub(r"\s+([,.;:)'\"])", r"\1", text)
    text = re.sub(r"([’”])\s+([.,;:])", r"\1\2", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r",(?!\d{3}(?!\d))", ", ", text)
    text = re.sub(r"(?<=[a-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[a-z])", " ", text)
    text = re.sub(r"([‘“])\s+", r"\1", text)
    text = re.sub(r"\s+([’”])", r"\1", text)
    text = re.sub(r"(?<=[A-Za-z]):(?=[A-Za-z0-9])", ": ", text)
    text = re.sub(r"(?<=[a-z])\.(?=[A-Z][a-z])", ". ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _tidy(text: str) -> str:
    text = re.sub(r"`\s*s\b", "'s", text)
    # ``label:value`` and ``sentence.Next`` are common after phrase-wise
    # translation; restore the missing spaces without touching decimals/URLs.
    text = re.sub(r"(?<=[A-Za-z]):(?=[A-Za-z0-9])", ": ", text)
    text = re.sub(r"(?<=[a-z])\.(?=[A-Z][a-z])", ". ", text)
    text = re.sub(r"\s+([,.;:)'\"])", r"\1", text)
    text = re.sub(r"([’”])\s+([.,;:])", r"\1\2", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r",(?!\d{3}(?!\d))", ", ", text)
    text = re.sub(r"(?<=[a-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[a-z])", " ", text)
    text = re.sub(r"([‘“])\s+", r"\1", text)
    text = re.sub(r"\s+([’”])", r"\1", text)
    text = re.sub(r" {2,}", " ", text)
    return text


def _load() -> None:
    global _CACHE, _PATTERN
    path = Path(__file__).with_name("translations_en.json")
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    _CACHE = {
        str(key): str(value)
        for key, value in data.items()
        if key and value and str(value).strip()
    }
    keys = sorted(
        (
            key
            for key in _CACHE
            if len(key) >= 2 or key in _SINGLE_OK
        ),
        key=len,
        reverse=True,
    )
    if keys:
        _PATTERN = re.compile("|".join(re.escape(key) for key in keys))


def set_lang(lang: str) -> None:
    _LANG.set("en" if str(lang).lower().startswith("en") else "zh")


def get_lang() -> str:
    return _LANG.get()


def tr(text: str) -> str:
    if not isinstance(text, str) or not text or _LANG.get() != "en":
        return text
    if not _HAS_CJK_ALL.search(text):
        return text
    key = ("en", text)
    cached = _MEMORY.get(key)
    if cached is not None:
        return cached
    exact = _CACHE.get(text)
    if exact:
        translated = exact
    elif _PATTERN is not None:
        translated = _PATTERN.sub(
            lambda match: _CACHE.get(match.group(0), match.group(0)), text
        )
    else:
        translated = text
    if _HAS_CJK_ALL.search(translated):
        translated = _cleanup(translated)
    else:
        translated = _tidy(translated)
    _MEMORY[key] = translated
    return translated


_ORIGINAL_SET_TEXT = Text.set_text


def _set_text(self, s):  # noqa: ANN001
    if isinstance(s, str):
        s = tr(s)
    return _ORIGINAL_SET_TEXT(self, s)


# Patch once at import time; ``tr`` returns the original text in Chinese mode.
if Text.set_text is not _set_text:
    Text.set_text = _set_text

_load()
