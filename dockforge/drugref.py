"""FDA 已上市药物参考集与候选分子的结构、靶点与 ADMET 参考分析。

数据来源：公开 FDA 处方标签（DailyMed）、PubChem 与 ChEMBL 汇总，仅作参考。

本模块提供三件事：

1. ``FDA_DRUGS``：45 个结构简单、SMILES 确定、靶点明确的 FDA 已上市药物，
   字段覆盖药物类别、HGNC 靶点、适应证、批准年份、给药途径、典型半衰期与
   CYP 代谢酶。所有 SMILES 均经 RDKit 解析校验；半衰期与批准年份只在有把握
   时填写，否则为 ``None``。
2. ``compare``：把候选分子与参考集逐一比较（Morgan 指纹 Tanimoto、Bemis–Murcko
   骨架、最大公共子结构 MCS、靶点 Jaccard、ADMET 与理化属性相似度），输出可
   排序的参考列表。不伪造 pIC50，药效关联只依据靶点重叠与机制邻近描述。
3. ``drugref_pdf``：A4 竖版 CNS 风格多面板图（结构相似度条形图、结构–靶点散点、
   ADMET 雷达、属性表、参考药二维结构），可直接并入最终报告。

所有 ADMET 数值均来自 :mod:`dockforge.admet` 的透明经验模型，是计算值，
不能替代临床或实验数据；综合分是启发式排序指标，不是概率或置信度。
"""
from __future__ import annotations

import math
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import rdFMCS, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import admet, i18n

# ---------------------------------------------------------------------------
# reference set
# ---------------------------------------------------------------------------
# 说明：SMILES 已逐个用 RDKit 解析并用分子式/分子量核对；half_life_h 为公开
# 标签中的典型值（小时），无把握者置 None；approval_year 为 FDA 首次批准年份，
# 历史久远或口径不一者置 None。
FDA_DRUGS: list[dict] = [
    {
        "name": "Aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "drug_class": "非甾体抗炎药（COX-1/2 抑制剂）",
        "primary_targets": ["PTGS1", "PTGS2"],
        "indication": "解热镇痛、抗血小板（低剂量）",
        "approval_year": None,
        "route": "口服",
        "half_life_h": 0.3,
        "metabolized_by": [],
        "notes": "血浆酯酶快速水解为水杨酸；对血小板 COX-1 的抑制不可逆。",
    },
    {
        "name": "Ibuprofen",
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "drug_class": "非甾体抗炎药（COX 抑制剂）",
        "primary_targets": ["PTGS1", "PTGS2"],
        "indication": "镇痛、抗炎、退热",
        "approval_year": 1974,
        "route": "口服",
        "half_life_h": 2.0,
        "metabolized_by": ["CYP2C9"],
        "notes": "手性药物，S-构型对 COX 抑制活性更强。",
    },
    {
        "name": "Acetaminophen",
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "drug_class": "解热镇痛药",
        "primary_targets": ["PTGS1", "PTGS2"],
        "indication": "解热镇痛（中枢为主）",
        "approval_year": None,
        "route": "口服",
        "half_life_h": 2.5,
        "metabolized_by": ["CYP2E1", "CYP1A2", "CYP3A4"],
        "notes": "过量时 CYP2E1 生成 NAPQI，耗竭谷胱甘肽可致肝损伤。",
    },
    {
        "name": "Celecoxib",
        "smiles": "Cc1ccc(cc1)c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F",
        "drug_class": "选择性 COX-2 抑制剂",
        "primary_targets": ["PTGS2"],
        "indication": "骨关节炎、类风湿关节炎镇痛",
        "approval_year": 1998,
        "route": "口服",
        "half_life_h": 11.0,
        "metabolized_by": ["CYP2C9"],
        "notes": "含磺胺结构；需评估心血管与胃肠道风险。",
    },
    {
        "name": "Warfarin",
        "smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
        "drug_class": "维生素 K 拮抗剂",
        "primary_targets": ["VKORC1"],
        "indication": "抗凝（房颤、静脉血栓栓塞）",
        "approval_year": 1954,
        "route": "口服",
        "half_life_h": 40.0,
        "metabolized_by": ["CYP2C9", "CYP3A4", "CYP1A2"],
        "notes": "治疗窗窄需 INR 监测；S-华法林主要经 CYP2C9 代谢。",
    },
    {
        "name": "Clopidogrel",
        "smiles": "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
        "drug_class": "P2Y12 受体拮抗剂（前药）",
        "primary_targets": ["P2RY12"],
        "indication": "抗血小板（急性冠脉综合征、支架术后）",
        "approval_year": 1997,
        "route": "口服",
        "half_life_h": 6.0,
        "metabolized_by": ["CYP2C19", "CYP3A4"],
        "notes": "需经 CYP2C19 等活化；慢代谢者活性代谢物暴露降低。",
    },
    {
        "name": "Rivaroxaban",
        "smiles": "O=C1COCCN1c1ccc(N2C[C@H](CNC(=O)c3ccc(Cl)s3)OC2=O)cc1",
        "drug_class": "Xa 因子抑制剂",
        "primary_targets": ["F10"],
        "indication": "抗凝（房颤、静脉血栓栓塞）",
        "approval_year": 2011,
        "route": "口服",
        "half_life_h": 9.0,
        "metabolized_by": ["CYP3A4", "CYP2J2"],
        "notes": "口服直接 Xa 因子抑制剂，通常无需常规凝血监测。",
    },
    {
        "name": "Metformin",
        "smiles": "CN(C)C(=N)NC(=N)N",
        "drug_class": "双胍类降糖药",
        "primary_targets": [],
        "indication": "2 型糖尿病一线用药",
        "approval_year": 1994,
        "route": "口服",
        "half_life_h": 6.0,
        "metabolized_by": [],
        "notes": "无公认单一蛋白靶点，主要经线粒体复合物 I/AMPK 途径起效；原型经肾排泄。",
    },
    {
        "name": "Atorvastatin",
        "smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
        "drug_class": "他汀类（HMG-CoA 还原酶抑制剂）",
        "primary_targets": ["HMGCR"],
        "indication": "高脂血症、动脉粥样硬化一级/二级预防",
        "approval_year": 1996,
        "route": "口服",
        "half_life_h": 14.0,
        "metabolized_by": ["CYP3A4"],
        "notes": "亲脂性他汀，活性代谢物同样抑制 HMGCR。",
    },
    {
        "name": "Simvastatin",
        "smiles": "CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12",
        "drug_class": "他汀类（HMG-CoA 还原酶抑制剂）",
        "primary_targets": ["HMGCR"],
        "indication": "高脂血症",
        "approval_year": 1991,
        "route": "口服",
        "half_life_h": 2.0,
        "metabolized_by": ["CYP3A4"],
        "notes": "内酯前药，经 CYP3A4 水解为活性羟酸；与葡萄柚汁相互作用明显。",
    },
    {
        "name": "Rosuvastatin",
        "smiles": "CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1/C=C/[C@@H](O)C[C@@H](O)CC(=O)O",
        "drug_class": "他汀类（HMG-CoA 还原酶抑制剂）",
        "primary_targets": ["HMGCR"],
        "indication": "高脂血症",
        "approval_year": 2003,
        "route": "口服",
        "half_life_h": 19.0,
        "metabolized_by": ["CYP2C9"],
        "notes": "亲水性他汀，多以原型经胆汁与肾排泄。",
    },
    {
        "name": "Amlodipine",
        "smiles": "CCOC(=O)C1=C(COCCN)NC(C)=C(C1c1ccccc1Cl)C(=O)OC",
        "drug_class": "二氢吡啶类钙通道阻滞剂",
        "primary_targets": ["CACNA1C"],
        "indication": "高血压、稳定型心绞痛",
        "approval_year": 1990,
        "route": "口服",
        "half_life_h": 40.0,
        "metabolized_by": ["CYP3A4"],
        "notes": "半衰期长，可每日给药一次。",
    },
    {
        "name": "Losartan",
        "smiles": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(cc1)c1ccccc1c1[nH]nnn1",
        "drug_class": "血管紧张素 II AT1 受体拮抗剂",
        "primary_targets": ["AGTR1"],
        "indication": "高血压、糖尿病肾病",
        "approval_year": 1995,
        "route": "口服",
        "half_life_h": 2.0,
        "metabolized_by": ["CYP2C9", "CYP3A4"],
        "notes": "活性代谢物 EXP3174 亲和力更强、半衰期更长。",
    },
    {
        "name": "Valsartan",
        "smiles": "CCCCC(=O)N(Cc1ccc(cc1)c1ccccc1c1[nH]nnn1)[C@@H](C(C)C)C(=O)O",
        "drug_class": "血管紧张素 II AT1 受体拮抗剂",
        "primary_targets": ["AGTR1"],
        "indication": "高血压、心力衰竭",
        "approval_year": 1996,
        "route": "口服",
        "half_life_h": 6.0,
        "metabolized_by": ["CYP2C9"],
        "notes": "非前药，CYP 代谢占比很低，多以原型经胆汁排泄。",
    },
    {
        "name": "Metoprolol",
        "smiles": "COCCc1ccc(OCC(O)CNC(C)C)cc1",
        "drug_class": "β1 选择性肾上腺素受体阻滞剂",
        "primary_targets": ["ADRB1"],
        "indication": "高血压、心绞痛、慢性心力衰竭",
        "approval_year": 1978,
        "route": "口服",
        "half_life_h": 4.0,
        "metabolized_by": ["CYP2D6"],
        "notes": "CYP2D6 多态性显著影响血药浓度。",
    },
    {
        "name": "Furosemide",
        "smiles": "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl",
        "drug_class": "袢利尿剂",
        "primary_targets": ["SLC12A1"],
        "indication": "水肿、心力衰竭、高血压",
        "approval_year": 1966,
        "route": "口服",
        "half_life_h": 2.0,
        "metabolized_by": [],
        "notes": "抑制髓袢 Na+-K+-2Cl- 共转运体 NKCC2；主要经葡萄糖醛酸化排泄。",
    },
    {
        "name": "Hydrochlorothiazide",
        "smiles": "NS(=O)(=O)c1cc2c(cc1Cl)NCNS2(=O)=O",
        "drug_class": "噻嗪类利尿剂",
        "primary_targets": ["SLC12A3"],
        "indication": "高血压、水肿",
        "approval_year": 1959,
        "route": "口服",
        "half_life_h": 10.0,
        "metabolized_by": [],
        "notes": "抑制远曲小管 Na+-Cl- 共转运体 NCC；原型经肾排泄。",
    },
    {
        "name": "Omeprazole",
        "smiles": "CC1=CN=C(C(=C1OC)CS(=O)c1nc2cc(OC)ccc2[nH]1)C",
        "drug_class": "质子泵抑制剂（前药）",
        "primary_targets": ["ATP4A", "ATP4B"],
        "indication": "胃食管反流、消化性溃疡",
        "approval_year": 1989,
        "route": "口服",
        "half_life_h": 1.0,
        "metabolized_by": ["CYP2C19", "CYP3A4"],
        "notes": "酸性环境中活化后与 H+/K+-ATP 酶共价结合。",
    },
    {
        "name": "Pantoprazole",
        "smiles": "COc1ccnc(CS(=O)c2nc3cc(OC(F)F)ccc3[nH]2)c1OC",
        "drug_class": "质子泵抑制剂（前药）",
        "primary_targets": ["ATP4A", "ATP4B"],
        "indication": "胃食管反流、消化性溃疡",
        "approval_year": 2000,
        "route": "口服",
        "half_life_h": 1.0,
        "metabolized_by": ["CYP2C19", "CYP3A4"],
        "notes": "CYP 介导的药物相互作用相对较少。",
    },
    {
        "name": "Cetirizine",
        "smiles": "OC(=O)COCCN1CCN(CC1)C(c1ccccc1)c1ccc(Cl)cc1",
        "drug_class": "第二代 H1 受体拮抗剂",
        "primary_targets": ["HRH1"],
        "indication": "过敏性鼻炎、慢性荨麻疹",
        "approval_year": 1995,
        "route": "口服",
        "half_life_h": 8.0,
        "metabolized_by": [],
        "notes": "羟嗪的活性羧酸代谢物，多以原型经肾排泄。",
    },
    {
        "name": "Loratadine",
        "smiles": "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc23)CC1",
        "drug_class": "第二代 H1 受体拮抗剂",
        "primary_targets": ["HRH1"],
        "indication": "过敏性鼻炎、慢性荨麻疹",
        "approval_year": 1993,
        "route": "口服",
        "half_life_h": 12.0,
        "metabolized_by": ["CYP3A4", "CYP2D6"],
        "notes": "经 CYP 生成活性代谢物地氯雷他定。",
    },
    {
        "name": "Amoxicillin",
        "smiles": "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O",
        "drug_class": "β-内酰胺类抗生素（氨基青霉素）",
        "primary_targets": [],
        "indication": "呼吸道、耳鼻喉及幽门螺杆菌联合根除治疗",
        "approval_year": 1974,
        "route": "口服",
        "half_life_h": 1.3,
        "metabolized_by": [],
        "notes": "靶点为细菌青霉素结合蛋白（PBP），无人类 HGNC 对应基因；原型经肾排泄。",
    },
    {
        "name": "Ciprofloxacin",
        "smiles": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
        "drug_class": "氟喹诺酮类抗生素",
        "primary_targets": [],
        "indication": "革兰阴性菌感染、尿路感染",
        "approval_year": 1987,
        "route": "口服",
        "half_life_h": 4.0,
        "metabolized_by": ["CYP1A2"],
        "notes": "靶点为细菌 DNA 旋转酶/拓扑异构酶 IV；可抑制 CYP1A2。",
    },
    {
        "name": "Fluconazole",
        "smiles": "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",
        "drug_class": "三唑类抗真菌药",
        "primary_targets": [],
        "indication": "念珠菌、隐球菌感染",
        "approval_year": 1990,
        "route": "口服",
        "half_life_h": 30.0,
        "metabolized_by": ["CYP2C9", "CYP3A4"],
        "notes": "靶点为真菌 CYP51（ERG11）；同时是人类 CYP2C9/CYP3A4 抑制剂。",
    },
    {
        "name": "Oseltamivir",
        "smiles": "CCC(CC)O[C@@H]1C=C(C(=O)OCC)C[C@H](N)[C@@H]1NC(C)=O",
        "drug_class": "神经氨酸酶抑制剂（前药）",
        "primary_targets": [],
        "indication": "甲型/乙型流感治疗与预防",
        "approval_year": 1999,
        "route": "口服",
        "half_life_h": 6.0,
        "metabolized_by": [],
        "notes": "经肝酯酶转化为活性羧酸；靶点为流感病毒神经氨酸酶。",
    },
    {
        "name": "Imatinib",
        "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(n1)c1cccnc1",
        "drug_class": "BCR-ABL 酪氨酸激酶抑制剂",
        "primary_targets": ["ABL1", "KIT", "PDGFRA"],
        "indication": "慢性髓性白血病、胃肠道间质瘤",
        "approval_year": 2001,
        "route": "口服",
        "half_life_h": 18.0,
        "metabolized_by": ["CYP3A4"],
        "notes": "2-苯氨基嘧啶骨架，ATP 竞争性抑制剂。",
    },
    {
        "name": "Erlotinib",
        "smiles": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        "drug_class": "EGFR 酪氨酸激酶抑制剂",
        "primary_targets": ["EGFR"],
        "indication": "EGFR 突变非小细胞肺癌、胰腺癌",
        "approval_year": 2004,
        "route": "口服",
        "half_life_h": 36.0,
        "metabolized_by": ["CYP3A4", "CYP1A2"],
        "notes": "喹唑啉骨架，可逆 ATP 竞争性抑制 EGFR。",
    },
    {
        "name": "Gefitinib",
        "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        "drug_class": "EGFR 酪氨酸激酶抑制剂",
        "primary_targets": ["EGFR"],
        "indication": "EGFR 突变非小细胞肺癌",
        "approval_year": 2003,
        "route": "口服",
        "half_life_h": 48.0,
        "metabolized_by": ["CYP3A4", "CYP2D6"],
        "notes": "首个获批的 EGFR TKI，喹唑啉骨架。",
    },
    {
        "name": "Sorafenib",
        "smiles": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3cc(C(F)(F)F)ccc3Cl)cc2)ccn1",
        "drug_class": "多激酶抑制剂",
        "primary_targets": ["BRAF", "RAF1", "KDR", "FLT3", "PDGFRB"],
        "indication": "肝细胞癌、肾细胞癌、甲状腺癌",
        "approval_year": 2005,
        "route": "口服",
        "half_life_h": 30.0,
        "metabolized_by": ["CYP3A4"],
        "notes": "同时抑制 RAF 与 VEGFR/PDGFR；UGT1A9 也参与代谢。",
    },
    {
        "name": "Sitagliptin",
        "smiles": "N[C@@H](CC(=O)N1CCn2c(C1)nnc2C(F)(F)F)Cc1cc(F)c(F)cc1F",
        "drug_class": "DPP-4 抑制剂",
        "primary_targets": ["DPP4"],
        "indication": "2 型糖尿病",
        "approval_year": 2006,
        "route": "口服",
        "half_life_h": 12.0,
        "metabolized_by": [],
        "notes": "多以原型经肾排泄，CYP 代谢占比很小。",
    },
    {
        "name": "Empagliflozin",
        "smiles": "OC[C@H]1O[C@@H](c2ccc(Cl)c(Cc3ccc(O[C@H]4CCOC4)cc3)c2)[C@H](O)[C@@H](O)[C@@H]1O",
        "drug_class": "SGLT2 抑制剂",
        "primary_targets": ["SLC5A2"],
        "indication": "2 型糖尿病、心力衰竭",
        "approval_year": 2014,
        "route": "口服",
        "half_life_h": 12.0,
        "metabolized_by": [],
        "notes": "葡萄糖苷类结构，阻断肾小管葡萄糖重吸收；主要经 UGT 葡萄糖醛酸化。",
    },
    {
        "name": "Prednisone",
        "smiles": "CC12C=CC(=O)C=C1CCC1C2C(=O)CC2(C)C1CCC2(O)C(=O)CO",
        "drug_class": "糖皮质激素（前药）",
        "primary_targets": ["NR3C1"],
        "indication": "炎症、自身免疫病、血液系统疾病",
        "approval_year": None,
        "route": "口服",
        "half_life_h": 3.0,
        "metabolized_by": ["CYP3A4"],
        "notes": "经 11β-羟基类固醇脱氢酶转化为活性泼尼松龙。",
    },
    {
        "name": "Dexamethasone",
        "smiles": "C[C@@]12[C@](O)(C(=O)CO)CC[C@]1([C@H](C[C@H]3[C@H]2C[C@H](C)[C@@]4(C)C=CC(=O)C=C34)O)F",
        "drug_class": "糖皮质激素",
        "primary_targets": ["NR3C1"],
        "indication": "炎症、脑水肿、化疗相关恶心呕吐",
        "approval_year": None,
        "route": "口服",
        "half_life_h": 4.0,
        "metabolized_by": ["CYP3A4"],
        "notes": "9α-氟-16α-甲基结构，抗炎活性强而盐皮质激素作用弱。",
    },
    {
        "name": "Methotrexate",
        "smiles": "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1",
        "drug_class": "叶酸拮抗剂",
        "primary_targets": ["DHFR", "TYMS"],
        "indication": "类风湿关节炎、肿瘤、异位妊娠",
        "approval_year": None,
        "route": "口服",
        "half_life_h": None,
        "metabolized_by": [],
        "notes": "多聚谷氨酸化后在细胞内长期滞留；主要经肾排泄，半衰期随剂量与肾功能变化大。",
    },
    {
        "name": "Tamoxifen",
        "smiles": "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
        "drug_class": "选择性雌激素受体调节剂（SERM）",
        "primary_targets": ["ESR1"],
        "indication": "雌激素受体阳性乳腺癌",
        "approval_year": 1977,
        "route": "口服",
        "half_life_h": 168.0,
        "metabolized_by": ["CYP2D6", "CYP3A4"],
        "notes": "经 CYP2D6 生成 4-羟基他莫昔芬等活性代谢物。",
    },
    {
        "name": "Letrozole",
        "smiles": "N#Cc1ccc(cc1)C(n1cncn1)c1ccc(C#N)cc1",
        "drug_class": "芳香化酶抑制剂",
        "primary_targets": ["CYP19A1"],
        "indication": "绝经后雌激素受体阳性乳腺癌",
        "approval_year": 1997,
        "route": "口服",
        "half_life_h": 48.0,
        "metabolized_by": ["CYP2A6", "CYP3A4"],
        "notes": "三唑类结构，抑制雄激素向雌激素转化。",
    },
    {
        "name": "Sildenafil",
        "smiles": "CCCc1nn(C)c2c1nc(-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1)c(=O)[nH]2",
        "drug_class": "PDE5 抑制剂",
        "primary_targets": ["PDE5A"],
        "indication": "勃起功能障碍、肺动脉高压",
        "approval_year": 1998,
        "route": "口服",
        "half_life_h": 4.0,
        "metabolized_by": ["CYP3A4", "CYP2C9"],
        "notes": "吡唑并嘧啶酮骨架；与硝酸酯类药物联用为禁忌。",
    },
    {
        "name": "Diazepam",
        "smiles": "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1",
        "drug_class": "苯二氮卓类",
        "primary_targets": ["GABRA1"],
        "indication": "焦虑、癫痫持续状态、肌肉痉挛",
        "approval_year": None,
        "route": "口服",
        "half_life_h": 48.0,
        "metabolized_by": ["CYP2C19", "CYP3A4"],
        "notes": "正性调节 GABA-A 受体；活性代谢物去甲西泮半衰期长。",
    },
    {
        "name": "Sertraline",
        "smiles": "CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc21",
        "drug_class": "选择性 5-HT 再摄取抑制剂（SSRI）",
        "primary_targets": ["SLC6A4"],
        "indication": "抑郁症、强迫症、创伤后应激障碍",
        "approval_year": 1991,
        "route": "口服",
        "half_life_h": 26.0,
        "metabolized_by": ["CYP2B6", "CYP2C19", "CYP3A4"],
        "notes": "抑制 5-HT 转运体 SERT，代谢物去甲舍曲林活性较弱。",
    },
    {
        "name": "Escitalopram",
        "smiles": "CN(C)CCCC1(c2ccc(F)cc2)OCc2cc(C#N)ccc21",
        "drug_class": "选择性 5-HT 再摄取抑制剂（SSRI）",
        "primary_targets": ["SLC6A4"],
        "indication": "抑郁症、广泛性焦虑障碍",
        "approval_year": 2002,
        "route": "口服",
        "half_life_h": 30.0,
        "metabolized_by": ["CYP2C19", "CYP3A4", "CYP2D6"],
        "notes": "西酞普兰的 S-对映体，SERT 抑制作用更强。",
    },
    {
        "name": "Olanzapine",
        "smiles": "Cc1cc2c(s1)Nc1ccccc1N=C2N1CCN(C)CC1",
        "drug_class": "非典型抗精神病药",
        "primary_targets": ["DRD2", "HTR2A", "HRH1"],
        "indication": "精神分裂症、双相障碍",
        "approval_year": 1996,
        "route": "口服",
        "half_life_h": 33.0,
        "metabolized_by": ["CYP1A2", "CYP2D6"],
        "notes": "噻吩并苯二氮平骨架；吸烟诱导 CYP1A2 可降低暴露。",
    },
    {
        "name": "Risperidone",
        "smiles": "CC1=C(CCN2CCC(CC2)C3=NOC4=C3C=CC(=C4)F)C(=O)N5CCCCC5=N1",
        "drug_class": "非典型抗精神病药",
        "primary_targets": ["DRD2", "HTR2A"],
        "indication": "精神分裂症、双相躁狂",
        "approval_year": 1993,
        "route": "口服",
        "half_life_h": 3.0,
        "metabolized_by": ["CYP2D6"],
        "notes": "活性代谢物 9-羟基利培酮由 CYP2D6 生成，半衰期更长。",
    },
    {
        "name": "Donepezil",
        "smiles": "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
        "drug_class": "乙酰胆碱酯酶抑制剂",
        "primary_targets": ["ACHE"],
        "indication": "阿尔茨海默病",
        "approval_year": 1996,
        "route": "口服",
        "half_life_h": 70.0,
        "metabolized_by": ["CYP2D6", "CYP3A4"],
        "notes": "哌啶类可逆性乙酰胆碱酯酶抑制剂。",
    },
    {
        "name": "Carbamazepine",
        "smiles": "NC(=O)N1c2ccccc2C=Cc2ccccc21",
        "drug_class": "抗癫痫药（钠通道阻滞）",
        "primary_targets": ["SCN1A"],
        "indication": "癫痫、三叉神经痛、双相障碍",
        "approval_year": None,
        "route": "口服",
        "half_life_h": 25.0,
        "metabolized_by": ["CYP3A4", "CYP2C8"],
        "notes": "自身诱导代谢，连续用药后半衰期缩短，相互作用多。",
    },
    {
        "name": "Allopurinol",
        "smiles": "O=c1[nH]cnc2[nH]ncc12",
        "drug_class": "黄嘌呤氧化酶抑制剂",
        "primary_targets": ["XDH"],
        "indication": "痛风、高尿酸血症",
        "approval_year": 1966,
        "route": "口服",
        "half_life_h": 2.0,
        "metabolized_by": [],
        "notes": "经黄嘌呤氧化酶转化为活性氧嘌呤醇，后者半衰期更长。",
    },
]


def list_drugs() -> list[dict]:
    """精简参考药清单，供前端下拉选择。"""
    return [
        {
            "name": item["name"],
            "drug_class": item["drug_class"],
            "primary_targets": list(item["primary_targets"]),
            "indication": item["indication"],
            "approval_year": item["approval_year"],
            "route": item["route"],
        }
        for item in FDA_DRUGS
    ]


# ---------------------------------------------------------------------------
# similarity machinery
# ---------------------------------------------------------------------------
_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# 标准化属性距离使用的尺度（约为常见药物分布的一个典型跨度）
_PROP_SCALES = {
    "mw": 150.0,
    "logp": 2.0,
    "tpsa": 40.0,
    "hbd": 2.0,
    "hba": 3.0,
    "qed": 0.30,
}

_SCORE_WEIGHTS = {
    "tanimoto": 0.45,
    "target": 0.25,
    "admet": 0.20,
    "property": 0.10,
}

_RADAR_COLORS = ("#0369a1", "#b45309", "#7e22ce")

_CACHE: list[dict] | None = None


def _normalize_targets(targets) -> list[str]:
    if not targets:
        return []
    result = []
    for target in targets:
        if not target:
            continue
        symbol = str(target).strip().upper()
        if symbol and symbol not in result:
            result.append(symbol)
    return result


def _scaffold(mol: Chem.Mol) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


def _admet_block(profile: dict) -> dict:
    values = {item["key"]: float(item["value"]) for item in profile.get("admet", [])}
    return {
        "herg": values.get("herg"),
        "cyp3a4": values.get("cyp3a4"),
        "cyp2d6": values.get("cyp2d6"),
        "ames": values.get("ames"),
        "half_life": values.get("half_life"),
    }


def _compact_profile(profile: dict) -> dict:
    phys = profile["physchem"]
    drug = profile["druglike"]
    return {
        "physchem": {
            "mw": phys["mw"],
            "logp": phys["logp"],
            "tpsa": phys["tpsa"],
            "hbd": phys["hbd"],
            "hba": phys["hba"],
            "rotb": phys["rotb"],
            "rings": phys["rings"],
            "formula": phys["formula"],
            "qed": drug["qed"],
        },
        "admet": _admet_block(profile),
        "radar": profile["radar"],
    }


def _build_cache() -> list[dict]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    records = []
    for entry in FDA_DRUGS:
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol is None:  # 理论上不会发生，SMILES 已校验
            continue
        profile = admet.profile_smiles(entry["smiles"], entry["name"])
        records.append(
            {
                "entry": entry,
                "mol": mol,
                "fp": _FP_GEN.GetFingerprint(mol),
                "targets": _normalize_targets(entry["primary_targets"]),
                "scaffold": _scaffold(mol),
                "compact": _compact_profile(profile),
            }
        )
    _CACHE = records
    return records


def _mcs_stats(mol_a: Chem.Mol, mol_b: Chem.Mol) -> tuple[int | None, float | None]:
    """最大公共子结构原子数与相对比例（相对较小分子重原子数）。"""
    try:
        result = rdFMCS.FindMCS(
            [mol_a, mol_b],
            timeout=2,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
        )
    except Exception:
        return None, None
    atoms = int(getattr(result, "numAtoms", 0) or 0)
    if atoms <= 0:
        return 0, 0.0
    heavy = min(mol_a.GetNumHeavyAtoms(), mol_b.GetNumHeavyAtoms())
    return atoms, round(atoms / max(1, heavy), 3)


def _property_distance(phys_a: dict, phys_b: dict) -> float:
    """MW/logP/TPSA/HBD/HBA/QED 的标准化绝对差，0–100。"""
    diffs = []
    for key, scale in _PROP_SCALES.items():
        value_a = float(phys_a.get(key) or 0.0)
        value_b = float(phys_b.get(key) or 0.0)
        diffs.append(min(abs(value_a - value_b) / scale, 1.0))
    return round(100.0 * sum(diffs) / len(diffs), 2)


def _admet_similarity(admet_a: dict, admet_b: dict) -> float:
    """基于 hERG/CYP3A4/CYP2D6/Ames/半衰期差异的 0–100 相似度。"""

    def relative(key: str, scale: float) -> float:
        value_a = float(admet_a.get(key) or 0.0)
        value_b = float(admet_b.get(key) or 0.0)
        return min(abs(value_a - value_b) / scale, 1.0)

    half_a = max(float(admet_a.get("half_life") or 0.05), 0.05)
    half_b = max(float(admet_b.get("half_life") or 0.05), 0.05)
    half_diff = min(abs(math.log(half_a) - math.log(half_b)) / math.log(4.0), 1.0)
    difference = (
        0.30 * relative("herg", 40.0)
        + 0.20 * relative("cyp3a4", 40.0)
        + 0.15 * relative("cyp2d6", 40.0)
        + 0.15 * relative("ames", 30.0)
        + 0.20 * half_diff
    )
    return round(100.0 * (1.0 - difference), 2)


def compare(
    candidate_smiles: str,
    candidate_name: str = "候选分子",
    candidate_targets=None,
    top_n: int = 12,
) -> dict:
    """把候选分子与 FDA 参考集逐项比较，返回排序后的参考列表。"""
    cand_mol = Chem.MolFromSmiles(candidate_smiles)
    if cand_mol is None:
        raise ValueError(f"无法解析候选分子 SMILES：{str(candidate_smiles)[:80]}")

    cand_profile = admet.profile_smiles(candidate_smiles, candidate_name)
    cand_compact = _compact_profile(cand_profile)
    cand_targets = _normalize_targets(candidate_targets)
    cand_fp = _FP_GEN.GetFingerprint(cand_mol)
    cand_scaffold = _scaffold(cand_mol)

    records = _build_cache()
    pairs = [
        (DataStructs.TanimotoSimilarity(cand_fp, record["fp"]), record)
        for record in records
    ]
    pairs.sort(key=lambda item: item[0], reverse=True)

    # MCS 只对结构最接近的前 20 个计算，控制耗时
    mcs_names = {record["entry"]["name"] for _, record in pairs[:20]}

    references = []
    for tanimoto, record in pairs:
        entry = record["entry"]
        if entry["name"] in mcs_names:
            mcs_atoms, mcs_ratio = _mcs_stats(cand_mol, record["mol"])
        else:
            mcs_atoms, mcs_ratio = None, None

        shared = sorted(set(cand_targets) & set(record["targets"]))
        union = set(cand_targets) | set(record["targets"])
        jaccard = len(shared) / len(union) if union else 0.0

        prop_distance = _property_distance(
            cand_compact["physchem"], record["compact"]["physchem"]
        )
        prop_similarity = round(100.0 - prop_distance, 2)
        admet_similarity = _admet_similarity(
            cand_compact["admet"], record["compact"]["admet"]
        )

        components = {
            "tanimoto": float(tanimoto),
            "target": float(jaccard),
            "admet": admet_similarity / 100.0,
            "property": prop_similarity / 100.0,
        }
        if cand_targets:
            score = sum(
                _SCORE_WEIGHTS[key] * value for key, value in components.items()
            )
        else:
            # 缺少候选靶点时不把 target 项按 0 计入，改为在剩余权重上归一化
            active = {k: w for k, w in _SCORE_WEIGHTS.items() if k != "target"}
            score = sum(active[k] * components[k] for k in active) / sum(active.values())

        references.append(
            {
                "name": entry["name"],
                "drug_class": entry["drug_class"],
                "targets": list(record["targets"]),
                "indication": entry["indication"],
                "approval_year": entry["approval_year"],
                "route": entry["route"],
                "half_life_h": entry["half_life_h"],
                "metabolized_by": list(entry["metabolized_by"]),
                "smiles": entry["smiles"],
                "tanimoto": round(float(tanimoto), 4),
                "same_scaffold": bool(
                    cand_scaffold and cand_scaffold == record["scaffold"]
                ),
                "mcs_atoms": mcs_atoms,
                "mcs_ratio": mcs_ratio,
                "shared_targets": shared,
                "target_jaccard": round(float(jaccard), 4),
                "property_distance": prop_distance,
                "property_similarity": prop_similarity,
                "admet_similarity": admet_similarity,
                "score": round(float(score), 4),
                "mechanism": (
                    f"{entry['drug_class']}；主要靶点 "
                    f"{'/'.join(record['targets']) or '未列'}"
                ),
                "physchem": record["compact"]["physchem"],
                "admet": record["compact"]["admet"],
                "radar": record["compact"]["radar"],
            }
        )

    references.sort(key=lambda item: item["score"], reverse=True)
    nearby_classes = []
    for reference in references[:5]:
        if reference["drug_class"] not in nearby_classes:
            nearby_classes.append(reference["drug_class"])

    candidate = {
        "name": candidate_name,
        "smiles": Chem.MolToSmiles(cand_mol),
        "targets": cand_targets,
        "scaffold": cand_scaffold,
        "physchem": cand_compact["physchem"],
        "admet": cand_compact["admet"],
        "radar": cand_compact["radar"],
        "shared_target_count": sum(1 for item in references if item["shared_targets"]),
        "nearby_classes": nearby_classes,
    }

    top = [
        {
            "rank": index + 1,
            "name": item["name"],
            "drug_class": item["drug_class"],
            "score": item["score"],
            "tanimoto": item["tanimoto"],
            "target_jaccard": item["target_jaccard"],
            "admet_similarity": item["admet_similarity"],
            "shared_targets": item["shared_targets"],
        }
        for index, item in enumerate(references[: max(1, int(top_n))])
    ]

    return {
        "candidate": candidate,
        "references": references,
        "top": top,
        "method": (
            "结构：RDKit Morgan 指纹（radius=2, 2048 bit）Tanimoto + Bemis–Murcko 骨架 + "
            "最大公共子结构（rdFMCS，仅对相似度前 20 名计算，timeout 2 s）；"
            "靶点：候选靶点与各药 HGNC 靶点集合的 Jaccard；"
            "ADMET：admet.profile_smiles 的 hERG/CYP3A4/CYP2D6/Ames/预测半衰期，"
            "按归一化差异折算 0–100 相似度；属性：MW、logP、TPSA、HBD、HBA、QED 的"
            "标准化绝对差。综合分 = 0.45×Tanimoto + 0.25×靶点 Jaccard + "
            "0.20×ADMET 相似度 + 0.10×属性相似度（未提供候选靶点时在剩余权重上归一化），"
            "属启发式排序，不是概率或置信度。"
        ),
        "limitations": (
            f"参考集仅含 {len(records)} 个结构简单、靶点明确的 FDA 已上市药物，"
            "不代表全部上市药物；靶点取自说明书中的已知主要靶点，缺少脱靶谱与定量亲和力，"
            "因此不给出 pIC50 等药效数值，只用靶点重叠与机制邻近做参考。ADMET、QED 与"
            "半衰期均为计算预测，可能与实测值有数量级差异；Tanimoto 与 MCS 只反映二维"
            "结构相似，不反映三维构象或结合模式。综合分为经验权重，仅用于排序与假设"
            "生成，不能作为疗效或安全性结论，也不能替代临床或实验数据。"
        ),
    }


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def _truncate(text: str, limit: int) -> str:
    text = i18n.tr(str(text or ""))
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def _target_text(targets: list[str], limit: int = 3) -> str:
    if not targets:
        return "—"
    text = "/".join(targets[:limit])
    if len(targets) > limit:
        text += "…"
    return text


def _fmt(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def drugref_pdf(data: dict, outpath) -> Path:
    """输出 A4 竖版 CNS 风格参考分析图（多面板）。"""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    from . import figures

    figures._style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    candidate = data.get("candidate") or {}
    references = list(data.get("references") or [])
    name = str(candidate.get("name") or "候选分子")

    fig = figures._a4_page(
        "图 19　FDA 已上市药物参考分析",
        f"{_truncate(name, 28)}　与 {len(references)} 个已上市药物的结构、靶点与 ADMET 比较",
        "Fig. 19　FDA reference panel",
    )

    # ---- a 结构相似度条形图 ------------------------------------------------
    by_tanimoto = sorted(
        references, key=lambda item: float(item.get("tanimoto") or 0.0), reverse=True
    )[:15]
    ax = fig.add_axes([0.315, 0.690, 0.610, 0.184])
    if by_tanimoto:
        labels = [_truncate(item["name"], 16) for item in by_tanimoto][::-1]
        values = [float(item["tanimoto"]) for item in by_tanimoto][::-1]
        xmax = max(0.22, max(values) * 1.22)
        colors = [
            figures.TEAL if value >= 0.40 else figures.BLUE if value >= 0.25 else figures.MUTED
            for value in values
        ]
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=7.2)
        ax.barh(labels, values, color=colors, height=0.62, edgecolor="none")
        for y, value in enumerate(values):
            ax.text(
                value + xmax * 0.012, y, f"{value:.2f}",
                va="center", fontsize=9.8, color=figures.INK, fontweight="bold",
            )
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=9.6)
        figures._compact_frame(ax)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "无可用参考药物", ha="center", va="center",
                fontsize=12, color=figures.MUTED)
    ax.set_title("结构相似度（Morgan 指纹 Tanimoto，前 15 位）", loc="left",
                 fontsize=13, pad=10)
    figures._panel_tag(ax, "a", x=-0.20, y=1.14)

    # ---- b Tanimoto vs 靶点 Jaccard ---------------------------------------
    ax = fig.add_axes([0.085, 0.470, 0.400, 0.163])
    if references:
        xs = [float(item.get("target_jaccard") or 0.0) for item in references]
        ys = [float(item.get("tanimoto") or 0.0) for item in references]
        jitter = [((index % 5) - 2) * 0.007 for index in range(len(xs))]
        ax.scatter(
            [x + j for x, j in zip(xs, jitter)], ys, s=46, color=figures.BLUE,
            alpha=0.82, edgecolor="white", linewidth=0.8, zorder=3,
        )
        if candidate.get("targets"):
            ax.scatter(
                [1.0], [1.0], marker="*", s=240, color=figures.AMBER,
                edgecolor="white", linewidth=1.0, zorder=5,
            )
            ax.annotate(
                "候选", (1.0, 1.0), xytext=(-8, -18), textcoords="offset points",
                fontsize=9.5, color=figures.AMBER, ha="right", fontweight="bold",
            )
        else:
            ax.text(
                0.04, 0.92, "未提供候选靶点\nJaccard 按 0 计",
                transform=ax.transAxes, fontsize=8.6, color=figures.MUTED,
                va="top", linespacing=1.5,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                      "edgecolor": figures.GRID, "linewidth": 0.8},
            )
        label_index = 0
        for item in sorted(
            references, key=lambda entry: -float(entry.get("score") or 0.0)
        )[:3]:
            jaccard = float(item.get("target_jaccard") or 0.0)
            tanimoto = float(item.get("tanimoto") or 0.0)
            if jaccard >= 0.999 and tanimoto >= 0.999:
                continue  # 候选自身，已由星标表示
            if tanimoto > 0.70:
                dy = -16
            elif tanimoto < 0.26:
                # 贴近横轴的点只向上标注，避免文字压到坐标轴
                dy = [14, 42, 70][label_index % 3]
            else:
                dy = [14, -22, 40][label_index % 3]
            label_index += 1
            ax.annotate(
                _truncate(item["name"], 13),
                (jaccard, tanimoto),
                xytext=(7, dy), textcoords="offset points",
                fontsize=7.2, color=figures.MUTED, ha="left",
                arrowprops=(
                    {"arrowstyle": "-", "color": figures.GRID, "linewidth": 0.7}
                    if abs(dy) >= 20 else None
                ),
            )
        ax.set_xlim(-0.06, 1.10)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("靶点 Jaccard（与候选）", fontsize=11.5)
        ax.set_ylabel("Tanimoto", fontsize=11.5)
        ax.tick_params(labelsize=9.5)
        figures._compact_frame(ax, "both")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "无可用参考药物", ha="center", va="center",
                fontsize=12, color=figures.MUTED)
    ax.set_title("结构相似度与靶点重叠", loc="left", fontsize=13, pad=8)
    figures._panel_tag(ax, "b", x=-0.22, y=1.16)

    # ---- c ADMET 雷达 ------------------------------------------------------
    ax = fig.add_axes([0.555, 0.455, 0.355, 0.165], polar=True)
    radar = candidate.get("radar") or []
    if radar:
        labels = [str(item["axis"]) for item in radar]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        def draw(series_values, color, label, width=1.8, alpha=0.0):
            values = list(series_values) + [series_values[0]]
            ax.plot(angles, values, color=color, linewidth=width, label=label)
            if alpha:
                ax.fill(angles, values, color=color, alpha=alpha)

        draw([float(item["score"]) for item in radar], figures.INK,
             "候选", width=2.2, alpha=0.10)
        top3 = sorted(
            references, key=lambda item: -float(item.get("score") or 0.0)
        )[:3]
        for index, item in enumerate(top3):
            series = item.get("radar") or []
            if len(series) != len(labels):
                continue
            draw([float(point["score"]) for point in series],
                 _RADAR_COLORS[index % len(_RADAR_COLORS)],
                 _truncate(item["name"], 7), width=1.7)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9.6, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7.2,
                           color=figures.MUTED)
        ax.grid(color=figures.GRID)
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            # The caption identifies the candidate (black) and top-3 (colour)
            # traces; a legend here would collide with the radial labels.
            pass
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "无 ADMET 数据", ha="center", va="center",
                fontsize=11, color=figures.MUTED)
    ax.set_title(
        "ADMET 雷达（候选 vs 综合分前 3）" if i18n.get_lang() != "en"
        else "ADMET radar",
        fontsize=11.8, pad=28,
    )
    figures._panel_tag(ax, "c", x=-0.14, y=1.16)

    # ---- d 参考药属性表 ----------------------------------------------------
    table_rows = [[
        "药物", "类别", "靶点", "相似度", "QED", "logP", "TPSA",
        "hERG", "CYP3A4", "t½", "综合分",
    ]]
    for item in sorted(
        references, key=lambda entry: -float(entry.get("score") or 0.0)
    )[:8]:
        phys = item.get("physchem") or {}
        endpoint = item.get("admet") or {}
        table_rows.append(
            [
                _truncate(item["name"], 16),
                _truncate(item["drug_class"], 12),
                _truncate(_target_text(item.get("targets") or []), 15),
                f"{float(item.get('tanimoto') or 0.0):.2f}",
                f"{float(phys.get('qed') or 0.0):.2f}",
                _fmt(phys.get("logp"), 1),
                _fmt(phys.get("tpsa"), 0),
                _fmt(endpoint.get("herg"), 0),
                _fmt(endpoint.get("cyp3a4"), 0),
                _fmt(endpoint.get("half_life"), 1),
                f"{float(item.get('score') or 0.0):.3f}",
            ]
        )
    from matplotlib.lines import Line2D

    fig.add_artist(
        Line2D([0.07, 0.93], [0.414, 0.414], transform=fig.transFigure,
               color=figures.GRID, linewidth=0.8)
    )
    xs = [0.070, 0.190, 0.315, 0.425, 0.495, 0.545, 0.595, 0.650, 0.710, 0.775, 0.835]
    y = 0.408
    for row_index, row in enumerate(table_rows):
        for col_index, (x, cell) in enumerate(zip(xs, row)):
            text = str(cell)
            if col_index == 0:
                text = _truncate(text, 15)
            elif col_index == 1:
                text = _truncate(text, 16)
            elif col_index == 2:
                text = _truncate(text, 13)
            fig.text(
                x, y, text, fontsize=6.8, va="top",
                color=figures.INK if row_index else figures.MUTED,
                fontweight="bold" if row_index == 0 else "normal",
            )
        y -= 0.0140

    # ---- e 前 3 参考药二维结构 --------------------------------------------
    top_mols = sorted(
        references, key=lambda entry: -float(entry.get("score") or 0.0)
    )[:3]
    for index, item in enumerate(top_mols):
        ax = fig.add_axes([0.070 + index * 0.290, 0.166, 0.240, 0.078])
        try:
            mol = Chem.MolFromSmiles(item["smiles"])
            figures._draw_molecule(ax, mol, compact=True)
        except Exception:
            ax.axis("off")
            ax.text(0.5, 0.5, "结构不可用", ha="center", va="center",
                    fontsize=9, color=figures.MUTED)
        ax.set_title(
            _truncate(item["name"], 18), fontsize=9.4, color=figures.INK,
            fontweight="bold", pad=5,
        )
        if index == 0:
            figures._panel_tag(ax, "e", x=-0.08, y=1.12)

    english = i18n.get_lang() == "en"
    caption = (
        "Top 15 FDA-approved drugs by Morgan (radius 2, 2048-bit) Tanimoto "
        "similarity to the candidate. Joint distribution of structural similarity "
        "and target Jaccard; the star marks the candidate itself and the small "
        "horizontal jitter only prevents overlapping points. ADMET radar for the "
        "candidate and the top three drugs by composite score. (d) Reference-drug "
        "property table; hERG and CYP3A4 are 0-100 relative risk scores (not "
        "probabilities) and predicted t1/2 is in hours. (e) 2D structures of the "
        "top three drugs. Composite = 0.45 x Tanimoto + 0.25 x target Jaccard + "
        "0.20 x ADMET similarity + 0.10 x property similarity, a heuristic ranking "
        "only. Drug facts are curated from public FDA labels, PubChem and "
        "ChEMBL; ADMET values are computed and cannot replace "
        "clinical or experimental data."
        if english else
        (
            "Morgan 指纹（radius 2、2048 bit）Tanimoto 相似度前 15 位。"
            "结构相似度与靶点 Jaccard 联合分布，星标为候选自身锚点，横向微抖动仅为避免点重叠。"
            "候选与综合分前 3 药物的 ADMET 雷达（0–100）。"
            "（d）参考药属性汇总；hERG 与 CYP3A4 为 0–100 相对风险评分（非概率），预测 t½ 单位为小时。"
            "（e）综合分前 3 药物的二维结构。综合分 = 0.45×Tanimoto + 0.25×靶点 Jaccard + "
            "0.20×ADMET 相似度 + 0.10×属性相似度，仅为启发式排序。"
            "药物信息汇总自公开 FDA 标签、PubChem 与 ChEMBL；ADMET 为计算值，"
            "不能替代临床或实验数据。"
        )
    )
    figures._caption(
        fig,
        caption,
        y=0.150,
        width=112 if english else 104,
    )

    fig.savefig(outpath, dpi=600, facecolor="white")
    plt.close(fig)
    return outpath
