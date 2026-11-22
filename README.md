# DrugHX · 分子对接与药物设计工作台

DrugHX（作者：HX）是一个面向 Windows 11 的本地分子对接与药物设计软件，提供从受体准备到报告输出的一体化工作流（独立实现）。所有操作都可以在图形界面中完成：获取/导入受体、准备 PDBQT、生成配体 3D 构象、定义结合位点、批量对接、查看打分与构象、分析蛋白–配体作用力、ADMET、LigPlot、靶点垂钓、网络药理学、可合成性、FDA 参考、药物互作，并输出中英文完整报告。

界面为本地 Web UI，启动后自动在浏览器打开，计算全部在本机完成，不上传任何结构。

---

## 1. 与 SailVina 工作流的差异

> DrugHX 为独立实现，未使用 SailVina 源码；此处仅作技术流程对比，名称归其作者所有。

| SailVina 工作流的限制 | DrugHX 的处理 |
| --- | --- |
| 依赖 MGLTools 的 Python 2 脚本准备受体 | 使用 **Meeko 0.8 + RDKit**，纯 Python 3，无需安装 MGLTools |
| 依赖 Open Babel 2.4 做格式转换 | 内置 RDKit 读写 SDF/MOL/MOL2/PDB/SMILES，不依赖外部 Open Babel |
| 安装路径不能有中文或空格 | 全流程使用 Unicode 路径；已在 `中文 项目测试` 这类目录下完成端到端验证 |
| 对接时界面无响应 | 准备与对接全部在后台线程/子进程中运行，UI 实时显示进度和日志，可取消 |
| 旧版 Vina 1.1.2，盒子受 27000 Å³ 限制 | 使用 **AutoDock Vina 1.2.5**，支持 Vinardo、较大盒子、随机种子和 CPU 控制 |
| 需要手工准备 config.txt、手工摆放盒子 | 自动识别共晶配体或预测口袋并生成盒子，也可手动输入 |
| 没有 3D 视图，需要另开 PyMOL | 内置 3Dmol.js：蛋白卡通、配体、口袋、盒子、构象切换、氢键/盐桥虚线 |
| 没有作用力分析，推荐上传 PLIP | 内置氢键、疏水接触、盐桥、π-堆积、卤键、金属配位的几何分析 |
| 结果靠 txt/Excel 手工整理 | 结果表按打分排序，可导出带 BOM 的 CSV（Excel 直接打开） |
| 只支持配体文件，画分子要另用 ChemDraw | 直接粘贴 SMILES，自动加氢、生成 3D 构象、MMFF94 最小化、质子化 |
| 外部 PDBQT 的残基编号不规范（如 C000）会导致分析崩溃 | 自动规范 PDBQT 的原子序号/链/残基号，并提供无键级 PDBQT 的几何回退分析 |
| 只有单一 Vina 打分，结果不够稳健 | Vina + Vinardo 双打分、共识排名；GPU 结果用标准 Vina 复核 |
| 深色界面不符合论文/办公场景 | 全浅色专业主题，白底 3D 视图与论文图输出 |
| 缺少可复现性 | 项目目录保存输入、PDBQT、每次运行的 config、随机种子、日志和构象 |

---

## 2. 功能一览

- **受体**：RCSB PDB ID 下载、本地 PDB/PDBQT 上传、多受体管理；去除水/杂原子、保留金属离子、选择链、模板电荷或 Gasteiger 电荷；Meeko 生成 PDBQT。
- **配体（批量输入）**：多行 SMILES（`名称 SMILES` 或 `SMILES 名称` 自动识别；`.smi` 采用“第一个空白前为 SMILES，其余整段为名称”，`Salvianolic acid G` 这类含空格名称会完整保留）、SDF（自动拆分全部记录）、CSV/TSV（识别 `smiles`/`name` 列）、MOL、MOL2、PDB、现成 PDBQT，可一次上传多个文件；准备前自动做成药性过滤（默认严格 Lipinski/Veber 空间：PAINS、Brenk/NIH、MW > 500、cLogP > 4.5、TPSA > 140、HBD > 5、HBA > 10、可旋转键 > 10、环数 > 6、最大环 ≥ 8 元、QED < 0.30；**手动与自动模式均有总开关，10 项阈值全部可调**）；pH 7.4 酸性/碱性基团处理；ETKDGv3 构象；MMFF94 能量最小化；批量准备。宏环配体以**刚性宏环**模式准备，避免 Vina-GPU 无法解析的 `G0/CG0` glue 类型；旧配体在对接前会自动重新准备。
- **结合位点**：共晶配体定位、内置 LIGSITE 风格口袋预测（Top 5）、手动盒子；实时显示体积与 3D 盒子。
- **对接**：单/多受体 × 单/多配体矩阵；默认参数按准确性优先设置（exhaustiveness 16、输出 9 个构象、能量范围 4 kcal/mol、随机种子 42；GPU search depth 40），仍可在界面调整；**CPU 多任务并行**或 **GPU（Vina-GPU 2.0）加速**；后台运行、进度、日志、取消。
- **打分**：Vina 主打分 + Vinardo 交叉打分 + 共识排名；GPU 结果会用标准 Vina 重新打分，保证 CPU/GPU 可比。
- **结果**：打分排序、每个配体的构象、RMSD l.b./u.b.、作用力分析、构象 3D 预览、复合物 PDB 导出、CSV 导出、Vina 日志。
- **构象物理合理性（PoseBusters）**：对每个结合构象做 20 项物理有效性检查（键长/键角、内部原子冲突、环平面性、内部应变能、与蛋白及辅因子的距离与体积重叠），结果页可逐构象查看，也可一键校验该记录前 50 个构象并按配体汇总，用于识别“打分虚高但不合理”的结合模式。
- **论文图**：白底 2D 结构、作用力示意图、Vina/Vinardo 打分图、作用力热图、双面板汇总图；全部以 **PDF 矢量图**输出，并附 SVG 备份、PyMOL 脚本与图注草稿。
- **ADMET / 类药性（12 项端点）**：12 项端点（HIA、Caco-2、logBB、蛋白结合、hERG、CYP3A4/2D6、Ames、清除率、半衰期、P-gp、logS）+ Lipinski/Veber/Egan/类先导规则 + 结构警示 + QED；给出 LE、LipE、BEI、SEI 等配体效率指标，并输出 A4 雷达图＋端点条形图 PDF。
- **药效团互补性（几何特征匹配）**：从对接构象提取氢键供体/受体、芳香、疏水、正/负电与金属结合特征，逐个匹配 5 Å 内残基，计算特征覆盖率与未满足特征，输出 2D 特征映射 + 残基互补表 PDF。
- **自由能分解（Prime MM-GBSA 风格）**：12-6 Lennard-Jones + 距离相关介电 Coulomb + Born 型极性去溶剂化 + 非极性项 + 几何氢键，按残基分解结合能，识别热区残基并输出能量项、残基贡献与表格 PDF。
- **药物优化**：一级菜单“药物优化”，基于残基作用力谱自动生成骨架跃迁/生物电子等排/卤素/极性修饰候选，自动准备 3D 构象、重新对接、比较 ΔVina，并输出 PDF 改进报告与作用力对比图；候选表直接列出 QED、LE 与类药性通过数，兼顾活性与成药性。
- **分子设计**：一级菜单“分子设计”，针对选定蛋白口袋枚举核心/取代基生成类药分子，自动对接筛选，并在本地 SMILES 数据库（内置参考集 + 项目配体 + 可导入库）中做 Morgan/Tanimoto 相似性检索；设计表同样附带 QED 与 LE。
- **靶点垂钓（多源证据）**：输入 SMILES，先做 ChEMBL 精确匹配，再以 Tanimoto ≥ 40% 检索最多 120 个类似物，合并 ChEMBL 活性/作用机制、DGIdb、BindingDB、PubChem、Open Targets、SwissTargetPrediction 与 RCSB PDB 共晶配体证据，输出最多 220 个潜在靶点与基因集；结果可直接用于网络药理学。
- **网络药理学**：把垂钓得到的基因集（默认 80 个，上限 150）提交 STRING 构建蛋白互作网络，再用 Enrichr 完成 KEGG、GO、Reactome、WikiPathway、DisGeNET 与 Hallmark 富集，输出 PPI 网络、气泡图、C-T-D 网络、桑基图与完整富集表。
- **在线智能资源**：AlphaFold DB / ESMFold 在线获取或预测蛋白结构，并可一键登记为项目受体自动准备 PDBQT；一次查询 UniProt 注释、Open Targets 疾病关联与 RCSB 实验结构；PubChem 化合物性质与 2D 相似物检索（可加入配体库）；Enrichr 通路富集。全部免费、无需 API Key、无需本地 GPU。
- **系列可开发性**：一次汇总全部成功对接分子的 BOILED-Egg 吸收/入脑判别、Wager CNS MPO、logD7.4、Ertl SAscore 与 PAINS/Brenk/NIH 结构警示（图 11），并在结果详情提供“可开发性 PDF”一键导出。
- **化学空间与骨架多样性**：Morgan/Tanimoto 聚类热图、经典 MDS 投影与 Murcko 骨架频次，用于判断系列是集中单一骨架还是覆盖多个化学型（图 12）。
- **代谢软点预测**：用 SMARTS 规则标注 CYP 氧化、N-/O-脱烷基、硫氧化等潜在位点，给出 0–100 风险分并在二维结构上高亮（图 13）。
- **自动模式**：上传受体文件、输入 PDB ID / 基因或 UniProt 登录号 / 氨基酸序列，加入一批配体即可全流程运行；默认同时输出中文与英文两份完整报告；精细对接完成后暂停，由用户在候选分子面板中查看最多 **20 个**候选的结构、对接打分、理化与成药性参数后手动确认，确认后才继续优化、设计、靶点垂钓、网络药理学与报告，并在结束时提供“打开报告文件夹”。自动模式面板提供成药性过滤总开关与 10 项阈值。候选项先按**药物化学分层**过滤（QED ≥ 0.45、SAscore ≤ 4.5、LE ≥ 0.25、重原子 12–45、无 PAINS/Brenk、PoseBusters 通过率 ≥ 50% 优先），再按**综合决策分数**排序（对接 45%、构象物理有效性 20%、配体效率 12%、QED 13%、合成可及性 10%，结构警示扣分），而不是只看 Vina 打分；优先层不足 20 个时才由高分分子补齐。
- **可复现记录**：每次完整报告生成同目录的 `*_manifest.json`（软件与引擎版本、全部参数、随机种子、输入文件 SHA-256），报告内含“可复现性记录”页；`GET /api/runs/{id}/manifest?verify=true` 可复核输入结构是否被替换。
- **兽药扩展（非人物种）**：支持猪、牛、羊、犬、猫、大鼠；常见兽药类别的示例 MRL、基于 logP/分子量的休药期估计、肠道菌群还原/水解规则枚举的代谢产物，并汇总为报告中的“兽药残留与菌群代谢”页；犬、猫、大鼠为非食用动物，不计算 MRL/休药期。所有数值均标注为筛查级估计。
- **种属创新算法**：异速药代外推（CL ∝ BW^-0.25、Vd ∝ BW^-0.05，给出各物种 CL/Vd/半衰期）、种属加权兽药可开发性指数 SVDI（反刍动物计过瘤胃、食用动物计休药期、伴侣动物计适用域）、适用域判断（45 个上市药 Morgan/Tanimoto 最近邻）、口袋可成药性评分（体积、埋藏度、疏水残基比例），并分别进入报告 C8 页与口袋排序。
- **主动设计**：分子设计先生成 2 倍候选，再按“质量 60% + 多样性 40%”挑选对接批次，减少无效对接；多引擎共识采用加权排名，引擎分歧过大时标记低置信度。
- **研发决策（科研院所与药厂预研发）**：结果分析页内置三张决策卡——**MPO 多参数优化**（效力 30%、配体效率 15%、QED 20%、可合成性 10%、安全性 25%，输出 0–100 分、起点/命中/先导/候选阶段、Pareto 前沿与相对 FDA 药物集的新颖性，可导出 CSV）、**实验数据回填与打分校准**（导入 IC50/Kd/EC50/Ki，按 SMILES 或名称匹配配体，给出 Spearman/Pearson 与“能否按打分排序”的明确结论）、**起始剂量预测**（按效价、分子量、F、清除与分布容积估算人及猪、牛、羊、犬、猫、大鼠的维持/负荷剂量与半衰期）。
- **可信度与弃权**：把受体结构质量（分辨率/R-free/缺失残基/altloc/pLDDT）、适用域、多引擎一致性、实验校准与构象有效性合成 0–100 可信度；低于 45 分明确标记“不建议据此决策”。
- **项目内 SAR 学习（MMP）**：从本项目自己的对接/实测数据挖掘取代基改造规则（支持对数、平均 Δ、方向、置信度），只在本项目内迁移，不把通用规则强加给新骨架。
- **七物种 PBPK 与休药期**：血浆–肝–肾–肌–脂肪灌注限制模型，模拟口服/肌注/静注浓度–时间曲线，按示例 MRL 估算组织残留超标时间与休药期，并给出 Monte-Carlo 中位/95 分位，达到模拟上限会显式标记。
- **主动学习实验推荐**：以实测 pActivity 为标签的 Morgan/Tanimoto + 对接打分高斯过程，按 Expected Improvement 与多样性约束推荐下一批 8–16 个分子；实测不足 4 条或候选全部已测时明确退回，不给出伪预测。
- **回溯校准与数据驱动权重**：把同一次对接中的已知活性物与其余配体当作诱饵，自动计算 EF1%/EF5%/AUC（含 bootstrap 95% CI）/BEDROC（RDKit Truchon–Bayly），并用逻辑回归拟合权重与专家 MPO 权重对比；只有交叉验证 AUC ≥0.60 且提升 ≥0.03 才建议采用新权重，接近随机时直接判定“本靶点/库不具备排序能力”。
- **完整报告**：一键把对接结果、优化候选、设计分子、靶点垂钓、网络药理学、方法与全部 CNS 风格 PDF 矢量图合并成一份完整 PDF 报告；若项目里尚无垂钓/网络药理学记录，点击报告链接时会先以后台任务补齐这两步再打开 PDF，避免下载请求长时间无响应。
- **专家分析 PDF**：结果详情一键把 ADMET、药效团、自由能分解三页合并为一份 A4 PDF，便于组会与投稿补充材料使用。
- **验证**：可从共晶配体一键提取配体（使用 RCSB ideal SDF 保证键级和芳香性正确），重新对接后计算对称校正 RMSD。
- **工具**：R 基团化合物库枚举；**分子性质 / ADMET 计算器**（粘贴任意 SMILES 即时给出理化性质、类药规则、ADMET 预测与 QED，无需对接）；**批量工具箱**（结构标准化并导出 SDF、配体描述符 CSV、按 Tanimoto 最大最小距离挑选多样配体）；最近项目；系统信息；项目文件夹直达。

---

## 3. 快速开始

### 3.1 一键启动（推荐）

双击 `DrugHX.bat`。脚本会自动：

1. 创建 `.venv` 虚拟环境；
2. 安装 `requirements.txt` 中的依赖；
3. 启动本地服务并打开浏览器。

首次安装需要联网，耗时取决于网速。

### 3.2 手动启动

```powershell
cd DockForge
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

要求 Python 3.10+（推荐 3.12）。默认地址为 `http://127.0.0.1:8765`，若端口被占用会自动顺延。

### 3.3 Vina 可执行文件

`bin/vina.exe` 已随项目提供（AutoDock Vina 1.2.5）。如果缺失，`app.py` 会尝试从官方 GitHub Release 下载；也可以手动下载：

```
https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_win.exe
```

保存为 `DockForge/bin/vina.exe`，或用环境变量 `VINA_PATH` 指向其他位置。

---

## 4. CPU 与 GPU 双引擎

DrugHX 内置两套计算后端，在“分子对接”步骤的“计算引擎”中选择：

| 模式 | 引擎 | 适用场景 | 说明 |
| --- | --- | --- | --- |
| CPU | AutoDock Vina 1.2.5 | 所有机器，最终打分与验证 | 支持 1–16 个配体任务并行；每个任务自动分配 CPU 线程 |
| GPU | Vina-GPU 2.0（随软件提供） | 大批量筛选、批量对接 | 需要 OpenCL 驱动；NVIDIA / AMD 均可，实测 RTX 4070 Laptop 可用 |

- GPU 二进制位于 `bin/vina-gpu.exe`，OpenCL 内核位于 `bin/vina-gpu-kernels/`。如果替换为新版本，保持这两个路径即可，或用 `VINA_GPU_PATH` 指定可执行文件。
- GPU 负责构象搜索；找到构象后会再用 **标准 Vina** 对该构象重新打分，并可选 Vinardo 交叉打分。因此 CPU 与 GPU 结果在同一打分尺度上可比，避免不同实现的分值漂移。
- Vina-GPU 是随机搜索，单次运行可能落入局部极小。DrugHX 默认对每个配体做 **8 次独立起点搜索**（可在界面调整 1–8），再对 GPU 打分最好的两个构象做 CPU Vina 复核并保留最优；在 3PTB/aspirin 测试中，多起点把 GPU 单次 −3.8 kcal/mol 的结果提升到 −4.3 至 −5.1 kcal/mol。
- GPU 模式的定位是**快速筛选**：它能在约 1 秒/次的量级完成搜索，但受固定 search depth 限制，命中全局最优的概率低于 CPU 完整搜索。最终投稿数据建议用 CPU 模式复跑关键配体（同一盒子、同一打分函数），或把 GPU 结果作为优先级排序依据。
- 批量任务建议：GPU 模式 + 并行任务数 1（GPU 已高度并行）；CPU 模式 + 并行任务数 2–4（取决于核心数）。
- 结果表同时报告 **Vina 主打分、Vinardo 交叉打分、共识排名**（两项排名之和，越小越好）与 **GPU 原始打分**。论文中建议以 Vina 为主，Vinardo/共识仅作为稳健性排序依据，并在方法中写明打分函数与盒子参数。
- 未检测到 GPU 引擎时，GPU 选项会自动置灰，不影响 CPU 使用。

---

## 5. 九步工作流

1. **受体**：输入 PDB ID 或上传文件 → 选择准备选项 → 准备。完成后自动识别口袋，若有共晶配体则优先用它生成默认盒子。
2. **配体**：上传 SDF/MOL2 或粘贴 SMILES → 准备。状态变为“已准备”后即可参与对接。
3. **结合位点**：在“口袋预测 / 共晶配体 / 手动”中选择一个盒子。盒子应覆盖配体可能占据的区域。
4. **分子对接**：勾选受体和配体，设置参数，开始。任务在后台运行，可以切换步骤查看 3D 结构。
5. **结果分析**：按打分排序查看结果，切换构象，运行作用力分析；可继续做 ADMET、药效团互补性与自由能分解，或一键导出专家分析 PDF；可下载构象、导出复合物和 CSV。
6. **药物优化**：选择母体结果，自动生成骨架跃迁/生物电子等排候选，重新对接并输出 PDF 改进报告与残基建议。
7. **分子设计**：选择靶点口袋，枚举类药核心/取代基，自动对接筛选并做数据库相似性检索。
8. **靶点垂钓**：粘贴小分子 SMILES → 设置相似度阈值与检索规模 → 程序检索 ChEMBL 相似分子的活性记录，按 pChEMBL 排序潜在人源靶点，并映射基因符号。
9. **网络药理学**：直接使用上一步的基因集（可调整参与基因数与 STRING 置信度）→ 构建 PPI 网络并完成 KEGG/GO 富集，结果自动并入完整 PDF 报告。

### 共晶配体验证（可选）

在“结合位点”的共晶配体列表中点击 **提取**，配体会进入“配体”列表；准备后与受体对接，再在结果详情点击 **RMSD** 即可比较对接构象与晶体构象。

在 3PTB / BEN（苯甲脒）测试中，DrugHX 的 redocking RMSD 为 **0.17 Å**，并正确识别 Asp189 盐桥与 Ser190 氢键。

---

## 6. 药物优化模块

“药物优化”是一级菜单（STEP 06），用于把一次成功对接转化为可执行的分子改进方案：

1. 选择母体对接记录和配体；
2. 设置候选数量（3–10）、每个候选的 exhaustiveness、CPU/GPU 与并行数；
3. 点击“自动生成改进方案并跑分”。

模块会：

- 从母体 SMILES/构象识别羧酸、酯、芳环、甲基、羟基等可改造位点；
- 按作用力谱生成 **羧酸→四唑**、**酯→酰胺**、**苯环→吡啶（骨架跃迁）**、**芳环氟代**、**甲基→CF3**、**引入酚羟基** 等候选；
- 对每个候选重新加氢、生成 3D 构象、MMFF94 最小化、Meeko 转 PDBQT；
- 在同一位点、同一打分函数下自动重新对接，计算 ΔVina、氢键/疏水计数；
- 输出残基级改进建议（哪个残基、当前作用、建议策略）；
- 生成 **PDF 矢量图**：打分对比、作用力热图、母体/最佳候选 2D 结构、母体/最佳候选作用力示意图、完整改进报告。

示例结果（3PTB / aspirin）：

| 候选 | 改造 | Vina (kcal/mol) | ΔVina |
| --- | --- | --- | --- |
| 羧酸→四唑 | 生物电子等排 | −5.01 | −0.46 |
| 酯→酰胺 | 稳定性改造 | −4.95 | −0.40 |
| 苯环→吡啶 | 骨架跃迁 | −4.62 | −0.07 |

这些是计算假设，需实验验证；不要将 Vina 分值解释为实测结合自由能。

---

## 7. 分子设计模块

“分子设计”是一级菜单（STEP 07），用于在指定蛋白口袋中从头生成类药分子：

1. 选择已准备并设置盒子的靶点；
2. 设置设计数量（4–24）、Exhaustiveness、CPU/GPU 与并行数；
3. 可选：粘贴本地化合物库（每行 `SMILES 名称`），点击“保存数据库”；
4. 点击“生成分子并自动筛选”。

处理流程：

- 从苯、吡啶、嘧啶、噻吩、呋喃、吡咯、咪唑、萘、喹啉、吲哚等核心出发；
- 枚举氟、氯、羟基、氨基、甲基、CF3、甲氧基、氰基、酰胺、羧酸、磺酰胺等取代基；
- 按 MW、cLogP、TPSA、HBD/HBA、可旋转键做类药性预筛与指纹去重；
- 对候选生成 3D 构象、MMFF94 最小化、Meeko 转 PDBQT；
- 在靶点口袋中自动对接，按 Vina 打分排序；
- 对每个设计分子计算 Morgan 指纹（半径 2，2048 bit）与数据库的 Tanimoto 相似度，给出最近邻。

未导入数据库时，默认使用内置药物/片段参考集和项目配体；可导入 ChEMBL、ZINC 子集或企业自建库。

---

## 8. 专家分析：ADMET / 药效团 / 自由能分解

这三项功能补充了 ADMET、药效团与残基能量分解分析，作为**结果分析的扩展**集成，不新增一级菜单。打开任一条对接结果，详情操作行即可看到 **ADMET**、**药效团**、**自由能分解** 与 **专家分析 PDF** 四个按钮。

### 8.1 ADMET / 类药性

- 理化性质：MW、cLogP、cLogS（Delaney ESOL）、TPSA、HBD/HBA、可旋转键、芳香比例、pKa 类型、QED。
- 类药规则：Lipinski、Veber、Egan、类先导（lead-like）四套规则逐条判定与通过数；结构警示命中（硝基、酰卤、Michael 受体、环氧化物等）。
- 12 项 ADMET 端点（经验模型，0–1 为归一化“良好概率”）：人肠吸收 HIA、Caco-2 通透性、logBB、血浆蛋白结合、hERG 抑制、CYP3A4/CYP2D6 抑制、Ames 致突变、清除率、半衰期、P-gp 底物、logS。
- 配体效率：LE（−ΔG/重原子数）、LipE（pIC50 − cLogP 近似）、BEI、SEI，用于判断“增大分子是否值得”。
- 输出：A4 PDF，含六轴雷达图（吸收/分布/代谢/排泄/毒性/类药性）与 12 端点条形图。
- 工具弹窗中的 **分子性质 / ADMET 计算器** 支持对任意 SMILES 直接计算，不依赖对接记录。

### 8.2 药效团互补性

- 用 RDKit 的 BaseFeatures 定义从对接构象提取特征：氢键供体、氢键受体、芳香中心、疏水中心、正电中心、负电中心、金属结合位点。
- 对每个特征检索 5 Å 内的蛋白残基，按几何关系判定“互补类型”：互补氢键、π-堆积、疏水接触、盐桥、邻近。
- 输出特征总数、已满足数、覆盖率与未满足特征清单，并给出“哪些残基已经满足、哪些子口袋仍空着”的可操作提示。
- A4 PDF：左侧二维特征映射（按家族着色标注），右侧残基互补表（前 8 个）。

### 8.3 自由能分解（MM-GBSA 近似）

- 能量模型：12-6 Lennard-Jones（Amber 类联合原子参数）+ Coulomb 静电（距离相关介电 ε = 4r）+ Born 型极性去溶剂化（缩放 0.15，按埋藏度计）+ 非极性项（碳接触数）+ 几何氢键（每残基 −1.6，封顶 −3.2 kcal/mol）。
- 按残基分解配体–蛋白相互作用能，排序得到热区残基（hot spot），并给出各能量项占比。
- A4 PDF：能量项条形图、残基贡献条形图（前若干位）、残基明细表与图例。
- 说明：这是快速、透明的近似力场计算，**绝对值仅供参考**；残基排序与同系列类似物的相对比较才是有效输出。需要严谨数值时请使用 Amber/GROMACS + MMPBSA.py 或 Amber/GROMACS + MMPBSA.py、商业 MM/PBSA 实现复算。

### 8.4 专家分析 PDF

结果详情的 **专家分析 PDF** 会把上述三页按统一 A4 版面合并为单个文件（`exports/professional/`），可直接作为组会材料或投稿补充材料。

---

## 9. 完整 PDF 报告

点击顶部栏 **“中文报告”** 或 **“英文报告”**，两种版本内容、图号与页数完全一致，仅显示语言不同；程序会自动选用最新一次药物优化与分子设计记录，并优先使用对应的英文模块报告。也可在结果分析、药物优化、分子设计页面点击 **“完整 PDF 报告”**。生成的合并 PDF：

- 封面：项目、靶点、结果最优配体、优化母体与最佳候选、设计最佳；
- 方法页（1.1–1.21）：受体/配体准备、Vina/GPU、打分函数、作用力定义、优化与设计规则、数据组织、术语、靶点垂钓与网络药理学的数据来源，以及类药性、化学空间与代谢软点的判据；
- 摘要页：背景、主要发现、结论与关键词；
- 讨论页（2.1–2.10）：结合亲和力、打分可靠性、作用力热点、类药性、优化、设计、靶点垂钓、网络药理学与验证计划；
- 图 1–图 6（A4 单页矢量，面板 a/b/c）：结合亲和力全景（Vina vs Vinardo，含 Pearson r 与 Spearman ρ）、配体效率与类药化学空间、相互作用指纹热图与残基接触频率、构象能量谱、优化哑铃图（ΔVina 相对优化母体）、骨架枚举设计评价与类药性雷达；
- 图 7–图 8：代表配体的 2D 结构与作用力示意；
- 图 9 与附表 F1：靶点垂钓排序条形图、配体–靶点网络，以及按中位 pChEMBL 排序的靶点完整数据；
- 图 10 与附表 N1：STRING 蛋白互作网络、KEGG/GO 富集气泡图，以及全部富集条目与基因列表；
- 图 11：BOILED-Egg 吸收/入脑判别、CNS MPO、SAscore–logD7.4 可开发性窗口与 PAINS/Brenk/NIH 结构警示；
- 图 12：Tanimoto 相似性聚类热图、经典 MDS 化学空间与 Murcko 骨架频次；
- 图 13：代谢软点风险排序与二维结构高亮；
- 附图 A1–A3：ADMET 与类药性、药效团特征、MM-GBSA 残基能量分解（配体缺少 SMILES 时自动跳过）；
- 附图 B1–B2：药物优化报告与骨架枚举设计报告，二者之前附一页“模块报告图注”；
- 第 2.11–2.17 页：多参数优化与候选分层（MPO/Pareto/阶段/新颖性）、实验数据回填与打分校准、可信度与弃权、项目内匹配分子对（MMP）、七物种 PBPK 与休药期、主动学习实验推荐、回溯校准与数据驱动权重；
- 第 4 章“结论与实验指导”（紧随结果与讨论）：Tier 1/2/3 总体可行性判定、7 项关键指标表、主要发现、分阶段实验验证路线（蛋白/化合物质控 → 生化结合 → 细胞活性与选择性 → 成药性与安全 → 体内 PK/药效，含兽药残留）、Go/No-Go 判据与数据驱动的结构优化建议；
- 附录 A（数据表与相互作用计数）、附录 B（运行参数与补充说明）、附录 C（文件与图像索引）。

所有图表页自动排版为统一 A4 版面，PDF 矢量输出（字体 Type 42 子集嵌入，可在 Illustrator/Inkscape 中编辑）；**图注紧随各图放置**，不再集中到独立图注页。靶点垂钓、网络药理学与系列可开发性为可选模块：有数据时自动并入，无数据时报告其余部分不受影响（系列可开发性页需要至少两个带 SMILES 的成功对接结果）。字体按 CNS 风格放大（图题 16 pt、轴标签 13 pt、刻度 11 pt、面板字母 21 pt；正文 11.6–12 pt）。图 1–6 为中文标注版本，投稿英文期刊时可据此重排标签，数据与坐标无需改动。论文图表包中的双面板汇总图仍可单独下载。

---

## 10. 项目目录

所有数据保存在软件数据目录的 `data\projects\<项目名>`（例如 `<软件根目录>\data\projects\EGFR_抑制剂筛选`），可通过环境变量 `DOCKFORGE_PROJECTS` 改根目录；首次启动会把旧的 `C:\Users\<用户名>\DockForgeProjects` 项目复制进来并自动切换到新路径：

```
project.json                 项目状态、参数、运行记录
receptors/<id>/              source.pdb、receptor.pdb、receptor_clean.pdb、receptor.pdbqt
ligands/<id>/                source.sdf、ligand.sdf、ligand.pdbqt
runs/<run_id>/               001_受体__配体.pdbqt / .log / .txt / receptor_<id>.pdb
exports/                    导出的复合物 PDB、PDF 矢量图、优化报告
exports/fishing/<id>/       fishing.json（ChEMBL 靶点垂钓原始汇总）
exports/netpharm/<id>/      netpharm.json（STRING 网络与 KEGG/GO 富集结果）
```

`project.json` 可直接备份或迁移；迁移后通过“工具 → 打开已有项目”选择目录即可。

---

## 11. 技术架构

- 后端：FastAPI + Uvicorn，任务线程 + Vina 子进程，日志流式回传。
- 化学：RDKit（结构标准化、构象、描述符、作用力几何）、Meeko（PDBQT、受体模板）。
- 对接：AutoDock Vina 1.2.5 命令行（`--scoring vina|vinardo`）。
- 前端：原生 HTML/CSS/JS + 3Dmol.js，本地静态资源，无前端构建步骤。
- 口袋预测：网格埋藏度 + 连通空腔的启发式算法（无需 fpocket）。

```
DockForge/                  安装目录名（软件名 DrugHX）
├─ app.py                    启动器
├─ DrugHX.bat                一键安装/启动
├─ DockForge.bat             兼容旧快捷方式
├─ requirements.txt
├─ bin/vina.exe
├─ dockforge/
│  ├─ api.py                 FastAPI 路由与任务
│  ├─ docking.py             Vina 调用与结果解析
│  ├─ receptors.py           PDB 下载、受体准备
│  ├─ ligands.py             配体读取、标准化、3D、PDBQT
│  ├─ boxes.py               对接盒与口袋预测
│  ├─ analysis.py            作用力分析与 RMSD
│  ├─ admet.py               ADMET、类药规则、QED、配体效率与雷达图 PDF
│  ├─ pharmacophore.py       药效团特征提取与残基互补性分析
│  ├─ mmgbsa.py              MM-GBSA 近似与残基能量分解
│  ├─ optimization.py        骨架跃迁、残基建议与自动跑分
│  ├─ design.py              靶向分子生成与数据库相似性检索
│  ├─ fishing.py             ChEMBL 反向靶点垂钓与 UniProt 基因映射
│  ├─ netpharm.py            STRING PPI 与 Enrichr KEGG/GO 富集
│  ├─ profiling.py           可开发性、化学空间、Murcko 骨架与代谢软点
│  ├─ figures.py             PDF 矢量图、优化/设计/垂钓/网络药理学/完整报告
│  ├─ pdbio.py               PDB/PDBQT 解析与转换
│  ├─ project.py             项目持久化
│  └─ jobs.py                后台任务
└─ web/                      index.html / app.js / style.css / 3Dmol
```

---

## 12. 使用建议与限制

- **打分不是结合自由能**。Vina 打分适合同一靶点内的相对排序，不能直接换算成实验亲和力。
- **ADMET 与 MM-GBSA 是经验近似**。ADMET 端点为基于理化性质与子结构的启发式模型，MM-GBSA 为快速力场近似；两者适合做同系列分子的相对排序与假设生成，不能替代实验测定或严谨的自由能计算。
- **口袋预测是启发式**。有共晶配体时优先用它；无共晶结构时建议结合文献、突变数据或 fpocket 结果确认位点。
- **缺失残基不会自动补全**。Meeko 会跳过无法识别的残基；若结合位点附近有缺失，建议先用 PDBFixer / MODELLER 修复。
- **金属离子**：勾选“保留金属离子”后会以 HETATM 写入 PDBQT；请确认离子类型和电荷符合你的体系。
- **pH 处理是规则式**：羧酸、磺酸、磷酸、四唑去质子化，脂肪胺、脒/胍基质子化；组氨酸、半胱氨酸等仍按模板处理。特殊体系请自行检查质子化状态。
- **大盒子**：Vina 1.2.5 支持大于 27000 Å³ 的盒子，但计算时间会明显增加；建议先用较小盒子做试探。
- **在线数据模块需要联网**：靶点垂钓依赖 ChEMBL 与 UniProt，网络药理学依赖 STRING 与 Enrichr；结果反映数据库当前收录的已知活性与注释，阴性结果不能排除靶点，阳性排序也不等于本分子的实测亲和力。
- **靶点垂钓只针对非共价小分子**：软件以 AutoDock Vina 为核心，不建模共价键、金属配位键或蛋白–蛋白界面；共价抑制剂与共价靶点需改用专用程序复核。
- **Windows 安全软件**可能误报下载的 `vina.exe`，请从官方 Release 获取或添加信任。

---

## 13. 论文级图输出

结果详情中的 **论文图** 按钮提供当前构象的 PDF 矢量图（2D 结构 + 作用力示意图）；**论文图表包** 会生成：

```
fig_publication_panel.pdf / .svg     双面板汇总图（a 打分排序，b 作用力谱），约 183 mm 双栏宽
fig_score_ranking.pdf / .svg         Vina 与 Vinardo 分组打分图
fig_interaction_heatmap.pdf / .svg   各配体的作用力计数热图
figure_binding_mode.pml              PyMOL 场景脚本（可选的光线追踪成图）
figure_caption.txt                   建议图注（含 CPU/GPU 方法与参数）
figure_source_data.csv               两个面板的源数据，便于统计与复现
```

- 图表使用 Arial/Helvetica 风格无衬线字体、白底、无多余网格线，尺寸按双栏图（约 183 mm）设计；所有交付图均为 **PDF 矢量图**（附 SVG 备份），不再输出 PNG。
- PyMOL 脚本使用相对路径，解压后在该目录运行 `pymol figure_binding_mode.pml`，可选地生成三维光线追踪图；投稿排版请优先使用 PDF/SVG 的 2D 结构、作用力与打分图。
- 打分图与热图基于当前对接记录的全部成功结果（最多 30 个）生成；作用力为几何规则计算，建议在论文方法中注明参数。
- 图注会区分 CPU 与 GPU 方法；GPU 记录会注明 Vina-GPU 搜索和 CPU Vina 复核步骤。

---

## 14. 引用

如果 DrugHX 对你的研究有帮助，请引用其调用的底层软件：

- Eberhardt J, Santos-Martins D, Tillack AF, Forli S. AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings. *J Chem Inf Model*. 2021. DOI: 10.1021/acs.jcim.1c00203
- Trott O, Olson AJ. AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. *J Comput Chem*. 2010. DOI: 10.1002/jcc.21334
- Ding J, Tang S, Mei Z, et al. Vina-GPU 2.0: Further Accelerating AutoDock Vina and Its Derivatives with Graphics Processing Units. *J Chem Inf Model*. 2023. DOI: 10.1021/acs.jcim.2c01504
- RDKit: Open-source cheminformatics. https://www.rdkit.org
- Meeko: preparation of small molecules and receptors for AutoDock. https://github.com/forlilab/Meeko

---

## 15. 致谢

感谢以下软件、数据库与方法作者（完整列表见软件“致谢”页）：

- **对接与计算**：AutoDock Vina、Vina-GPU、smina、QuickVina 2、Meeko、RDKit、Open Babel。
- **数据库与网络资源**：RCSB PDB、ChEMBL、BindingDB、DGIdb、PubChem、Open Targets、STRING、Enrichr、DisGeNET、DrugBank 与 FDA 标签。
- **评分与可视化**：SAscore、BOILED-Egg、CNS MPO、PAINS/Brenk/NIH、Bemis–Murcko、rdFMCS、matplotlib、PyMuPDF。
- **软件作者**：HX。
