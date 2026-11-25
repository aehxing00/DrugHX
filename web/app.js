/* DrugHX frontend */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const STEP_META = {
    auto: {
      eyebrow: "AUTO",
      title: "自动模式",
      subtitle: "上传一个受体和一个或多个配体，自动完成受体准备、口袋识别、配体准备、对接与全部分析，并把完整报告直接写入数据目录。",
    },
    receptor: {
      eyebrow: "STEP 01",
      title: "受体准备",
      subtitle: "导入 PDB 结构，自动去水、补氢并转换为 Vina 可用的 PDBQT。",
    },
    ligands: {
      eyebrow: "STEP 02",
      title: "配体准备",
      subtitle: "支持 SMILES、SDF、MOL、MOL2、PDB 与现成 PDBQT，自动生成 3D 构象并做能量最小化。",
    },
    site: {
      eyebrow: "STEP 03",
      title: "结合位点",
      subtitle: "优先使用共晶配体定义盒子，也可以采用内置口袋预测或手动坐标。",
    },
    dock: {
      eyebrow: "STEP 04",
      title: "分子对接",
      subtitle: "选择受体与配体，设置 Vina 参数后提交后台任务，界面全程保持响应。",
    },
    results: {
      eyebrow: "STEP 05",
      title: "结果分析",
      subtitle: "浏览打分与构象、分析蛋白–配体作用力，导出 CSV 或复合物结构。",
    },
    optimize: {
      eyebrow: "STEP 06",
      title: "药物优化",
      subtitle: "基于残基作用力谱生成骨架跃迁与生物电子等排方案，自动重新对接并输出 PDF 矢量图。",
    },
    design: {
      eyebrow: "STEP 07",
      title: "分子设计",
      subtitle: "针对指定蛋白口袋生成类药分子，自动对接筛选，并在本地化合物数据库中检索相似分子。",
    },
    fishing: {
      eyebrow: "STEP 08",
      title: "靶点垂钓",
      subtitle: "输入小分子，基于 ChEMBL 已知活性数据反向寻找潜在蛋白靶点。",
    },
    netpharm: {
      eyebrow: "STEP 09",
      title: "网络药理学",
      subtitle: "直接读取靶点垂钓的基因集，完成 STRING 网络、多库富集、C-T-D 网络与靶点–通路桑基图。",
    },
    synthesis: {
      eyebrow: "STEP 10",
      title: "可合成性",
      subtitle: "计算 SAscore、手性/大环硬步骤与模板逆合成路线，输出可合成性画像和合成路线图。",
    },
    admet: {
      eyebrow: "STEP 11",
      title: "ADMET",
      subtitle: "独立的吸收、分布、代谢、排泄与毒性分析，支持单分子与批量 SMILES，并对接完整报告。",
    },
    experts: {
      eyebrow: "STEP 12",
      title: "专家分析",
      subtitle: "代谢软点、组织分布、药效团、MM-GBSA、接触指纹、口袋画像与重对接验证，结果均可并入完整报告。",
    },
    ddi: {
      eyebrow: "STEP 13",
      title: "药物互作",
      subtitle: "默认与全部 FDA 参考药物比对，基于共享靶点、CYP 竞争、hERG 叠加与结构类别相似度评估联用风险。",
    },
    drugref: {
      eyebrow: "STEP 14",
      title: "FDA 参考",
      subtitle: "把候选分子与 FDA 已上市药物做结构、靶点、药动与 ADMET 对照，给出可解释的参考排序。",
    },
    pharm: {
      eyebrow: "STEP 15",
      title: "综合药理预测",
      subtitle: "物种感知的口服、过瘤胃、肠道菌群、稳定性、剂型、核酸结合、组织分布与综合风险，不依赖分子对接。",
    },
    online: {
      eyebrow: "STEP 16",
      title: "在线智能资源",
      subtitle: "无需 GPU 或 API Key：AlphaFold/ESMFold 结构获取、靶点情报、化合物相似性与通路富集。",
    },
  };

  const STATUS = {
    raw: ["待准备", "info"],
    preparing: ["准备中", "warn"],
    ready: ["已准备", "ok"],
    error: ["失败", "err"],
  };

  const RUN_STATUS = {
    running: "运行中",
    done: "完成",
    error: "失败",
    cancelled: "已取消",
    interrupted: "已中断",
  };

  const LIGAND_COLORS = {
    carbon: "#0f766e",
    oxygen: "#dc2626",
    nitrogen: "#1d4ed8",
    sulfur: "#b45309",
    hydrogen: "#f8fafc",
    fluorine: "#65a30d",
    chlorine: "#15803d",
    bromine: "#92400e",
  };
  const PROTEIN_COLORS = {
    carbon: "#94a3b8",
    oxygen: "#dc2626",
    nitrogen: "#1d4ed8",
    sulfur: "#b45309",
  };

  const S = {
    data: null,
    step: "receptor",
    receptorId: null,
    ligandId: null,
    ligandCache: {},
    runId: null,
    resultIndex: 0,
    mode: 1,
    analysis: null,
    analysisKey: "",
    rmsd: null,
    rmsdKey: "",
    poseCheck: null,
    poseCheckKey: "",
    poseBatch: null,
    poseBusy: false,
    mpo: null,
    mpoBusy: false,
    assay: null,
    assayProjectKey: null,
    assayBusy: false,
    dose: null,
    doseBusy: false,
    trust: null,
    trustBusy: false,
    mmp: null,
    mmpBusy: false,
    pbpkData: null,
    pbpkBusy: false,
    active: null,
    activeBusy: false,
    benchmark: null,
    benchmarkBusy: false,
    selectedReceptors: new Set(),
    selectedLigands: new Set(),
    knownReceptors: new Set(),
    knownLigands: new Set(),
    selectionInit: false,
    projectKey: null,
    autoSelect: true,
    ligandMode: "fine",
    ligandModeInit: false,
    fineLigands: null,
    fineTotal: 0,
    fineOffset: 0,
    fineQuery: "",
    bulkPresetId: null,
    bulkReceptorId: null,
    bulkTop: null,
    bulkTopN: 50,
    recOptions: null,
    ligOptions: null,
    job: null,
    jobTimer: null,
    jobFailures: 0,
    pendingRunId: null,
    logs: [],
    viewerSurface: false,
    viewerToken: 0,
    modalOpen: false,
    enumResults: [],
    optResult: null,
    designResult: null,
    designReceptorId: null,
    fishingResult: null,
    fishingSmiles: "",
    fishingName: "",
    fishingTouched: false,
    fishingLigandId: null,
    fishingExternalGenes: "",
    netpharmResult: null,
    netpharmFishingId: null,    synthesisResult: null,
    admetResult: null,
    admetBatch: [],
    admetErrors: [],
    expertsData: null,
    expertsKey: "",
    expertsLoading: false,
    validationData: null,
    ddiResult: null,
    drugrefResult: null,
    pharmResult: null,
    pharmGenes: "",
    pharmSpecies: "human",
    onlineMode: "structure",
    onlineStructure: null,
    onlineTarget: null,
    onlineCompound: null,
    onlineEnrich: null,
    onlineBusy: "",
    autoResult: null,
    autoPick: null,
    pendingReportUrl: null,
  };

  const FINE_LIMIT = 200;

  const FILTER_LIMIT_DEFAULTS = {
    mw: 500, logp: 4.5, tpsa: 140, hbd: 5, hba: 10,
    rotb: 10, heavy: 45, rings: 6, max_ring: 8, qed: 0.30,
  };

  function filterLimitGrid(limits) {
    const values = { ...FILTER_LIMIT_DEFAULTS, ...(limits || {}) };
    const fields = [
      ["mw", "分子量上限", 10],
      ["logp", "cLogP 上限", 0.1],
      ["tpsa", "TPSA 上限", 5],
      ["hbd", "氢键供体上限", 1],
      ["hba", "氢键受体上限", 1],
      ["rotb", "可旋转键上限", 1],
      ["heavy", "重原子上限", 1],
      ["rings", "环数上限", 1],
      ["max_ring", "最大环上限", 1],
      ["qed", "QED 下限", 0.05],
    ];
    return fields.map(([key, label, step]) => `<div class="field">
      <label>${label}</label>
      <input class="input" type="number" data-filter-limit="${key}" step="${step}" value="${Number(values[key])}">
    </div>`).join("");
  }


  let viewer = null;

  /* ------------------------------------------------------------------ utils */
  const esc = (v) =>
    String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function fmtTime(value) {
    if (!value) return "—";
    const d = new Date(String(value).replace(" ", "T"));
    if (Number.isNaN(d.getTime())) return value;
    return `${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  }

  const fmt = (v, n = 2) => (typeof v === "number" ? v.toFixed(n) : esc(v));

  function toast(message, type = "info", ms = 4200) {
    const root = $("#toast-root");
    const node = document.createElement("div");
    node.className = `toast ${type}`;
    node.textContent = message;
    root.appendChild(node);
    setTimeout(() => {
      node.style.opacity = "0";
      node.style.transform = "translateY(6px)";
      node.style.transition = "0.2s";
      setTimeout(() => node.remove(), 220);
    }, ms);
  }

  async function api(path, options = {}) {
    const opts = { ...options };
    if (opts.body && !(opts.body instanceof FormData) && typeof opts.body !== "string") {
      opts.body = JSON.stringify(opts.body);
    }
    if (opts.body && !(opts.body instanceof FormData) && !opts.headers) {
      opts.headers = { "Content-Type": "application/json" };
    }
    const response = await fetch(path, opts);
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const data = await response.json();
        detail = data.detail || detail;
      } catch (_) {
        try { detail = (await response.text()) || detail; } catch (__) { /* ignore */ }
      }
      throw new Error(detail);
    }
    const type = response.headers.get("content-type") || "";
    return type.includes("application/json") ? response.json() : response.text();
  }

  async function fetchText(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`加载失败（HTTP ${response.status}）`);
    return response.text();
  }

  const ligandById = (id) => {
    if (!id) return null;
    return (
      (S.data?.ligands || []).find((l) => l.id === id)
      || (S.data?.fine_ligands || []).find((l) => l.id === id)
      || S.ligandCache[id]
      || null
    );
  };

  const knownLigands = () => {
    const map = new Map();
    [
      ...(S.data?.ligands || []),
      ...(S.data?.fine_ligands || []),
      ...Object.values(S.ligandCache || {}),
    ].forEach((item) => {
      if (item?.id) map.set(item.id, { ...(map.get(item.id) || {}), ...item });
    });
    return [...map.values()];
  };

  const resultLigand = (result) => {
    if (!result) return null;
    const local = ligandById(result.ligand_id);
    const smiles = local?.smiles || local?.meta?.smiles || result.ligand_smiles || "";
    if (local) return { ...local, smiles };
    if (!result.ligand_id && !result.ligand_name) return null;
    return {
      id: result.ligand_id || "",
      name: result.ligand_name || "",
      smiles,
      meta: {},
    };
  };

  const activeReceptor = () => (S.data?.receptors || []).find((r) => r.id === S.receptorId) || null;
  const activeLigand = () => resultLigand(activeResult()) || ligandById(S.ligandId);
  const activeRun = () => (S.data?.runs || []).find((r) => r.id === S.runId) || null;
  const activeResult = () => {
    const run = activeRun();
    return run?.results?.[S.resultIndex] || null;
  };
  const dockableReceptor = (r) => r.status === "ready" && r.has_pdbqt && Boolean(r.box);

  // Best-scoring successful row (largest |affinity|); falls back to row 0.
  function bestResultIndex(run) {
    const rows = run?.results || [];
    let best = -1;
    let bestScore = -1;
    rows.forEach((row, index) => {
      const value = Number(row?.affinity);
      if (row?.status !== "ok" || !Number.isFinite(value)) return;
      if (best < 0 || Math.abs(value) > bestScore) {
        best = index;
        bestScore = Math.abs(value);
      }
    });
    return best >= 0 ? best : 0;
  }

  // Selecting a result updates the global ligand default for every downstream
  // module (optimisation, design, fishing, ADMET, DDI, FDA reference, pharm).
  function applyResultSelection(index) {
    const result = activeRun()?.results?.[Number(index)] || null;
    S.resultIndex = Number(index);
    if (result?.ligand_id) {
      S.ligandId = result.ligand_id;
      S.fishingLigandId = result.ligand_id;
      S.fishingName = result.ligand_name || "";
      S.fishingSmiles = result.ligand_smiles || "";
    } else {
      S.fishingLigandId = null;
      S.fishingName = "";
      S.fishingSmiles = "";
    }
    S.fishingTouched = false;
    S.mode = 1;
    S.analysis = null;
    S.analysisKey = "";
    S.rmsd = null;
    S.poseCheck = null;
    S.poseCheckKey = "";
  }

  function badge(status) {
    const [label, cls] = STATUS[status] || [status || "未知", "info"];
    return `<span class="badge ${cls}">${esc(label)}</span>`;
  }

  function emptyState(title, text, icon = "flask") {
    const iconSvg = icon === "search"
      ? `<svg viewBox="0 0 24 24" fill="none"><circle cx="10.5" cy="10.5" r="6.5" stroke="currentColor" stroke-width="1.5"/><path d="M15.5 15.5L21 21" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="none"><path d="M9 3h6M10 3v6.2L4.8 18a2 2 0 001.7 3h11a2 2 0 001.7-3L14 9.2V3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M7 15h10" stroke="currentColor" stroke-width="1.3"/></svg>`;
    return `<div class="empty-state"><div class="empty-icon">${iconSvg}</div><h4>${esc(title)}</h4><p>${text}</p></div>`;
  }

  /* ------------------------------------------------------------------ boot */
  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    initViewer();
    bindEvents();
    const saved = localStorage.getItem("dockforge_step");
    if (saved && STEP_META[saved]) S.step = saved;
    await refresh();
    if (S.step === "results") loadAnalysis(true);
    if (S.data?.active_jobs?.length) {
      const job = S.data.active_jobs[0];
      S.job = { id: job.id, kind: job.kind, title: job.title, since: job.log_index || 0 };
      openConsole(true);
      setJobUi(job);
      appendLogs(job.logs || []);
      schedulePoll(300);
    }
  }

  function initViewer() {
    const el = $("#viewer");
    if (!window.$3Dmol) {
      el.innerHTML = `<div style="padding:24px;color:#f87171;font-size:12px">3Dmol.js 加载失败，请检查 web/vendor 目录。</div>`;
      return;
    }
    viewer = window.$3Dmol.createViewer(el, {
      backgroundColor: "#ffffff",
      antialias: true,
      backgroundAlpha: 1,
    });
    viewer.setBackgroundColor("#ffffff");
    window.addEventListener("resize", () => viewer && viewer.resize());
    if (window.ResizeObserver) {
      let timer = null;
      new ResizeObserver(() => {
        clearTimeout(timer);
        timer = setTimeout(() => {
          if (!viewer) return;
          viewer.resize();
          viewer.render();
        }, 90);
      }).observe(el);
    }
  }

  async function refresh() {
    try {
      S.data = await api("/api/state");
      hydrate();
      render();
    } catch (error) {
      $("#panel-body").innerHTML = `<div class="note err">无法连接后端：${esc(error.message)}</div>`;
    }
  }

  function hydrate() {
    const d = S.data;
    if (!d) return;

    // A paused automatic pipeline survives a page refresh: rebuild the
    // candidate panel from the project so the run can still be confirmed.
    const pending = d.auto_pending || {};
    if (!S.autoResult && pending.run_id && (pending.candidates || []).length) {
      const first = pending.candidates[0] || {};
      S.autoResult = {
        awaiting: true,
        run_id: pending.run_id,
        project: d.name,
        project_path: d.path,
        report_dir: `${d.path}\\exports\\full_report`,
        screened: pending.screened,
        fine_count: pending.fine_count,
        best_ligand: first.name,
        best_affinity: first.affinity,
        candidates: pending.candidates,
        online: pending.online || {},
      };
      S.step = "auto";
    }

    const projectKey = String(d.path || d.name || "");
    if (S.projectKey && S.projectKey !== projectKey) {
      S.selectedReceptors.clear();
      S.selectedLigands.clear();
      S.knownReceptors.clear();
      S.knownLigands.clear();
      S.selectionInit = false;
      S.autoSelect = true;
      S.fineLigands = null;
      S.bulkTop = null;
      S.ligandId = null;
      S.ligandCache = {};
      S.runId = null;
      S.resultIndex = 0;
      S.fishingLigandId = null;
      S.fishingName = "";
      S.fishingSmiles = "";
      S.fishingTouched = false;
    }
    S.projectKey = projectKey;
    if (S.assayProjectKey !== projectKey) {
      S.assayProjectKey = projectKey;
      S.assay = null;
      S.mpo = null;
      S.dose = null;
      S.trust = null;
      S.mmp = null;
      S.active = null;
      S.benchmark = null;
      S.pbpkData = d.pbpk && Object.keys(d.pbpk).length ? d.pbpk : null;
      api("/api/projects/current/assays")
        .then((data) => {
          if (data && (data.rows || data.summary)) {
            S.assay = data;
            if (S.step === "results") render({ viewer: false });
          }
        })
        .catch(() => {});
    }

    if (!S.ligandModeInit && d.ligand_total !== undefined) {
      S.ligandMode = Number(d.ligand_total) > FINE_LIMIT ? "bulk" : "fine";
      S.ligandModeInit = true;
    }

    if (!S.recOptions) S.recOptions = { ...(d.receptor_options || {}) };
    if (!S.ligOptions) S.ligOptions = { ...(d.ligand_options || {}) };

    if (!d.receptors.some((r) => r.id === S.receptorId)) {
      const preferred = d.receptors.find((r) => r.status === "ready") || d.receptors[0];
      S.receptorId = preferred?.id || null;
    }
    if (!S.ligandId) {
      const preferred = d.ligands.find((l) => l.status === "ready") || d.ligands[0];
      S.ligandId = preferred?.id || null;
    }

    if (!S.selectionInit) {
      d.receptors.filter(dockableReceptor).forEach((r) => S.selectedReceptors.add(r.id));
      d.ligands.filter((l) => l.status === "ready").slice(0, 10).forEach((l) => S.selectedLigands.add(l.id));
      d.receptors.forEach((r) => S.knownReceptors.add(r.id));
      d.ligands.forEach((l) => S.knownLigands.add(l.id));
      S.selectionInit = true;
    } else {
      d.receptors.forEach((r) => {
        if (dockableReceptor(r) && !S.knownReceptors.has(r.id)) S.selectedReceptors.add(r.id);
        S.knownReceptors.add(r.id);
      });
      d.ligands.forEach((l) => {
        if (
          l.status === "ready"
          && !S.knownLigands.has(l.id)
          && S.selectedLigands.size < 10
        ) {
          S.selectedLigands.add(l.id);
        }
        S.knownLigands.add(l.id);
      });
    }
    if (S.autoSelect) {
      S.selectedReceptors = new Set(d.receptors.filter(dockableReceptor).map((r) => r.id));
      S.selectedLigands = new Set(
        d.ligands.filter((l) => l.status === "ready").slice(0, 10).map((l) => l.id)
      );
    }

    const validRec = new Set(d.receptors.map((r) => r.id));
    S.selectedReceptors = new Set([...S.selectedReceptors].filter((id) => validRec.has(id)));
    // d.ligands is only the current 300-row page; keep off-page selections
    // loaded from a bulk-screening top-N list.

    const runsWithResults = d.runs.filter((r) => (r.results || []).length);
    if (!d.runs.some((r) => r.id === S.runId)) {
      S.runId = runsWithResults[0]?.id || d.runs[0]?.id || null;
      S.resultIndex = bestResultIndex(activeRun());
      S.mode = 1;
    }
    const run = activeRun();
    if (run && S.resultIndex >= (run.results || []).length) {
      S.resultIndex = bestResultIndex(run);
      S.mode = 1;
    }
    // The ligand chosen in "结果分析" is the single source of truth for every
    // downstream module; keep it even when it is outside the 300-row page.
    const picked = activeResult();
    if (picked?.ligand_id) {
      S.ligandId = picked.ligand_id;
      if (!S.fishingTouched) {
        S.fishingLigandId = picked.ligand_id;
        S.fishingName = picked.ligand_name || "";
        S.fishingSmiles = picked.ligand_smiles || "";
      }
    }
  }

  /* ------------------------------------------------------------------ render */
  function render(options = {}) {
    const d = S.data;
    if (!d) return;
    const system = d.system || {};
    $("#project-name").textContent = d.name;
    $("#project-chip").title = d.path;
    $("#sys-version").textContent = `DrugHX v${system.app_version || "1.0.0"}`;
    $("#sys-engine").textContent = `RDKit ${system.rdkit || "?"} · Meeko ${system.meeko || "?"}`;

    const vinaOk = Boolean(system.vina_version);
    const pill = $("#vina-status");
    pill.classList.toggle("bad", !vinaOk);
    pill.querySelector("span:last-child").textContent = vinaOk ? "Vina 就绪" : "Vina 未找到";
    pill.title = vinaOk ? system.vina_version : `请将 vina.exe 放入 ${system.vina_path || "bin/"}`;

    renderNav();
    renderStepHead();
    renderPanel();
    if (options.viewer !== false) syncViewer();
    if (S.job) updateJobCard();
  }

  function renderNav() {
    const d = S.data;
    $$(".step").forEach((btn) => btn.classList.toggle("active", btn.dataset.step === S.step));
    const states = {
      auto: S.autoResult ? "complete" : "",
      receptor: d.receptors.some((r) => r.status === "ready") ? "complete" : d.receptors.length ? "partial" : "",
      ligands: d.ligands.some((l) => l.status === "ready") ? "complete" : d.ligands.length ? "partial" : "",
      site: d.receptors.some((r) => r.box) ? "complete" : "",
      dock: d.runs.length ? "complete" : "",
      results: d.runs.some((r) => (r.results || []).length) ? "complete" : "",
      optimize: (d.optimizations || []).length ? "complete" : "",
      design: (d.designs || []).length ? "complete" : "",
      fishing: (d.fishing || []).length ? "complete" : "",
      netpharm: (d.netpharm || []).length ? "complete" : "",
      synthesis: (d.synthesis || []).length ? "complete" : "",
      admet: S.admetResult || (S.admetBatch || []).length ? "complete" : "",
      experts: d.runs.some((r) => (r.results || []).length) ? "complete" : "",
      ddi: (d.ddi || []).length ? "complete" : "",
      drugref: (d.drugref || []).length ? "complete" : "",
      pharm: S.pharmResult ? "complete" : "",
    };
    $$(".step").forEach((btn) => {
      const dot = btn.querySelector(".step-state");
      dot.className = `step-state ${states[btn.dataset.step] || ""}`;
    });
  }

  function renderStepHead() {
    const meta = STEP_META[S.step];
    $("#step-eyebrow").textContent = meta.eyebrow;
    $("#step-title").textContent = meta.title;
    $("#step-subtitle").textContent = meta.subtitle;
    const actions = $("#step-actions");
    if (S.step === "results") {
      actions.innerHTML = "";
    } else if (S.step === "receptor" && S.data.receptors.length) {
      actions.innerHTML = `<span class="chip">${S.data.receptors.filter((r) => r.status === "ready").length}/${S.data.receptors.length} 已准备</span>`;
    } else if (S.step === "ligands" && S.data.ligands.length) {
      actions.innerHTML = `<span class="chip">${S.data.ligands.filter((l) => l.status === "ready").length}/${S.data.ligands.length} 已准备</span>`;
    } else {
      actions.innerHTML = "";
    }
  }

  function renderPanel() {
    const fn = {
      auto: renderAutoStep,
      receptor: renderReceptorStep,
      ligands: renderLigandStep,
      site: renderSiteStep,
      dock: renderDockStep,
      results: renderResultsStep,
      optimize: renderOptimizeStep,
      design: renderDesignStep,
      fishing: renderFishingStep,
      netpharm: renderNetpharmStep,
      synthesis: renderSynthesisStep,
      admet: renderAdmetStep,
      experts: renderExpertsStep,
      ddi: renderDdiStep,
      drugref: renderDrugrefStep,
      pharm: renderPharmStep,
      online: renderOnlineStep,
    }[S.step];
    $("#panel-body").innerHTML = fn ? fn() : "";
  }

  /* ------------------------------------------------------------------ auto step */
  function renderAutoStep() {
    const d = S.data || {};
    const running = S.job && S.job.status === "running";
    const jobCard = S.job
      ? `<div class="job-card" id="auto-job">
           <div class="job-icon ${running ? "spin" : ""}">
             <svg viewBox="0 0 24 24" fill="none"><path d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6l2.1 2.1m0-12.8l-2.1 2.1M7.7 16.3l-2.1 2.1" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>
           </div>
           <div class="job-main">
             <div class="job-title">${esc(S.job.title || "全自动分析")}</div>
             <div class="job-msg" id="auto-job-msg">${esc(S.job.message || "")}</div>
             <div class="progress" style="margin-top:7px;max-width:none"><div class="progress-fill" id="auto-progress" style="width:${Math.round((S.job.progress || 0) * 100)}%"></div></div>
           </div>
           <button class="btn danger sm" data-action="cancel-job" ${running ? "" : "disabled"}>取消</button>
         </div>`
      : "";
    const result = S.autoResult;
    const awaiting = Boolean(result?.awaiting);
    const candidates = result?.candidates || [];
    const chosen = candidates.find((item) => item.ligand_id === S.autoPick) || null;
    const candidateCard = (item, active) => `
      <div class="candidate${active ? " active" : ""}" data-action="auto-candidate" data-id="${esc(item.ligand_id)}" role="button" tabindex="0">
        <img src="${esc(item.svg)}" alt="${esc(item.name)}" loading="lazy">
        <div class="candidate-name" title="${esc(item.name)}">${esc(item.name)}</div>
        <div class="candidate-meta">综合 ${item.decision_score != null ? Number(item.decision_score).toFixed(1) : "—"} · ${item.affinity != null ? `${Number(item.affinity).toFixed(2)} kcal/mol` : "—"}</div>
        <div class="candidate-meta">MW ${item.mw != null ? Number(item.mw).toFixed(0) : "—"} · cLogP ${item.logp != null ? Number(item.logp).toFixed(2) : "—"}</div>
        <div class="candidate-meta">QED ${item.qed != null ? Number(item.qed).toFixed(2) : "—"} · LE ${item.le != null ? Number(item.le).toFixed(2) : "—"}</div>
        <div class="candidate-meta">构象校验 ${item.pose_pass != null ? `${Math.round(Number(item.pose_pass) * 100)}%` : "未校验"}</div>
      </div>`;
    const candidateSection = awaiting
      ? `<div class="divider"></div>
         <div class="row" style="justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px">
           <div><strong>候选分子（${candidates.length}）</strong>
             <div class="hint">按综合决策分数排序（对接 45%、构象有效性 20%、配体效率 12%、QED 13%、合成可及性 10%）。点击卡片查看结构、对接与成药性参数。当前选择：<strong>${chosen ? esc(chosen.name) : "未选择"}</strong></div>
           </div>
           <div class="row" style="gap:8px">
             <button class="btn ghost" data-action="auto-pick-best" ${candidates.length ? "" : "disabled"}>选打分最优</button>
             <button class="btn primary" data-action="auto-continue" ${S.autoPick ? "" : "disabled"}>确认并继续</button>
           </div>
         </div>
         <div class="candidate-grid">${candidates.map((item) => candidateCard(item, S.autoPick === item.ligand_id)).join("")}</div>`
      : (result?.report
        ? `<div class="note" style="margin-top:10px">报告路径：<span class="mono" style="word-break:break-all">${esc(result.report || "")}</span></div>`
        : "");
    const resultCard = result
      ? `<div class="card">
           <div class="card-head">
             <div><h3>${awaiting ? "精细对接完成，请确认候选分子" : "全自动分析完成"}</h3>
               <p>${awaiting ? "点击卡片查看化学结构、对接打分与成药性参数；确认后才会继续优化、设计与报告。" : "项目与完整报告已写入数据目录，可直接下载；也可以继续在其它模块中深入分析。"}</p></div>
             <span class="badge">${awaiting ? "待确认" : "已生成"}</span>
           </div>
           <div class="card-body">
             <div class="summary-strip">
               <div class="item"><span>项目</span><strong>${esc(result.project || "")}</strong></div>
               <div class="item"><span>精细对接</span><strong class="mono">${esc(result.run_id || "")}</strong></div>
               <div class="item"><span>配体</span><strong>${result.screened ?? ""} → ${result.fine_count ?? ""}</strong></div>
               <div class="item"><span>最优分子</span><strong>${esc(result.best_ligand || "")}${result.best_affinity != null ? ` · ${esc(String(result.best_affinity))} kcal/mol` : ""}</strong></div>
               ${result.optimization_id ? `<div class="item"><span>优化记录</span><strong class="mono">${esc(result.optimization_id)}</strong></div>` : ""}
               ${result.design_id ? `<div class="item"><span>设计记录</span><strong class="mono">${esc(result.design_id)}</strong></div>` : ""}
             </div>
             ${candidateSection}
             ${(result.online?.receptor || result.online?.target || result.online?.compound)
               ? `<div class="note" style="margin-top:10px">
                    <strong>在线情报：</strong>
                    ${result.online?.receptor ? `${esc(result.online.receptor.source || "")}（${esc(result.online.receptor.gene || result.online.receptor.accession || "")}，pLDDT ${esc(result.online.receptor.plddt ?? "—")}）` : ""}
                    ${result.online?.target ? `；靶点 ${esc(result.online.target.gene || "")}，疾病关联 ${esc(result.online.target.disease_count ?? "—")} 条，实验结构 ${esc(result.online.target.pdb_count ?? "—")} 个` : ""}
                    ${result.online?.compound ? `；最优分子 PubChem CID ${esc(result.online.compound.cid ?? "—")}，相似物 ${(result.online.compound.similar || []).length} 个` : ""}
                  </div>`
               : ""}
             ${awaiting ? "" : `<div class="row" style="margin-top:12px;flex-wrap:wrap">
               ${result.reports?.zh ? `<a class="btn primary" href="/api/auto/report?run_id=${encodeURIComponent(result.run_id)}&lang=zh" target="_blank" rel="noopener">下载中文报告</a>` : ""}
               ${result.reports?.en ? `<a class="btn" href="/api/auto/report?run_id=${encodeURIComponent(result.run_id)}&lang=en" target="_blank" rel="noopener">下载英文报告</a>` : ""}
               <button class="btn ghost" data-action="auto-reveal">打开项目目录</button>
             </div>`}
             <div class="note" style="margin-top:10px">项目目录：<span class="mono" style="word-break:break-all">${esc(result.project_path || "")}</span></div>
             ${awaiting ? "" : `<div class="note" style="margin-top:10px;border:1px solid rgba(0,0,0,.12);border-radius:10px;padding:10px">
               <strong>报告输出文件夹</strong>
               <div class="mono" style="word-break:break-all;margin-top:4px">${esc(result.report_dir || "")}</div>
               <button class="btn primary sm" style="margin-top:8px" data-action="auto-reveal-report">打开报告文件夹</button>
             </div>`}
           </div>
         </div>`
      : "";
    return `
      ${d.system?.vina_version ? "" : `<div class="note err" style="margin-bottom:14px">未找到 Vina 可执行文件，自动模式无法对接。请将 vina.exe 放入 <span class="mono">bin/</span> 或设置 VINA_PATH。</div>`}
      <div class="card">
        <div class="card-head"><div><h3>一键全流程</h3><p>上传一个受体和一个或多个配体，软件会自动完成全部步骤并把完整报告写入项目的数据目录。</p></div></div>
        <div class="card-body">
          <form data-form="auto-run">
            <div class="params-grid">
              <div class="field"><label>项目名称</label><input class="input" id="auto-project-name" placeholder="留空则自动命名"></div>
              <div class="field"><label>报告语言</label>
                <select class="select" id="auto-lang">
                  <option value="both" selected>中英双语（两份 PDF）</option>
                  <option value="zh">仅中文</option>
                  <option value="en">仅英文</option>
                </select>
              </div>
            </div>
            <div class="field" style="margin-top:12px"><label>受体文件（1 个，.pdb / .ent / .pdbqt）</label><input class="input" type="file" id="auto-receptor" accept=".pdb,.ent,.pdbqt,.txt"></div>
            <div class="params-grid" style="margin-top:10px">
              <div class="field"><label>或：基因 / UniProt 登录号（AlphaFold DB）</label><input class="input mono" id="auto-receptor-query" placeholder="例如 EGFR 或 P00533"></div>
              <div class="field"><label>或：PDB ID（RCSB 下载，4 位）</label><input class="input mono" id="auto-receptor-pdb" placeholder="例如 6LU7" maxlength="8" style="text-transform:uppercase"></div>
              <div class="field"><label>物种</label>
                <select class="select" id="auto-species">
                  <option value="human">人</option><option value="rat">大鼠</option><option value="pig">猪</option><option value="cattle">牛</option><option value="sheep">羊</option><option value="dog">犬</option><option value="cat">猫</option><option value="dog">犬</option><option value="cat">猫</option>
                </select>
              </div>
            </div>
            <div class="field" style="margin-top:8px"><label>或：氨基酸序列（ESMFold 在线预测，建议 ≤600 aa；请勿提交人类遗传资源或个人信息）</label><textarea class="textarea mono" id="auto-receptor-sequence" rows="3" placeholder="MKTAYIAKQRQISFVKSHFSRQ..."></textarea></div>
            <label class="check" style="margin-top:10px"><input type="checkbox" id="auto-online-intel" checked> 自动获取在线靶点与最优分子情报（需联网，可关闭）</label>
            <label class="check" style="margin-top:10px"><input type="checkbox" id="auto-filter" checked> 配体准备前启用成药性过滤（总开关）</label>
            <div id="auto-filter-grid" class="params-grid" style="margin-top:8px">${filterLimitGrid(S.ligOptions?.filter_limits)}</div>
            <div class="field" style="margin-top:12px"><label>配体文件（可多选，.sdf / .mol / .mol2 / .smi / .pdb / .pdbqt）</label><input class="input" type="file" id="auto-ligands" accept=".sdf,.sd,.mol,.mol2,.smi,.smiles,.pdb,.pdbqt" multiple></div>
            <div class="row" style="margin-top:14px;flex-wrap:wrap">
              <button class="btn primary" type="submit" ${running ? "disabled" : ""}>开始全自动分析</button>
              <span class="hint">自动选择最可能的口袋；配体超过 10 个时先用${d.system?.gpu_available ? " GPU" : "多线程 CPU"} 粗筛前十，再进行多工具共识精细对接。自动模式固定使用经过验证的默认参数；需要调整盒子、引擎、exhaustiveness、质子化等参数时，请使用左侧逐步模块。</span>
            </div>
          </form>
        </div>
      </div>

      <div class="card">
        <div class="card-head"><div><h3>自动执行的实验</h3><p>无需逐步点击，下面所有环节都会按顺序自动完成。</p></div></div>
        <div class="card-body">
          <div class="summary-strip">
            <div class="item"><span>1 受体</span><strong>文件 / PDB ID / AlphaFold / ESMFold，自动准备 PDBQT</strong></div>
            <div class="item"><span>2 口袋</span><strong>自动选择最可能的口袋与对接盒</strong></div>
            <div class="item"><span>3 配体</span><strong>去盐、质子化、构象与 PDBQT</strong></div>
            <div class="item"><span>4 粗筛</span><strong>&gt;10 个配体时 GPU 粗对接取前十</strong></div>
            <div class="item"><span>5 精细</span><strong>多工具共识对接，选最优分子</strong></div>
            <div class="item"><span>6 确认</span><strong>人工查看结构、对接与成药性参数后选定分子</strong></div>
            <div class="item"><span>7 拓展</span><strong>优化、设计、在线情报、ADMET 与综合分析</strong></div>
            <div class="item"><span>8 输出</span><strong>完整 PDF 报告写入项目 exports</strong></div>
          </div>
        </div>
      </div>

      ${jobCard}
      ${resultCard}`;
  }

  /* ------------------------------------------------------------------ receptor step */
  function renderReceptorStep() {
    const d = S.data;
    const rec = activeReceptor();
    const fetchCard = `
      <div class="card">
        <div class="card-head"><div><h3>从 RCSB PDB 获取</h3><p>输入 4 位 PDB ID，自动下载结构文件。</p></div></div>
        <div class="card-body">
          <form class="row" data-form="fetch-pdb">
            <input class="input mono" id="pdb-id" placeholder="例如 3PTB" maxlength="8" style="max-width:180px;text-transform:uppercase">
            <button class="btn primary" type="submit">获取结构</button>
          </form>
          <div class="hint" style="margin-top:9px">需要联网；也可以直接上传本地文件。</div>
        </div>
      </div>`;
    const uploadCard = `
      <div class="card">
        <div class="card-head"><div><h3>上传受体结构</h3><p>支持 .pdb / .ent（原始结构）或 .pdbqt（已准备）。</p></div></div>
        <div class="card-body">
          <label class="dropzone" id="receptor-drop">
            <input type="file" data-upload="receptor" accept=".pdb,.ent,.pdbqt,.txt" multiple>
            <svg viewBox="0 0 24 24" fill="none"><path d="M12 16V4m0 0L7 9m5-5l5 5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
            <strong>拖拽文件到这里，或点击选择</strong>
            <span>可一次选择多个受体</span>
          </label>
        </div>
      </div>`;

    const list = d.receptors.length
      ? `<div class="card">
           <div class="card-head"><div><h3>受体列表</h3><p>点击卡片切换当前受体；准备完成后会自动识别口袋并生成默认对接盒。</p></div><span class="badge">${d.receptors.length}</span></div>
           <div class="card-body"><div class="entity-list">${d.receptors.map(receptorCard).join("")}</div></div>
         </div>`
      : `<div class="card"><div class="card-body">${emptyState("还没有受体", "从 PDB 下载或上传本地结构开始。", "flask")}</div></div>`;

    return `<div class="grid-2">${fetchCard}${uploadCard}</div>${list}${rec ? receptorOptionsCard(rec) : ""}`;
  }

  function receptorCard(rec) {
    const meta = rec.meta || {};
    const chains = meta.chains || [];
    const bits = [
      chains.length ? `${chains.length} 条链 (${chains.join(", ")})` : "",
      meta.residues ? `${meta.residues} 残基` : "",
      meta.atoms ? `${meta.atoms} 原子` : "",
      meta.resolution ? `分辨率 ${Number(meta.resolution).toFixed(2)} Å` : "",
    ].filter(Boolean);
    const box = rec.box;
    const boxText = box
      ? `对接盒：${esc(box.label || "已设置")} · ${box.size.map((v) => Number(v).toFixed(1)).join(" × ")} Å`
      : "尚未设置对接盒";
    return `
      <div class="entity-card ${rec.id === S.receptorId ? "active" : ""}" data-action="rec-select" data-id="${rec.id}">
        <div class="entity-main">
          <div class="entity-title">${esc(rec.name)} ${badge(rec.status)}${rec.pdb_id ? `<span class="mono">${esc(rec.pdb_id)}</span>` : ""}</div>
          <div class="entity-meta">${bits.map((b) => `<span>${esc(b)}</span>`).join("")}</div>
          <div class="entity-sub">${boxText}${rec.pockets?.length ? ` · ${rec.pockets.length} 个候选口袋` : ""}${rec.message ? ` · ${esc(rec.message)}` : ""}</div>
        </div>
        <div class="entity-actions">
          <button class="btn ghost" data-action="rec-info" data-id="${rec.id}">信息</button>
          ${rec.meta?.from_pdbqt
            ? `<span class="chip">外部 PDBQT</span>`
            : rec.status === "raw" || rec.status === "error"
              ? `<button class="btn primary" data-action="prep-receptor" data-id="${rec.id}">准备</button>`
              : `<button class="btn ghost" data-action="prep-receptor" data-id="${rec.id}">重跑</button>`}
          <button class="btn danger" data-action="rec-delete" data-id="${rec.id}">删除</button>
        </div>
      </div>`;
  }

  function receptorOptionsCard(rec) {
    if (rec.meta?.from_pdbqt) {
      return `<div class="card"><div class="card-head"><div><h3>准备选项 · ${esc(rec.name)}</h3><p>该受体直接来自已准备的 PDBQT 文件，DrugHX 不会再改写其电荷与原子类型。</p></div><span class="badge ok">外部 PDBQT</span></div>
        <div class="card-body"><div class="note"><svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M12 8h.01M11 12h1v5h1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg><span>可直接在“结合位点”步骤设置对接盒，然后参与对接。</span></div></div></div>`;
    }
    const opts = S.recOptions || {};
    const chains = rec.meta?.chains || [];
    const selected = opts.chains || [];
    const chainButtons = chains.length
      ? `<div class="chain-chips">
           <button type="button" class="chain-chip ${selected.length === 0 ? "on" : ""}" data-action="chain-all" data-id="${rec.id}">全部</button>
           ${chains.map((c) => `<button type="button" class="chain-chip ${selected.includes(c) ? "on" : ""}" data-action="toggle-chain" data-chain="${esc(c)}" data-id="${rec.id}">${esc(c)}</button>`).join("")}
         </div>`
      : `<span class="muted" style="font-size:11px">准备后显示链信息</span>`;
    return `
      <div class="card">
        <div class="card-head">
          <div><h3>准备选项 · ${esc(rec.name)}</h3><p>移除水与杂原子后，Meeko 会按残基模板补全极性氢并分配 Gasteiger 电荷。</p></div>
          <span class="badge ${rec.status === "ready" ? "ok" : "info"}">${rec.status === "ready" ? "已准备" : "待运行"}</span>
        </div>
        <div class="card-body">
          <div class="options-grid">
            <div>
              <div class="toggle-row">
                <div><div class="t-label">移除水分子</div><div class="t-sub">HOH / WAT / DOD</div></div>
                <label class="switch"><input type="checkbox" data-opt="remove_water" ${opts.remove_water ? "checked" : ""}><span></span></label>
              </div>
              <div class="toggle-row">
                <div><div class="t-label">移除杂原子与配体</div><div class="t-sub">共晶配体会先用于定位，再被移除</div></div>
                <label class="switch"><input type="checkbox" data-opt="remove_hetero" ${opts.remove_hetero ? "checked" : ""}><span></span></label>
              </div>
              <div class="toggle-row">
                <div><div class="t-label">保留金属离子</div><div class="t-sub">Zn / Mg / Ca / Fe 等，写入 PDBQT</div></div>
                <label class="switch"><input type="checkbox" data-opt="keep_ions" ${opts.keep_ions ? "checked" : ""}><span></span></label>
              </div>
            </div>
            <div>
              <div class="field" style="margin-bottom:12px">
                <label>保留链</label>
                ${chainButtons}
              </div>
              <div class="field">
                <label>电荷模型</label>
                <select class="select" data-opt-select="charge_model">
                  <option value="template" ${opts.charge_model === "template" ? "selected" : ""}>模板电荷（快，推荐）</option>
                  <option value="gasteiger" ${opts.charge_model === "gasteiger" ? "selected" : ""}>Gasteiger 重新计算（慢）</option>
                </select>
              </div>
            </div>
          </div>
          <div class="divider"></div>
          <div class="row">
            <div class="note" style="flex:1">
              <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M12 8h.01M11 12h1v5h1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
              <span>准备过程在后台运行。若结构含缺失残基或非常规残基，Meeko 会跳过无法识别的部分并在日志中提示。</span>
            </div>
            <button class="btn primary" data-action="prep-receptor" data-id="${rec.id}">${rec.status === "ready" ? "重新准备" : "准备受体"}</button>
          </div>
        </div>
      </div>`;
  }

  /* ------------------------------------------------------------------ ligand step */
  function renderLigandStep() {
    const d = S.data;
    const opts = S.ligOptions || {};
    const mode = S.ligandMode === "bulk" ? "bulk" : "fine";
    const stats = d.ligand_stats || {};
    const total = Number(d.ligand_total ?? d.ligands.length);
    const readyCount = Number(stats.ready || 0);
    const addCard = `
      <div class="card">
        <div class="card-head"><div><h3>添加配体</h3><p>支持标准 .smi（SMILES + 名称）、SDF、CSV/TSV；大文件建议走“大规模筛选”。</p></div></div>
        <div class="card-body">
          <div class="grid-2">
            <label class="dropzone" id="ligand-drop">
              <input type="file" data-upload="ligand" accept=".sdf,.sd,.mol,.mdl,.mol2,.pdb,.pdbqt,.smi,.txt,.csv,.tsv" multiple>
              <svg viewBox="0 0 24 24" fill="none"><path d="M12 16V4m0 0L7 9m5-5l5 5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
              <strong>拖拽配体文件（可多选）</strong>
              <span>SDF（多记录）· CSV/TSV · SMI（名称可含空格）· MOL2 · PDBQT</span>
            </label>
            <form data-form="add-smiles" style="display:flex;flex-direction:column;gap:9px">
              <div class="field">
                <label>直接输入 SMILES（每行一个，<span class="mono">SMILES 名称</span> 或 <span class="mono">名称 SMILES</span> 自动识别；名称可含空格，如 <span class="mono">Salvianolic acid G</span>）</label>
                <textarea class="textarea" id="smiles-input" placeholder="aspirin CC(=O)Oc1ccccc1C(=O)O&#10;caffeine Cn1c(=O)c2c(ncn2C)n(C)c1=O&#10;CC(=O)Oc1ccccc1C(=O)O aspirin_2&#10;CC(=O)O Salvianolic acid G"></textarea>
              </div>
              <button class="btn" type="submit" style="align-self:flex-start">添加 SMILES</button>
            </form>
          </div>
        </div>
      </div>`;

    const switcher = `
      <div class="card">
        <div class="card-head"><div><h3>配体准备模式</h3><p>精细模式用于关键分子；大规模模式只做批量准备与粗筛，不逐条渲染列表。</p></div><span class="badge">${total} 个配体</span></div>
        <div class="card-body">
          <div class="row" style="gap:8px;flex-wrap:wrap">
            <button class="btn ${mode === "fine" ? "primary" : "ghost"}" data-action="ligand-mode" data-mode="fine">精细准备（≤${FINE_LIMIT} 个）</button>
            <button class="btn ${mode === "bulk" ? "primary" : "ghost"}" data-action="ligand-mode" data-mode="bulk">大规模筛选（&gt;${FINE_LIMIT} 个）</button>
            <div class="spacer"></div>
            <span class="chip">待准备 <strong>${Number(stats.raw || 0) + Number(stats.error || 0)}</strong></span>
            <span class="chip">已准备 <strong>${readyCount}</strong></span>
            <span class="chip">成药性过滤 <strong>${Number(stats.filtered || 0)}</strong></span>
            <span class="chip">失败 <strong>${Number(stats.error || 0)}</strong></span>
          </div>
        </div>
      </div>`;

    const fineOptions = `
      <div class="card">
        <div class="card-head"><div><h3>精细准备选项</h3><p>逐条生成 3D 构象、按 pH 7.4 校正质子化并做 MMFF94 最小化，适合关键配体的精细对接。</p></div></div>
        <div class="card-body">
          <div class="params-grid">
            <div class="field">
              <label>质子化状态</label>
              <select class="select" data-opt-select="lig_protonation">
                <option value="ph7.4" ${opts.protonation === "ph7.4" ? "selected" : ""}>pH 7.4（推荐）</option>
                <option value="neutral" ${opts.protonation === "neutral" ? "selected" : ""}>中性化</option>
                <option value="keep" ${opts.protonation === "keep" ? "selected" : ""}>保持原样</option>
              </select>
            </div>
            <div class="field">
              <label>随机种子</label>
              <input class="input" type="number" data-opt-input="lig_seed" value="${Number(opts.seed ?? 42)}" step="1">
            </div>
            <label class="check" style="align-self:end;padding-bottom:10px">
              <input type="checkbox" data-opt="lig_minimize" ${opts.minimize ? "checked" : ""}>
              生成后进行能量最小化
            </label>
            <label class="check" style="align-self:end;padding-bottom:10px">
              <input type="checkbox" data-opt="lig_filter" ${opts.filter === false ? "" : "checked"}>
              启用成药性过滤（总开关）
            </label>
          </div>
          <div class="divider"></div>
          <div class="row" style="margin-bottom:6px">
            <strong style="font-size:11.5px">成药性过滤参数</strong>
            <span class="muted" style="font-size:10.5px">关闭总开关后不生效；警告阈值按上限自动收紧</span>
          </div>
          <div class="params-grid" style="${opts.filter === false ? "opacity:.5" : ""}">${filterLimitGrid(opts.filter_limits)}</div>
          <div class="divider"></div>
          <div class="row">
            <span class="chip">已选 <strong>${S.selectedLigands.size}</strong> / ${FINE_LIMIT}</span>
            <button class="btn ghost" data-action="lig-select-none">清空选择</button>
            <div class="spacer"></div>
            <button class="btn primary" data-action="prep-fine">准备所选（≤${FINE_LIMIT}）</button>
          </div>
          <div class="note" style="margin-top:10px">精细准备完成后，可直接在“分子对接”步骤中选择这些配体，使用“单配体精细（Vina CPU）”预设。</div>
        </div>
      </div>`;

    const fineRows = (S.fineLigands || d.ligands.slice(0, 30));
    const fineTotal = S.fineLigands ? S.fineTotal : total;
    const fineList = fineRows.length
      ? `<div class="card">
           <div class="card-head"><div><h3>精细配体列表</h3><p>只显示当前页；勾选最多 ${FINE_LIMIT} 个参与精细准备与对接。</p></div><span class="badge">${fineRows.length} / ${fineTotal}</span></div>
           <div class="card-body" style="padding-top:8px">
             <div class="row" style="margin-bottom:10px;gap:8px">
               <input class="input" id="fine-query" placeholder="按名称或 SMILES 搜索" value="${esc(S.fineQuery || "")}" style="flex:1;min-width:200px">
               <button class="btn" data-action="fine-search">搜索</button>
               <button class="btn ghost" data-action="fine-page" data-offset="${Math.max(0, S.fineOffset - 30)}" ${S.fineOffset <= 0 ? "disabled" : ""}>上一页</button>
               <button class="btn ghost" data-action="fine-page" data-offset="${S.fineOffset + 30}" ${S.fineOffset + 30 >= fineTotal ? "disabled" : ""}>下一页</button>
             </div>
             <div style="overflow:auto">
               <table class="lig-table">
                 <thead><tr><th></th><th>配体</th><th>分子量</th><th>logP</th><th>可旋转键</th><th>状态</th><th></th></tr></thead>
                 <tbody>${fineRows.map(ligandRow).join("")}</tbody>
               </table>
             </div>
           </div>
         </div>`
      : `<div class="card"><div class="card-body">${emptyState("还没有配体", "上传 SDF/MOL2/SMI 文件，或直接粘贴 SMILES。", "flask")}</div></div>`;

    const bulkOptions = `
      <div class="card">
        <div class="card-head"><div><h3>大规模准备选项</h3><p>批量生成 3D 构象，默认跳过能量最小化以获得最高吞吐；准备好后统一进入大规模对接。</p></div></div>
        <div class="card-body">
          <div class="params-grid">
            <div class="field">
              <label>质子化状态</label>
              <select class="select" id="bulk-protonation">
                <option value="ph7.4" selected>pH 7.4（推荐）</option>
                <option value="neutral">中性化</option>
                <option value="keep">保持原样</option>
              </select>
            </div>
            <div class="field"><label>随机种子</label><input class="input" id="bulk-seed" type="number" value="42" step="1"></div>
            <div class="field"><label>并行线程（1–32）</label><input class="input" id="bulk-threads" type="number" min="1" max="32" value="${Number(opts.threads ?? 8)}"></div>
            <label class="check" style="align-self:end;padding-bottom:10px">
              <input type="checkbox" id="bulk-minimize">
              执行能量最小化（更慢，通常关闭）
            </label>
            <label class="check" style="align-self:end;padding-bottom:10px">
              <input type="checkbox" id="bulk-filter" checked>
              启用成药性过滤（总开关）
            </label>
            <div class="params-grid" style="grid-column:1/-1;margin-top:6px">${filterLimitGrid(opts.filter_limits)}</div>
          </div>
          <div class="divider"></div>
          <div class="row" style="flex-wrap:wrap;gap:8px">
            <span class="muted">准备结果可储存为 ZIP，在其它项目中直接载入复用，无需重新准备。</span>
            <div class="spacer"></div>
            ${readyCount ? `<a class="btn ghost sm" href="/api/ligands/bulk/export" download>储存准备结果（ZIP）</a>` : `<button class="btn ghost sm" disabled>储存准备结果（ZIP）</button>`}
            <input type="file" id="bulk-import-file" accept=".zip" style="display:none">
            <button class="btn ghost sm" data-action="lig-import">载入准备结果（ZIP）</button>
          </div>
          <div class="divider"></div>
          <div class="row">
            <span class="chip">待准备 <strong>${Number(stats.raw || 0) + Number(stats.error || 0)}</strong></span>
            <div class="spacer"></div>
            <button class="btn danger" data-action="lig-clear" data-scope="unprepared">清除未准备（${Number(stats.raw || 0) + Number(stats.error || 0)}）</button>
            <button class="btn danger" data-action="lig-clear" data-scope="prepared">清除已准备（${readyCount}）</button>
            <button class="btn primary" data-action="prep-bulk" ${total ? "" : "disabled"}>一键准备全部（后台）</button>
          </div>
          <div class="note" style="margin-top:10px">准备过程在后台按设定线程并行运行，列表不再逐条渲染，因此数万配体也不会卡住界面。可在“任务日志”查看进度。</div>
        </div>
      </div>`;

    const bulkReceptors = d.receptors.filter(dockableReceptor);
    const bulkPresets = (d.dock_presets || []).filter((item) =>
      ["ultra-fast", "qvina-fast", "gpu-fast"].includes(item.id)
    );
    const qvinaReady = Boolean(
      (d.system?.engines || []).find((item) => item.id === "qvina2")?.available
    );
    const gpuReady = Boolean(
      (d.system?.engines || []).find((item) => item.id === "vina-gpu")?.available
    );
    let selectedBulkPreset = S.bulkPresetId
      || (gpuReady ? "gpu-fast" : qvinaReady ? "qvina-fast" : "ultra-fast");
    if (!bulkPresets.some((item) => item.id === selectedBulkPreset)) {
      selectedBulkPreset = bulkPresets[0]?.id || "";
    }
    const bulkPresetInfo = bulkPresets.find((item) => item.id === selectedBulkPreset);
    const isGpuPreset = bulkPresetInfo?.params?.docking_engine === "vina-gpu";
    const gpuThreads = Number(bulkPresetInfo?.params?.gpu_threads ?? 2000);
    const gpuDepth = Number(bulkPresetInfo?.params?.gpu_search_depth ?? 1);
    const gpuChunk = Number(bulkPresetInfo?.params?.gpu_chunk ?? 1500);
    const bulkRun = (d.runs || []).find(
      (run) => run.params?.bulk && Number(run.result_count || 0) > 0
    ) || (d.runs || []).find((run) => run.params?.bulk);
    const summary = bulkRun?.bulk_summary;
    const topLoaded = Boolean(S.bulkTop && bulkRun && S.bulkTop.runId === bulkRun.id);
    const topItems = topLoaded ? S.bulkTop.items : [];
    const topN = Number(S.bulkTopN || 50);
    const bulkCard = `
      <div class="card">
        <div class="card-head"><div><h3>大规模对接（极速粗筛）</h3><p>对所有已准备配体并行快速打分，结果用于挑选进入精细对接的分子；个别损坏配体会被自动跳过。</p></div><span class="badge">${readyCount} 个可用</span></div>
        <div class="card-body">
          ${readyCount ? `
          <div class="params-grid">
            <div class="field"><label>受体</label>
              <select class="select" id="bulk-receptor">
                ${bulkReceptors.map((r) => `<option value="${esc(r.id)}" ${r.id === S.bulkReceptorId ? "selected" : ""}>${esc(r.name)}</option>`).join("")}
              </select>
            </div>
            <div class="field"><label>预设</label>
              <select class="select" id="bulk-preset">
                ${bulkPresets.map((item) => `<option value="${esc(item.id)}" ${item.id === selectedBulkPreset ? "selected" : ""}>${esc(item.label)}</option>`).join("")}
              </select>
            </div>
            ${isGpuPreset ? `
            <div class="field"><label>GPU lane（--thread）</label><input class="input" id="bulk-gpu-threads" type="number" min="1000" max="20000" step="500" value="${gpuThreads}"></div>
            <div class="field"><label>搜索深度（search_depth）</label><input class="input" id="bulk-gpu-depth" type="number" min="1" max="20" value="${gpuDepth}"></div>
            <div class="field"><label>每批配体数</label><input class="input" id="bulk-gpu-chunk" type="number" min="200" max="5000" step="100" value="${gpuChunk}"></div>
            ` : ""}
          </div>
          ${isGpuPreset ? `<div class="note" style="margin-top:8px">GPU 批量模式为单卡单进程引擎：解析配体与精修结果阶段 GPU 会短暂空闲，正式对接阶段利用率通常 90–100%、显存约 6 GB。4070 笔记本上 search_depth 1 约 0.16 秒/配体，12000 个配体约 30 分钟；运行日志每 30 秒打印一次真实 GPU 利用率、显存与功耗，可直接核对。</div>` : ""}
          <div class="row" style="margin-top:12px">
            <div class="spacer"></div>
            <button class="btn primary" data-action="dock-bulk" ${bulkReceptors.length ? "" : "disabled"}>开始大规模对接</button>
          </div>
          ${bulkReceptors.length ? "" : `<div class="note warn" style="margin-top:10px">请先在“受体”步骤准备一个带对接盒的受体。</div>`}
          ` : `<div class="note warn">还没有已准备的配体。先执行“一键准备全部”。</div>`}
          ${bulkRun ? `
            <div class="divider"></div>
            <div class="row" style="flex-wrap:wrap;gap:8px">
              <span class="chip">已完成 <strong>${summary?.ok ?? 0}</strong> / ${bulkRun.result_count ?? 0}</span>
              <span class="chip">中位 <strong>${summary?.median ?? "—"}</strong> kcal/mol</span>
              <span class="chip">最优 <strong>${summary?.best ?? "—"}</strong></span>
              <span class="chip">≤ −8 <strong>${summary?.below_8 ?? 0}</strong></span>
              <span class="chip">≤ −9 <strong>${summary?.below_9 ?? 0}</strong></span>
              <div class="spacer"></div>
              <a class="btn ghost sm" href="/api/runs/${esc(bulkRun.id)}/bulk/figures.pdf" target="_blank" rel="noopener">下载 7 张统计图（PDF）</a>
              <a class="btn ghost sm" href="/api/runs/${esc(bulkRun.id)}/bulk.png" target="_blank" rel="noopener">下载筛查图</a>
            </div>
            <div style="margin-top:12px;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fff">
              <img src="/api/runs/${esc(bulkRun.id)}/bulk.png" alt="大规模对接分数分布" style="display:block;width:100%">
            </div>
            <div class="divider"></div>
            <div class="row" style="flex-wrap:wrap;gap:8px;align-items:end">
              <div class="field" style="margin:0;width:170px">
                <label>候选数量（5–200）</label>
                <input class="input" id="bulk-top-n" type="number" min="5" max="200" value="${topN}">
              </div>
              <button class="btn ghost" data-action="load-top" data-run="${esc(bulkRun.id)}">获取排名前 N 个</button>
              <button class="btn primary" data-action="use-top" ${topItems.length ? "" : "disabled"}>载入到精细准备（${topItems.length || 0}）</button>
              <div class="spacer"></div>
              <span class="muted">按打分从低到高排序，可直接进入精细准备与对接。</span>
            </div>
            ${topItems.length ? `
            <div style="margin-top:10px;max-height:340px;overflow:auto;border:1px solid var(--line);border-radius:10px">
              <table class="lig-table">
                <thead><tr><th>#</th><th>配体</th><th>打分</th><th>MW</th><th>配体效率</th><th>状态</th></tr></thead>
                <tbody>
                  ${topItems.map((item) => `
                    <tr>
                      <td class="num">${item.rank}</td>
                      <td><div class="lig-name">${esc(item.ligand_name)}</div></td>
                      <td class="num"><strong>${Number(item.affinity).toFixed(2)}</strong></td>
                      <td class="num">${item.mw ? Number(item.mw).toFixed(0) : "—"}</td>
                      <td class="num">${item.le !== null && item.le !== undefined ? Number(item.le).toFixed(2) : "—"}</td>
                      <td>${badge(item.status)}</td>
                    </tr>`).join("")}
                </tbody>
              </table>
            </div>` : ""}
            <div class="note" style="margin-top:8px">载入后会自动切换到“精细准备”并选中这些配体；精细准备支持最多 ${FINE_LIMIT} 个配体，完成后即可精细对接。</div>
          ` : `<div class="note" style="margin-top:10px">完成一次大规模对接后，这里会显示分数分布、配体效率分布与最优候选。</div>`}
        </div>
      </div>`;

    return `${switcher}${addCard}${mode === "fine" ? fineOptions + fineList : bulkOptions + bulkCard}`;
  }

  function ligandRow(lig) {
    const meta = lig.meta || {};
    const smiles = lig.smiles || meta.smiles || "";
    const selected = S.selectedLigands.has(lig.id);
    const actions = [];
    actions.push(`<button class="btn ghost sm" data-action="lig-select" data-id="${lig.id}">查看</button>`);
    if (lig.status === "raw" || lig.status === "error") {
      actions.push(`<button class="btn primary sm" data-action="prep-one" data-id="${lig.id}">准备</button>`);
    }
    actions.push(`<a class="btn ghost sm" href="/api/ligands/${lig.id}/download" download>下载</a>`);
    actions.push(`<button class="btn danger sm" data-action="lig-delete" data-id="${lig.id}">删除</button>`);
    return `
      <tr class="${selected ? "selected" : ""}">
        <td><label class="check" style="justify-content:center"><input type="checkbox" data-check="ligand" data-id="${lig.id}" ${selected ? "checked" : ""}></label></td>
        <td><div class="lig-name">${esc(lig.name)} ${badge(lig.status)}${meta.formula ? `<small>${esc(meta.formula)}</small>` : ""}</div>${smiles ? `<div class="entity-sub mono" style="word-break:break-all">SMILES：${esc(smiles)} <button class="btn ghost tiny" data-action="lig-copy-smiles" data-smiles="${esc(smiles)}">复制</button></div>` : ""}${lig.message ? `<div class="entity-sub">${esc(lig.message)}</div>` : ""}</td>
        <td class="num">${meta.mw ? Number(meta.mw).toFixed(1) : "—"}</td>
        <td class="num">${meta.logp !== undefined ? Number(meta.logp).toFixed(2) : "—"}</td>
        <td class="num">${meta.rotb ?? "—"}</td>
        <td>${badge(lig.status)}</td>
        <td><div class="actions-cell">${actions.join("")}</div></td>
      </tr>`;
  }

  /* ------------------------------------------------------------------ site step */
  function renderSiteStep() {
    const d = S.data;
    const rec = activeReceptor();
    if (!rec) {
      return `<div class="card"><div class="card-body">${emptyState("请先添加受体", "在“受体”步骤中导入并准备受体结构。", "search")}</div></div>`;
    }
    if (rec.status !== "ready") {
      return `<div class="card"><div class="card-body">${emptyState("受体尚未准备", "准备完成后会自动识别口袋，并可用共晶配体定义对接盒。", "search")}</div></div>`;
    }
    const switcher = d.receptors.length > 1
      ? `<div class="row" style="margin-bottom:14px">${d.receptors.map((r) => `<button class="run-chip ${r.id === S.receptorId ? "active" : ""}" data-action="rec-select" data-id="${r.id}">${esc(r.name)} ${badge(r.status)}</button>`).join("")}</div>`
      : "";

    const pockets = rec.pockets || [];
    const pocketItems = pockets.length
      ? pockets.map((p) => `
        <div class="pocket-item ${rec.box?.source === "pocket" && rec.box?.label === p.label ? "applied" : ""}">
          <div class="pocket-rank">${p.rank}</div>
          <div class="pocket-info">
            <div class="pocket-name">${esc(p.label)}</div>
            <div class="pocket-meta">中心 ${p.center.map((v) => Number(v).toFixed(1)).join(", ")} · ${p.size.map((v) => Number(v).toFixed(0)).join("×")} Å${p.druggability != null ? ` · 可成药性 ${Number(p.druggability).toFixed(0)}` : ""}</div>
          </div>
          <div class="pocket-score" title="可成药性评分">${p.druggability != null ? Number(p.druggability).toFixed(0) : Number(p.score).toFixed(0)}</div>
          <button class="btn ghost sm" data-action="apply-pocket" data-id="${rec.id}" data-index="${p.rank - 1}">使用</button>
        </div>`).join("")
      : `<div class="note warn">未检测到明显的封闭口袋。可改用共晶配体或手动设置盒子。</div>`;

    const hetero = rec.hetero || [];
    const heteroItems = hetero.length
      ? hetero.map((h, index) => `
        <div class="pocket-item">
          <div class="pocket-rank">${index + 1}</div>
          <div class="pocket-info">
            <div class="pocket-name">${esc(h.name)} <span class="muted mono" style="font-size:9px">${esc(h.key)}</span></div>
            <div class="pocket-meta">${h.heavy} 个重原子 · 中心 ${h.center.map((v) => Number(v).toFixed(1)).join(", ")}</div>
          </div>
          <button class="btn ghost sm" data-action="extract-ref" data-id="${rec.id}" data-index="${index}">提取</button>
          <button class="btn ghost sm" data-action="apply-ref" data-id="${rec.id}" data-index="${index}">定位</button>
        </div>`).join("")
      : `<div class="note warn">该受体没有可用于定位的共晶小分子配体。</div>`;

    const box = rec.box || { center: [0, 0, 0], size: [20, 20, 20] };
    const volume = box.size[0] * box.size[1] * box.size[2];
    const manual = `
      <form data-form="box" data-id="${rec.id}">
        <div class="box-grid">
          ${["cx", "cy", "cz"].map((key, i) => `<div class="field"><label>中心 ${"XYZ"[i]}</label><input class="input" id="box-${key}" type="number" step="0.1" value="${Number(box.center[i]).toFixed(3)}"></div>`).join("")}
          ${["sx", "sy", "sz"].map((key, i) => `<div class="field"><label>边长 ${"XYZ"[i]}</label><input class="input" id="box-${key}" type="number" step="0.5" min="5" max="100" value="${Number(box.size[i]).toFixed(2)}"></div>`).join("")}
        </div>
        <div class="box-readout ${volume > 40000 ? "warn" : ""}">
          <span>盒子体积 <strong>${Math.round(volume).toLocaleString()} Å³</strong>${volume > 40000 ? " · 体积较大，计算会变慢" : ""}</span>
          <span>Vina 1.2.5 支持大盒子</span>
        </div>
        <div class="row" style="margin-top:12px">
          <button class="btn primary" type="submit">保存对接盒</button>
          <button class="btn ghost" type="button" data-action="detect-pockets" data-id="${rec.id}">重新预测口袋</button>
        </div>
      </form>`;

    return `${switcher}
      <div class="site-grid">
        <div class="card">
          <div class="card-head"><div><h3>内置口袋预测</h3><p>基于埋藏度与连通空腔的启发式检测，无需外部程序。</p></div><span class="badge info">${pockets.length}</span></div>
          <div class="card-body"><div class="pocket-list">${pocketItems}</div></div>
        </div>
        <div class="card">
          <div class="card-head"><div><h3>共晶配体定位</h3><p>晶体结构中的小分子通常最接近真实结合位点。</p></div><span class="badge info">${hetero.length}</span></div>
          <div class="card-body"><div class="pocket-list">${heteroItems}</div></div>
        </div>
      </div>
      <div class="card">
        <div class="card-head"><div><h3>对接盒 · ${esc(rec.name)}</h3><p>盒子应完整覆盖配体可能占据的区域，并留出约 5 Å 余量。</p></div><span class="badge ${rec.box ? "ok" : "warn"}">${rec.box ? esc(rec.box.label || "已设置") : "未设置"}</span></div>
        <div class="card-body">${manual}</div>
      </div>`;
  }

  /* ------------------------------------------------------------------ dock step */
  function applyDockPreset(preset) {
    S.dockPreset = preset.id;
    const params = preset.params || {};
    const fields = {
      exhaustiveness: "param-exhaustiveness",
      num_modes: "param-num-modes",
      energy_range: "param-energy-range",
      cpu: "param-cpu",
      seed: "param-seed",
      parallel_jobs: "param-parallel",
      gpu_threads: "param-gpu-threads",
      gpu_search_depth: "param-gpu-depth",
      gpu_replicas: "param-gpu-replicas",
    };
    Object.entries(fields).forEach(([key, id]) => {
      const input = $(`#${id}`);
      if (input && params[key] !== undefined) input.value = params[key];
    });
    const scoring = $("#param-scoring");
    if (scoring && params.scoring) scoring.value = params.scoring;
    const engine = $("#param-docking-engine");
    if (engine && params.docking_engine) engine.value = params.docking_engine;
    const rescore = $("#param-rescore");
    if (rescore) rescore.checked = params.rescore !== false;
    document.querySelectorAll('[data-action="dock-preset"]').forEach((button) => {
      const active = button.dataset.preset === preset.id;
      button.classList.toggle("primary", active);
      button.classList.toggle("ghost", !active);
    });
    const desc = $("#dock-preset-desc");
    if (desc) desc.textContent = preset.description || "";
    toast(`已套用预设：${preset.label}`, "success", 3000);
  }

  async function loadFineLigands(query = S.fineQuery, offset = 0) {
    const data = await api(
      `/api/ligands?offset=${offset}&limit=30&query=${encodeURIComponent(query || "")}`,
    );
    S.fineLigands = data.items || [];
    S.fineTotal = Number(data.total || 0);
    S.fineOffset = Number(data.offset || 0);
    S.fineQuery = query || "";
    render();
  }

  function renderDockStep() {
    const d = S.data;
    const recs = d.receptors.filter(dockableReceptor);
    const ligs = d.ligands.filter((l) => l.status === "ready" && l.has_pdbqt);
    const params = d.params || {};

    const recList = recs.length
      ? recs.map((r) => `
        <label class="select-row">
          <input type="checkbox" data-check="dock-rec" data-id="${r.id}" ${S.selectedReceptors.has(r.id) ? "checked" : ""}>
          <span class="name">${esc(r.name)}</span>
          <span class="meta">${r.box ? r.box.size.map((v) => Number(v).toFixed(0)).join("×") : "无盒子"}</span>
        </label>`).join("")
      : `<div class="note warn">没有已准备且设置对接盒的受体。</div>`;

    const ligList = ligs.length
      ? ligs.map((l) => `
        <label class="select-row">
          <input type="checkbox" data-check="dock-lig" data-id="${l.id}" ${S.selectedLigands.has(l.id) ? "checked" : ""}>
          <span class="name">${esc(l.name)}</span>
          <span class="meta">${l.meta?.mw ? Number(l.meta.mw).toFixed(0) : "—"} Da</span>
        </label>`).join("")
      : `<div class="note warn">没有已准备的配体。</div>`;

    const total = S.selectedReceptors.size * S.selectedLigands.size;
    const hiddenSelected = [...S.selectedLigands].filter(
      (id) => !ligs.some((l) => l.id === id)
    ).length;
    const vinaOk = Boolean(d.system?.vina_version);
    const engines = d.system?.engines || [{ id: "vina", label: "AutoDock Vina", available: vinaOk, description: "" }];
    const selectedEngineId = params.docking_engine || "vina";
    const selectedEngine = engines.find((item) => item.id === selectedEngineId) || engines[0];
    const engineOk = Boolean(selectedEngine?.available);
    const engineOptions = engines.map((item) => `<option value="${esc(item.id)}" ${item.id === selectedEngineId ? "selected" : ""}>${esc(item.label)}${item.available ? "" : "（未安装）"}</option>`).join("");
    const jobRunning = S.job && ["running"].includes(S.job.status);
    // This module is reserved for fine docking only; bulk screening lives in
    // the ligand step, so batch/ultra-fast presets are hidden here.
    const finePresets = ["single-fine", "gpu-fine", "smina-fine", "qvina-fine"];
    const presets = (d.dock_presets || []).filter((item) => finePresets.includes(item.id));
    const activePreset = S.dockPreset || "";
    const presetButtons = presets.length
      ? presets.map((item) => `<button class="btn ${item.id === activePreset ? "primary" : "ghost"} sm" data-action="dock-preset" data-preset="${esc(item.id)}" title="${esc(item.description || "")}">${esc(item.label)}</button>`).join("")
      : "";
    const activePresetInfo = presets.find((item) => item.id === activePreset);
    const jobCard = S.job
      ? `<div class="job-card" id="dock-job">
           <div class="job-icon ${jobRunning ? "spin" : ""}">
             <svg viewBox="0 0 24 24" fill="none"><path d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6l2.1 2.1m0-12.8l-2.1 2.1M7.7 16.3l-2.1 2.1" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>
           </div>
           <div class="job-main">
             <div class="job-title">${esc(S.job.title || "后台任务")}</div>
             <div class="job-msg" id="dock-job-msg">${esc(S.job.message || "")}</div>
             <div class="progress" style="margin-top:7px;max-width:none"><div class="progress-fill" id="dock-progress" style="width:${Math.round((S.job.progress || 0) * 100)}%"></div></div>
           </div>
           <button class="btn danger sm" data-action="cancel-job" ${jobRunning ? "" : "disabled"}>取消</button>
         </div>`
      : "";

    return `
      ${vinaOk ? "" : `<div class="note err" style="margin-bottom:14px">未找到 Vina 可执行文件。请将 vina.exe 放入 <span class="mono">bin/</span>，或设置环境变量 VINA_PATH；也可以在“工具 → 系统信息”中确认路径。</div>`}
      ${d.system?.gpu_available ? "" : `<div class="note" style="margin-bottom:14px">GPU 模式需要 Vina-GPU 2.0：将可执行文件放到 <span class="mono">bin/vina-gpu.exe</span>（可选内核目录 <span class="mono">bin/vina-gpu-kernels/</span>），或设置 VINA_GPU_PATH。未检测到时 GPU 选项会置灰，CPU 模式仍可正常使用。</div>`}
      <div class="card">
        <div class="card-head"><div><h3>选择受体与配体</h3><p>将按“受体 × 配体”逐一计算；每对组合使用独立随机种子以便复现。</p></div><span class="badge">${total} 个组合</span></div>
        <div class="card-body">
          <div class="selection-grid">
            <div><div class="field" style="margin-bottom:8px"><label>受体（${S.selectedReceptors.size}/${recs.length}）</label></div><div class="select-list">${recList}</div></div>
            <div><div class="field" style="margin-bottom:8px"><label>配体（${S.selectedLigands.size}/${ligs.length}）</label></div><div class="select-list">${ligList}</div></div>
          </div>
          ${hiddenSelected ? `<div class="note" style="margin-top:10px">已选 ${S.selectedLigands.size} 个配体，其中 ${hiddenSelected} 个不在当前列表页（来自大规模粗筛候选），仍会全部参与对接。</div>` : ""}
        </div>
      </div>

      <div class="card">
        <div class="card-head"><div><h3>精细对接预设</h3><p>本模块只负责少量分子的精细对接；上万个配体的粗筛请回到“配体准备 → 大规模筛选”。</p></div></div>
        <div class="card-body">
          <div class="row" style="flex-wrap:wrap;gap:6px">${presetButtons || `<span class="muted">未提供预设</span>`}</div>
          <div class="note" style="margin-top:10px" id="dock-preset-desc">${esc(activePresetInfo?.description || "选择预设后自动套用针对 16 核 CPU / RTX 4070 调优的参数。")}</div>
        </div>
      </div>

      <div class="card">
        <div class="card-head"><div><h3>Vina 参数</h3><p>exhaustiveness 越高搜索越充分；首次试用建议 8–16。</p></div></div>
        <div class="card-body">
          <div class="params-grid">
            <div class="field"><label>Exhaustiveness</label><input class="input" id="param-exhaustiveness" type="number" min="1" max="128" value="${Number(params.exhaustiveness ?? 16)}"></div>
            <div class="field"><label>输出构象数</label><input class="input" id="param-num-modes" type="number" min="1" max="20" value="${Number(params.num_modes ?? 9)}"></div>
            <div class="field"><label>能量范围 (kcal/mol)</label><input class="input" id="param-energy-range" type="number" min="1" max="20" step="0.5" value="${Number(params.energy_range ?? 4)}"></div>
            <div class="field"><label>CPU 线程（0 = 全部）</label><input class="input" id="param-cpu" type="number" min="0" max="128" value="${Number(params.cpu ?? 0)}"></div>
            <div class="field"><label>随机种子（0 = 每对随机）</label><input class="input" id="param-seed" type="number" min="0" value="${Number(params.seed ?? 0)}"></div>
            <div class="field"><label>随机种子重复次数</label><input class="input" id="param-seed-replicates" type="number" min="1" max="10" value="${Number(params.seed_replicates ?? 1)}"><div class="muted" style="font-size:9.5px;margin-top:4px">每个组合用不同种子重复对接，取最优并报告标准差；1 = 不重复。</div></div>
            <div class="field"><label>打分函数</label>
              <select class="select" id="param-scoring">
                <option value="vina" ${params.scoring !== "vinardo" ? "selected" : ""}>Vina</option>
                <option value="vinardo" ${params.scoring === "vinardo" ? "selected" : ""}>Vinardo</option>
              </select>
            </div>
            <div class="field"><label>对接引擎</label>
              <select class="select" id="param-docking-engine">${engineOptions}</select>
              <div class="row" style="margin-top:5px;gap:5px">
                <button class="btn ghost tiny" data-action="engine-install" data-engine="${esc(selectedEngineId)}">安装 / 检测</button>
                <button class="btn ghost tiny" data-action="engine-path" data-engine="${esc(selectedEngineId)}">指定路径</button>
              </div>
            </div>
            <div class="field"><label>并行任务数</label><input class="input" id="param-parallel" type="number" min="1" max="16" value="${Number(params.parallel_jobs ?? 1)}"></div>
            <div class="field"><label>GPU 线程 / lane</label><input class="input" id="param-gpu-threads" type="number" min="1000" max="10000" step="500" value="${Number(params.gpu_threads ?? 8000)}"></div>
            <div class="field"><label>GPU 搜索深度</label><input class="input" id="param-gpu-depth" type="number" min="5" max="200" step="5" value="${Number(params.gpu_search_depth ?? 20)}"></div>
            <div class="field"><label>GPU 多起点复算</label><input class="input" id="param-gpu-replicas" type="number" min="1" max="8" value="${Number(params.gpu_replicas ?? 8)}"></div>
            <label class="check" style="align-self:end;padding-bottom:10px">
              <input type="checkbox" id="param-rescore" ${params.rescore !== false ? "checked" : ""}>
              Vinardo 交叉打分并计算共识排名
            </label>
            <label class="check" style="align-self:end;padding-bottom:10px">
              <input type="checkbox" id="param-consensus" ${params.consensus || (total > 0 && total <= 30) ? "checked" : ""}>
              多引擎共识（Vina + smina / QuickVina 2，已安装的才运行）
            </label>
          </div>
          <div class="note" style="margin-top:10px">多引擎共识默认在 ≤ 30 个“受体 × 配体”组合时开启，约增加 1–2 倍计算时间；超过 30 个组合默认关闭，避免精细对接过慢。追加引擎只用于排序复核，不改变主对接结果。</div>
          <div class="note" style="margin-top:12px">
            <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M12 8h.01M11 12h1v5h1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
            <span>${esc(selectedEngine?.description || "")}${selectedEngine?.available ? "" : ` 当前引擎未安装：${esc(selectedEngine?.path || "bin/")}；可放入可执行文件或设置对应环境变量。`} CPU 模式适合最终结果；GPU 模式用 Vina-GPU 快速筛选，并自动用 CPU Vina 复核。追求最高精度时请用 Vina CPU 复跑关键配体。</span>
          </div>
          <div class="divider"></div>
          <div class="summary-strip">
            <div class="item"><span>受体</span><strong>${S.selectedReceptors.size}</strong></div>
            <div class="item"><span>配体</span><strong>${S.selectedLigands.size}</strong></div>
            <div class="item"><span>对接次数</span><strong class="accent">${total}</strong></div>
            <div class="spacer"></div>
            <div class="item" style="justify-content:center"><button class="btn primary" data-action="dock-start" ${total && engineOk ? "" : "disabled"}>开始对接</button></div>
          </div>
          ${jobCard}
          <div class="note" style="margin-top:12px">
            <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M12 8h.01M11 12h1v5h1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
            <span>任务在后台运行，可切换到其他步骤查看结构；取消会终止当前 Vina 进程。结果会自动保存到项目目录。</span>
          </div>
        </div>
      </div>`;
  }

  /* ------------------------------------------------------------------ results step */
  function renderDecisionCards(run) {
    const mpo = S.mpo && S.mpo.run_id === run.id ? S.mpo : null;
    const items = (mpo?.items || []).slice(0, 15);
    const mpoRows = items.map((item) => `<tr>
      <td class="rank">${item.rank ?? "—"}</td>
      <td>${esc(item.name || "")}${item.pareto ? ` <span class="badge ok">Pareto</span>` : ""}</td>
      <td class="aff">${item.affinity != null ? Number(item.affinity).toFixed(2) : "—"}</td>
      <td class="num">${item.le != null ? Number(item.le).toFixed(2) : "—"}</td>
      <td class="num">${item.qed != null ? Number(item.qed).toFixed(2) : "—"}</td>
      <td class="num">${item.sa != null ? Number(item.sa).toFixed(1) : "—"}</td>
      <td class="num">${item.safety != null ? Number(item.safety).toFixed(0) : "—"}</td>
      <td class="num"><strong>${item.mpo != null ? Number(item.mpo).toFixed(1) : "—"}</strong></td>
      <td><span class="badge ${item.stage === "候选" ? "ok" : item.stage === "先导" ? "warn" : "info"}">${esc(item.stage || "—")}</span></td>
      <td>${esc(item.novelty?.label || "—")}</td>
    </tr>`).join("");
    const mpoHtml = `<div class="card" style="margin:12px 0">
      <div class="card-head"><div><h3>多参数优化与候选分层（MPO / Pareto）</h3><p>效力 30%、配体效率 15%、QED 20%、可合成性 10%、安全性 25%；给出命中/先导/候选阶段与上市药相似度，不重新对接。</p></div>
        <div class="row" style="gap:6px">
          <button class="btn primary sm" data-action="mpo-run" data-id="${esc(run.id)}" ${S.mpoBusy ? "disabled" : ""}>${S.mpoBusy ? "计算中…" : mpo ? "重新计算" : "计算 MPO"}</button>
          ${mpo ? `<button class="btn ghost sm" data-action="mpo-csv">导出 CSV</button>` : ""}
        </div>
      </div>
      <div class="card-body">
        ${mpo ? `<div class="note" style="margin-bottom:10px">${esc(mpo.summary || "")}</div>
        <div class="table-scroll">
          <table class="result-table">
            <thead><tr><th>排名</th><th>配体</th><th>Vina</th><th>LE</th><th>QED</th><th>SA</th><th>安全性</th><th>MPO</th><th>阶段</th><th>新颖性</th></tr></thead>
            <tbody>${mpoRows || `<tr><td colspan="10"><div class="note warn" style="margin:8px">没有可评分的配体（需要 SMILES）。</div></td></tr>`}</tbody>
          </table>
        </div>
        <p class="hint" style="margin-top:9px">阶段判定：候选需 MPO ≥70、打分 ≤−9、LE ≥0.35、QED ≥0.60；先导需 MPO ≥55、打分 ≤−8、LE ≥0.28。新颖性为与内置 FDA 药物集的 Morgan/Tanimoto 最大相似度：≥0.60 低、0.40–0.60 中、<0.40 高。</p>`
        : `<p class="hint">点击“计算 MPO”后给出 0–100 综合评分、Pareto 前沿、阶段判定与新颖性；计算只读取现有对接结果与 ADMET，不会重新对接。</p>`}
      </div>
    </div>`;

    const assay = S.assay;
    const assaySummary = assay?.summary;
    const assayRows = (assay?.rows || []).slice(0, 8).map((row) => `<tr>
      <td>${esc(row.name || "")}</td>
      <td class="num">${esc(`${row.value ?? ""} ${row.unit || ""}`.trim())}</td>
      <td class="num">${row.p_activity != null ? Number(row.p_activity).toFixed(2) : "—"}</td>
      <td class="aff">${row.affinity != null ? Number(row.affinity).toFixed(2) : "—"}</td>
      <td>${esc(row.matched_by || "—")}</td>
    </tr>`).join("");
    const assayHtml = `<div class="card" style="margin:12px 0">
      <div class="card-head"><div><h3>实验数据回填与打分校准</h3><p>上传实测 IC50/Kd/EC50/Ki，自动匹配项目配体并计算与对接打分的秩相关，判断本系列能否按打分排序。</p></div>
        <div class="row" style="gap:6px">
          <span class="badge ${assaySummary ? (assaySummary.level === "ok" ? "ok" : assaySummary.level === "bad" ? "err" : "warn") : "info"}">${assaySummary ? `匹配 ${assaySummary.matched}/${assaySummary.total}` : "未导入"}</span>
          ${assaySummary ? `<button class="btn ghost sm" data-action="assay-clear">清除</button>` : ""}
        </div>
      </div>
      <div class="card-body">
        <div class="row" style="flex-wrap:wrap;gap:8px">
          <input class="input" type="file" id="assay-file" accept=".csv,.tsv,.txt" style="max-width:320px">
          <button class="btn primary sm" data-action="assay-upload" ${S.assayBusy ? "disabled" : ""}>${S.assayBusy ? "校准中…" : "上传并校准"}</button>
        </div>
        <p class="hint" style="margin-top:8px">CSV/TSV 至少包含名称或 SMILES 列，以及 IC50/Kd/EC50/Ki 数值列和单位（nM、μM、mM、M）；也接受 name,smiles,value,unit 四列格式。</p>
        ${assaySummary ? `<div class="note" style="margin-top:8px">匹配 ${assaySummary.matched}/${assaySummary.total} 条 · Spearman ρ ${assaySummary.spearman ?? "—"} · Pearson r ${assaySummary.pearson ?? "—"} · ${esc(assaySummary.verdict || "")}</div>
        <div class="table-scroll" style="margin-top:10px">
          <table class="result-table">
            <thead><tr><th>配体</th><th>实测值</th><th>pActivity</th><th>Vina</th><th>匹配</th></tr></thead>
            <tbody>${assayRows}</tbody>
          </table>
        </div>
        <p class="hint" style="margin-top:8px">pActivity = 9 − log10(IC50/nM)；预期相关性为负（打分越负、活性越高）。ρ ≤ −0.60 可用于本系列排序，−0.60 至 −0.30 仅可粗排，|ρ| < 0.30 或方向为正时不应按打分排序。</p>`
        : ""}
      </div>
    </div>`;

    const picked = activeResult();
    const pickedLigand = picked ? ligandById(picked.ligand_id) : null;
    const defaultMw = pickedLigand?.meta?.mw ?? "";
    const pickedSmiles =
      pickedLigand?.smiles
      || pickedLigand?.meta?.smiles
      || resultLigand(picked)?.smiles
      || "";

    const trust = S.trust && S.trust.run_id === run.id ? S.trust : null;
    const trustRows = (trust?.components || []).map((item) => `<tr>
      <td>${esc(item.label || item.key)}</td>
      <td class="num">${item.score != null ? Number(item.score).toFixed(0) : "—"}</td>
      <td class="num">${item.weight != null ? `${(Number(item.weight) * 100).toFixed(0)}%` : "—"}</td>
      <td>${esc(item.note || "")}</td>
    </tr>`).join("");
    const trustHtml = `<div class="card" style="margin:12px 0">
      <div class="card-head"><div><h3>可信度与弃权判定</h3><p>结构质量 30%、适用域 20%、引擎一致性 25%、实验校准 15%、构象有效性 10%；低于 45 分明确弃权。</p></div>
        <div class="row" style="gap:6px">
          ${trust ? `<span class="badge ${trust.abstain ? "err" : trust.level === "高" ? "ok" : "warn"}">可信度 ${Number(trust.score).toFixed(0)} · ${esc(trust.level)}</span>` : ""}
          <button class="btn primary sm" data-action="trust-run" data-id="${esc(run.id)}" ${S.trustBusy ? "disabled" : ""}>${S.trustBusy ? "评估中…" : trust ? "重新评估" : "评估可信度"}</button>
        </div>
      </div>
      <div class="card-body">
        ${trust ? `<div class="note ${trust.abstain ? "err" : ""}" style="margin-bottom:9px">${esc(trust.advice || "")}</div>
        <div class="table-scroll">
          <table class="result-table">
            <thead><tr><th>分项</th><th>得分</th><th>权重</th><th>说明</th></tr></thead>
            <tbody>${trustRows}</tbody>
          </table>
        </div>` : `<p class="hint">综合受体结构质量、模型适用域、引擎/构象一致性与实验校准，给出 0–100 可信度；低分时明确标记“不建议据此决策”。</p>`}
      </div>
    </div>`;

    const mmpData = S.mmp && S.mmp.run_id === run.id ? S.mmp : null;
    const mmpRows = (mmpData?.rules || []).slice(0, 10).map((rule) => `<tr>
      <td class="mono" style="font-size:10px;word-break:break-all">${esc(rule.from)} → ${esc(rule.to)}</td>
      <td class="rank">${rule.n ?? "—"}</td>
      <td class="num">${rule.mean_delta != null ? Number(rule.mean_delta).toFixed(2) : "—"}</td>
      <td>${esc(rule.direction || "—")}</td>
      <td>${esc(rule.confidence || "—")}</td>
    </tr>`).join("");
    const mmpHtml = `<div class="card" style="margin:12px 0">
      <div class="card-head"><div><h3>项目内匹配分子对（MMP）与 SAR 规则</h3><p>从项目自己的对接/实测数据中挖掘“某个取代基改造带来多少活性变化”，支持对数越多越可信。</p></div>
        <div class="row" style="gap:6px">
          ${mmpData ? `<span class="badge ok">${mmpData.rule_count || 0} 条规则 / ${mmpData.pairs || 0} 对</span>` : ""}
          <button class="btn primary sm" data-action="mmp-run" data-id="${esc(run.id)}" ${S.mmpBusy ? "disabled" : ""}>${S.mmpBusy ? "挖掘中…" : mmpData ? "重新挖掘" : "挖掘 SAR 规则"}</button>
        </div>
      </div>
      <div class="card-body">
        ${mmpData ? `<div class="note" style="margin-bottom:9px">${esc(mmpData.note || "")}</div>
        <div class="table-scroll">
          <table class="result-table">
            <thead><tr><th>改造</th><th>对数</th><th>平均 Δ</th><th>方向</th><th>置信度</th></tr></thead>
            <tbody>${mmpRows || `<tr><td colspan="5"><div class="note warn" style="margin:8px">项目内还没有可配对的分子（需要同一核心的多个取代基变体）。</div></td></tr>`}</tbody>
          </table>
        </div>` : `<p class="hint">仅使用项目内分子，不依赖通用规则库；Δ 正值为“后者改善”，实测优先，否则用 −Vina 差值。</p>`}
      </div>
    </div>`;

    const pbpk = S.pbpkData;
    const pbpkRows = (pbpk?.mrl_rows || []).map((row) => `<tr>
      <td>${esc(row.tissue || "")}</td>
      <td class="num">${row.mrl_ug_kg ?? "—"}</td>
      <td class="num">${row.cmax_mg_kg != null ? Number(row.cmax_mg_kg).toFixed(4) : "—"}</td>
      <td class="num">${row.days_above_mrl != null ? Number(row.days_above_mrl).toFixed(2) : "—"}</td>
      <td>${row.exceeded ? `<span class="badge warn">超标</span>` : `<span class="badge ok">未超标</span>`}</td>
    </tr>`).join("");
    const pbpkHtml = `<div class="card" style="margin:12px 0">
      <div class="card-head"><div><h3>七物种 PBPK 与休药期</h3><p>血浆–肝–肾–肌–脂肪灌注限制模型，模拟浓度–时间曲线，按示例 MRL 估算休药期并做 Monte-Carlo 不确定性。</p></div>
        ${pbpk ? `<span class="badge ${pbpk.withdrawal_days > 0 ? "warn" : "ok"}">休药期 ${pbpk.withdrawal_days ?? 0} 天${pbpk.censored ? "（达模拟上限）" : ""}${pbpk.monte_carlo ? `（95% ${Number(pbpk.monte_carlo.p95_days || 0).toFixed(0)} 天）` : ""}</span>` : ""}
      </div>
      <div class="card-body">
        <div class="params-grid">
          <div class="field"><label>物种</label><select class="select" id="pbpk-species">
            ${[["human", "人（非食用）"], ["pig", "猪"], ["cattle", "牛"], ["sheep", "羊"], ["dog", "犬（非食用）"], ["cat", "猫（非食用）"], ["rat", "大鼠（非食用）"]].map(([value, label]) => `<option value="${value}" ${pbpk?.species === value ? "selected" : ""}>${label}</option>`).join("")}
          </select></div>
          <div class="field"><label>给药途径</label><select class="select" id="pbpk-route">
            ${[["oral", "口服"], ["im", "肌注"], ["iv", "静注"]].map(([value, label]) => `<option value="${value}" ${pbpk?.route === value ? "selected" : ""}>${label}</option>`).join("")}
          </select></div>
          <div class="field"><label>剂量 (mg/kg)</label><input class="input" id="pbpk-dose" type="number" min="0.01" step="any" value="${pbpk?.dose_mg_kg ?? 5}"></div>
          <div class="field"><label>F (%)</label><input class="input" id="pbpk-f" type="number" min="1" max="100" value="${pbpk?.f_percent ?? 50}"></div>
          <div class="field"><label>模拟天数</label><input class="input" id="pbpk-days" type="number" min="1" max="90" value="${pbpk?.duration_days ?? 14}"></div>
        </div>
        <div class="row" style="margin-top:10px;flex-wrap:wrap;gap:10px">
          <button class="btn primary sm" data-action="pbpk-run" ${S.pbpkBusy ? "disabled" : ""}>${S.pbpkBusy ? "模拟中…" : "运行 PBPK"}</button>
          ${pbpk ? `<span class="chip">Cmax ${Number(pbpk.cmax_mg_l || 0).toFixed(3)} mg/L</span>
          <span class="chip">Tmax ${Number(pbpk.tmax_h || 0).toFixed(1)} h</span>
          <span class="chip">t½(表观) ${pbpk.half_life_h ?? "—"} h</span>
          <span class="chip">AUC ${Number(pbpk.auc_mg_h_l || 0).toFixed(1)} mg·h/L</span>` : ""}
        </div>
        ${pbpk ? `<div class="table-scroll" style="margin-top:10px">
          <table class="result-table">
            <thead><tr><th>组织</th><th>MRL (µg/kg)</th><th>Cmax (mg/kg)</th><th>超标天数</th><th>状态</th></tr></thead>
            <tbody>${pbpkRows || `<tr><td colspan="5"><div class="note" style="margin:8px">非食用动物或未提供 MRL，不计算休药期。</div></td></tr>`}</tbody>
          </table>
        </div>
        <p class="hint" style="margin-top:8px">${esc(pbpk.note || "")}</p>` : `<p class="hint" style="margin-top:8px">需要先选中一个配体（用于分子量与 logP）；MRL 为示例值，注册前必须核对目标市场法规。</p>`}
      </div>
    </div>`;

    const active = S.active && S.active.run_id === run.id ? S.active : null;
    const activeRows = (active?.items || []).slice(0, 12).map((item) => `<tr>
      <td>${esc(item.name || "")}</td>
      <td class="aff">${item.affinity != null ? Number(item.affinity).toFixed(2) : "—"}</td>
      <td class="num">${item.predicted_pactivity != null ? Number(item.predicted_pactivity).toFixed(2) : "—"}</td>
      <td class="num">${item.sd != null ? `±${Number(item.sd).toFixed(2)}` : "—"}</td>
      <td class="num">${item.ei != null ? Number(item.ei).toFixed(2) : "—"}</td>
      <td>${esc(item.reason || "")}</td>
    </tr>`).join("");
    const activeHtml = `<div class="card" style="margin:12px 0">
      <div class="card-head"><div><h3>下一轮实验推荐（主动学习）</h3><p>以实测 pActivity 为标签的高斯过程，按 Expected Improvement 与多样性挑选下一批 8–16 个分子；实测点不足时退回对接排序。</p></div>
        <div class="row" style="gap:6px">
          ${active ? `<span class="badge ${active.mode === "active" ? "ok" : "warn"}">${active.mode === "active" ? `训练 ${active.train} · 推荐 ${active.batch}` : active.mode === "exhausted" ? "已全部实测" : "等待实测数据"}</span>` : ""}
          <button class="btn primary sm" data-action="active-run" data-id="${esc(run.id)}" ${S.activeBusy ? "disabled" : ""}>${S.activeBusy ? "计算中…" : active ? "重新推荐" : "推荐下一批"}</button>
        </div>
      </div>
      <div class="card-body">
        ${active ? `<div class="note" style="margin-bottom:9px">${esc(active.note || "")}</div>
        <div class="table-scroll">
          <table class="result-table">
            <thead><tr><th>配体</th><th>Vina</th><th>预测 pActivity</th><th>不确定度</th><th>EI</th><th>说明</th></tr></thead>
            <tbody>${activeRows || `<tr><td colspan="6"><div class="note warn" style="margin:8px">没有可推荐的候选。</div></td></tr>`}</tbody>
          </table>
        </div>` : `<p class="hint">导入至少 4 条实测 IC50/Kd 后即可启用；推荐结果用于下一轮实验排序，不是活性结论。</p>`}
      </div>
    </div>`;

    const bench = S.benchmark && S.benchmark.run_id === run.id
      ? S.benchmark
      : (S.data?.benchmarks?.[run.id] || null);
    const benchMetrics = bench?.metrics;
    const benchRows = bench?.weights
      ? Object.entries(bench.weights).map(([key, value]) => `<tr>
          <td>${esc(bench.feature_labels?.[key] || key)}</td>
          <td class="num">${Number(value).toFixed(2)}</td>
          <td class="num">${({ potency: 0.30, le: 0.15, qed: 0.20, sa: 0.10, safety: 0.25 }[key] ?? 0).toFixed(2)}</td>
        </tr>`).join("")
      : "";
    const benchHtml = `<div class="card" style="margin:12px 0">
      <div class="card-head"><div><h3>回溯校准与数据驱动权重</h3><p>用同一次对接中的“已知活性物 + 诱饵”计算 EF1%/EF5%/AUC/BEDROC，并用逻辑回归拟合决策权重与专家权重对比。</p></div>
        ${benchMetrics ? `<span class="badge ${benchMetrics.auc >= 0.8 ? "ok" : benchMetrics.auc >= 0.65 ? "warn" : "err"}">AUC ${benchMetrics.auc ?? "—"} · BEDROC ${benchMetrics.bedroc ?? "—"}</span>` : ""}
      </div>
      <div class="card-body">
        <div class="row" style="flex-wrap:wrap;gap:8px">
          <input class="input" type="file" id="benchmark-file" accept=".csv,.tsv,.txt" style="max-width:320px">
          <button class="btn primary sm" data-action="benchmark-run" data-id="${esc(run.id)}" ${S.benchmarkBusy ? "disabled" : ""}>${S.benchmarkBusy ? "校准中…" : "运行回溯校准"}</button>
        </div>
        <p class="hint" style="margin-top:8px">活性物 CSV 需要名称或 SMILES + IC50/Kd/EC50/Ki 与单位；同一次对接中的其余配体自动作为诱饵。建议 ≥20 个活性物 + ≥200 个诱饵。</p>
        ${benchMetrics ? `<div class="row" style="margin-top:10px;flex-wrap:wrap;gap:10px">
          <span class="chip">活性物 ${benchMetrics.n_active}/${benchMetrics.n_total}</span>
          <span class="chip">EF1% ${benchMetrics.ef1}</span>
          <span class="chip">EF5% ${benchMetrics.ef5}</span>
          <span class="chip">AUC ${benchMetrics.auc}${benchMetrics.auc_ci ? ` [${benchMetrics.auc_ci.join(", ")}]` : ""}</span>
          <span class="chip">BEDROC ${benchMetrics.bedroc}</span>
          <span class="chip">最佳活性物排名 ${benchMetrics.best_active_rank}</span>
        </div>
        ${benchRows ? `<div class="table-scroll" style="margin-top:10px">
          <table class="result-table">
            <thead><tr><th>特征</th><th>拟合权重</th><th>专家权重</th></tr></thead>
            <tbody>${benchRows}</tbody>
          </table>
        </div>
        <div class="note" style="margin-top:8px">${esc(bench.comparison?.verdict || "")}${bench.comparison?.cv_auc != null ? `（5 折 AUC ${bench.comparison.cv_auc}，专家 ${bench.comparison.expert_auc}）` : ""}</div>` : ""}` : ""}
        ${bench ? `<p class="hint" style="margin-top:8px">${esc(bench.note || "")}</p>` : ""}
      </div>
    </div>`;

    const dose = S.dose;
    const doseHtml = `<div class="card" style="margin:12px 0">
      <div class="card-head"><div><h3>起始剂量预测（人体 / 目标动物）</h3><p>按目标游离浓度 = 2× 效价、口服生物利用度、肝清除与 1.5 L/kg 分布容积估算维持剂量与负荷剂量，用于立项前剂量可行性判断。</p></div>
        ${dose ? `<span class="badge ${dose.level === "ok" ? "ok" : dose.level === "warn" ? "warn" : "err"}">${esc(dose.verdict || "")}</span>` : ""}
      </div>
      <div class="card-body">
        <div class="params-grid">
          <div class="field"><label>目标效价 IC50/Kd (nM)</label><input class="input" id="dose-potency" type="number" min="0.001" step="any" value="${dose?.potency_nm ?? 100}"></div>
          <div class="field"><label>分子量</label><input class="input" id="dose-mw" type="number" min="1" step="any" value="${dose?.mw ?? defaultMw}"></div>
          <div class="field"><label>物种</label><select class="select" id="dose-species">
            ${[["human", "人"], ["pig", "猪"], ["cattle", "牛"], ["sheep", "羊"], ["dog", "犬"], ["cat", "猫"], ["rat", "大鼠"]].map(([value, label]) => `<option value="${value}" ${dose?.species === value ? "selected" : ""}>${label}</option>`).join("")}
          </select></div>
          <div class="field"><label>口服生物利用度 F (%)</label><input class="input" id="dose-f" type="number" min="1" max="100" step="1" value="${dose?.f_percent ?? 50}"></div>
        </div>
        <div class="row" style="margin-top:10px;flex-wrap:wrap;gap:10px">
          <button class="btn primary sm" data-action="dose-run" ${S.doseBusy ? "disabled" : ""}>${S.doseBusy ? "计算中…" : "预测剂量"}</button>
          ${dose ? `<span class="chip">维持剂量 <strong>${Number(dose.dose_mg_day).toFixed(1)}</strong> mg/day</span>
          <span class="chip">${Number(dose.dose_mg_kg).toFixed(2)} mg/kg/day</span>
          <span class="chip">负荷剂量 ${Number(dose.loading_mg).toFixed(1)} mg</span>
          <span class="chip">t½ ${Number(dose.half_life_h).toFixed(1)} h</span>` : ""}
        </div>
        ${dose ? `<p class="hint" style="margin-top:8px">Ceff ${Number(dose.ceff_nm).toFixed(1)} nM（${Number(dose.ceff_mg_l).toFixed(4)} mg/L） · CL ${Number(dose.cl_l_h).toFixed(2)} L/h · Vd ${Number(dose.vd_l).toFixed(1)} L · ${esc(dose.note || "")}</p>` : `<p class="hint" style="margin-top:8px">若分子量留空，请先在上方结果中选择一个配体，或手动填写分子量。</p>`}
      </div>
    </div>`;
    return mpoHtml + assayHtml + trustHtml + mmpHtml + pbpkHtml + activeHtml + benchmarkHtml + doseHtml;
  }

  function renderResultsStep() {
    const d = S.data;
    if (!d.runs.length) {
      return `<div class="card"><div class="card-body">${emptyState("还没有对接结果", "完成一次对接后，打分、构象与作用力分析会显示在这里。", "search")}</div></div>`;
    }
    const run = activeRun();
    if (!run) return "";
    const results = run.results || [];
    const totalResults = Number(run.result_count || results.length);
    const resultsTruncated = totalResults > results.length;
    // Sort by the absolute value of the docking score: the strongest binder
    // (most negative affinity) always sits on top, independent of sign
    // conventions used by different engines.
    const ordered = results
      .map((r, i) => ({ r, i }))
      .sort((a, b) => {
        const av = a.r.status === "ok" && a.r.affinity != null ? Math.abs(Number(a.r.affinity)) : -1;
        const bv = b.r.status === "ok" && b.r.affinity != null ? Math.abs(Number(b.r.affinity)) : -1;
        return bv - av;
      });
    const picked = results[S.resultIndex];
    const pickedName = picked && picked.status === "ok" ? picked.ligand_name : "未选择";
    const failedCount = results.filter((item) => item.status !== "ok").length;
    const consensusMeta = run.consensus || {};
    const consensusEngines = consensusMeta.engines || [];
    const engineNames = { vina: "Vina", smina: "smina", qvina2: "QuickVina 2" };
    const consensusTop = results
      .filter((item) => item.status === "ok" && item.consensus_rank != null)
      .sort((a, b) => Number(a.consensus_rank) - Number(b.consensus_rank))
      .slice(0, 10);
    const correlationText = Object.entries(consensusMeta.correlations || {})
      .map(([key, value]) => `${key.split("|").map((name) => engineNames[name] || name).join("–")} ρ=${value}`)
      .join(" · ");
    const consensusHtml = consensusEngines.length >= 2
      ? `<div class="card" style="margin:12px 0">
          <div class="card-head"><div><h3>多引擎共识结果</h3><p>${consensusEngines.map((name) => engineNames[name] || name).join(" · ")} · ${consensusMeta.count || consensusTop.length} 个配体参与加权共识（Vina 1.0、Vinardo 0.6、smina 0.8、QuickVina 2 0.6）；分歧大的候选标记为低置信度。</p></div><span class="badge ${consensusMeta.low_confidence ? "warn" : "ok"}">${consensusMeta.low_confidence ? `${consensusMeta.low_confidence} 个低置信度` : "可并入报告"}</span></div>
          <div class="card-body">
            <div class="table-scroll">
              <table class="result-table">
                <thead><tr><th>共识</th><th>配体</th>${consensusEngines.map((name) => `<th>${esc(engineNames[name] || name)}</th>`).join("")}<th>平均排名</th></tr></thead>
                <tbody>${consensusTop.map((item, idx) => `<tr><td class="rank">${idx + 1}</td><td>${esc(item.ligand_name)}</td>${consensusEngines.map((name) => `<td class="aff">${item.engine_scores?.[name] != null ? Number(item.engine_scores[name]).toFixed(2) : "—"}</td>`).join("")}<td class="rank">${Number(item.consensus_rank).toFixed(2)}</td></tr>`).join("")}</tbody>
              </table>
            </div>
            <div class="row" style="margin-top:12px;flex-wrap:wrap;gap:10px">
              <a class="btn ghost sm" href="/api/runs/${encodeURIComponent(run.id)}/consensus.pdf" target="_blank" rel="noopener">下载多引擎共识 PDF</a>
              ${correlationText ? `<span class="muted" style="font-size:11px">引擎秩相关：${esc(correlationText)}</span>` : ""}
            </div>
            <p class="hint" style="margin-top:10px">共识排名为加权名次（权重见标题）；引擎分歧过大时自动标记低置信度，建议对低置信度候选做 MM-GBSA 或分子动力学复核。不同引擎打分尺度不同，请比较排名与一致性，不要直接比较绝对值。</p>
          </div>
        </div>`
      : "";

    const poseBatch = S.poseBatch && S.poseBatch.run_id === run.id ? S.poseBatch : null;
    const poseBatchRows = (poseBatch?.rows || []).filter((row) => row.available !== false);
    const poseBatchHtml = poseBatch
      ? `<div class="card" style="margin:12px 0">
          <div class="card-head"><div><h3>全系列物理合理性 · PoseBusters ${esc(poseBatch.module_version || "")}</h3><p>${poseBatchRows.filter((row) => row.all_passed).length}/${poseBatchRows.length} 个构象通过全部检查；只评估构象的物理有效性，不改变对接打分。</p></div><span class="badge ${poseBatchRows.length && poseBatchRows.every((row) => row.all_passed) ? "ok" : "warn"}">${poseBatchRows.filter((row) => row.all_passed).length} 通过</span></div>
          <div class="card-body">
            <div class="table-scroll">
              <table class="result-table">
                <thead><tr><th>配体</th><th>Vina</th><th>通过项</th><th>未通过检查</th></tr></thead>
                <tbody>${poseBatchRows.map((row) => `<tr>
                  <td>${esc(row.ligand || row.name)}</td>
                  <td class="aff">${row.affinity != null ? Number(row.affinity).toFixed(2) : "—"}</td>
                  <td class="rank">${row.passed}/${row.total}</td>
                  <td>${row.all_passed ? `<span class="badge ok">通过</span>` : esc((row.failed_labels || []).join("、") || "—")}</td>
                </tr>`).join("")}</tbody>
              </table>
            </div>
            <p class="hint" style="margin-top:10px">检查项涵盖键长/键角、内部原子冲突、环平面性、内部应变能、与蛋白及辅因子的距离与体积重叠。未通过的构象并非一定错误，但通常提示打分虚高或结合模式不合理，建议优先复核或重新对接。</p>
          </div>
        </div>`
      : "";

    const runOptions = d.runs.map((r) => `<option value="${r.id}" ${r.id === run.id ? "selected" : ""}>${fmtTime(r.time)} · ${r.results?.length || 0} 个结果 · ${RUN_STATUS[r.status] || r.status}</option>`).join("");
    const rows = ordered.length
      ? ordered.map(({ r, i }, idx) => {
          const ok = r.status === "ok";
          const active = i === S.resultIndex;
          return `<tr class="${active ? "active" : ""} ${ok ? "" : "err-row"}" data-action="select-result" data-index="${i}">
            <td><button class="btn ${active ? "primary" : "ghost"} sm" data-action="pick-result" data-index="${i}" ${ok ? "" : "disabled"}>${active ? "已选择" : "选择"}</button></td>
            <td class="rank">${ok ? idx + 1 : "—"}</td>
            <td>${esc(r.ligand_name)}</td>
            <td>${esc(r.receptor_name)}</td>
            <td class="aff">${ok && r.affinity != null ? Number(r.affinity).toFixed(2) : "失败"}</td>
            <td class="num" style="color:var(--muted)">${ok && r.vinardo != null ? Number(r.vinardo).toFixed(2) : "—"}</td>
            <td class="rank">${ok ? (r.consensus_rank ?? "—") : "—"}</td>
            <td class="rank">${ok ? (r.modes || []).length : "—"}</td>
            <td>${ok ? `<span class="badge ok">完成</span>` : `<span class="badge err">错误</span>`}</td>
          </tr>`;
        }).join("")
      : `<tr><td colspan="9"><div class="note warn" style="margin:12px">暂无结果，任务可能仍在运行。</div></td></tr>`;

    return `
      <div class="run-bar">
        <select class="select" id="run-select" style="max-width:330px">${runOptions}</select>
        <span class="badge ${run.status === "done" ? "ok" : run.status === "running" ? "warn" : "info"}">${RUN_STATUS[run.status] || esc(run.status)}</span>
        <span class="chip">${fmtTime(run.time)}</span>
        <span class="chip">当前配体 <strong>${esc(pickedName)}</strong></span>
        <div class="spacer"></div>
        <a class="btn ghost sm" href="/api/runs/${encodeURIComponent(run.id)}/export.csv">导出 CSV</a>
        <button class="btn ghost sm" data-action="posebatch" data-id="${run.id}" ${S.poseBusy ? "disabled" : ""}>${S.poseBusy ? "校验中…" : "全系列物理校验"}</button>
        ${failedCount ? `<button class="btn ghost sm" data-action="resume-run" data-id="${run.id}">续跑失败组合 (${failedCount})</button>` : ""}
        <a class="btn ghost sm" href="/api/reports/full.pdf?run_id=${encodeURIComponent(run.id)}&result_index=${S.resultIndex}&species=${encodeURIComponent(S.pharmSpecies || "human")}" target="_blank" rel="noopener">完整 PDF 报告</a>
        <button class="btn danger sm" data-action="delete-run" data-id="${run.id}">删除记录</button>
      </div>
      <div class="note" style="margin:10px 0">按得分绝对值从高到低排序。点“选择”后，优化、设计、ADMET、药物互作与完整报告都默认使用该配体；进入各模块后仍可再改。</div>
      ${resultsTruncated ? `<div class="note warn" style="margin:10px 0">本次对接共 ${totalResults} 条结果，为保持页面流畅仅加载前 ${results.length} 条；完整排序请在大规模筛选面板查看 Top 命中或导出 CSV。</div>` : ""}
      ${consensusHtml}
      ${poseBatchHtml}
      ${renderDecisionCards(run)}
      <div class="result-layout">
        <div class="result-table-wrap">
          <table class="result-table">
            <thead><tr><th>选择</th><th>排名</th><th>配体</th><th>受体</th><th>Vina</th><th>Vinardo</th><th>共识</th><th>构象</th><th>状态</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        <div id="result-detail">${renderResultDetail(run)}</div>
      </div>`;
  }

  /* ------------------------------------------------------------------ optimize step */
  function renderOptimizeStep() {
    const d = S.data;
    const runs = (d.runs || []).filter((r) =>
      (r.results || []).some((item) => item.status === "ok")
    );
    if (!runs.length) {
      return `<div class="card"><div class="card-body">${emptyState(
        "还没有可优化的对接结果",
        "先完成一次成功对接，再回到这里进行骨架跃迁、残基优化和自动跑分。",
        "flask"
      )}</div></div>`;
    }
    const run = runs.find((item) => item.id === S.runId) || runs[0];
    const okResults = (run.results || [])
      .map((item, index) => ({ item, index }))
      .filter(({ item }) => item.status === "ok");
    const selected = okResults.find(({ index }) => index === S.resultIndex) || okResults[0];
    if (S.runId !== run.id || S.resultIndex !== selected.index) {
      S.runId = run.id;
      applyResultSelection(selected.index);
    }
    const parent = selected.item;
    const params = d.params || {};
    const runOptions = runs.map((item) =>
      `<option value="${item.id}" ${item.id === run.id ? "selected" : ""}>${fmtTime(item.time)} · ${item.results?.length || 0} 个结果</option>`
    ).join("");
    const resultOptions = okResults.map(({ item, index }) =>
      `<option value="${index}" ${index === selected.index ? "selected" : ""}>${esc(item.ligand_name)} · ${Number(item.affinity).toFixed(2)} kcal/mol</option>`
    ).join("");

    const opt = S.optResult;
    const latest = !opt && (d.optimizations || []).length ? d.optimizations[0] : null;
    let reportHtml = "";
    if (opt) {
      const best = opt.best;
      const candidates = opt.candidates || [];
      const candidateRows = candidates.map((item) => `
        <tr>
          <td>${esc(item.label)}</td>
          <td>${esc(item.modification)}</td>
          <td class="aff">${item.affinity != null ? Number(item.affinity).toFixed(2) : "—"}</td>
          <td class="${item.delta != null && item.delta < 0 ? "good" : "bad"}">${item.delta != null ? Number(item.delta).toFixed(2) : "—"}</td>
          <td class="rank">${item.counts?.hbonds ?? "—"}</td>
          <td class="rank">${item.counts?.hydrophobic ?? "—"}</td>
          <td class="rank">${item.props?.qed != null ? Number(item.props.qed).toFixed(2) : "—"}</td>
          <td class="rank">${item.props?.efficiency?.le != null ? Number(item.props.efficiency.le).toFixed(2) : "—"}</td>
          <td>${item.props && !item.props.lipinski ? `<span class="badge warn">违反</span>` : item.props ? `<span class="badge ok">通过</span>` : "—"}</td>
          <td>${item.index != null ? `<a class="link" href="/api/runs/${encodeURIComponent(opt.run_id)}/result/${item.index}/figures.pdf" target="_blank" rel="noopener">矢量图</a>` : "—"}</td>
        </tr>`).join("");
      const suggestionRows = (opt.suggestions || []).map((item) => `
        <tr><td><strong>${esc(item.residue)}</strong></td><td>${esc(item.interaction)}</td><td>${esc(item.strategy)}</td></tr>`).join("");
      reportHtml = `
        <div class="card">
          <div class="card-head">
            <div><h3>优化结果 · 母体 → 候选</h3><p>自动生成的类似物已在同一位点重新对接；所有图均为 PDF 矢量图。</p></div>
            <span class="badge ok">PDF 矢量输出</span>
          </div>
          <div class="card-body">
            <div class="opt-hero">
              <div><span class="eyebrow">最佳候选</span><h2>${esc(best?.label || "—")}</h2><p>${esc(best?.modification || "")}</p></div>
              <div class="opt-metric"><strong>${best?.affinity != null ? Number(best.affinity).toFixed(2) : "—"}</strong><span>Vina kcal/mol</span></div>
              <div class="opt-metric ${best?.delta != null && best.delta < 0 ? "good" : "bad"}"><strong>${best?.delta != null ? Number(best.delta).toFixed(2) : "—"}</strong><span>相对母体</span></div>
              <div class="opt-metric"><strong>${opt.parent?.affinity != null ? Number(opt.parent.affinity).toFixed(2) : "—"}</strong><span>母体 Vina</span></div>
            </div>
            <div class="row" style="margin:14px 0;flex-wrap:wrap">
              <a class="btn primary sm" href="/api/optimizations/${encodeURIComponent(opt.run_id)}/report.pdf" target="_blank" rel="noopener">下载优化报告 PDF</a>
              <a class="btn ghost sm" href="/api/reports/full.pdf?run_id=${encodeURIComponent(run.id)}&optimization_id=${encodeURIComponent(opt.run_id)}&species=${encodeURIComponent(S.pharmSpecies || "human")}" target="_blank" rel="noopener">完整 PDF 报告</a>
              ${best?.index != null ? `<button class="btn sm" data-action="opt-refine" data-run="${opt.run_id}" data-index="${best.index}">继续优化最佳候选</button>` : ""}
              <a class="btn sm" href="/api/runs/${encodeURIComponent(run.id)}/result/${selected.index}/figures.pdf?mode=1" target="_blank" rel="noopener">母体结构/作用力 PDF</a>
              ${best?.index != null ? `<a class="btn ghost sm" href="/api/runs/${encodeURIComponent(opt.run_id)}/result/${best.index}/figures.pdf?mode=1" target="_blank" rel="noopener">最佳候选结构/作用力 PDF</a>` : ""}
            </div>
            <div class="table-scroll">
              <table class="result-table">
                <thead><tr><th>候选</th><th>改造</th><th>Vina</th><th>ΔVina</th><th>氢键</th><th>疏水</th><th>QED</th><th>LE</th><th>类药</th><th>结构图</th></tr></thead>
                <tbody>${candidateRows}</tbody>
              </table>
            </div>
          </div>
        </div>
        <div class="card">
          <div class="card-head"><div><h3>残基层面的改进方案</h3><p>根据当前构象中的氢键、疏水、盐桥与 π-堆积给出的位点级建议。</p></div></div>
          <div class="card-body table-scroll">
            <table class="result-table opt-suggest">
              <thead><tr><th>残基</th><th>当前作用</th><th>改进策略</th></tr></thead>
              <tbody>${suggestionRows}</tbody>
            </table>
          </div>
        </div>`;
    } else if (latest) {
      reportHtml = `
        <div class="card">
          <div class="card-head"><div><h3>最近一次优化</h3><p>${esc(latest.parent_ligand || "")} → ${esc(latest.best?.label || "—")}</p></div><span class="badge ok">PDF 可下载</span></div>
          <div class="card-body">
            <div class="row"><a class="btn primary sm" href="/api/optimizations/${encodeURIComponent(latest.run_id)}/report.pdf" target="_blank" rel="noopener">下载优化报告 PDF</a></div>
            <div class="note" style="margin-top:12px">${esc(latest.best?.modification || "")} · ΔVina ${latest.best?.delta != null ? Number(latest.best.delta).toFixed(2) : "—"} kcal/mol</div>
          </div>
        </div>`;
    }

    return `
      <div class="card">
        <div class="card-head">
          <div><h3>选择母体并结合改造策略</h3><p>基于残基接触生成骨架跃迁、生物电子等排和卤素/极性修饰候选，并自动对接排序。</p></div>
          <span class="badge">${okResults.length} 个可选结果</span>
        </div>
        <div class="card-body">
          <div class="params-grid">
            <div class="field"><label>母体对接记录</label><select class="select" id="opt-run-select">${runOptions}</select></div>
            <div class="field"><label>母体配体</label><select class="select" id="opt-result-select">${resultOptions}</select></div>
            <div class="field"><label>候选数量</label><input class="input" id="opt-candidates" type="number" min="3" max="60" value="16"></div>
            <div class="field"><label>每个候选 Exhaustiveness</label><input class="input" id="opt-exhaustiveness" type="number" min="1" max="32" value="${Math.min(8, Number(params.exhaustiveness ?? 8))}"></div>
            <div class="field"><label>计算引擎</label>
              <select class="select" id="opt-engine">
                <option value="cpu" selected>CPU · AutoDock Vina</option>
                <option value="gpu" ${d.system?.gpu_available ? "" : "disabled"}>GPU · Vina-GPU${d.system?.gpu_available ? "" : "（未安装）"}</option>
              </select>
            </div>
            <div class="field"><label>并行任务数</label><input class="input" id="opt-parallel" type="number" min="1" max="8" value="2"></div>
          </div>
          <div class="parent-strip">
            <span>母体 <strong>${esc(parent.ligand_name)}</strong></span>
            <span>受体 <strong>${esc(parent.receptor_name)}</strong></span>
            <span>Vina <strong>${parent.affinity != null ? Number(parent.affinity).toFixed(2) : "—"}</strong> kcal/mol</span>
            <span>Vinardo <strong>${parent.vinardo != null ? Number(parent.vinardo).toFixed(2) : "—"}</strong></span>
          </div>
          <div class="row" style="margin-top:14px">
            <button class="btn primary" data-action="opt-start">自动生成改进方案并跑分</button>
            <span class="muted" style="font-size:11px">生成 3–60 个候选，覆盖生物电子等排、骨架跃迁、取代基行走、连接子改造与组合改造，逐个准备 3D 构象、重新对接并输出 PDF 报告。</span>
          </div>
          <div class="note" style="margin-top:12px">
            <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M12 8h.01M11 12h1v5h1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
            <span>改造规则包括：羧酸→四唑/酰基磺酰胺/异羟肟酸、酰胺/酯改造、苯环→吡啶/嘧啶/二氮杂环、五元杂环杂原子替换、芳环多位置氟/氯/甲基/甲氧基/羟基/氨基/氰基行走、亚甲基→醚/胺/羰基、末端甲基→CF3，以及对优选候选的二次组合改造。结果用于假设生成，最终需实验验证。</span>
          </div>
        </div>
      </div>
      ${reportHtml}`;
  }

  /* ------------------------------------------------------------------ design step */
  function renderDesignStep() {
    const d = S.data;
    const receptors = (d.receptors || []).filter((r) => r.box && r.has_pdbqt);
    if (!receptors.length) {
      return `<div class="card"><div class="card-body">${emptyState(
        "还没有可设计的靶点",
        "先准备受体并设置结合位点盒子，再回到分子设计页。",
        "flask"
      )}</div></div>`;
    }
    const selectedId = S.designReceptorId && receptors.some((r) => r.id === S.designReceptorId)
      ? S.designReceptorId
      : receptors[0].id;
    S.designReceptorId = selectedId;
    const receptor = receptors.find((r) => r.id === selectedId);
    const receptorOptions = receptors.map((r) =>
      `<option value="${r.id}" ${r.id === selectedId ? "selected" : ""}>${esc(r.name)} · ${esc(r.box?.label || "自定义盒子")}</option>`
    ).join("");
    const preferredReference = resultLigand(activeResult());
    const referenceCandidates = [...(d.fine_ligands || [])].filter(
      (item) =>
        (item.smiles || item.meta?.smiles)
        && !String(item.name || "").toUpperCase().startsWith("DESIGN")
    );
    if (
      preferredReference?.smiles
      && !referenceCandidates.some((item) => item.id === preferredReference.id)
    ) {
      referenceCandidates.unshift(preferredReference);
    }
    const preferredReferenceId = preferredReference?.id || "";
    const referenceOptions = referenceCandidates
      .map((item) => `<option value="${item.id}" ${item.id === preferredReferenceId ? "selected" : ""}>${esc(item.name || item.id)}</option>`)
      .join("");
    const result = S.designResult;
    const latest = !result && (d.designs || []).length ? d.designs[0] : null;
    let resultHtml = "";
    if (result) {
      const designs = result.designs || [];
      const rows = designs.map((item) => {
        const hit = (item.database_hits || [])[0] || {};
        return `<tr>
          <td>${esc(item.name || item.label)}</td>
          <td>${esc(item.core)}</td>
          <td>${esc((item.substituents || []).join("、"))}</td>
          <td class="aff">${item.affinity != null ? Number(item.affinity).toFixed(2) : "—"}</td>
          <td class="rank">${item.props?.qed != null ? Number(item.props.qed).toFixed(2) : "—"}</td>
          <td class="rank">${item.props?.efficiency?.le != null ? Number(item.props.efficiency.le).toFixed(2) : "—"}</td>
          <td>${esc(hit.name || "—")}</td>
          <td class="rank">${hit.tanimoto != null ? Number(hit.tanimoto).toFixed(2) : "—"}</td>
          <td>${item.index != null ? `<a class="link" href="/api/runs/${encodeURIComponent(result.run_id)}/result/${item.index}/figures.pdf" target="_blank" rel="noopener">矢量图</a>` : "—"}</td>
        </tr>`;
      }).join("");
      resultHtml = `
        <div class="card">
          <div class="card-head"><div><h3>设计结果 · ${esc(result.receptor)}</h3><p>片段生长候选已自动对接，并完成本地数据库相似性检索。</p></div><span class="badge ok">PDF 矢量输出</span></div>
          <div class="card-body">
            <div class="opt-hero">
              <div><span class="eyebrow">最佳设计</span><h2>${esc(result.best?.label || "—")}</h2><p>${designs.length} 个设计通过对接筛选 · 模式 ${esc(result.mode || "auto")} · 骨架跃迁 ${result.scaffold_hops ?? 0} 个</p></div>
              <div class="opt-metric"><strong>${result.best?.affinity != null ? Number(result.best.affinity).toFixed(2) : "—"}</strong><span>Vina kcal/mol</span></div>
            </div>
            <div class="row" style="margin:14px 0;flex-wrap:wrap">
              <a class="btn primary sm" href="/api/designs/${encodeURIComponent(result.run_id)}/report.pdf" target="_blank" rel="noopener">下载设计报告 PDF</a>
              <a class="btn ghost sm" href="/api/reports/full.pdf?run_id=${encodeURIComponent(result.run_id)}&design_id=${encodeURIComponent(result.run_id)}&species=${encodeURIComponent(S.pharmSpecies || "human")}" target="_blank" rel="noopener">完整 PDF 报告</a>
            </div>
            <div class="table-scroll">
              <table class="result-table">
                <thead><tr><th>设计分子</th><th>核心</th><th>取代基</th><th>Vina</th><th>QED</th><th>LE</th><th>最近邻</th><th>Tanimoto</th><th>结构图</th></tr></thead>
                <tbody>${rows}</tbody>
              </table>
            </div>
          </div>
        </div>`;
    } else if (latest) {
      resultHtml = `
        <div class="card">
          <div class="card-head"><div><h3>最近一次设计</h3><p>${esc(latest.receptor_name || "")} · ${latest.count || 0} 个设计</p></div><span class="badge ok">PDF 可下载</span></div>
          <div class="card-body"><div class="row"><a class="btn primary sm" href="/api/designs/${encodeURIComponent(latest.run_id)}/report.pdf" target="_blank" rel="noopener">下载设计报告 PDF</a></div></div>
        </div>`;
    }
    return `
      <div class="card">
        <div class="card-head"><div><h3>针对蛋白口袋生成分子</h3><p>从芳环/杂环核心出发枚举取代基，类药性预筛后自动对接，并在数据库中检索最近邻。</p></div><span class="badge">${receptors.length} 个靶点</span></div>
        <div class="card-body">
          <div class="params-grid">
            <div class="field"><label>靶点 / 口袋</label><select class="select" id="design-receptor">${receptorOptions}</select></div>
            <div class="field"><label>设计模式</label>
              <select class="select" id="design-mode">
                <option value="auto" selected>自动（片段生长 + 骨架跃迁）</option>
                <option value="fragment">仅片段生长</option>
                <option value="scaffold">仅骨架跃迁</option>
                <option value="hybrid">混合探索</option>
              </select>
            </div>
            <div class="field"><label>参考配体（可选，用于骨架跃迁）</label>
              <select class="select" id="design-reference">
                <option value="">不使用参考配体</option>
                ${referenceOptions}
              </select>
            </div>
            <div class="field"><label>设计数量</label><input class="input" id="design-count" type="number" min="6" max="60" value="20"></div>
            <div class="field"><label>Exhaustiveness</label><input class="input" id="design-exhaustiveness" type="number" min="1" max="16" value="8"></div>
            <div class="field"><label>计算引擎</label>
              <select class="select" id="design-engine">
                <option value="cpu" selected>CPU · AutoDock Vina</option>
                <option value="gpu" ${d.system?.gpu_available ? "" : "disabled"}>GPU · Vina-GPU${d.system?.gpu_available ? "" : "（未安装）"}</option>
              </select>
            </div>
            <div class="field"><label>并行任务数</label><input class="input" id="design-parallel" type="number" min="1" max="8" value="2"></div>
            <div class="field"><label>随机种子</label><input class="input" id="design-seed" type="number" min="1" value="42"></div>
          </div>
          <div class="parent-strip">
            <span>靶点 <strong>${esc(receptor.name)}</strong></span>
            <span>口袋 <strong>${esc(receptor.box?.label || "自定义")}</strong></span>
            <span>盒子中心 <strong>${(receptor.box?.center || []).map((v) => Number(v).toFixed(1)).join(", ")}</strong></span>
          </div>
          <div class="field" style="margin-top:14px">
            <label>本地化合物数据库（可选，每行 <span class="mono">SMILES 名称</span>；留空则使用内置参考集 + 项目配体）</label>
            <textarea class="textarea" id="design-db" rows="3" placeholder="CC(=O)Oc1ccccc1C(=O)O aspirin&#10;Cn1c(=O)c2c(ncn2C)n(C)c1=O caffeine"></textarea>
          </div>
          <div class="row" style="margin-top:12px;flex-wrap:wrap">
            <button class="btn primary" data-action="design-start">生成分子并自动筛选</button>
            <button class="btn ghost sm" data-action="design-save-db">保存数据库</button>
            <span class="muted" style="font-size:11px">生成 6–60 个类药分子；片段生长覆盖单/双/三取代，参考配体模式下同时执行骨架跃迁、杂环替换与连接子改造。</span>
          </div>
          <div class="note" style="margin-top:12px">
            <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M12 8h.01M11 12h1v5h1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
            <span>片段生长覆盖 29 类芳环/杂环核心与 22 类取代基；参考配体模式会把苯环→吡啶/嘧啶、五元杂环替换、连接子改造等骨架跃迁候选一并纳入筛选。设计结果为计算假设；数据库检索使用 Morgan 指纹（半径 2，2048 bit）与 Tanimoto 相似度。</span>
          </div>
        </div>
      </div>
      ${resultHtml}`;
  }

  function renderFishingStep() {
    const d = S.data;
    const latest = (d.fishing || [])[0];
    const result = S.fishingResult;
    const record = result || null;
    const targets = record?.targets || [];
    const runId = d.runs?.[0]?.id || "";
    const pickedResult = activeResult();
    const pickedLigand = resultLigand(pickedResult);
    // Candidate list: only ligands that actually took part in fine docking.
    // The ligand selected in "结果分析" is always included, even if it came
    // from a bulk screen.
    const fineLigands = (d.fine_ligands || []).filter((item) => item.smiles);
    const preferredLigandId = S.fishingLigandId || pickedLigand?.id || S.ligandId;
    const activeLigand = ligandById(preferredLigandId) || pickedLigand || fineLigands[0] || null;
    const candidates = [...fineLigands];
    if (activeLigand && !candidates.some((item) => item.id === activeLigand.id)) {
      candidates.unshift(activeLigand);
    }
    const ligandOptions = candidates
      .map((item) => `<option value="${esc(item.id)}" ${item.id === activeLigand?.id ? "selected" : ""}>${esc(item.name || item.id)}</option>`)
      .join("");
    const activeSmiles = activeLigand?.smiles || activeLigand?.meta?.smiles || "";
    const sourceSummary = Object.entries(record?.stats?.sources || {}).map(([name, count]) => `${esc(name)} ${count}`).join(" · ");
    const external = record?.external_sources || {};
    const externalText = Object.entries(external).map(([name, info]) => `${esc(name)}: ${info.available ? `可用 ${info.count || 0}` : "不可用"}${info.error ? `（${esc(info.error)}）` : ""}`).join("；");
    const rows = targets.slice(0, 20).map((item) => `
      <tr>
        <td><strong>${esc(item.gene || "—")}</strong></td>
        <td>${esc(item.name || "")}</td>
        <td>${esc((item.sources || ["ChEMBL"]).join(" / "))}</td>
        <td class="rank">${item.n ?? "—"}</td>
        <td class="aff">${item.median_pchembl != null ? Number(item.median_pchembl).toFixed(2) : "—"}</td>
        <td class="rank">${item.stp_probability != null ? Number(item.stp_probability).toFixed(2) : "—"}</td>
      </tr>`).join("");
    const resultCard = targets.length
      ? `<div class="card">
          <div class="card-head"><div><h3>垂钓结果 · ${esc(record.compound?.name || "查询分子")}</h3><p>${record.target_count || targets.length} 个候选靶点 / ${record.gene_count || "—"} 个基因；来源：${sourceSummary || "ChEMBL"}。</p></div><div class="row" style="gap:6px"><span class="badge ok">可并入完整报告</span><a class="btn ghost sm" href="/api/fishing/${encodeURIComponent(record.id)}/report.pdf" target="_blank" rel="noopener">独立报告 PDF</a></div></div>
          <div class="card-body">
            ${externalText ? `<div class="note" style="margin-bottom:10px">外部源状态：${externalText}</div>` : ""}
            <div class="table-scroll">
              <table class="result-table">
                <thead><tr><th>基因</th><th>靶点</th><th>证据来源</th><th>记录数</th><th>pChEMBL</th><th>STP 概率</th></tr></thead>
                <tbody>${rows}</tbody>
              </table>
            </div>
            <div class="row" style="margin-top:14px;flex-wrap:wrap">
              <button class="btn primary sm" data-action="goto-netpharm">生成网络药理学</button>
              ${runId ? `<a class="btn ghost sm" href="/api/reports/full.pdf?run_id=${encodeURIComponent(runId)}&fishing_id=${encodeURIComponent(record.id)}&species=${encodeURIComponent(S.pharmSpecies || "human")}" target="_blank" rel="noopener">完整 PDF 报告</a>` : ""}
            </div>
          </div>
        </div>`
      : latest
        ? `<div class="card"><div class="card-head"><div><h3>最近一次垂钓</h3><p>${esc(latest.compound?.name || "")} · ${latest.target_count || latest.targets?.length || 0} 个靶点</p></div></div>
           <div class="card-body"><button class="btn ghost sm" data-action="fishing-load" data-id="${esc(latest.id)}">载入结果</button></div></div>`
        : "";
    return `
      <div class="card">
        <div class="card-head"><div><h3>多源反向靶点垂钓</h3><p>ChEMBL 相似性 + SwissTargetPrediction 2D/3D + BindingDB + PubChem + DGIdb + Open Targets + PDB，并支持导入 PharmMapper / SEA / COMET 等结果。</p></div><span class="badge">多源融合</span></div>
        <div class="card-body">
          <div class="row" style="flex-wrap:wrap;align-items:end">
            <div class="field" style="flex:1;min-width:200px"><label>分子名称</label><input class="input" id="fishing-name" placeholder="例如 Imatinib" value="${esc(S.fishingTouched ? S.fishingName : (activeLigand?.name || ""))}"></div>
            ${ligandOptions ? `<div class="field" style="flex:1;min-width:220px"><label>从精细对接配体选择（默认跟随结果分析）</label><select class="input" id="fishing-ligand">${ligandOptions}</select></div>` : ""}
            <button class="btn ghost sm" data-action="fishing-use-ligand" data-smiles="${esc(activeSmiles)}" data-name="${esc(activeLigand?.name || "")}">填入当前项目配体${activeLigand?.name ? `（${esc(activeLigand.name)}）` : ""}</button>
          </div>
          <div class="field"><label>SMILES（默认使用项目配体，可手动覆盖）</label><textarea class="textarea" id="fishing-smiles" rows="3" placeholder="CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CC=CN=C5">${esc(S.fishingTouched ? S.fishingSmiles : activeSmiles)}</textarea></div>
          <div class="row" style="margin-top:9px;flex-wrap:wrap">
            <input type="file" id="fishing-sdf" accept=".sdf,.sd,.mol,.smi,.txt,.csv,.tsv">
            <button class="btn ghost sm" data-action="fishing-sdf">从 SDF/SMILES 提取</button>
            <button class="btn ghost sm" data-action="fishing-to-sdf">SMILES → SDF</button>
          </div>
          <div class="params-grid" style="margin-top:10px">
            <div class="field"><label>垂钓深度</label><select class="input" id="fishing-depth"><option value="deep" selected>深度（220 靶点，适合网络药理学）</option><option value="standard">标准（60 靶点）</option></select></div>
            <div class="field"><label>相似度阈值 (%)</label><input class="input" id="fishing-similarity" type="number" min="40" max="100" value="40"></div>
            <div class="field"><label>相似分子上限</label><input class="input" id="fishing-molecules" type="number" min="10" max="200" value="120"></div>
            <div class="field"><label>输出靶点数</label><input class="input" id="fishing-targets" type="number" min="20" max="300" value="220"></div>
          </div>
          <div class="params-grid" style="margin-top:8px">
            <div class="field"><label>SwissTargetPrediction 物种（可多选，默认全部）</label>
              <div class="row" style="flex-wrap:wrap;gap:14px;margin-top:6px">
                <label class="check"><input type="checkbox" class="fishing-organism" value="Homo_sapiens" checked> 人</label>
                <label class="check"><input type="checkbox" class="fishing-organism" value="Mus_musculus" checked> 小鼠</label>
                <label class="check"><input type="checkbox" class="fishing-organism" value="Rattus_norvegicus" checked> 大鼠</label>
              </div>
            </div>
            <div class="field" style="grid-column:span 2"><label>外部预测基因导入（可选，逗号/空格/换行分隔）</label><textarea class="textarea" id="fishing-external-genes" rows="3" placeholder="来自 PharmMapper / SEA / idTarget / COMET / LigTMap 的基因列表，例如：EGFR, PTGS2, CA2">${esc(S.fishingExternalGenes || "")}</textarea></div>
          </div>
          <div class="row" style="margin-top:12px;flex-wrap:wrap">
            <button class="btn primary" data-action="fishing-start">开始多源垂钓</button>
            <button class="btn ghost sm" data-action="fishing-methods">生成论文方法段落</button>
          </div>
          <p class="hint" style="margin-top:10px">SwissTargetPrediction 已直接集成（人/小鼠/大鼠）；PharmMapper、SEA、idTarget、TarFisDock、COMET、LigTMap、PLATO、ReverseDock 没有稳定公开 API，可在对应网站运行后把基因列表粘贴到“外部预测基因导入”，程序会合并去重。若 ChEMBL/SwissTargetPrediction 暂时不可用，程序会自动用本地 FDA 相似药物靶点、DGIdb、PubChem、BindingDB、PDB 同源结构回退，保证至少给出可用的靶点假设。所有结果仅为 in silico 预筛选，需 SPR/CETSA/亲和捕获等实验验证。</p>
          <div class="row" style="margin-top:8px;flex-wrap:wrap;font-size:11px">
            <a href="https://swisstargetprediction.ch/" target="_blank" rel="noopener">SwissTargetPrediction</a>
            <a href="http://www.lilab-ecust.cn/pharmmapper/" target="_blank" rel="noopener">PharmMapper</a>
            <a href="https://sea.bkslab.org/" target="_blank" rel="noopener">SEA</a>
            <a href="http://idtarget.rcas.sinica.edu.tw/" target="_blank" rel="noopener">idTarget</a>
            <a href="http://www.dddc.ac.cn/tarfisdock/" target="_blank" rel="noopener">TarFisDock</a>
            <a href="https://www.pdbbind-plus.org.cn/comet" target="_blank" rel="noopener">COMET</a>
            <a href="https://cbbio.online/LigTMap" target="_blank" rel="noopener">LigTMap</a>
            <a href="http://plato.uniba.it/" target="_blank" rel="noopener">PLATO</a>
            <a href="https://reversedock.biologie.uni-freiburg.de/" target="_blank" rel="noopener">ReverseDock</a>
          </div>
        </div>
      </div>
      ${resultCard}`;
  }

  function renderNetpharmStep() {
    const d = S.data;
    const fishRecords = d.fishing || [];
    const selectedFishId = S.netpharmFishingId || fishRecords[0]?.id || "";
    const selectedFish = fishRecords.find((item) => item.id === selectedFishId) || fishRecords[0];
    const fishOptions = fishRecords.map((item) => `<option value="${esc(item.id)}" ${item.id === selectedFishId ? "selected" : ""}>${esc(item.compound?.name || "查询分子")} · ${item.target_count || item.targets?.length || 0} 靶点 · ${fmtTime(item.created)}</option>`).join("");
    const latestNet = (d.netpharm || [])[0];
    const result = S.netpharmResult;
    const record = result || null;
    const runId = d.runs?.[0]?.id || "";
    const keggRows = (record?.kegg || []).slice(0, 6).map((item) => `
      <tr><td>${esc(item.term)}</td><td class="rank">${Number(item.p).toExponential(1)}</td><td class="rank">${item.fdr != null ? Number(item.fdr).toExponential(1) : "—"}</td></tr>`).join("");
    const goRows = (record?.go || []).slice(0, 6).map((item) => `
      <tr><td>${esc(item.term)}</td><td class="rank">${Number(item.p).toExponential(1)}</td><td class="rank">${item.fdr != null ? Number(item.fdr).toExponential(1) : "—"}</td></tr>`).join("");
    const libTitles = { kegg: "KEGG", go_bp: "GO BP", go_cc: "GO CC", go_mf: "GO MF", reactome: "Reactome", wikipathway: "WikiPathway", disgenet: "DisGeNET", hallmark: "Hallmark" };
    const libRows = Object.entries(record?.enrichment || {}).map(([key, block]) => {
      const first = (block?.terms || [])[0] || {};
      return `<tr><td>${esc(libTitles[key] || key)}</td><td class="rank">${block?.significant ?? 0}</td><td>${esc(first.term || "—")}</td><td class="rank">${first.p != null ? Number(first.p).toExponential(1) : "—"}</td></tr>`;
    }).join("");
    const resultCard = record
      ? `<div class="card">
          <div class="card-head"><div><h3>网络药理学结果</h3><p>${record.genes?.length || 0} 个基因 · ${record.ppi?.edges?.length || 0} 条互作边</p></div><div class="row" style="gap:6px"><span class="badge ok">可并入完整报告</span><a class="btn ghost sm" href="/api/netpharm/${encodeURIComponent(record.id)}/report.pdf" target="_blank" rel="noopener">独立报告 PDF</a></div></div>
          <div class="card-body">
            <div class="opt-hero">
              <div><span class="eyebrow">KEGG 最显著通路</span><h2 style="font-size:16px">${esc(record.kegg?.[0]?.term || "—")}</h2><p>${record.stats?.kegg_significant || 0} 条 FDR &lt; 0.05</p></div>
              <div class="opt-metric"><strong>${record.stats?.go_significant || 0}</strong><span>GO 显著条目</span></div>
              <div class="opt-metric"><strong>${record.ctd?.nodes || 0}</strong><span>C-T-D 节点</span></div>
              <div class="opt-metric"><strong>${record.sankey?.pathways || 0}</strong><span>桑基通路</span></div>
            </div>
            ${libRows ? `<div class="table-scroll" style="margin-top:12px">
              <table class="result-table">
                <thead><tr><th>Enrichr 库</th><th>显著条目</th><th>最显著条目</th><th>P</th></tr></thead>
                <tbody>${libRows}</tbody>
              </table>
            </div>` : ""}
            <div class="table-scroll" style="margin-top:12px">
              <table class="result-table">
                <thead><tr><th>KEGG 通路</th><th>P</th><th>FDR</th></tr></thead>
                <tbody>${keggRows}</tbody>
              </table>
            </div>
            <div class="table-scroll" style="margin-top:12px">
              <table class="result-table">
                <thead><tr><th>GO 生物学过程</th><th>P</th><th>FDR</th></tr></thead>
                <tbody>${goRows}</tbody>
              </table>
            </div>
            <div class="row" style="margin-top:14px;flex-wrap:wrap">
              ${runId ? `<a class="btn ghost sm" href="/api/reports/full.pdf?run_id=${encodeURIComponent(runId)}&netpharm_id=${encodeURIComponent(record.id)}&species=${encodeURIComponent(S.pharmSpecies || "human")}" target="_blank" rel="noopener">完整 PDF 报告</a>` : ""}
            </div>
          </div>
        </div>`
      : latestNet
        ? `<div class="card"><div class="card-head"><div><h3>最近一次网络药理学</h3><p>${latestNet.genes?.length || 0} 个基因 · ${latestNet.edges || 0} 条边</p></div></div>
           <div class="card-body"><button class="btn ghost sm" data-action="netpharm-load" data-id="${esc(latestNet.id)}">载入结果</button></div></div>`
        : "";
    return `
      <div class="card">
        <div class="card-head"><div><h3>网络药理学分析</h3><p>读取靶点垂钓的基因集，构建 STRING 互作网络，完成多库富集、C-T-D 网络与靶点–通路桑基图。</p></div><span class="badge">STRING + Enrichr 多库</span></div>
        <div class="card-body">
          ${fishRecords.length
            ? `<div class="field"><label>使用哪一次靶点垂钓结果</label><select class="input" id="netpharm-fishing">${fishOptions}</select></div>
               <div class="note" style="margin-top:10px">当前基因集来自：<strong>${esc(selectedFish?.compound?.name || "查询分子")}</strong> · ${selectedFish?.target_count || selectedFish?.targets?.length || 0} 个候选靶点。网络药理学不再单独输入 SMILES/SDF，直接用这次垂钓的基因符号建网与富集。</div>`
            : `<div class="note warn">还没有靶点垂钓结果。请先完成“靶点垂钓”，网络药理学将直接使用其靶点与基因集。</div>
               <div class="row" style="margin-top:10px"><button class="btn primary" data-action="goto-fishing">去靶点垂钓</button></div>`}
          <div class="field" style="margin-top:9px"><label>补充基因（可选，逗号/空格分隔；用于文献或实验靶点）</label><textarea class="textarea" id="netpharm-extra-genes" rows="2" placeholder="EGFR, AKT1, MTOR"></textarea></div>
          <div class="params-grid" style="margin-top:12px">
            <div class="field"><label>参与基因数</label><input class="input" id="netpharm-genes" type="number" min="5" max="150" value="80"></div>
            <div class="field"><label>STRING 置信度</label><input class="input" id="netpharm-score" type="number" min="150" max="900" step="50" value="400"></div>
            <div class="field"><label>一阶邻居扩展</label><input class="input" id="netpharm-expand" type="number" min="0" max="50" value="15"></div>
          </div>
          <div class="row" style="margin-top:14px"><button class="btn primary" data-action="netpharm-start" ${fishRecords.length ? "" : "disabled"}>开始网络药理学分析</button></div>
          <p class="hint" style="margin-top:10px">将完成 STRING PPI、KEGG / GO BP / GO CC / GO MF / Reactome / WikiPathways / DisGeNET / Hallmark 多库富集，并生成 C-T-D 网络与靶点–通路桑基图。若 STRING 无互作边或所有库均无显著条目，会直接报错而不生成空图；富集结果只用于机制假设。</p>
        </div>
      </div>
      ${resultCard}`;
  }

  function renderSynthesisStep() {
    const d = S.data;
    const latest = (d.synthesis || [])[0];
    const record = S.synthesisResult || null;
    const result = activeResult();
    const ligand = resultLigand(result);
    const activeLigand = ligand || ligandById(S.ligandId) || (d.ligands || [])[0];
    const ligandSmiles = activeLigand?.smiles || activeLigand?.meta?.smiles || ligand?.smiles || ligand?.meta?.smiles || "";
    const best = record?.best_route || null;
    const badgeTone = record
      ? (record.sa != null && record.sa <= 3.0 ? "ok" : record.sa != null && record.sa <= 4.0 ? "" : "warn")
      : "";
    const routeRows = (record?.routes || []).map((route) => `
      <tr>
        <td class="rank">#${route.rank ?? "—"}</td>
        <td>${esc(route.label || route.reaction || "—")}</td>
        <td class="rank">${route.step_count ?? (route.steps || []).length}</td>
        <td class="rank">${route.overall_yield != null ? Number(route.overall_yield).toFixed(1) + "%" : "—"}</td>
        <td class="rank">${route.score != null ? Number(route.score).toFixed(1) : "—"}</td>
      </tr>`).join("");
    const stepRows = (best?.steps || []).map((step, index) => `
      <tr>
        <td class="rank">${index + 1}</td>
        <td>${esc(step.reaction)}</td>
        <td>${esc(step.conditions || "—")}</td>
        <td class="rank">${step.yield != null ? (Number(step.yield) * 100).toFixed(0) + "%" : "—"}</td>
      </tr>`).join("");
    const precursorRows = (best?.leaves || []).map((leaf) => `
      <tr>
        <td class="mono" style="font-size:10px;word-break:break-all">${esc(leaf.smiles)}</td>
        <td>${esc(leaf.formula || "—")}</td>
        <td class="rank">${leaf.heavy ?? "—"}</td>
        <td class="rank">${leaf.sa != null ? Number(leaf.sa).toFixed(2) : "—"}</td>
        <td class="rank">${leaf.common ? "是" : "否"}</td>
      </tr>`).join("");
    const metricCells = record
      ? [
          ["MW", record.metrics?.mw?.toFixed(2)],
          ["重原子", record.metrics?.heavy],
          ["环数", record.metrics?.rings],
          ["芳香环", record.metrics?.aromatic_rings],
          ["可旋转键", record.metrics?.rotb],
          ["手性中心", record.metrics?.stereocenters],
          ["Fsp3", record.metrics?.fsp3?.toFixed(3)],
          ["Bertz", record.metrics?.bertz?.toFixed(1)],
        ].map(([label, value]) => `<div class="count-cell"><span>${label}</span><strong>${value ?? "—"}</strong></div>`).join("")
      : "";
    const resultCard = record
      ? `<div class="card">
          <div class="card-head"><div><h3>可合成性评估 · ${esc(record.compound?.name || "候选分子")}</h3><p class="mono" style="font-size:10px;word-break:break-all">${esc(record.compound?.smiles || "")}</p></div><span class="badge ${badgeTone}">${esc(record.verdict || "")}</span></div>
          <div class="card-body">
            <div class="opt-hero">
              <div><span class="eyebrow">SAscore（1 易 – 10 难）</span><h2 style="font-size:22px">${record.sa != null ? Number(record.sa).toFixed(2) : "—"}</h2><p>${esc(record.verdict || "")}</p></div>
              <div class="opt-metric"><strong>${best?.step_count ?? (best?.steps || []).length ?? 0}</strong><span>首选路线步数</span></div>
              <div class="opt-metric"><strong>${best?.overall_yield != null ? Number(best.overall_yield).toFixed(0) + "%" : "—"}</strong><span>预计总产率</span></div>
              <div class="opt-metric"><strong>${(record.routes || []).length}</strong><span>候选路线</span></div>
            </div>
            <div class="count-grid" style="margin-top:12px">${metricCells}</div>
            ${routeRows ? `<div class="table-scroll" style="margin-top:14px">
              <table class="result-table">
                <thead><tr><th>排名</th><th>路线</th><th>步数</th><th>总产率</th><th>评分</th></tr></thead>
                <tbody>${routeRows}</tbody>
              </table>
            </div>` : `<div class="note warn" style="margin-top:12px">当前结构未匹配到反应模板，也没有可用的 BRICS 断键方案。可尝试补充立体信息或简化目标分子。</div>`}
            ${stepRows ? `<h3 style="margin:18px 0 8px;font-size:13px">首选路线分步</h3><div class="table-scroll">
              <table class="result-table">
                <thead><tr><th>步骤</th><th>反应类型</th><th>建议条件</th><th>经验产率</th></tr></thead>
                <tbody>${stepRows}</tbody>
              </table>
            </div>` : ""}
            ${precursorRows ? `<h3 style="margin:18px 0 8px;font-size:13px">建议起始砌块</h3><div class="table-scroll">
              <table class="result-table">
                <thead><tr><th>SMILES</th><th>分子式</th><th>重原子</th><th>SAscore</th><th>常见砌块</th></tr></thead>
                <tbody>${precursorRows}</tbody>
              </table>
            </div>` : ""}
            ${(record.flags || []).length ? `<div class="note warn" style="margin-top:14px"><span>${record.flags.map((item) => esc(item)).join(" ")}</span></div>` : ""}
            ${(record.recommendations || []).length ? `<div style="margin-top:12px"><h3 style="margin:0 0 6px;font-size:13px">建议</h3><ul style="margin:0;padding-left:18px;font-size:11px;line-height:1.7">${record.recommendations.map((item) => `<li>${esc(item)}</li>`).join("")}</ul></div>` : ""}
            <div class="row" style="margin-top:14px;flex-wrap:wrap">
              <a class="btn ghost sm" href="/api/synthesis.pdf?record_id=${encodeURIComponent(record.id)}" target="_blank" rel="noopener">可合成性 PDF</a>
            </div>
            <p class="hint" style="margin-top:10px">路线来自有限反应模板库与 BRICS 断键，产率为经验估计；请把结果当作路线假设，再用文献/供应商核实。</p>
          </div>
        </div>`
      : latest
        ? `<div class="card"><div class="card-head"><div><h3>最近一次可合成性评估</h3><p>${esc(latest.compound?.name || "候选分子")} · ${esc(latest.verdict || "")}</p></div></div>
           <div class="card-body"><button class="btn ghost sm" data-action="synthesis-load" data-id="${esc(latest.id)}">载入结果</button></div></div>`
        : "";
    return `
      <div class="card">
        <div class="card-head"><div><h3>可合成性与合成路线</h3><p>计算 SAscore、识别手性/大环/反应性硬步骤，并用反应模板给出逆合成路线与起始砌块。</p></div><span class="badge">SAscore + 模板逆合成</span></div>
        <div class="card-body">
          <div class="field"><label>化合物名称</label><input class="input" id="synth-name" placeholder="例如 Imatinib" value="${esc(activeLigand?.name || record?.compound?.name || "")}"></div>
          <div class="field" style="margin-top:9px"><label>SMILES（默认使用结果分析选中的配体，可手动覆盖）</label><input class="input mono" id="synth-smiles" placeholder="例如 CC(=O)Oc1ccccc1C(=O)O" value="${esc(ligandSmiles || record?.compound?.smiles || "")}"></div>
          ${ligandSmiles ? `<div class="row" style="margin-top:9px"><button class="btn ghost sm" data-action="synth-use-ligand" data-smiles="${esc(ligandSmiles)}" data-name="${esc(activeLigand?.name || ligand?.name || "")}">填入当前项目配体${activeLigand?.name ? `（${esc(activeLigand.name)}）` : ""}</button></div>` : ""}
          <div class="row" style="margin-top:14px"><button class="btn primary" data-action="synthesis-start">评估可合成性</button></div>
          <p class="hint" style="margin-top:10px">SAscore 为 RDKit 贡献包实现的相对可合成性评分；路线由酰胺、酯、Suzuki、还原胺化等模板逐层拆解，未命中模板时回退到 BRICS 断键。</p>
        </div>
      </div>
      ${resultCard}`;
  }

  function renderAdmetStep() {
    const d = S.data;
    const result = activeResult();
    const ligand = resultLigand(result);
    const activeLigand = ligand || ligandById(S.ligandId) || (d.ligands || [])[0];
    const ligandSmiles = activeLigand?.smiles || activeLigand?.meta?.smiles || ligand?.smiles || ligand?.meta?.smiles || "";
    const profile = S.admetResult;
    const batch = S.admetBatch || [];
    const endpoint = (item, key) => {
      const found = (item?.admet || []).find((entry) => entry.key === key);
      return found ? `${found.value}${found.unit ? " " + found.unit : ""}` : "—";
    };
    const batchRows = batch.map((item) => `
      <tr>
        <td><strong>${esc(item.name)}</strong><div class="mono" style="font-size:9px;color:var(--muted-2);word-break:break-all">${esc(item.smiles)}</div></td>
        <td class="rank">${Number(item.physchem.mw).toFixed(1)}</td>
        <td class="rank">${Number(item.physchem.logp).toFixed(2)}</td>
        <td class="rank">${Number(item.physchem.tpsa).toFixed(0)}</td>
        <td class="rank">${Number(item.druglike.qed).toFixed(3)}</td>
        <td class="rank">${endpoint(item, "herg")}</td>
        <td class="rank">${endpoint(item, "cyp3a4")}</td>
        <td class="rank">${endpoint(item, "cyp2d6")}</td>
        <td class="rank">${endpoint(item, "ames")}</td>
        <td class="rank">${endpoint(item, "half_life")}</td>
        <td class="rank">${item.druglike.rules.lipinski.pass ? "通过" : `${item.druglike.rules.lipinski.violations} 违反`}</td>
      </tr>`).join("");
    const singleCard = profile
      ? `<div class="card">
          <div class="card-head"><div><h3>ADMET 结果 · ${esc(profile.name)}</h3><p>${esc(profile.physchem.formula)} · ${esc(profile.smiles)}</p></div><span class="badge ok">可导出 PDF</span></div>
          <div class="card-body" style="padding:0">
            ${admetBlocks(profile)}
            <div class="row" style="padding:0 18px 16px;flex-wrap:wrap">
              <a class="btn primary sm" href="/api/admet.pdf?smiles=${encodeURIComponent(profile.smiles)}&name=${encodeURIComponent(profile.name)}" target="_blank" rel="noopener">下载 ADMET PDF</a>
              <a class="btn ghost sm" href="/api/reports/full.pdf?run_id=${encodeURIComponent(activeRun()?.id || d.runs?.[0]?.id || "")}&result_index=${S.resultIndex}&species=${encodeURIComponent(S.pharmSpecies || "human")}" target="_blank" rel="noopener">并入完整报告</a>
            </div>
          </div>
        </div>`
      : "";
    return `
      <div class="card">
        <div class="card-head"><div><h3>ADMET 与类药性</h3><p>独立的吸收、分布、代谢、排泄与毒性分析；支持单分子和批量 SMILES。</p></div><span class="badge">RDKit + 经验模型</span></div>
        <div class="card-body">
          <div class="field"><label>分子名称</label><input class="input" id="admet-name" placeholder="例如 Imatinib" value="${esc(activeLigand?.name || profile?.name || "")}"></div>
          <div class="field" style="margin-top:9px"><label>SMILES（默认使用结果分析选中的配体，可手动覆盖）</label><input class="input mono" id="admet-single-smiles" placeholder="例如 CC(=O)Oc1ccccc1C(=O)O" value="${esc(ligandSmiles || profile?.smiles || "")}"></div>
          <div class="row" style="margin-top:9px;flex-wrap:wrap">
            ${ligandSmiles ? `<button class="btn ghost sm" data-action="admet-use-ligand" data-smiles="${esc(ligandSmiles)}" data-name="${esc(activeLigand?.name || "")}">填入当前项目配体${activeLigand?.name ? `（${esc(activeLigand.name)}）` : ""}</button>` : ""}
            <button class="btn primary sm" data-action="admet-run">计算 ADMET</button>
            <button class="btn ghost sm" data-action="admet-pdf-single">下载当前 PDF</button>
          </div>
          <div class="field" style="margin-top:14px"><label>批量 SMILES（每行一个，可写“名称, SMILES”）</label><textarea class="textarea mono" id="admet-batch-input" rows="5" placeholder="Aspirin, CC(=O)Oc1ccccc1C(=O)O&#10;Ibuprofen, CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"></textarea></div>
          <div class="row" style="margin-top:9px"><button class="btn ghost sm" data-action="admet-batch">批量计算</button></div>
          ${(S.admetErrors || []).length ? `<div class="note warn" style="margin-top:10px">${S.admetErrors.map((item) => `${esc(item.name)}：${esc(item.error)}`).join("；")}</div>` : ""}
          ${batchRows ? `<div class="table-scroll" style="margin-top:14px">
            <table class="result-table">
              <thead><tr><th>分子</th><th>MW</th><th>cLogP</th><th>TPSA</th><th>QED</th><th>hERG</th><th>CYP3A4</th><th>CYP2D6</th><th>Ames</th><th>t½</th><th>Lipinski</th></tr></thead>
              <tbody>${batchRows}</tbody>
            </table>
          </div>` : ""}
          <p class="hint" style="margin-top:10px">ADMET 端点来自 RDKit 描述符与公开经验相关式，用于系列内排序和趋势判断，不替代实验测定。批量结果最多 60 个分子。</p>
        </div>
      </div>
      ${singleCard}`;
  }

  function renderPharmStep() {
    const d = S.data;
    const p = S.pharmResult;
    const pickedResult = activeResult();
    const pickedLigand = resultLigand(pickedResult);
    const ligands = [...(d.fine_ligands || [])].filter(
      (item) => item.smiles || item.meta?.smiles
    );
    if (pickedLigand && !ligands.some((item) => item.id === pickedLigand.id)) {
      ligands.unshift(pickedLigand);
    }
    const pickedLigandId = pickedLigand?.id || "";
    const ligandOptions = ligands.map((item) => `<option value="${esc(item.id)}" ${item.id === pickedLigandId ? "selected" : ""}>${esc(item.name || item.id)}${item.formula || item.meta?.formula ? ` · ${esc(item.formula || item.meta.formula)}` : ""}</option>`).join("");
    const speciesOptions = [["human", "人"], ["pig", "猪"], ["cattle", "牛"], ["sheep", "羊"], ["dog", "犬"], ["cat", "猫"], ["rat", "大鼠"]]
      .map(([value, label]) => `<option value="${value}" ${value === (p?.species || S.pharmSpecies || "human") ? "selected" : ""}>${label}</option>`)
      .join("");
    const conf = p?.confidence || {};
    const risk = p?.risk || {};
    const oral = p?.oral || {};
    const rumen = p?.rumen || {};
    const microbiota = p?.microbiota || {};
    const nucleic = p?.nucleic_acid || {};
    const stability = p?.stability || {};
    const formulation = p?.formulation || {};
    const tissue = p?.tissue || {};
    const onlineData = p?.online || {};
    const endpointRows = (risk.endpoints || []).map((item) => `<tr><td>${esc(item.name)}</td><td class="rank">${Number(item.value).toFixed(1)}</td><td><span class="badge ${item.level === "高" ? "err" : item.level === "中" ? "warn" : "ok"}">${esc(item.level)}</span></td><td>${esc(item.note)}</td></tr>`).join("");
    const tissueRows = (tissue.kp_rows || []).slice(0, 8).map((item) => `<tr><td>${esc(item.tissue)}</td><td class="rank">${Number(item.kp).toFixed(2)}</td><td class="rank">${Number(item.share).toFixed(1)}%</td></tr>`).join("");
    const targetRows = (tissue.target_rows || []).map((item) => `<tr><td>${esc(item.gene)}</td><td>${esc(item.tissue || "—")}</td><td class="rank">${esc(item.ntpm || "—")}</td><td>${esc(item.specificity || "")}</td></tr>`).join("");
    const onlineRows = [];
    const pubchem = onlineData.compound?.pubchem || {};
    const chembl = onlineData.compound?.chembl || {};
    if (pubchem.available) onlineRows.push(`<tr><td>PubChem</td><td>CID ${esc(pubchem.cid)}</td><td>${esc(pubchem.formula || "")} · XLogP ${esc(pubchem.xlogp ?? "—")}</td></tr>`);
    if (chembl.available) onlineRows.push(`<tr><td>ChEMBL</td><td>${esc(chembl.chembl_id)}</td><td>${esc(chembl.pref_name || "")} · phase ${esc(chembl.max_phase ?? "—")}</td></tr>`);
    for (const item of (onlineData.targets?.targets || []).slice(0, 6)) {
      const hpa = item.hpa || {};
      onlineRows.push(`<tr><td>${esc(item.gene)}</td><td>${item.available ? "UniProt" : "不可用"}</td><td>${esc(hpa.rna_specificity || item.protein_name || item.error || "")}</td></tr>`);
    }
    const resultCard = p ? `
      <div class="card" style="margin-top:14px">
        <div class="card-head"><div><h3>综合药理结果 · ${esc(p.name)}</h3><p>物种：${esc(p.species_label)} · ${esc(p.smiles)}</p></div><span class="badge ok">置信度 ${esc(conf.level || "—")}</span></div>
        <div class="card-body">
          <div class="row" style="flex-wrap:wrap;gap:8px">
            <span class="chip">口服 F ≈ ${Number(oral.f_percent || 0).toFixed(0)}%</span>
            <span class="chip">BCS ${esc(oral.bcs || "—")}</span>
            <span class="chip">${rumen.applicable ? `过瘤胃 ≈ ${Number(rumen.bypass_percent || 0).toFixed(0)}%` : "非反刍物种"}</span>
            <span class="chip">菌群 ${esc(microbiota.category || "—")}</span>
            <span class="chip">核酸结合 ${esc(nucleic.category || "—")}</span>
            <span class="chip">综合风险 ${esc(risk.level || "—")}（${Number(risk.overall || 0).toFixed(0)}）</span>
          </div>
          <div class="pharm-grid" style="margin-top:14px">
            <div>
              <h4>口服、瘤胃与菌群</h4>
              <p><strong>BCS ${esc(oral.bcs || "—")}</strong>；HIA ${Number(oral.hia || 0).toFixed(1)}%；Caco-2 ${Number(oral.caco2 || 0).toFixed(2)}；剂量数 ${Number(oral.dose_number || 0).toFixed(2)}。</p>
              <p>${esc(oral.food_effect || "")}</p>
              <p>${rumen.applicable ? `天然过瘤胃 ${Number(rumen.bypass_percent || 0).toFixed(0)}%，保护后约 ${Number(rumen.protected_bypass_percent || 0).toFixed(0)}%。${esc(rumen.strategy || "")}` : esc(rumen.note || "该物种不适用过瘤胃评估。")}</p>
              <p>菌群：${esc(microbiota.direction || "")}；结构警示：${esc((microbiota.alerts || []).join("、") || "无")}。</p>
            </div>
            <div>
              <h4>稳定性与剂型</h4>
              <p>稳定性 ${esc(stability.category || "—")}（${Number(stability.overall || 0).toFixed(0)}/100）。${esc(stability.recommendation || "")}</p>
              <p>建议剂型：${esc(formulation.form || "—")}；制剂风险 ${esc(formulation.risk || "—")}。</p>
              <p>${esc(formulation.salt_or_cocrystal || "")}</p>
              <p>辅料：${esc((formulation.excipient || []).join("；"))}</p>
              <p>核酸结合：嵌入 ${Number(nucleic.intercalation || 0).toFixed(0)}、沟槽 ${Number(nucleic.groove || 0).toFixed(0)}；${esc(nucleic.note || "")}</p>
            </div>
          </div>
          <h4 style="margin-top:14px">综合风险谱</h4>
          <div class="table-scroll"><table class="result-table"><thead><tr><th>端点</th><th>风险分</th><th>等级</th><th>依据</th></tr></thead><tbody>${endpointRows}</tbody></table></div>
          <div class="pharm-grid" style="margin-top:14px">
            <div>
              <h4>化合物组织分布（经验 Kp）</h4>
              <div class="table-scroll"><table class="result-table"><thead><tr><th>组织</th><th>Kp</th><th>分布占比</th></tr></thead><tbody>${tissueRows}</tbody></table></div>
            </div>
            <div>
              <h4>靶点组织表达（HPA/UniProt）</h4>
              ${targetRows ? `<div class="table-scroll"><table class="result-table"><thead><tr><th>基因</th><th>高表达组织</th><th>nTPM</th><th>特异性</th></tr></thead><tbody>${targetRows}</tbody></table></div>` : `<p class="hint">填写靶点基因（逗号分隔）后可查询 HPA/UniProt 组织表达；非人源物种仅显示 UniProt 组织特异性文本。</p>`}
              <p class="hint" style="margin-top:8px">化合物–靶点组织重叠评分 ${Number(tissue.overlap_score || 0).toFixed(0)}/100。Kp 为经验分配模型，不是 PBPK。</p>
            </div>
          </div>
          ${onlineRows.length ? `<h4 style="margin-top:14px">在线证据</h4><div class="table-scroll"><table class="result-table"><thead><tr><th>来源</th><th>标识</th><th>关键结果</th></tr></thead><tbody>${onlineRows.join("")}</tbody></table></div>` : ""}
          <p class="hint" style="margin-top:10px">所有数值为筛查级规则/经验模型；在线查询失败会自动降级，不影响报告生成。</p>
        </div>
      </div>` : "";
    return `
      <div class="card">
        <div class="card-head"><div><h3>综合药理与风险预测</h3><p>不依赖分子对接；可单独输出化合物综合药理报告。</p></div><span class="badge">物种感知</span></div>
        <div class="card-body">
          <div class="row" style="flex-wrap:wrap;align-items:end">
            <div class="field" style="min-width:220px;flex:1"><label>从精细对接配体选择（默认跟随结果分析）</label><select class="input" id="pharm-ligand">${ligandOptions || `<option value="">项目中暂无可用配体</option>`}</select></div>
            <button class="btn ghost sm" data-action="pharm-use-ligand">填入配体</button>
          </div>
          <div class="row" style="margin-top:10px;flex-wrap:wrap">
            <div class="field" style="flex:1;min-width:160px"><label>分子名称</label><input class="input" id="pharm-name" placeholder="例如 Aspirin" value="${esc(pickedLigand?.name || p?.name || "")}"></div>
            <div class="field" style="flex:2;min-width:260px"><label>SMILES</label><input class="input mono" id="pharm-smiles" placeholder="例如 CC(=O)Oc1ccccc1C(=O)O" value="${esc(pickedLigand?.smiles || p?.smiles || "")}"></div>
          </div>
          <div class="row" style="margin-top:10px;flex-wrap:wrap;align-items:end">
            <div class="field" style="min-width:120px"><label>物种</label><select class="input" id="pharm-species">${speciesOptions}</select></div>
            <div class="field" style="flex:1;min-width:240px"><label>靶点基因（逗号分隔，可选）</label><input class="input mono" id="pharm-genes" placeholder="例如 EGFR, PTGS2, MAPT" value="${esc(S.pharmGenes || "")}"></div>
            <label class="row" style="gap:6px;align-items:center;font-size:12px"><input type="checkbox" id="pharm-online" checked> 查询在线证据</label>
          </div>
          <div class="row" style="margin-top:12px;flex-wrap:wrap">
            <button class="btn primary" data-action="pharm-run">计算综合药理</button>
            <button class="btn ghost sm" data-action="pharm-pdf" data-lang="zh">下载药理 PDF（中文）</button>
            <button class="btn ghost sm" data-action="pharm-pdf" data-lang="en">Download PDF（English）</button>
            <button class="btn ghost sm" data-action="pharm-report" data-lang="zh">下载化合物综合报告（中文）</button>
            <button class="btn ghost sm" data-action="pharm-report" data-lang="en">Compound report（English）</button>
          </div>
          <p class="hint" style="margin-top:10px">口服、瘤胃、菌群、稳定性、剂型、核酸结合、风险与组织分布均为规则/经验模型；在线证据来自 PubChem、ChEMBL、UniProt 与 Human Protein Atlas。</p>
        </div>
      </div>
      ${resultCard}`;
  }

  /* ------------------------------------------------------------------ online step */
  function plddtHint(mean) {
    const value = Number(mean);
    if (!Number.isFinite(value)) return "该结构没有可用的 pLDDT 置信度列。";
    if (value >= 90) return "平均 pLDDT ≥ 90：主链与侧链取向可信，可直接用于对接与位点分析。";
    if (value >= 70) return "平均 pLDDT 70–90：主链整体可信，柔性环区与侧链取向需谨慎解释。";
    if (value >= 50) return "平均 pLDDT 50–70：仅主链拓扑可信，对接结果应作为待验证假设。";
    return "平均 pLDDT < 50：模型置信度低，建议改用实验结构或同源模板。";
  }

  function renderOnlineStructure() {
    const r = S.onlineStructure;
    const plddt = r?.plddt || {};
    const detail = r?.available
      ? `<div class="card" style="margin-top:14px">
           <div class="card-head"><div><h3>结构结果 · ${esc(r.gene || r.accession || "预测模型")}</h3><p>${esc(r.protein_name || r.source || "")}</p></div><span class="badge ok">${esc(r.source || "在线")}</span></div>
           <div class="card-body">
             <div class="summary-strip">
               <div class="item"><span>长度</span><strong>${esc(r.length ?? "—")} aa</strong></div>
               <div class="item"><span>平均 pLDDT</span><strong>${esc(plddt.mean ?? "—")}</strong></div>
               <div class="item"><span>高置信残基</span><strong>${esc(plddt.very_high ?? "—")}%</strong></div>
               <div class="item"><span>低置信残基</span><strong>${esc(plddt.low ?? "—")}%</strong></div>
               <div class="item"><span>模型版本</span><strong>${esc(r.model_version ?? "—")}</strong></div>
             </div>
             <p class="hint" style="margin-top:10px">${esc(plddtHint(plddt.mean))}</p>
             <p class="hint">${esc(r.caveat || "")}</p>
             <div class="row" style="margin-top:12px;flex-wrap:wrap">
               <button class="btn primary" data-action="online-structure-save">登记为受体并准备</button>
             </div>
             <p class="hint" style="margin-top:8px">登记后会自动加入“受体准备”，完成即得到可对接的 PDBQT 与候选口袋。</p>
           </div>
         </div>`
      : r
        ? `<div class="note err" style="margin-top:14px">${esc(r.error || "结构获取失败")}</div>`
        : "";
    return `
      <div class="card" style="margin-top:14px">
        <div class="card-head"><div><h3>蛋白质结构获取</h3><p>已知靶点优先下载 AlphaFold DB；新序列用 ESMFold 在线预测。</p></div><span class="badge">AI 结构</span></div>
        <div class="card-body">
          <div class="row" style="flex-wrap:wrap;align-items:end">
            <div class="field" style="min-width:220px"><label>获取方式</label><select class="input" id="online-structure-mode">
              <option value="alphafold">基因 / UniProt → AlphaFold DB</option>
              <option value="esmfold">氨基酸序列 → ESMFold</option>
            </select></div>
            <div class="field" style="flex:1;min-width:220px"><label>基因名或 UniProt 登录号</label><input class="input mono" id="online-gene" placeholder="例如 EGFR 或 P00533"></div>
          </div>
          <div class="field" style="margin-top:10px"><label>或粘贴氨基酸序列（ESMFold，建议 ≤600 aa；只提交目标结构域；请勿提交人类遗传资源或个人信息）</label><textarea class="textarea mono" id="online-sequence" rows="4" placeholder="MKTAYIAKQRQISFVKSHFSRQ..."></textarea></div>
          <div class="row" style="margin-top:12px"><button class="btn primary" data-action="online-structure-run">获取 / 预测结构</button></div>
          <p class="hint" style="margin-top:10px">AlphaFold DB 使用已发布的单链模型；ESMFold 为在线单链预测，不含配体、辅因子与多聚体界面。两者均为预测结构，需实验验证。</p>
        </div>
      </div>
      ${detail}`;
  }

  function renderOnlineTarget() {
    const r = S.onlineTarget;
    const u = r?.uniprot || {};
    const pdb = r?.pdb || {};
    const ot = r?.opentargets || {};
    const diseaseRows = (ot.diseases || []).map((item) => `<tr><td>${esc(item.name)}</td><td class="rank">${Number(item.score || 0).toFixed(3)}</td><td class="mono">${esc(item.id)}</td></tr>`).join("");
    const pdbRows = (pdb.entries || []).map((item) => {
      const ligands = (item.ligands || []).map((lig) => `${lig.id}${lig.name ? ` (${lig.name})` : ""}`).join("；");
      return `<tr><td class="mono">${esc(item.pdb_id)}</td><td class="rank">${item.resolution != null ? Number(item.resolution).toFixed(2) + " Å" : "—"}</td><td>${esc(item.title)}</td><td>${esc(ligands || "—")}</td></tr>`;
    }).join("");
    const tractability = (ot.tractability || []).map((item) => `<span class="chip">${esc(item)}</span>`).join("");
    const detail = r?.available
      ? `<div class="card" style="margin-top:14px">
           <div class="card-head"><div><h3>靶点情报 · ${esc(u.gene || r.query)}</h3><p>${esc(u.protein_name || "")} · ${esc(u.organism || "")}</p></div><span class="badge ok">在线证据</span></div>
           <div class="card-body">
             <div class="summary-strip">
               <div class="item"><span>UniProt</span><strong class="mono">${esc(u.accession || "—")}</strong></div>
               <div class="item"><span>长度</span><strong>${esc(u.length ?? "—")} aa</strong></div>
               <div class="item"><span>Ensembl</span><strong class="mono">${esc(u.ensembl || "—")}</strong></div>
               <div class="item"><span>实验结构</span><strong>${esc(pdb.total ?? (pdb.entries || []).length)}</strong></div>
             </div>
             ${u.function ? `<p style="margin-top:10px"><strong>功能：</strong>${esc(u.function)}</p>` : ""}
             ${(u.subcellular || []).length ? `<p class="hint">亚细胞定位：${esc((u.subcellular || []).join("、"))}</p>` : ""}
             ${tractability ? `<div class="row" style="flex-wrap:wrap;gap:6px;margin-top:8px">${tractability}</div>` : ""}
             <h4 style="margin-top:14px">疾病关联（Open Targets）</h4>
             ${diseaseRows ? `<div class="table-scroll"><table class="result-table"><thead><tr><th>疾病</th><th>关联评分</th><th>ID</th></tr></thead><tbody>${diseaseRows}</tbody></table></div>` : `<p class="hint">${esc(ot.error || "暂无疾病关联数据")}</p>`}
             <h4 style="margin-top:14px">实验结构（RCSB PDB）</h4>
             ${pdbRows ? `<div class="table-scroll"><table class="result-table"><thead><tr><th>PDB</th><th>分辨率</th><th>标题</th><th>共晶配体</th></tr></thead><tbody>${pdbRows}</tbody></table></div>` : `<p class="hint">${esc(pdb.error || "暂无实验结构")}</p>`}
           </div>
         </div>`
      : r
        ? `<div class="note err" style="margin-top:14px">${esc(r.error || "靶点查询失败")}</div>`
        : "";
    return `
      <div class="card" style="margin-top:14px">
        <div class="card-head"><div><h3>靶点情报</h3><p>UniProt 注释、Open Targets 疾病证据与 RCSB 实验结构一次查全。</p></div><span class="badge">靶点验证</span></div>
        <div class="card-body">
          <div class="row" style="flex-wrap:wrap;align-items:end">
            <div class="field" style="flex:1;min-width:220px"><label>基因名或 UniProt 登录号</label><input class="input mono" id="online-target-query" placeholder="例如 EGFR、P00533"></div>
            <div class="field" style="min-width:110px"><label>物种</label><select class="input" id="online-target-species">
              <option value="human">人</option><option value="rat">大鼠</option><option value="pig">猪</option><option value="cattle">牛</option><option value="sheep">羊</option><option value="dog">犬</option><option value="cat">猫</option>
            </select></div>
            <button class="btn primary" data-action="online-target-run">查询靶点</button>
          </div>
          <p class="hint" style="margin-top:10px">疾病关联评分来自 Open Targets 综合证据（遗传、表达、文献、动物模型等），不是临床有效性结论。</p>
        </div>
      </div>
      ${detail}`;
  }

  function renderOnlineCompound() {
    const r = S.onlineCompound;
    const record = r?.record || null;
    const similar = r?.similar || null;
    const rows = (similar?.rows || []).map((item) => `<tr>
      <td class="mono">${esc(item.cid)}</td>
      <td>${esc(item.title || "—")}</td>
      <td>${esc(item.formula || "—")}</td>
      <td class="rank">${esc(item.mw || "—")}</td>
      <td class="rank">${esc(item.xlogp ?? "—")}</td>
      <td class="mono" style="max-width:260px;word-break:break-all">${esc(item.smiles || "")}</td>
      <td><button class="btn ghost tiny" data-action="online-compound-add" data-smiles="${esc(item.smiles || "")}" data-name="${esc(item.title || ("CID" + item.cid))}">加入配体</button></td>
    </tr>`).join("");
    const recordCard = record?.available
      ? `<div class="card" style="margin-top:14px">
           <div class="card-head"><div><h3>化合物记录 · ${esc(record.title || record.query)}</h3><p>PubChem CID ${esc(record.cid)}</p></div><span class="badge ok">PubChem</span></div>
           <div class="card-body">
             <div class="summary-strip">
               <div class="item"><span>分子式</span><strong>${esc(record.formula || "—")}</strong></div>
               <div class="item"><span>分子量</span><strong>${esc(record.mw || "—")}</strong></div>
               <div class="item"><span>XLogP</span><strong>${esc(record.xlogp ?? "—")}</strong></div>
               <div class="item"><span>TPSA</span><strong>${esc(record.tpsa ?? "—")}</strong></div>
               <div class="item"><span>HBD / HBA</span><strong>${esc(record.hbd ?? "—")} / ${esc(record.hba ?? "—")}</strong></div>
             </div>
             <p class="hint" style="margin-top:8px;word-break:break-all">SMILES：${esc(record.smiles || "—")}</p>
             ${(record.synonyms || []).length ? `<p class="hint">同义词：${esc((record.synonyms || []).join("、"))}</p>` : ""}
             <h4 style="margin-top:14px">相似化合物${similar?.threshold ? `（2D Tanimoto ≥ ${Math.round(Number(similar.threshold) * 100)}%）` : ""}</h4>
             ${rows ? `<div class="table-scroll"><table class="result-table"><thead><tr><th>CID</th><th>名称</th><th>分子式</th><th>MW</th><th>XLogP</th><th>SMILES</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>` : `<p class="hint">${esc(similar?.error || "没有找到满足阈值的相似化合物")}</p>`}
           </div>
         </div>`
      : record
        ? `<div class="note err" style="margin-top:14px">${esc(record.error || "化合物查询失败")}</div>`
        : "";
    return `
      <div class="card" style="margin-top:14px">
        <div class="card-head"><div><h3>化合物情报与类似物检索</h3><p>PubChem 实验记录、理化性质与 2D 相似化合物，可直接加入配体库。</p></div><span class="badge">类似物</span></div>
        <div class="card-body">
          <div class="row" style="flex-wrap:wrap;align-items:end">
            <div class="field" style="min-width:150px"><label>输入类型</label><select class="input" id="online-compound-kind">
              <option value="name">化合物名称</option><option value="smiles">SMILES</option>
            </select></div>
            <div class="field" style="flex:1;min-width:240px"><label>名称或 SMILES</label><input class="input mono" id="online-compound-query" placeholder="例如 aspirin 或 CC(=O)Oc1ccccc1C(=O)O"></div>
            <div class="field" style="min-width:150px"><label>相似度阈值</label><select class="input" id="online-compound-threshold">
              <option value="0.85">≥ 85%</option><option value="0.90" selected>≥ 90%</option><option value="0.95">≥ 95%</option>
            </select></div>
            <button class="btn primary" data-action="online-compound-run">查询化合物</button>
          </div>
          <p class="hint" style="margin-top:10px">相似化合物基于 PubChem 2D 指纹，不保证可购买或具有相同活性；加入配体后可在“配体准备”中统一质子化与构象生成。</p>
        </div>
      </div>
      ${recordCard}`;
  }

  function renderOnlineEnrich() {
    const r = S.onlineEnrich;
    const rows = (r?.rows || []).map((item) => `<tr>
      <td>${esc(item.term)}</td>
      <td class="rank">${Number(item.adjusted_p || 0).toExponential(2)}</td>
      <td class="rank">${Number(item.combined_score || 0).toFixed(1)}</td>
      <td>${esc((item.overlap_genes || []).join("、"))}</td>
    </tr>`).join("");
    const detail = r?.available
      ? `<div class="card" style="margin-top:14px">
           <div class="card-head"><div><h3>通路富集 · ${esc(r.library_label || r.library)}</h3><p>输入 ${esc(r.input_count)} 个基因，按调整后 P 值排序</p></div><span class="badge ok">Enrichr</span></div>
           <div class="card-body">
             ${rows ? `<div class="table-scroll"><table class="result-table"><thead><tr><th>通路 / 条目</th><th>调整 P</th><th>综合评分</th><th>重叠基因</th></tr></thead><tbody>${rows}</tbody></table></div>` : `<p class="hint">没有显著富集条目。</p>`}
             <p class="hint" style="margin-top:10px">富集结果依赖输入基因集与背景库；建议与网络药理学的 STRING/本地富集结果交叉验证。</p>
           </div>
         </div>`
      : r
        ? `<div class="note err" style="margin-top:14px">${esc(r.error || "富集失败")}</div>`
        : "";
    return `
      <div class="card" style="margin-top:14px">
        <div class="card-head"><div><h3>通路与功能富集</h3><p>把靶点垂钓、网络药理学或实验得到的基因集直接做 KEGG / GO / Reactome 富集。</p></div><span class="badge">Enrichr</span></div>
        <div class="card-body">
          <div class="field"><label>基因列表（逗号、空格或换行分隔，至少 3 个）</label><textarea class="textarea mono" id="online-enrich-genes" rows="4" placeholder="EGFR, PTGS2, MAPT, AKT1, TP53"></textarea></div>
          <div class="row" style="margin-top:10px;flex-wrap:wrap;align-items:end">
            <div class="field" style="min-width:220px"><label>背景库</label><select class="input" id="online-enrich-library">
              <option value="KEGG_2021_Human">KEGG 2021 Human</option>
              <option value="GO_Biological_Process_2023">GO Biological Process 2023</option>
              <option value="Reactome_2022">Reactome 2022</option>
              <option value="WikiPathway_2023_Human">WikiPathway 2023 Human</option>
            </select></div>
            <button class="btn primary" data-action="online-enrich-run">运行富集</button>
          </div>
        </div>
      </div>
      ${detail}`;
  }

  function renderOnlineStep() {
    const mode = S.onlineMode || "structure";
    const tabs = [
      ["structure", "结构获取"],
      ["target", "靶点情报"],
      ["compound", "化合物情报"],
      ["enrich", "通路富集"],
    ].map(([value, label]) => `<button class="btn ${mode === value ? "primary" : "ghost"} sm" data-action="online-mode" data-mode="${value}">${label}</button>`).join("");
    const busy = S.onlineBusy
      ? `<div class="note warn" style="margin-top:10px">正在查询 ${esc(S.onlineBusy)}，请稍候（在线结构预测可能需要数十秒）...</div>`
      : "";
    return `
      <div class="card">
        <div class="card-head"><div><h3>在线智能资源</h3><p>全部接口免费、无需 API Key；查询失败不影响本地对接与报告生成。</p></div><span class="badge">在线 · 低算力</span></div>
        <div class="card-body">
          <div class="row" style="flex-wrap:wrap;gap:8px">${tabs}</div>
          <div class="note" style="margin-top:10px">数据来源：AlphaFold DB、ESMFold（ESM Atlas）、UniProt、RCSB PDB、Open Targets、PubChem、Enrichr。结果用于假设生成，需实验验证。</div>
          ${busy}
        </div>
      </div>
      ${mode === "structure" ? renderOnlineStructure() : ""}
      ${mode === "target" ? renderOnlineTarget() : ""}
      ${mode === "compound" ? renderOnlineCompound() : ""}
      ${mode === "enrich" ? renderOnlineEnrich() : ""}`;
  }

  function renderExpertsStep() {
    const run = activeRun();
    const result = activeResult();
    if (!run || !result) {
      return `<div class="card"><div class="card-body">${emptyState("先选择一个对接结果", "专家分析会针对结果分析中选定的配体计算代谢软点、药效团、自由能分解与口袋画像。", "search")}</div></div>`;
    }
    if (result.status !== "ok") {
      return `<div class="card"><div class="card-body">${emptyState("该结果没有成功构象", "请选择状态为“完成”的结果。", "search")}</div></div>`;
    }
    const key = `${run.id}:${S.resultIndex}`;
    if (S.expertsKey !== key) {
      S.expertsKey = key;
      S.expertsData = null;
      S.validationData = null;
    }
    if (!S.expertsData && !S.expertsLoading) {
      S.expertsLoading = true;
      api(`/api/runs/${encodeURIComponent(run.id)}/profiling.json`)
        .then((data) => { S.expertsData = data; })
        .catch((error) => { S.expertsData = { rows: [], error: error.message }; })
        .finally(() => { S.expertsLoading = false; render(); });
    }
    const base = `/api/runs/${encodeURIComponent(run.id)}/result/${S.resultIndex}`;
    const metabolismRows = (S.expertsData?.rows || []).slice(0, 10).map((item) => `<tr>
        <td><strong>${esc(item.name)}</strong></td>
        <td class="rank">${Number(item.soft_spot_score ?? 0).toFixed(1)}</td>
        <td>${esc((item.soft_spot_labels || []).join("、") || "—")}</td>
        <td class="rank">${item.cns_mpo != null ? Number(item.cns_mpo).toFixed(1) : "—"}</td>
        <td class="rank">${item.sa != null ? Number(item.sa).toFixed(2) : "—"}</td>
        <td class="rank">${item.logd != null ? Number(item.logd).toFixed(2) : "—"}</td>
      </tr>`).join("");
    const tissue = S.pharmResult?.tissue || null;
    const tissueRows = (tissue?.kp_rows || []).slice(0, 8).map((item) => `<tr><td>${esc(item.tissue)}</td><td class="rank">${Number(item.kp).toFixed(2)}</td><td class="rank">${Number(item.share).toFixed(1)}%</td></tr>`).join("");
    const validation = S.validationData;
    const validationRows = (validation?.rows || []).map((item) => `<tr><td>${esc(item.name || "—")}</td><td class="rank">${Number(item.rmsd ?? 0).toFixed(2)}</td><td><span class="badge ${item.passed ? "ok" : "warn"}">${item.passed ? "通过" : "未通过"}</span></td></tr>`).join("");
    const card = (title, desc, links, extra = "") => `
      <div class="card" style="box-shadow:none;margin:0">
        <div class="card-head" style="padding-bottom:8px"><div><h3 style="font-size:13px">${title}</h3><p>${desc}</p></div></div>
        <div class="card-body" style="padding-top:6px">
          <div class="row" style="flex-wrap:wrap;gap:6px">${links}</div>
          ${extra}
        </div>
      </div>`;
    const metaboliteCard = card(
      "代谢软点与可开发性",
      "按 CYP 反应位点规则给出 0–100 的代谢风险分，并列出类药性、CNS MPO、SAscore 与 logD；用于安排代谢稳定性实验和结构封闭。",
      `<a class="btn ghost sm" href="/api/runs/${encodeURIComponent(run.id)}/profiling.pdf" target="_blank" rel="noopener">下载代谢与可开发性 PDF</a>
       <button class="btn ghost sm" data-action="experts-refresh">刷新数据</button>`,
      S.expertsData?.error ? `<div class="note warn" style="margin-top:10px">${esc(S.expertsData.error)}</div>` : (metabolismRows ? `<div class="table-scroll" style="margin-top:10px"><table class="result-table"><thead><tr><th>配体</th><th>代谢风险</th><th>软点类型</th><th>CNS MPO</th><th>SAscore</th><th>logD</th></tr></thead><tbody>${metabolismRows}</tbody></table></div>` : `<div class="note" style="margin-top:10px">正在计算代谢软点 ...</div>`),
    );
    const tissueCard = card(
      "药物组织分布",
      "经验 Kp 与组织占比来自 logP/血浆蛋白结合分配模型；靶点组织表达来自 HPA/UniProt，仅作组织暴露假设。",
      `<button class="btn ghost sm" data-action="goto-pharm">打开综合药理</button>
       <a class="btn ghost sm" href="/api/reports/full.pdf?run_id=${encodeURIComponent(run.id)}&result_index=${S.resultIndex}" target="_blank" rel="noopener">并入完整报告</a>`,
      tissueRows ? `<div class="table-scroll" style="margin-top:10px"><table class="result-table"><thead><tr><th>组织</th><th>Kp</th><th>分布占比</th></tr></thead><tbody>${tissueRows}</tbody></table></div>` : `<div class="note" style="margin-top:10px">尚未计算组织分布。点击“打开综合药理”运行后，这里会显示 Kp 与占比表。</div>`,
    );
    const validationCard = card(
      "重对接验证",
      "把共晶配体重新对接回原位点，用对称校正重原子 RMSD 判断参数是否可靠；≤ 2 Å 视为通过。",
      `<button class="btn ghost sm" data-action="experts-validation">运行重对接验证</button>`,
      validationRows ? `<div class="table-scroll" style="margin-top:10px"><table class="result-table"><thead><tr><th>共晶配体</th><th>RMSD (Å)</th><th>结论</th></tr></thead><tbody>${validationRows}</tbody></table></div>` : (validation ? `<div class="note" style="margin-top:10px">没有可用的共晶配体，无法做重对接验证。</div>` : ""),
    );
    const consensusCard = ((run.consensus?.engines || []).length >= 2)
      ? card(
          "多引擎共识",
          "Vina、Vinardo 与 smina / QuickVina 2 的排序一致性与综合排名。",
          `<a class="btn ghost sm" href="/api/runs/${encodeURIComponent(run.id)}/consensus.pdf" target="_blank" rel="noopener">下载共识图 PDF</a>`,
        )
      : "";
    return `
      <div class="card">
        <div class="card-head"><div><h3>${esc(result.ligand_name)} · 专家分析</h3><p>${esc(result.receptor_name)} · 构象 ${S.mode} · 每个模块都可单独导出，并自动并入完整报告。</p></div><span class="badge">模块化</span></div>
        <div class="card-body" style="padding-top:0">
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px">
            ${metaboliteCard}
            ${tissueCard}
            ${card("药效团特征", "识别氢键供体/受体、芳环、疏水中心与离子中心及其空间分布。",
              `<a class="btn ghost sm" href="${base}/pharmacophore.pdf" target="_blank" rel="noopener">药效团 PDF</a>
               <a class="btn ghost sm" href="${base}/pharmacophore" target="_blank" rel="noopener">查看数据</a>`)}
            ${card("MM-GBSA 结合自由能分解", "残基级范德华、静电、去溶剂化与氢键贡献，未做熵校正，仅用于同系列相对比较。",
              `<a class="btn ghost sm" href="${base}/mmgbsa.pdf" target="_blank" rel="noopener">MM-GBSA PDF</a>`)}
            ${card("结合能多方法比较", "并列 Vina、Vinardo、经验项与 MM-GBSA，并给出仅基于对接尺度的共识估计。",
              `<a class="btn ghost sm" href="${base}/energy.pdf" target="_blank" rel="noopener">结合能比较 PDF</a>`)}
            ${card("接触指纹与口袋画像", "原子–残基接触指纹、空间接触图与口袋理化画像，用于判断作用原子与残基环境。",
              `<a class="btn ghost sm" href="${base}/fingerprint.pdf" target="_blank" rel="noopener">接触指纹 PDF</a>
               <a class="btn ghost sm" href="${base}/pocket.pdf" target="_blank" rel="noopener">口袋画像 PDF</a>`)}
            ${card("ADMET 与类药性", "理化性质、ADMET 端点、类药规则与综合雷达，适合与代谢软点一起判读。",
              `<a class="btn ghost sm" href="${base}/admet.pdf" target="_blank" rel="noopener">ADMET PDF</a>
               <a class="btn ghost sm" href="${base}/expert.pdf" target="_blank" rel="noopener">专家合并 PDF</a>`)}
            ${validationCard}
            ${consensusCard}
          </div>
        </div>
      </div>`;
  }

  function renderDdiStep() {
    const d = S.data;
    const result = activeResult();
    const ligand = resultLigand(result);
    const ligandSmiles = ligand?.smiles || ligand?.meta?.smiles || "";
    const record = S.ddiResult || null;
    const coDrugRows = [...(record?.co_drugs || [])].sort(
      (a, b) => Number(b.score ?? 0) - Number(a.score ?? 0)
    );
    const rows = coDrugRows.slice(0, 20).map((item) => {
      const tone = item.category === "高" ? "err" : item.category === "中" ? "warn" : item.category === "低" ? "ok" : "";
      return `<tr>
        <td><strong>${esc(item.name)}</strong><div class="muted" style="font-size:9px">${esc(item.drug_class || "")}</div></td>
        <td class="rank">${item.tanimoto != null ? Number(item.tanimoto).toFixed(3) : "—"}</td>
        <td>${(item.shared_targets || []).length ? esc((item.shared_targets || []).join("、")) : "—"}</td>
        <td class="rank">${item.components?.cyp_component != null ? Number(item.components.cyp_component).toFixed(0) : "—"}</td>
        <td>${esc(item.mechanism || "—")}</td>
        <td><span class="badge ${tone}">${item.score != null ? Number(item.score).toFixed(0) + " · " : ""}${esc(item.category || "—")}</span></td>
      </tr>`;
    }).join("");
    const resultCard = record
      ? `<div class="card">
          <div class="card-head"><div><h3>药物互作结果 · ${esc(record.candidate?.name || "候选分子")}</h3><p>已比对 ${record.co_drugs?.length || 0} 个 FDA 参考药物 · 高风险 ${record.summary?.high || 0} 个 · 中风险 ${record.summary?.medium || 0} 个 · 下表显示风险最高的 20 个</p></div><span class="badge ok">可导出 PDF</span></div>
          <div class="card-body">
            <div class="table-scroll">
              <table class="result-table">
                <thead><tr><th>联用药物</th><th>结构相似度</th><th>共享靶点</th><th>CYP 竞争</th><th>主要机制</th><th>风险</th></tr></thead>
                <tbody>${rows}</tbody>
              </table>
            </div>
            <div class="row" style="margin-top:14px;flex-wrap:wrap">
              <a class="btn ghost sm" href="/api/ddi.pdf?record_id=${encodeURIComponent(record.id || "")}" target="_blank" rel="noopener">下载互作 PDF</a>
            </div>
            <p class="hint" style="margin-top:10px">${esc(record.limitations || "")}</p>
          </div>
        </div>`
      : "";
    return `
      <div class="card">
        <div class="card-head"><div><h3>候选分子与其他药物的相互作用</h3><p>从共享靶点、CYP 竞争、hERG 叠加和结构类别相似度四个维度给出联用风险假设。</p></div><span class="badge">规则式 DDI</span></div>
        <div class="card-body">
          <div class="field"><label>候选分子名称</label><input class="input" id="ddi-name" placeholder="例如 Imatinib" value="${esc(ligand?.name || record?.candidate?.name || "")}"></div>
          <div class="field" style="margin-top:9px"><label>候选分子 SMILES（默认取自结果分析中选定的配体）</label><input class="input mono" id="ddi-smiles" placeholder="例如 CC(=O)Oc1ccccc1C(=O)O" value="${esc(ligandSmiles || record?.candidate?.smiles || "")}"></div>
          <div class="row" style="margin-top:9px;flex-wrap:wrap">
            ${ligandSmiles ? `<button class="btn ghost sm" data-action="ddi-use-ligand" data-smiles="${esc(ligandSmiles)}" data-name="${esc(ligand?.name || "")}">填入当前结果配体${ligand?.name ? `（${esc(ligand.name)}）` : ""}</button>` : ""}
            <button class="btn ghost sm" data-action="ddi-default-drugs">载入常用联用药物</button>
          </div>
          <label style="display:flex;gap:8px;align-items:center;margin-top:12px;font-size:12px"><input type="checkbox" id="ddi-fda-all" checked> <span>与全部 FDA 参考药物比对（默认，推荐；共 45 个已上市药物）</span></label>
          <div class="field" style="margin-top:9px"><label>补充/自定义联用药物（可选，每行一个名称或“名称, SMILES”）</label><textarea class="textarea" id="ddi-codrugs" rows="4" placeholder="Atorvastatin&#10;Warfarin&#10;自定义药物, CC(=O)O"></textarea></div>
          <div class="field" style="margin-top:9px"><label>候选靶点（可选；留空则自动读取同分子的靶点垂钓结果）</label><textarea class="textarea mono" id="ddi-targets" rows="2" placeholder="EGFR, AKT1, MTOR"></textarea></div>
          <div class="row" style="margin-top:14px"><button class="btn primary" data-action="ddi-start">分析药物相互作用</button></div>
          <p class="hint" style="margin-top:10px">默认与内置的 45 个 FDA 参考药物逐一比对（名称会自动解析结构、靶点与代谢酶）；未知药物可在下方补充 SMILES 与靶点。共享靶点不代表同向或拮抗，CYP 与 hERG 为计算风险，结果只用于优先核查。</p>
        </div>
      </div>
      ${resultCard}`;
  }

  function renderDrugrefStep() {
    const d = S.data;
    const result = activeResult();
    const ligand = resultLigand(result);
    const ligandSmiles = ligand?.smiles || ligand?.meta?.smiles || "";
    const record = S.drugrefResult || null;
    const top = record?.top || [];
    const rows = top.map((item) => `
      <tr>
        <td class="rank">#${item.rank}</td>
        <td><strong>${esc(item.name)}</strong><div class="muted" style="font-size:9px">${esc(item.drug_class || "")}</div></td>
        <td>${(item.shared_targets || []).length ? esc((item.shared_targets || []).join("、")) : "—"}</td>
        <td class="rank">${item.tanimoto != null ? Number(item.tanimoto).toFixed(3) : "—"}</td>
        <td class="rank">${item.target_jaccard != null ? Number(item.target_jaccard).toFixed(3) : "—"}</td>
        <td class="rank">${item.admet_similarity != null ? Number(item.admet_similarity).toFixed(1) : "—"}</td>
        <td class="rank">${item.score != null ? Number(item.score).toFixed(3) : "—"}</td>
      </tr>`).join("");
    const resultCard = record
      ? `<div class="card">
          <div class="card-head"><div><h3>FDA 参考分析 · ${esc(record.candidate?.name || "候选分子")}</h3><p>参考集 ${record.references?.length || 0} 个已上市药物 · 与 ${record.candidate?.shared_target_count || 0} 个参考药存在共享靶点</p></div><span class="badge ok">可导出 PDF</span></div>
          <div class="card-body">
            <div class="note" style="margin-bottom:12px">邻近药物类别：${esc((record.candidate?.nearby_classes || []).join("、") || "—")}。综合分为启发式排序，不是概率或疗效结论。</div>
            <div class="table-scroll">
              <table class="result-table">
                <thead><tr><th>排名</th><th>参考药</th><th>共享靶点</th><th>Tanimoto</th><th>靶点 Jaccard</th><th>ADMET 相似度</th><th>综合分</th></tr></thead>
                <tbody>${rows}</tbody>
              </table>
            </div>
            <div class="row" style="margin-top:14px;flex-wrap:wrap">
              <a class="btn ghost sm" href="/api/drugref.pdf?record_id=${encodeURIComponent(record.id || "")}" target="_blank" rel="noopener">下载 FDA 参考 PDF</a>
            </div>
            <p class="hint" style="margin-top:10px">${esc(record.limitations || "")}</p>
          </div>
        </div>`
      : "";
    return `
      <div class="card">
        <div class="card-head"><div><h3>与 FDA 已上市药物对照</h3><p>结构相似度、共享靶点、ADMET 相似度与理化性质距离的加权参考排序。</p></div><span class="badge">FDA 参考集</span></div>
        <div class="card-body">
          <div class="field"><label>候选分子名称</label><input class="input" id="drugref-name" placeholder="例如 Imatinib" value="${esc(ligand?.name || record?.candidate?.name || "")}"></div>
          <div class="field" style="margin-top:9px"><label>候选分子 SMILES（默认取自结果分析中选定的配体）</label><input class="input mono" id="drugref-smiles" placeholder="例如 CC(=O)Oc1ccccc1C(=O)O" value="${esc(ligandSmiles || record?.candidate?.smiles || "")}"></div>
          <div class="row" style="margin-top:9px;flex-wrap:wrap">
            ${ligandSmiles ? `<button class="btn ghost sm" data-action="drugref-use-ligand" data-smiles="${esc(ligandSmiles)}" data-name="${esc(ligand?.name || "")}">填入当前结果配体${ligand?.name ? `（${esc(ligand.name)}）` : ""}</button>` : ""}
          </div>
          <div class="field" style="margin-top:12px"><label>候选靶点（可选；留空则自动读取同分子的靶点垂钓结果）</label><textarea class="textarea mono" id="drugref-targets" rows="2" placeholder="EGFR, ABL1, KIT"></textarea></div>
          <div class="row" style="margin-top:14px"><button class="btn primary" data-action="drugref-start">与 FDA 药物比较</button></div>
          <p class="hint" style="margin-top:10px">参考集为 30–45 个结构简单、靶点明确的 FDA 已上市药物。药效只用靶点重叠与机制邻近表示，不伪造 pIC50；ADMET 与半衰期为计算值，不能替代临床或实验数据。</p>
        </div>
      </div>
      ${resultCard}`;
  }

  function renderResultDetail(run) {
    const result = activeResult();
    if (!result) {
      return `<div class="card"><div class="card-body">${emptyState("选择一个结果", "点击左侧任意一行查看构象与作用力。", "search")}</div></div>`;
    }
    if (result.status !== "ok") {
      return `<div class="card"><div class="card-body">
        <h3 style="margin:0 0 8px;font-size:14px">${esc(result.ligand_name)} × ${esc(result.receptor_name)}</h3>
        <div class="note err" style="margin-bottom:12px">${esc(result.error || "对接失败")}</div>
        <button class="btn ghost sm" data-action="view-log" data-run="${run.id}" data-index="${S.resultIndex}">查看日志</button>
      </div></div>`;
    }

    const modes = result.modes || [];
    const best = modes[0] || {};
    const key = `${run.id}:${S.resultIndex}:${S.mode}`;
    const analysis = S.analysisKey === key ? S.analysis : null;
    const counts = analysis?.counts || {};
    const rmsd = S.rmsdKey === key ? S.rmsd : null;
    const poseKey = `${run.id}:${S.resultIndex}:${S.mode}`;
    const poseStored = S.poseCheckKey === poseKey ? S.poseCheck : result.posebusters || null;
    const poseReady = poseStored && (poseStored.checks || poseStored.total != null);
    const poseHtml = poseStored && poseStored.available === false
      ? `<div class="note warn" style="margin:0 18px 12px">${esc(poseStored.error || "PoseBusters 不可用")}</div>`
      : poseReady
        ? `<div class="note ${poseStored.all_passed ? "" : "warn"}" style="margin:0 18px 12px;display:block">
            <div class="row" style="gap:8px;flex-wrap:wrap;align-items:center">
              <span class="badge ${poseStored.all_passed ? "ok" : "warn"}">PoseBusters ${poseStored.passed}/${poseStored.total}</span>
              <span style="font-size:11px">${poseStored.all_passed ? "全部物理合理性检查通过" : `未通过：${esc((poseStored.failed_labels || poseStored.failed || []).join("、"))}`}</span>
            </div>
            ${poseStored.checks?.length ? `<details style="margin-top:6px"><summary style="cursor:pointer;font-size:11px">查看 ${poseStored.total} 项检查</summary>
              <div class="row" style="gap:4px;flex-wrap:wrap;margin-top:6px">${poseStored.checks.map((c) => `<span class="chip" style="font-size:10px;${c.passed ? "" : "color:#b91c1c;border-color:#fca5a5"}">${c.passed ? "✓" : "✗"} ${esc(c.label_zh)}</span>`).join("")}</div>
            </details>` : ""}
          </div>`
        : "";
    const rec = S.data.receptors.find((r) => r.id === result.receptor_id);
    const hetero = rec?.hetero || [];

    const modeChips = modes.map((m) => `
      <button class="mode-chip ${m.mode === S.mode ? "active" : ""}" data-action="set-mode" data-mode="${m.mode}">
        #${m.mode} <b>${Number(m.affinity).toFixed(2)}</b>
      </button>`).join("");

    const countCells = [
      ["氢键", counts.hbonds],
      ["疏水残基", counts.hydrophobic],
      ["盐桥", counts.salt_bridges],
      ["π-堆积", counts.pi_stacking],
      ["卤键", counts.halogens],
      ["金属配位", counts.metals],
    ].map(([label, value]) => `<div class="count-cell"><span>${label}</span><strong>${value ?? "–"}</strong></div>`).join("");

    const summaryRows = analysis?.summary?.length
      ? analysis.summary.slice(0, 14).map((item) => {
          const tags = item.types.map((t) => {
            const cls = { "氢键": "hb", "疏水": "hyd", "盐桥": "salt", "π-堆积": "pi", "卤键": "hal", "金属配位": "metal" }[t] || "hyd";
            return `<span class="type-tag ${cls}">${esc(t)}</span>`;
          }).join(" ");
          return `<div class="interaction-row">
            <span class="res-tag">${esc(item.residue)} ${esc(item.chain)}${item.resseq}</span>
            <span style="flex:1">${tags}</span>
            <span class="num" style="color:var(--muted-2)">${Number(item.distance).toFixed(2)} Å</span>
          </div>`;
        }).join("")
      : `<div class="note" style="margin:10px 0 16px">${analysis ? "未检测到明显作用力。" : "点击“作用力分析”计算氢键、疏水接触、盐桥等。"}</div>`;

    const rmsdSelect = hetero.length > 1
      ? `<select class="select" id="rmsd-ref" style="height:28px;width:auto;font-size:10px;margin-left:8px">${hetero.map((h, i) => `<option value="${i}">${esc(h.name)} ${esc(h.key)}</option>`).join("")}</select>`
      : "";

    return `<div class="card detail-card">
      <div class="detail-head">
        <div class="detail-title">
          <div>
            <h3>${esc(result.ligand_name)}</h3>
            <div style="font-size:10.5px;color:var(--muted);margin-top:4px">受体 ${esc(result.receptor_name)} · 引擎 ${esc(result.engine || "cpu")} · 随机种子 ${result.seed ?? "—"}</div>
          </div>
          <div class="aff-big">${Number(best.affinity ?? result.affinity).toFixed(2)}<small>Vina · kcal/mol</small></div>
        </div>
        <div class="mode-list">${modeChips}</div>
        <div class="row" style="margin-top:9px;gap:6px">
          <span class="chip">Vinardo <strong>${result.vinardo != null ? Number(result.vinardo).toFixed(2) : "—"}</strong> kcal/mol</span>
          ${result.gpu_affinity != null ? `<span class="chip">GPU 原始 <strong>${Number(result.gpu_affinity).toFixed(2)}</strong> kcal/mol</span>` : ""}
          ${result.replicate_sd != null ? `<span class="chip">重复 <strong>${result.seed_replicates || (result.replicates || []).length}</strong> 次 · SD <strong>±${Number(result.replicate_sd).toFixed(2)}</strong></span>` : ""}
          <span class="chip">共识排名 <strong>${result.consensus_rank ?? "—"}</strong></span>
        </div>
      </div>
      <div class="count-grid">${countCells}</div>
      ${analysis?.method === "geometry" ? `<div class="note warn" style="margin:0 18px 14px"><svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M12 8h.01M11 12h1v5h1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg><span>该配体来自外部 PDBQT，文件未携带键级或 SMILES。作用力由三维坐标推断，盐桥、π-堆积等依赖键级的项可能偏少；如需精确定量，请上传 SDF/SMILES 后重新准备。</span></div>` : ""}
      ${analysis ? `<div class="interaction-list">${summaryRows}</div>` : `<div style="padding:0 18px 14px">${summaryRows}</div>`}
      ${rmsd ? `<div class="rmsd-box"><span>与参考配体对称校正 RMSD</span><strong>${rmsd.available ? Number(rmsd.rmsd).toFixed(2) + " Å" : esc(rmsd.error || "不可用")}</strong></div>` : ""}
      ${poseHtml}
      <div class="detail-actions">
        <button class="btn primary sm" data-action="analyze" data-run="${run.id}" data-index="${S.resultIndex}">作用力分析</button>
        <button class="btn sm" data-action="admet" data-run="${run.id}" data-index="${S.resultIndex}">ADMET</button>
        <button class="btn sm" data-action="pharmacophore" data-run="${run.id}" data-index="${S.resultIndex}">药效团</button>
        <button class="btn sm" data-action="mmgbsa" data-run="${run.id}" data-index="${S.resultIndex}">自由能分解</button>
        <button class="btn sm" data-action="posebusters" data-run="${run.id}" data-index="${S.resultIndex}" ${S.poseBusy ? "disabled" : ""}>${S.poseBusy ? "校验中…" : "物理合理性"}</button>
        <a class="btn sm" href="/api/runs/${encodeURIComponent(run.id)}/result/${S.resultIndex}/energy.pdf" target="_blank" rel="noopener">结合能比较</a>
        <button class="btn sm" data-action="synthesis-result" data-run="${run.id}" data-index="${S.resultIndex}">合成路线</button>
        <a class="btn ghost sm" href="/api/runs/${encodeURIComponent(run.id)}/result/${S.resultIndex}/expert.pdf?mode=${S.mode}" target="_blank" rel="noopener">专家分析 PDF</a>
        <a class="btn ghost sm" href="/api/runs/${encodeURIComponent(run.id)}/profiling.pdf" target="_blank" rel="noopener">可开发性 PDF</a>
        <button class="btn sm" data-action="figure-studio" data-run="${run.id}" data-index="${S.resultIndex}">论文图</button>
        <a class="btn ghost sm" href="/api/runs/${encodeURIComponent(run.id)}/figures" download>论文图表包</a>
        <span class="row" style="gap:4px;flex-wrap:nowrap">
          <button class="btn ghost sm" data-action="rmsd" data-run="${run.id}" data-index="${S.resultIndex}">RMSD</button>
          ${rmsdSelect}
        </span>
        <a class="btn ghost sm" href="/api/runs/${encodeURIComponent(run.id)}/result/${S.resultIndex}/pose?mode=${S.mode}" download>下载构象</a>
        <a class="btn ghost sm" href="/api/runs/${encodeURIComponent(run.id)}/result/${S.resultIndex}/complex?mode=${S.mode}" download>导出复合物</a>
        <button class="btn ghost sm" data-action="view-log" data-run="${run.id}" data-index="${S.resultIndex}">查看日志</button>
      </div>
    </div>`;
  }

  /* ------------------------------------------------------------------ viewer */
  function showViewerEmpty(text = "选择受体或配体后在此显示 3D 结构") {
    $("#viewer-empty").classList.remove("hidden");
    $("#viewer-empty").querySelector("p").textContent = text;
    $("#viewer-title").textContent = "3D 视图";
    $("#viewer-legend").innerHTML = "";
  }

  function hideViewerEmpty() {
    $("#viewer-empty").classList.add("hidden");
  }

  async function syncViewer() {
    if (!viewer) return;
    const token = ++S.viewerToken;
    try {
      if (S.step === "results") {
        const run = activeRun();
        const result = activeResult();
        if (run && result && result.status === "ok") {
          await showPoseViewer(run, result, token);
          return;
        }
      }
      if (S.step === "ligands" && activeLigand()) {
        await showLigandViewer(activeLigand(), token);
        return;
      }
      const rec = activeReceptor();
      if (rec && ["receptor", "site", "dock"].includes(S.step)) {
        await showReceptorViewer(rec, token);
        return;
      }
      if (token !== S.viewerToken) return;
      resetViewer();
      showViewerEmpty();
    } catch (error) {
      if (token !== S.viewerToken) return;
      resetViewer();
      showViewerEmpty(error.message || "无法加载 3D 结构");
    }
  }

  function resetViewer() {
    if (!viewer) return;
    viewer.removeAllModels();
    viewer.removeAllShapes();
    viewer.removeAllLabels();
    viewer.removeAllSurfaces();
    viewer.setBackgroundColor("#ffffff");
  }

  function settleViewer(token) {
    setTimeout(() => {
      if (token !== S.viewerToken || !viewer) return;
      viewer.resize();
      viewer.render();
    }, 110);
  }

  async function showReceptorViewer(rec, token) {
    const pdb = await fetchText(`/api/receptors/${rec.id}/pdb`);
    if (token !== S.viewerToken) return;
    resetViewer();
    hideViewerEmpty();
    viewer.addModel(pdb, "pdb");
    const cartoonStyle = S.step === "site"
      ? { color: "#7084a3", opacity: 0.9 }
      : { color: "#7c93b3", opacity: 0.95 };
    viewer.setStyle({ model: 0 }, { cartoon: cartoonStyle });
    if (S.viewerSurface) {
      viewer.addSurface(window.$3Dmol.SurfaceType.VDW, { opacity: 0.13, color: "#94a3b8" }, { model: 0 });
    }
    if (rec.box && (S.step === "site" || S.step === "dock" || S.step === "receptor")) {
      addBoxShape(rec.box);
      if (S.step === "site" || S.step === "dock") {
        const boxModel = addBoxFocusModel(rec.box);
        viewer.zoomTo({ model: boxModel });
        viewer.zoom(0.9);
      } else {
        viewer.zoomTo({ model: 0 });
      }
    } else {
      viewer.zoomTo({ model: 0 });
    }
    if (S.step === "site" && rec.pockets?.length) {
      rec.pockets.forEach((p) => {
        viewer.addSphere({ center: { x: p.center[0], y: p.center[1], z: p.center[2] }, radius: 1.4, color: "#0d9488", opacity: 0.35 });
        viewer.addLabel(String(p.rank), { position: { x: p.center[0], y: p.center[1], z: p.center[2] }, fontSize: 10, fontColor: "#0f766e", backgroundColor: "#ffffff", borderThickness: 1, borderColor: "#99f6e4", padding: 2 });
      });
    }
    $("#viewer-title").textContent = `受体 · ${rec.name}`;
    $("#viewer-legend").innerHTML = `
      <span class="legend-item"><span class="legend-swatch" style="background:#94a3b8"></span>蛋白</span>
      ${rec.box ? `<span class="legend-item"><span class="legend-swatch" style="background:#0f766e"></span>对接盒</span>` : ""}
      ${S.step === "site" ? `<span class="legend-item"><span class="legend-swatch" style="background:#0f766e;border-radius:50%"></span>候选口袋</span>` : ""}
    `;
    viewer.render();
    settleViewer(token);
  }

  async function showLigandViewer(lig, token) {
    let text = "";
    let format = "sdf";
    if (lig.has_sdf) {
      text = await fetchText(`/api/ligands/${lig.id}/sdf`);
    } else if (lig.has_pdbqt) {
      text = await fetchText(`/api/ligands/${lig.id}/download`);
      format = "pdbqt";
    }
    if (token !== S.viewerToken) return;
    resetViewer();
    if (!text.trim()) {
      showViewerEmpty("该配体还没有 3D 结构，请先准备。");
      return;
    }
    hideViewerEmpty();
    viewer.addModel(text, format);
    viewer.setStyle({ model: 0 }, { stick: { colorscheme: LIGAND_COLORS, radius: 0.18 } });
    $("#viewer-title").textContent = `配体 · ${lig.name}`;
    $("#viewer-legend").innerHTML = `<span class="legend-item"><span class="legend-swatch" style="background:#4ade80"></span>配体</span>`;
    viewer.zoomTo({ model: 0 });
    viewer.zoom(0.85);
    viewer.render();
    settleViewer(token);
  }

  async function showPoseViewer(run, result, token) {
    const [pdb, pose] = await Promise.all([
      fetchText(`/api/runs/${run.id}/result/${S.resultIndex}/receptor.pdb`),
      fetchText(`/api/runs/${run.id}/result/${S.resultIndex}/pose?mode=${S.mode}`),
    ]);
    if (token !== S.viewerToken) return;
    resetViewer();
    hideViewerEmpty();
    viewer.addModel(pdb, "pdb");
    viewer.setStyle({ model: 0 }, { cartoon: { color: "#94a3b8", opacity: 0.55 } });
    viewer.addModel(pose, "pdbqt");
    viewer.setStyle({ model: 1 }, { stick: { colorscheme: LIGAND_COLORS, radius: 0.2 } });

    const key = `${run.id}:${S.resultIndex}:${S.mode}`;
    const analysis = S.analysisKey === key ? S.analysis : null;
    if (analysis?.residues?.length) {
      const resi = analysis.residues.map((r) => r.resseq);
      viewer.setStyle({ model: 0, resi }, { cartoon: { color: "#94a3b8", opacity: 0.22 }, stick: { colorscheme: PROTEIN_COLORS, radius: 0.12 } });
    }
    if (analysis) {
      const draw = (item, color) => {
        if (!item.protein_coord || !item.ligand_coord) return;
        viewer.addLine({
          start: { x: item.protein_coord[0], y: item.protein_coord[1], z: item.protein_coord[2] },
          end: { x: item.ligand_coord[0], y: item.ligand_coord[1], z: item.ligand_coord[2] },
          color,
          dashed: true,
          linewidth: 2,
        });
      };
      (analysis.hbonds || []).forEach((item) => draw(item, "#b45309"));
      (analysis.salt_bridges || []).forEach((item) => draw(item, "#b91c1c"));
    }

    $("#viewer-title").textContent = `${result.ligand_name} · 构象 ${S.mode}`;
    $("#viewer-legend").innerHTML = `
      <span class="legend-item"><span class="legend-swatch" style="background:#94a3b8"></span>受体</span>
      <span class="legend-item"><span class="legend-swatch" style="background:#4ade80"></span>配体</span>
      <span class="legend-item"><span class="legend-swatch" style="background:#fbbf24"></span>氢键</span>
      <span class="legend-item"><span class="legend-swatch" style="background:#f87171"></span>盐桥</span>
    `;
    viewer.zoomTo({ model: 1 });
    viewer.zoom(0.72);
    const bounds = poseBounds(pose);
    if (bounds) {
      addFrameModel(
        [(bounds.min[0] + bounds.max[0]) / 2, (bounds.min[1] + bounds.max[1]) / 2, (bounds.min[2] + bounds.max[2]) / 2],
        [bounds.max[0] - bounds.min[0] + 16, bounds.max[1] - bounds.min[1] + 16, bounds.max[2] - bounds.min[2] + 16],
        2,
      );
      viewer.zoomTo({ model: 2 });
      viewer.zoom(0.95);
    }
    viewer.render();
    settleViewer(token);
  }

  function poseBounds(text) {
    const points = [];
    text.splitlines().forEach((line) => {
      if (!/^(ATOM|HETATM)/.test(line)) return;
      const x = Number(line.slice(30, 38));
      const y = Number(line.slice(38, 46));
      const z = Number(line.slice(46, 54));
      if ([x, y, z].every(Number.isFinite)) points.push([x, y, z]);
    });
    if (!points.length) return null;
    return {
      min: [0, 1, 2].map((i) => Math.min(...points.map((p) => p[i]))),
      max: [0, 1, 2].map((i) => Math.max(...points.map((p) => p[i]))),
    };
  }

  function addBoxShape(box) {
    const [cx, cy, cz] = box.center;
    const [sx, sy, sz] = box.size;
    const xs = [cx - sx / 2, cx + sx / 2];
    const ys = [cy - sy / 2, cy + sy / 2];
    const zs = [cz - sz / 2, cz + sz / 2];
    viewer.addBox({
      center: { x: cx, y: cy, z: cz },
      dimensions: { w: sx, h: sy, d: sz },
      color: "#0f766e",
      opacity: 0.06,
      wireframe: false,
    });
    const corners = [];
    xs.forEach((x) => ys.forEach((y) => zs.forEach((z) => corners.push({ x, y, z }))));
    const edges = [];
    corners.forEach((a, i) => {
      corners.forEach((b, j) => {
        if (j <= i) return;
        const diff = [a.x - b.x, a.y - b.y, a.z - b.z].filter((v) => Math.abs(v) > 1e-6).length;
        if (diff === 1) edges.push([a, b]);
      });
    });
    edges.forEach(([start, end]) => {
      viewer.addCylinder({ start, end, radius: 0.1, color: "#0f766e", opacity: 0.9 });
    });
  }

  function addFrameModel(center, size, modelIndex) {
    const [cx, cy, cz] = center;
    const [sx, sy, sz] = size;
    const corners = [];
    [cx - sx / 2, cx + sx / 2].forEach((x) =>
      [cy - sy / 2, cy + sy / 2].forEach((y) =>
        [cz - sz / 2, cz + sz / 2].forEach((z) => corners.push({ x, y, z })),
      ),
    );
    const xyz = `${corners.length}\nframe\n${corners
      .map((p) => `${p.x.toFixed(3)} ${p.y.toFixed(3)} ${p.z.toFixed(3)}`)
      .join("\n")}`;
    viewer.addModel(xyz, "xyz");
    viewer.setStyle({ model: modelIndex }, { sphere: { radius: 0.01, color: "#0f766e", opacity: 0.01 } });
  }

  function addBoxFocusModel(box) {
    // Frame the box with ~10 Å of surrounding protein for context.
    addFrameModel(box.center, box.size.map((v) => v + 20), 1);
    return 1;
  }

  /* ------------------------------------------------------------------ job handling */
  function openConsole(open) {
    $("#console").classList.toggle("collapsed", !open);
    setTimeout(() => viewer && viewer.resize(), 260);
  }

  function clearConsole() {
    S.logs = [];
    $("#console-body").innerHTML = "";
  }

  function appendLogs(lines) {
    if (!lines?.length) return;
    const body = $("#console-body");
    const frag = document.createDocumentFragment();
    lines.forEach((line) => {
      S.logs.push(line);
      const span = document.createElement("span");
      span.className = "log-line";
      if (/error|traceback|失败|错误/i.test(line)) span.classList.add("err");
      else if (/完成|✓|files written|success/i.test(line)) span.classList.add("hl");
      else if (/^\[DrugHX\]/.test(line)) span.classList.add("sys");
      span.textContent = line;
      frag.appendChild(span);
    });
    body.appendChild(frag);
    while (body.childElementCount > 2000) body.removeChild(body.firstElementChild);
    body.scrollTop = body.scrollHeight;
  }

  function setJobUi(job) {
    const stateEl = $("#job-state");
    const map = { running: ["运行中", "running"], done: ["完成", "done"], error: ["失败", "error"], cancelled: ["已取消", "error"] };
    const [label, cls] = map[job.status] || ["空闲", ""];
    stateEl.textContent = label;
    stateEl.className = `job-state ${cls}`;
    $("#job-progress").style.width = `${Math.round((job.progress || 0) * 100)}%`;
    $("#job-message").textContent = job.message || "";
    $("#job-cancel").classList.toggle("hidden", job.status !== "running");
    if (S.job) {
      S.job.status = job.status;
      S.job.progress = job.progress;
      S.job.message = job.message;
    }
    updateJobCard();
  }

  function updateJobCard() {
    if (!S.job) return;
    const msg = $("#dock-job-msg");
    if (msg) msg.textContent = S.job.message || "";
    const bar = $("#dock-progress");
    if (bar) bar.style.width = `${Math.round((S.job.progress || 0) * 100)}%`;
  }

  function schedulePoll(delay = 700) {
    clearTimeout(S.jobTimer);
    S.jobTimer = setTimeout(pollJob, delay);
  }

  async function pollJob() {
    if (!S.job) return;
    try {
      const job = await api(`/api/jobs/${S.job.id}?since=${S.job.since || 0}`);
      S.job.since = job.log_index;
      appendLogs(job.logs || []);
      setJobUi(job);
      S.jobFailures = 0;
      if (job.status === "running") {
        schedulePoll(700);
        return;
      }
      finishJob(job);
    } catch (error) {
      S.jobFailures += 1;
      if (S.jobFailures <= 3) {
        schedulePoll(1200);
      } else {
        toast(`任务状态获取失败：${error.message}`, "error");
        S.job = null;
      }
    }
  }

  async function finishJob(job) {
    if (job.status === "done") {
      toast(job.kind === "dock" ? "对接任务完成" : job.kind === "auto" ? "全自动分析完成，报告已写入项目目录" : "任务完成", "success");
    } else if (job.status === "cancelled") {
      toast("任务已取消", "warn");
    } else {
      toast(`任务失败：${job.error || "未知错误"}`, "error", 7000);
    }
    const wasDock = job.kind === "dock";
    const wasOptimize = job.kind === "optimize";
    const wasDesign = job.kind === "design";
    const wasFishing = job.kind === "fishing";
    const wasNetpharm = job.kind === "netpharm";
    const optimizeResult = wasOptimize ? job.result : null;
    const designResult = wasDesign ? job.result : null;
    const fishingResult = wasFishing ? job.result : null;
    const netpharmResult = wasNetpharm ? job.result : null;
    const runId = job.result?.run_id || S.pendingRunId;
    S.job = null;
    S.pendingRunId = null;
    await refresh();
    if (job.kind === "report") {
      const url = S.pendingReportUrl;
      S.pendingReportUrl = null;
      if (job.status === "done" && url) {
        const result = job.result || {};
        if (result.skipped || result.notice) {
          toast(
            result.reason || result.notice || "报告已生成，但未包含垂钓与网络药理学",
            "warn",
            7000,
          );
        }
        window.open(url, "_blank", "noopener");
      }
      render();
      openConsole(false);
      return;
    }
    if (S.step === "ligands" && S.ligandMode === "fine" && S.fineLigands) {
      await loadFineLigands(S.fineQuery, S.fineOffset);
    }
    if (wasFishing && job.status === "done" && fishingResult) {
      S.fishingResult = fishingResult;
      S.step = "fishing";
      localStorage.setItem("dockforge_step", S.step);
      render();
      openConsole(false);
      return;
    }
    if (wasNetpharm && job.status === "done" && netpharmResult) {
      S.netpharmResult = netpharmResult;
      S.step = "netpharm";
      localStorage.setItem("dockforge_step", S.step);
      render();
      openConsole(false);
      return;
    }
    if (wasDesign && job.status === "done" && designResult) {
      S.designResult = designResult;
      S.step = "design";
      localStorage.setItem("dockforge_step", S.step);
      render();
      openConsole(false);
      return;
    }
    if (wasOptimize && job.status === "done" && optimizeResult) {
      S.optResult = optimizeResult;
      S.step = "optimize";
      localStorage.setItem("dockforge_step", S.step);
      render();
      openConsole(false);
      return;
    }
    if (job.kind === "auto" && job.status === "done" && job.result) {
      S.autoResult = job.result;
      if (job.result.awaiting) S.autoPick = null;
      S.step = "auto";
      localStorage.setItem("dockforge_step", S.step);
      render();
      openConsole(false);
      return;
    }
    if (wasDock && runId && S.data.runs.some((r) => r.id === runId) && job.status === "done") {
      S.runId = runId;
      applyResultSelection(bestResultIndex(activeRun()));
      S.step = "results";
      localStorage.setItem("dockforge_step", S.step);
      render();
      openConsole(false);
      loadAnalysis(true);
    }
  }

  async function launchJob(path, body, title, kind) {
    const response = await api(path, { method: "POST", body });
    if (!response.job_id) throw new Error("后端未返回任务编号");
    S.job = { id: response.job_id, kind: kind || "", title, since: 0, status: "running", progress: 0, message: "已提交" };
    S.pendingRunId = response.run_id || null;
    openConsole(true);
    clearConsole();
    appendLogs([`[DrugHX] ${title}`]);
    setJobUi({ status: "running", progress: 0, message: "已提交任务 ..." });
    schedulePoll(300);
    return response;
  }

  /* ------------------------------------------------------------------ actions */
  function bindEvents() {
    // Full-report links: when fishing or network pharmacology has not been
    // computed yet, run the missing steps as a background job first instead of
    // letting a plain GET block for minutes.
    document.addEventListener("click", async (event) => {
      const link = event.target.closest('a[href^="/api/reports/full.pdf"]');
      if (!link || event.defaultPrevented) return;
      if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      const hasFishing = (S.data?.fishing || []).length > 0;
      const hasNetpharm = (S.data?.netpharm || []).length > 0;
      if (hasFishing && hasNetpharm) return;
      event.preventDefault();
      const url = link.getAttribute("href");
      let runId = "";
      try {
        runId = new URL(url, location.origin).searchParams.get("run_id") || "";
      } catch (error) {
        runId = "";
      }
      S.pendingReportUrl = url;
      try {
        await launchJob(
          "/api/reports/prepare",
          { run_id: runId },
          "补齐报告分析（靶点垂钓 + 网络药理学）",
          "report",
        );
      } catch (error) {
        S.pendingReportUrl = null;
        toast(error.message || "补齐分析失败", "error");
      }
    });

    document.addEventListener("click", async (event) => {
      const stepBtn = event.target.closest("[data-step]");
      if (stepBtn) {
        setStep(stepBtn.dataset.step);
        return;
      }
      const actionEl = event.target.closest("[data-action]");
      if (actionEl) {
        const action = actionEl.dataset.action;
        try {
          await handleAction(action, actionEl, event);
        } catch (error) {
          toast(error.message || String(error), "error", 6500);
        }
      }
    });

    document.addEventListener("change", async (event) => {
      const el = event.target;
      if (el.id === "pharm-species") {
        S.pharmSpecies = el.value || "human";
        return;
      }
      if (el.id === "netpharm-fishing") {
        S.netpharmFishingId = el.value;
        render();
        return;
      }
      if (el.id === "fishing-ligand") {
        const ligand = ligandById(el.value);
        S.fishingLigandId = el.value;
        S.fishingTouched = true;
        S.fishingName = ligand?.name || "";
        S.fishingSmiles = ligand?.smiles || ligand?.meta?.smiles || "";
        S.fishingResult = null;
        render();
        return;
      }
      if (el.id === "fishing-name") {
        const value = el.value.trim();
        S.fishingName = value;
        S.fishingTouched = true;
        const match = knownLigands().find(
          (item) => String(item.name || "").trim().toLowerCase() === value.toLowerCase()
        );
        if (match) {
          const smiles = match.smiles || match.meta?.smiles || "";
          if (smiles) {
            S.fishingLigandId = match.id;
            S.fishingSmiles = smiles;
            render();
            return;
          }
        }
        if (value && !match) {
          api(`/api/fishing/resolve?name=${encodeURIComponent(value)}`)
            .then((data) => {
              if (data.smiles) {
                S.fishingSmiles = data.smiles;
                S.fishingLigandId = null;
                render();
                toast(`已按名称匹配到 ${data.source === "FDA" ? "FDA 参考药物" : "项目配体"}的 SMILES`, "success");
              } else {
                S.fishingSmiles = "";
                S.fishingLigandId = null;
                render();
                toast("名称未在项目中匹配到结构，请粘贴该名称对应的 SMILES", "warn", 5000);
              }
            })
            .catch(() => {});
          return;
        }
      }
      if (el.id === "opt-run-select") {
        S.runId = el.value;
        applyResultSelection(bestResultIndex(activeRun()));
        S.optResult = null;
        render();
        return;
      }
      if (el.id === "opt-result-select") {
        applyResultSelection(Number(el.value));
        S.optResult = null;
        render();
        return;
      }
      if (el.id === "design-receptor") {
        S.designReceptorId = el.value;
        render();
        return;
      }
      if (el.matches('[data-check="ligand"]')) {
        const id = el.dataset.id;
        S.autoSelect = false;
        if (el.checked && S.ligandMode !== "bulk" && S.selectedLigands.size >= FINE_LIMIT) {
          el.checked = false;
          toast(`精细模式最多选择 ${FINE_LIMIT} 个配体`, "warn", 5000);
          return;
        }
        if (el.checked) S.selectedLigands.add(id); else S.selectedLigands.delete(id);
        render({ viewer: false });
      } else if (el.matches('[data-check="dock-rec"]')) {
        const id = el.dataset.id;
        S.autoSelect = false;
        if (el.checked) S.selectedReceptors.add(id); else S.selectedReceptors.delete(id);
        render({ viewer: false });
      } else if (el.matches('[data-check="dock-lig"]')) {
        const id = el.dataset.id;
        S.autoSelect = false;
        if (el.checked) S.selectedLigands.add(id); else S.selectedLigands.delete(id);
        render({ viewer: false });
      } else if (el.matches('[data-filter-limit]')) {
        const key = el.dataset.filterLimit;
        if (!el.value.trim()) return;
        const value = Number(el.value);
        if (key && Number.isFinite(value)) {
          S.ligOptions = S.ligOptions || {};
          S.ligOptions.filter_limits = {
            ...(S.ligOptions.filter_limits || {}),
            [key]: value,
          };
        }
      } else if (el.matches('[data-opt]')) {
        const key = el.dataset.opt;
        if (key === "lig_minimize") {
          S.ligOptions.minimize = el.checked;
        } else if (key === "lig_filter") {
          S.ligOptions.filter = el.checked;
        } else {
          S.recOptions[key] = el.checked;
        }
      } else if (el.matches('[data-opt-select="charge_model"]')) {
        S.recOptions.charge_model = el.value;
      } else if (el.matches('[data-opt-select="lig_protonation"]')) {
        S.ligOptions.protonation = el.value;
      } else if (el.matches('[data-opt-input="lig_seed"]')) {
        S.ligOptions.seed = Number(el.value) || 42;
      } else if (el.matches('[data-upload="receptor"]')) {
        await uploadFiles("/api/receptors/upload", el.files, "受体");
      } else if (el.matches('[data-upload="ligand"]')) {
        await uploadFiles("/api/ligands/upload", el.files, "配体");
      } else if (el.id === "bulk-import-file") {
        const file = el.files && el.files[0];
        el.value = "";
        if (file) await importBulkLigands(file);
      } else if (el.id === "bulk-preset") {
        S.bulkPresetId = el.value;
        render();
      } else if (el.id === "param-docking-engine") {
        const map = {
          vina: "single-fine",
          "vina-gpu": "gpu-fine",
          smina: "smina-fine",
          qvina2: "qvina-fine",
        };
        const preset = (S.data.dock_presets || []).find((item) => item.id === map[el.value]);
        if (preset) applyDockPreset(preset);
      } else if (el.id === "run-select") {
        S.runId = el.value;
        applyResultSelection(bestResultIndex(activeRun()));
        render();
      }
    });

    document.addEventListener("submit", async (event) => {
      const form = event.target;
      if (!form.matches("[data-form]")) return;
      event.preventDefault();
      try {
        if (form.dataset.form === "auto-run") {
          const receptorInput = $("#auto-receptor");
          const ligandInput = $("#auto-ligands");
          const receptorFile = receptorInput?.files?.[0];
          const receptorQuery = ($("#auto-receptor-query")?.value || "").trim();
          const receptorPdb = ($("#auto-receptor-pdb")?.value || "").trim();
          const receptorSequence = ($("#auto-receptor-sequence")?.value || "").trim();
          if (!receptorFile && !receptorQuery && !receptorPdb && !receptorSequence) return toast("请选择受体文件，或输入 PDB ID / 基因或 UniProt 登录号 / 氨基酸序列", "warn");
          if (!ligandInput?.files?.length) return toast("请选择至少一个配体文件", "warn");
          const body = new FormData();
          if (receptorFile) body.append("receptor", receptorFile);
          if (receptorPdb) body.append("receptor_pdb_id", receptorPdb);
          if (receptorQuery) body.append("receptor_query", receptorQuery);
          if (receptorSequence) body.append("receptor_sequence", receptorSequence);
          [...ligandInput.files].forEach((file) => body.append("ligands", file));
          body.append("species", $("#auto-species")?.value || "human");
          body.append("online_intel", $("#auto-online-intel")?.checked ? "true" : "false");
          const autoFilterLimits = {};
          $$("#auto-filter-grid [data-filter-limit]").forEach((el) => {
            if (!el.value.trim()) return;
            const value = Number(el.value);
            if (Number.isFinite(value)) autoFilterLimits[el.dataset.filterLimit] = value;
          });
          body.append("ligand_options", JSON.stringify({
            filter: $("#auto-filter")?.checked !== false,
            filter_limits: autoFilterLimits,
          }));
          body.append("project_name", $("#auto-project-name")?.value.trim() || "");
          body.append("report_lang", $("#auto-lang")?.value || "zh");
          S.autoResult = null;
          S.autoPick = null;
          await launchJob("/api/auto", body, "全自动分析", "auto");
          toast("已开始自动分析，完成后报告会写入项目目录", "success");
          await refresh();
        } else if (form.dataset.form === "fetch-pdb") {
          const value = $("#pdb-id")?.value.trim();
          if (!value) return toast("请输入 PDB ID", "warn");
          const response = await api("/api/receptors/fetch", { method: "POST", body: { pdb_id: value } });
          toast(`已获取 ${value.toUpperCase()}`, "success");
          S.receptorId = response.receptor_id;
          await refresh();
        } else if (form.dataset.form === "add-smiles") {
          const text = $("#smiles-input")?.value.trim();
          if (!text) return toast("请输入 SMILES", "warn");
          const response = await api("/api/ligands/smiles", { method: "POST", body: { text } });
          toast(`已添加 ${response.added} 个配体`, "success");
          await refresh();
        } else if (form.dataset.form === "box") {
          const id = form.dataset.id;
          const values = {};
          ["cx", "cy", "cz", "sx", "sy", "sz"].forEach((key) => {
            values[key] = Number($(`#box-${key}`).value);
          });
          const size = [values.sx, values.sy, values.sz];
          const response = await api(`/api/receptors/${id}/box`, {
            method: "POST",
            body: { center: [values.cx, values.cy, values.cz], size, source: "manual", label: "手动设置" },
          });
          toast(`对接盒已保存（${Math.round(response.volume).toLocaleString()} Å³）`, "success");
          await refresh();
        }
      } catch (error) {
        toast(error.message, "error", 6000);
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeModal();
      if (event.key === "Enter" && event.target.id === "pdb-id") {
        event.preventDefault();
        $('[data-form="fetch-pdb"]')?.requestSubmit();
      }
      const card = event.target.closest?.('[data-action="auto-candidate"]');
      if (card && (event.key === "Enter" || event.key === " ")) {
        event.preventDefault();
        card.click();
      }
    });

    document.addEventListener("dragover", (event) => {
      const zone = event.target.closest(".dropzone");
      if (zone) {
        event.preventDefault();
        zone.classList.add("drag");
      }
    });
    document.addEventListener("dragleave", (event) => {
      const zone = event.target.closest(".dropzone");
      if (zone) zone.classList.remove("drag");
    });
    document.addEventListener("drop", async (event) => {
      const zone = event.target.closest(".dropzone");
      if (!zone) return;
      event.preventDefault();
      zone.classList.remove("drag");
      const input = zone.querySelector("input[type=file]");
      if (!input) return;
      const type = input.dataset.upload;
      const files = event.dataTransfer.files;
      await uploadFiles(type === "receptor" ? "/api/receptors/upload" : "/api/ligands/upload", files, type === "receptor" ? "受体" : "配体");
    });
  }

  async function handleAction(action, el, event) {
    switch (action) {
      case "open-tools":
        openToolsModal();
        break;
      case "full-report": {
        const runs = S.data?.runs || [];
        const optimizations = S.data?.optimizations || [];
        const designs = S.data?.designs || [];
        const latestOpt = optimizations[0];
        const latestDesign = designs[0];
        const runCandidates = runs.filter((r) => (r.results || []).length);
        const pickedRun = S.runId ? runs.find((r) => r.id === S.runId && (r.results || []).length) : null;
        const baseRun = pickedRun || (latestOpt?.parent_run
          ? runs.find((r) => r.id === latestOpt.parent_run)
          : runCandidates[runCandidates.length - 1]);
        if (!baseRun) {
          toast("暂无可用于生成报告的对接结果", "warn");
          break;
        }
        const params = new URLSearchParams({ run_id: baseRun.id });
        if (pickedRun) params.set("result_index", String(S.resultIndex));
        if (latestOpt?.run_id) params.set("optimization_id", latestOpt.run_id);
        if (latestDesign?.run_id) params.set("design_id", latestDesign.run_id);
        params.set("lang", el.dataset.lang || "zh");
        params.set("species", $("#pharm-species")?.value || S.pharmSpecies || "human");
        window.open(`/api/reports/full.pdf?${params.toString()}`, "_blank", "noopener");
        break;
      }
      case "mpo-run": {
        const runId = el.dataset.id || activeRun()?.id;
        if (!runId) break;
        S.mpoBusy = true;
        render({ viewer: false });
        try {
          S.mpo = await api(`/api/runs/${encodeURIComponent(runId)}/mpo`);
          toast(
            `MPO 完成：${S.mpo.count} 个分子，Pareto ${S.mpo.pareto_count} 个`,
            "success",
          );
        } finally {
          S.mpoBusy = false;
          render({ viewer: false });
        }
        break;
      }
      case "mpo-csv": {
        const items = S.mpo?.items || [];
        if (!items.length) break;
        const header = [
          "name", "affinity", "le", "qed", "sa", "safety", "mpo",
          "stage", "pareto", "novelty", "novelty_max", "novelty_nearest",
        ];
        const lines = [header.join(",")].concat(
          items.map((item) => [
            JSON.stringify(item.name || ""),
            item.affinity ?? "",
            item.le ?? "",
            item.qed ?? "",
            item.sa ?? "",
            item.safety ?? "",
            item.mpo ?? "",
            item.stage ?? "",
            item.pareto ? "yes" : "no",
            JSON.stringify(item.novelty?.label || ""),
            item.novelty?.max ?? "",
            JSON.stringify((item.novelty?.nearest || []).map((hit) => `${hit.name} ${hit.similarity}`).join("; ")),
          ].join(",")),
        );
        const blob = new Blob(["\ufeff" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `mpo_${S.mpo?.run_id || "run"}.csv`;
        link.click();
        URL.revokeObjectURL(link.href);
        break;
      }
      case "assay-upload": {
        const file = $("#assay-file")?.files?.[0];
        if (!file) {
          toast("请先选择 CSV/TSV 实验数据文件", "warn");
          break;
        }
        const form = new FormData();
        form.append("file", file);
        const run = activeRun();
        if (run?.id) form.append("run_id", run.id);
        S.assayBusy = true;
        render({ viewer: false });
        try {
          S.assay = await api("/api/projects/current/assays", { method: "POST", body: form });
          toast(
            `已匹配 ${S.assay.summary?.matched || 0} 条实测数据`,
            S.assay.summary?.level === "bad" ? "warn" : "success",
            5200,
          );
        } finally {
          S.assayBusy = false;
          render({ viewer: false });
        }
        break;
      }
      case "assay-clear": {
        await api("/api/projects/current/assays", { method: "DELETE" });
        S.assay = null;
        render({ viewer: false });
        toast("已清除实验校准数据", "success");
        break;
      }
      case "dose-run": {
        const potency = Number($("#dose-potency")?.value || 0);
        const mw = Number($("#dose-mw")?.value || 0);
        const species = $("#dose-species")?.value || "human";
        const fPercent = Number($("#dose-f")?.value || 50);
        if (!(potency > 0) || !(mw > 0)) {
          toast("请填写有效的目标效价与分子量", "warn");
          break;
        }
        S.doseBusy = true;
        render({ viewer: false });
        try {
          S.dose = await api("/api/dose", {
            method: "POST",
            body: {
              potency_nm: potency,
              mw,
              species,
              f_percent: fPercent,
            },
          });
          toast(S.dose.verdict, S.dose.level === "ok" ? "success" : "warn", 5000);
        } finally {
          S.doseBusy = false;
          render({ viewer: false });
        }
        break;
      }
      case "trust-run": {
        const runId = el.dataset.id || activeRun()?.id;
        if (!runId) break;
        S.trustBusy = true;
        render({ viewer: false });
        try {
          S.trust = await api(`/api/runs/${encodeURIComponent(runId)}/trust`);
          toast(
            `可信度 ${Number(S.trust.score).toFixed(0)}（${S.trust.level}）${S.trust.abstain ? "：不建议据此决策" : ""}`,
            S.trust.abstain ? "warn" : "success",
            5200,
          );
        } finally {
          S.trustBusy = false;
          render({ viewer: false });
        }
        break;
      }
      case "mmp-run": {
        const runId = el.dataset.id || activeRun()?.id;
        if (!runId) break;
        S.mmpBusy = true;
        render({ viewer: false });
        try {
          S.mmp = await api(`/api/runs/${encodeURIComponent(runId)}/mmp`);
          toast(`已挖掘 ${S.mmp.rule_count || 0} 条 SAR 规则`, "success");
        } finally {
          S.mmpBusy = false;
          render({ viewer: false });
        }
        break;
      }
      case "active-run": {
        const runId = el.dataset.id || activeRun()?.id;
        if (!runId) break;
        S.activeBusy = true;
        render({ viewer: false });
        try {
          S.active = await api(`/api/runs/${encodeURIComponent(runId)}/active?batch=12`);
          toast(
            S.active.mode === "active"
              ? `已推荐 ${S.active.batch} 个分子（训练 ${S.active.train} 条实测）`
              : "实测数据不足，已按对接打分返回候选",
            S.active.mode === "active" ? "success" : "warn",
            5200,
          );
        } finally {
          S.activeBusy = false;
          render({ viewer: false });
        }
        break;
      }
      case "pbpk-run": {
        const species = $("#pbpk-species")?.value || "human";
        const route = $("#pbpk-route")?.value || "oral";
        const dose = Number($("#pbpk-dose")?.value || 5);
        const fPercent = Number($("#pbpk-f")?.value || 50);
        const days = Number($("#pbpk-days")?.value || 14);
        const ligand = activeLigand();
        const smiles = ligand?.smiles || ligand?.meta?.smiles || "";
        if (!(dose > 0)) {
          toast("请填写有效剂量", "warn");
          break;
        }
        S.pbpkBusy = true;
        render({ viewer: false });
        try {
          S.pbpkData = await api("/api/pbpk", {
            method: "POST",
            body: {
              smiles,
              mw: ligand?.meta?.mw || undefined,
              logp: ligand?.meta?.logp || undefined,
              dose_mg_kg: dose,
              species,
              route,
              f_percent: fPercent,
              duration_days: days,
            },
          });
          toast(`PBPK 完成：休药期 ${S.pbpkData.withdrawal_days ?? 0} 天`, "success", 5000);
        } finally {
          S.pbpkBusy = false;
          render({ viewer: false });
        }
        break;
      }
      case "benchmark-run": {
        const file = $("#benchmark-file")?.files?.[0];
        const runId = el.dataset.id || activeRun()?.id;
        if (!file) {
          toast("请先选择活性物 CSV 文件", "warn");
          break;
        }
        if (!runId) break;
        const form = new FormData();
        form.append("file", file);
        S.benchmarkBusy = true;
        render({ viewer: false });
        try {
          S.benchmark = await api(`/api/runs/${encodeURIComponent(runId)}/benchmark`, {
            method: "POST",
            body: form,
          });
          toast(
            `回溯校准完成：AUC ${S.benchmark.metrics?.auc ?? "—"}，BEDROC ${S.benchmark.metrics?.bedroc ?? "—"}`,
            "success",
            5200,
          );
        } finally {
          S.benchmarkBusy = false;
          render({ viewer: false });
        }
        break;
      }
      case "rec-select":
        S.receptorId = el.dataset.id;
        S.rmsd = null;
        render();
        break;
      case "prep-receptor":
        event.stopPropagation();
        await launchJob(`/api/receptors/${el.dataset.id}/prepare`, { options: S.recOptions }, "准备受体", "receptor");
        await refresh();
        break;
      case "rec-info":
        showReceptorInfo(el.dataset.id);
        break;
      case "rec-delete":
        if (!confirm("删除该受体及其准备文件？")) return;
        await api(`/api/receptors/${el.dataset.id}`, { method: "DELETE" });
        S.receptorId = null;
        await refresh();
        toast("受体已删除", "success");
        break;
      case "chain-all":
        S.recOptions.chains = [];
        render({ viewer: false });
        break;
      case "toggle-chain": {
        const chain = el.dataset.chain;
        const set = new Set(S.recOptions.chains || []);
        if (set.has(chain)) set.delete(chain); else set.add(chain);
        S.recOptions.chains = [...set];
        render({ viewer: false });
        break;
      }
      case "lig-select":
        S.ligandId = el.dataset.id;
        render();
        break;
      case "lig-copy-smiles": {
        const smiles = el.dataset.smiles || "";
        if (!smiles) return toast("该配体没有 SMILES", "warn");
        try {
          await navigator.clipboard.writeText(smiles);
          toast("SMILES 已复制", "success");
        } catch (_) {
          window.prompt("复制 SMILES：", smiles);
        }
        break;
      }
      case "ligand-mode": {
        S.ligandMode = el.dataset.mode === "bulk" ? "bulk" : "fine";
        render();
        if (S.ligandMode === "fine" && !S.fineLigands) {
          await loadFineLigands("", 0);
        }
        break;
      }
      case "fine-search":
        await loadFineLigands($("#fine-query")?.value || "", 0);
        break;
      case "fine-page":
        await loadFineLigands(S.fineQuery, Number(el.dataset.offset) || 0);
        break;
      case "lig-select-none":
        S.autoSelect = false;
        S.selectedLigands.clear();
        render();
        break;
      case "prep-fine": {
        const ids = [...S.selectedLigands];
        if (!ids.length) return toast("请先勾选要精细准备的配体", "warn");
        if (ids.length > FINE_LIMIT) return toast(`精细准备最多 ${FINE_LIMIT} 个配体`, "warn", 6000);
        await launchJob(
          "/api/ligands/prepare",
          { ids, options: S.ligOptions },
          `精细准备 ${ids.length} 个配体`,
          "ligand",
        );
        await refresh();
        break;
      }
      case "prep-bulk": {
        const total = Number(S.data.ligand_total || 0);
        if (!total) return toast("当前项目没有配体", "warn");
        const options = {
          protonation: $("#bulk-protonation")?.value || "ph7.4",
          minimize: Boolean($("#bulk-minimize")?.checked),
          filter: Boolean($("#bulk-filter")?.checked),
          filter_limits: S.ligOptions?.filter_limits || {},
          seed: Number($("#bulk-seed")?.value) || 42,
          threads: Math.max(1, Math.min(32, Number($("#bulk-threads")?.value) || 8)),
        };
        await launchJob(
          "/api/ligands/prepare",
          { scope: "all_raw", options },
          `大规模准备配体 · ${total} 个`,
          "ligand",
        );
        await refresh();
        break;
      }
      case "load-top": {
        const runId = el.dataset.run;
        if (!runId) return toast("请先完成大规模对接", "warn");
        const limit = Math.max(5, Math.min(200, Number($("#bulk-top-n")?.value) || 50));
        S.bulkTopN = limit;
        const response = await api(`/api/runs/${encodeURIComponent(runId)}/top?limit=${limit}`);
        S.bulkTop = { runId, items: response.items || [] };
        render();
        toast(`已获取前 ${S.bulkTop.items.length} 个候选`, "success");
        break;
      }
      case "use-top": {
        const items = S.bulkTop?.items || [];
        if (!items.length) return toast("请先获取候选列表", "warn");
        S.autoSelect = false;
        S.selectedLigands = new Set(
          items.map((item) => item.ligand_id).filter(Boolean)
        );
        S.ligandMode = "fine";
        S.fineLigands = null;
        setStep("ligands");
        await loadFineLigands("", 0);
        toast(
          `已载入 ${items.length} 个候选，点击“准备所选”开始精细准备`,
          "success",
          6500,
        );
        break;
      }
      case "dock-bulk": {
        const receptorId = $("#bulk-receptor")?.value;
        if (!receptorId) return toast("请选择受体", "warn");
        const presetId = $("#bulk-preset")?.value || "ultra-fast";
        const preset = (S.data.dock_presets || []).find((item) => item.id === presetId);
        const params = { ...(preset?.params || {}), bulk: true };
        if (params.docking_engine === "vina-gpu") {
          params.gpu_threads = Math.max(1000, Math.min(20000, Number($("#bulk-gpu-threads")?.value) || 2000));
          params.gpu_search_depth = Math.max(1, Math.min(20, Number($("#bulk-gpu-depth")?.value) || 1));
          params.gpu_chunk = Math.max(100, Math.min(5000, Number($("#bulk-gpu-chunk")?.value) || 1500));
        }
        params.engine = params.docking_engine === "vina-gpu" ? "gpu" : "cpu";
        S.bulkReceptorId = receptorId;
        S.bulkPresetId = presetId;
        const ready = Number(S.data.ligand_stats?.ready || 0);
        await launchJob(
          "/api/dock",
          { scope: "all_ready", receptor_ids: [receptorId], params },
          `大规模对接 · ${ready} 个配体`,
          "dock",
        );
        await refresh();
        break;
      }
      case "lig-import": {
        $("#bulk-import-file")?.click();
        break;
      }
      case "lig-clear": {
        const scope = el.dataset.scope === "prepared" ? "prepared" : "unprepared";
        const label = scope === "prepared" ? "已准备" : "未准备";
        const count = scope === "prepared"
          ? Number(S.data.ligand_stats?.ready || 0)
          : Number(S.data.ligand_total || 0) - Number(S.data.ligand_stats?.ready || 0);
        if (!count) return toast(`没有${label}的配体`, "warn");
        if (!window.confirm(`确定删除全部 ${count} 个${label}配体？该操作不可撤销。`)) return;
        try {
          const response = await api("/api/ligands/clear", {
            method: "POST",
            body: { scope },
          });
          S.fineLigands = null;
          S.selectedLigands.clear();
          S.knownLigands.clear();
          toast(`已移除 ${response.removed} 个${label}配体，后台正在清理文件`, "success", 5000);
          if (response.job_id) {
            S.job = {
              id: response.job_id,
              kind: "ligands",
              title: `清理${label}配体文件`,
              since: 0,
              status: "running",
              progress: 0,
              message: "正在删除配体目录",
            };
            openConsole(true);
            schedulePoll(300);
          }
          await refresh();
          if (S.step === "ligands" && S.ligandMode === "fine") {
            await loadFineLigands("", 0);
          }
        } catch (error) {
          toast(error.message, "error", 7000);
        }
        break;
      }
      case "prep-one":
        await launchJob("/api/ligands/prepare", { ids: [el.dataset.id], options: S.ligOptions }, "准备配体", "ligand");
        await refresh();
        break;
      case "prep-ligands": {
        const scope = el.dataset.scope;
        let ids;
        if (scope === "selected") {
          ids = [...S.selectedLigands].filter((id) => {
            const lig = S.data.ligands.find((l) => l.id === id);
            return lig && lig.status !== "ready";
          });
          if (!ids.length) ids = [...S.selectedLigands];
        } else {
          ids = S.data.ligands.filter((l) => l.status !== "ready").map((l) => l.id);
          if (!ids.length) ids = S.data.ligands.map((l) => l.id);
        }
        if (!ids.length) return toast("没有可准备的配体", "warn");
        await launchJob("/api/ligands/prepare", { ids, options: S.ligOptions }, `准备 ${ids.length} 个配体`, "ligand");
        await refresh();
        break;
      }
      case "lig-delete":
        if (!confirm("删除该配体？")) return;
        await api(`/api/ligands/${el.dataset.id}`, { method: "DELETE" });
        S.selectedLigands.delete(el.dataset.id);
        S.ligandId = null;
        await refresh();
        toast("配体已删除", "success");
        break;
      case "apply-pocket": {
        const rec = S.data.receptors.find((r) => r.id === el.dataset.id);
        const pocket = rec.pockets[Number(el.dataset.index)];
        if (!pocket) return;
        await api(`/api/receptors/${rec.id}/box`, {
          method: "POST",
          body: { center: pocket.center, size: pocket.size, source: "pocket", label: pocket.label },
        });
        await refresh();
        toast(`${pocket.label} 已应用`, "success");
        break;
      }
      case "apply-ref": {
        const rec = S.data.receptors.find((r) => r.id === el.dataset.id);
        const ref = rec.hetero[Number(el.dataset.index)];
        if (!ref) return;
        await api(`/api/receptors/${rec.id}/box/from-ligand`, { method: "POST", body: { ligand: ref } });
        await refresh();
        toast(`已用 ${ref.name} 定义对接盒`, "success");
        break;
      }
      case "extract-ref": {
        const rec = S.data.receptors.find((r) => r.id === el.dataset.id);
        const ref = rec.hetero[Number(el.dataset.index)];
        if (!ref) return;
        const response = await api(`/api/receptors/${rec.id}/extract-ligand`, { method: "POST", body: { ligand: ref } });
        S.ligandId = response.ligand_id;
        await refresh();
        toast(
          response.ideal_sdf
            ? `${ref.name} 已提取到配体列表（RCSB ideal SDF，键级正确）`
            : `${ref.name} 已用 PDB 坐标提取；离线模式下键级可能不准确，请检查`,
          response.ideal_sdf ? "success" : "warn",
          6000,
        );
        break;
      }
      case "detect-pockets":
        await api(`/api/receptors/${el.dataset.id}/pockets/detect`, { method: "POST", body: {} });
        await refresh();
        toast("口袋预测已更新", "success");
        break;
      case "dock-preset": {
        const preset = (S.data.dock_presets || []).find((item) => item.id === el.dataset.preset);
        if (!preset) break;
        applyDockPreset(preset);
        break;
      }
      case "engine-install": {
        const engineId = el.dataset.engine;
        toast(`正在安装 / 检测 ${engineId} ...`, "info");
        try {
          const response = await api(`/api/engines/${encodeURIComponent(engineId)}/install`, { method: "POST", body: {} });
          toast(`${engineId} 已可用：${response.path}`, "success", 6000);
          await refresh();
        } catch (error) {
          toast(error.message, "error", 9000);
        }
        break;
      }
      case "engine-path": {
        const engineId = el.dataset.engine;
        const path = window.prompt(`请输入 ${engineId} 可执行文件的完整路径`);
        if (!path) break;
        try {
          await api(`/api/engines/${encodeURIComponent(engineId)}/path`, { method: "POST", body: { path } });
          toast(`${engineId} 路径已保存`, "success");
          await refresh();
        } catch (error) {
          toast(error.message, "error", 8000);
        }
        break;
      }
      case "resume-run": {
        const runId = el.dataset.id;
        try {
          await launchJob(
            `/api/runs/${encodeURIComponent(runId)}/resume`,
            {},
            "续跑失败组合",
            "dock",
          );
          await refresh();
        } catch (error) {
          toast(error.message, "error", 9000);
        }
        break;
      }
      case "dock-start": {
        if (!S.selectedReceptors.size || !S.selectedLigands.size) return toast("请选择受体和配体", "warn");
        const params = {
          exhaustiveness: Number($("#param-exhaustiveness").value) || 16,
          num_modes: Number($("#param-num-modes").value) || 9,
          energy_range: Number($("#param-energy-range").value) || 4,
          cpu: Number($("#param-cpu").value) || 0,
          seed: Number($("#param-seed").value) || 0,
          seed_replicates: Number($("#param-seed-replicates")?.value) || 1,
          scoring: $("#param-scoring").value,
          docking_engine: $("#param-docking-engine").value,
          engine: $("#param-docking-engine").value === "vina-gpu" ? "gpu" : "cpu",
          parallel_jobs: Number($("#param-parallel").value) || 1,
          gpu_threads: Number($("#param-gpu-threads").value) || 8000,
          gpu_search_depth: Number($("#param-gpu-depth").value) || 20,
          gpu_replicas: Number($("#param-gpu-replicas").value) || 8,
          rescore: Boolean($("#param-rescore")?.checked),
          consensus: Boolean($("#param-consensus")?.checked),
        };
        await api("/api/project/params", { method: "POST", body: { params } });
        await launchJob(
          "/api/dock",
          { receptor_ids: [...S.selectedReceptors], ligand_ids: [...S.selectedLigands], params },
          `分子对接 · ${S.selectedReceptors.size} × ${S.selectedLigands.size}`,
          "dock",
        );
        await refresh();
        break;
      }
      case "auto-reveal":
        await api("/api/project/reveal", { method: "POST", body: {} });
        break;
      case "auto-candidate": {
        const item = (S.autoResult?.candidates || []).find(
          (row) => row.ligand_id === el.dataset.id,
        );
        if (!item) return;
        S.autoPick = item.ligand_id;
        render();
        const rows = [
          ["综合决策分", item.decision_score != null ? `${Number(item.decision_score).toFixed(1)} / 100` : "—"],
          ["对接打分", item.affinity != null ? `${item.affinity} kcal/mol` : "—"],
          ["构象物理校验", item.pose_pass != null ? `${Math.round(Number(item.pose_pass) * 100)}% 通过` : "未校验"],
          ["配体效率 LE", item.le != null ? Number(item.le).toFixed(3) : "—"],
          ["合成可及性 SA", item.sa != null ? Number(item.sa).toFixed(2) : "—"],
          ["分子式", item.formula || "—"],
          ["分子量", item.mw != null ? Number(item.mw).toFixed(2) : "—"],
          ["cLogP", item.logp != null ? Number(item.logp).toFixed(2) : "—"],
          ["TPSA", item.tpsa != null ? Number(item.tpsa).toFixed(1) : "—"],
          ["氢键供体 / 受体", `${item.hbd ?? "—"} / ${item.hba ?? "—"}`],
          ["可旋转键", item.rotb ?? "—"],
          ["重原子", item.heavy ?? "—"],
          ["环数 / 最大环", `${item.rings ?? "—"} / ${item.max_ring ?? "—"}`],
          ["QED", item.qed != null ? Number(item.qed).toFixed(3) : "—"],
        ];
        const warningHtml = [
          (item.decision_notes || []).length
            ? `<div class="note" style="margin-top:8px">评分说明：${esc(item.decision_notes.join("；"))}</div>`
            : "",
          (item.warnings || []).length
            ? `<div class="note" style="margin-top:8px;color:#b45309">结构提示：${esc(item.warnings.join("；"))}</div>`
            : "",
        ].join("");
        openModal(
          item.name || "候选分子",
          "化学结构、对接与成药性参数",
          `<div style="display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start">
             <img src="${esc(item.svg)}?width=420&height=320" alt="${esc(item.name)}" style="border:1px solid rgba(0,0,0,.08);border-radius:10px;background:#fff;max-width:100%">
             <table class="table" style="flex:1;min-width:240px">${rows
               .map(([key, value]) => `<tr><th style="text-align:left;padding:4px 10px 4px 0;white-space:nowrap">${esc(key)}</th><td class="mono">${esc(String(value))}</td></tr>`)
               .join("")}</table>
           </div>
           <div class="note" style="margin-top:10px;word-break:break-all">SMILES：<span class="mono">${esc(item.smiles || "—")}</span></div>
           ${warningHtml}`,
          `<button class="btn primary" data-action="auto-pick" data-id="${esc(item.ligand_id)}">选择此分子</button>`,
        );
        break;
      }
      case "auto-pick":
        S.autoPick = el.dataset.id;
        closeModal();
        render();
        break;
      case "auto-pick-best": {
        const first = (S.autoResult?.candidates || [])[0];
        if (!first) return toast("没有可选择的候选分子", "warn");
        S.autoPick = first.ligand_id;
        render();
        break;
      }
      case "auto-continue":
        if (!S.autoPick) return toast("请先选择一个候选分子", "warn");
        await launchJob(
          "/api/auto/continue",
          { ligand_id: S.autoPick },
          "自动分析（继续）",
          "auto",
        );
        break;
      case "auto-reveal-report":
        try {
          const response = await api("/api/project/reveal", { method: "POST", body: { kind: "report" } });
          toast(`已打开报告文件夹`, "success");
          if (response?.path) S.reportDir = response.path;
        } catch (error) {
          toast(error.message || "无法打开报告文件夹", "error");
        }
        break;
      case "cancel-job":
        if (S.job) {
          await api(`/api/jobs/${S.job.id}/cancel`, { method: "POST", body: {} });
          toast("正在取消 ...", "warn");
        }
        break;
      case "select-result":
        S.runId = activeRun()?.id || S.runId;
        applyResultSelection(Number(el.dataset.index));
        render();
        loadAnalysis(true);
        break;
      case "pick-result":
        S.runId = activeRun()?.id || S.runId;
        applyResultSelection(Number(el.dataset.index));
        render();
        loadAnalysis(true);
        toast("已选择该配体，后续模块与报告默认使用它", "success");
        break;
      case "set-mode":
        S.mode = Number(el.dataset.mode);
        S.analysis = null;
        S.analysisKey = "";
        render();
        loadAnalysis(true);
        break;
      case "analyze":
        await loadAnalysis(false);
        break;
      case "admet":
        await openAdmetModal(el.dataset.run, Number(el.dataset.index));
        break;
      case "pharmacophore":
        await openPharmacophoreModal(el.dataset.run, Number(el.dataset.index));
        break;
      case "mmgbsa":
        await openMmgbsaModal(el.dataset.run, Number(el.dataset.index));
        break;
      case "admet-calc":
        await runAdmetCalculator();
        break;
      case "opt-start": {
        const run = activeRun();
        const result = activeResult();
        if (!run || !result || result.status !== "ok") {
          return toast("请先选择一个成功的母体结果", "warn");
        }
        const options = {
          max_candidates: Number($("#opt-candidates")?.value || 6),
          exhaustiveness: Number($("#opt-exhaustiveness")?.value || 8),
          num_modes: 3,
          engine: $("#opt-engine")?.value || "cpu",
          parallel_jobs: Number($("#opt-parallel")?.value || 2),
          cpu: 0,
          seed: 42,
        };
        await launchJob(
          "/api/optimize",
          { run_id: run.id, index: S.resultIndex, options },
          "药物优化 · 骨架跃迁与自动跑分",
          "optimize"
        );
        break;
      }
      case "design-start": {
        const receptorId = $("#design-receptor")?.value;
        if (!receptorId) return toast("请选择靶点", "warn");
        const referenceId = $("#design-reference")?.value || "";
        const referenceLigand = (S.data?.ligands || []).find((item) => item.id === referenceId);
        const referenceSmiles = referenceLigand
          ? (referenceLigand.smiles || referenceLigand.meta?.smiles || "")
          : "";
        const options = {
          max_designs: Number($("#design-count")?.value || 20),
          exhaustiveness: Number($("#design-exhaustiveness")?.value || 8),
          num_modes: 3,
          engine: $("#design-engine")?.value || "cpu",
          parallel_jobs: Number($("#design-parallel")?.value || 2),
          cpu: 0,
          seed: Number($("#design-seed")?.value || 42),
          mode: $("#design-mode")?.value || "auto",
          reference_smiles: referenceSmiles,
        };
        await launchJob(
          "/api/design",
          { receptor_id: receptorId, options },
          "分子设计 · 靶向生成与数据库检索",
          "design"
        );
        break;
      }
      case "fishing-start": {
        const smiles = ($("#fishing-smiles")?.value || "").trim();
        if (!smiles) return toast("请输入 SMILES", "warn");
        const externalGenes = ($("#fishing-external-genes")?.value || "")
          .split(/[\s,;]+/)
          .map((item) => item.trim())
          .filter(Boolean);
        S.fishingSmiles = smiles;
        S.fishingName = ($("#fishing-name")?.value || "").trim();
        S.fishingExternalGenes = externalGenes.join(", ");
        const organisms = ["Homo_sapiens", "Mus_musculus", "Rattus_norvegicus"].filter(
          (value) => $(`.fishing-organism[value="${value}"]`)?.checked
        );
        const options = {
          depth: $("#fishing-depth")?.value || "deep",
          similarity: Number($("#fishing-similarity")?.value || 50),
          molecules: Number($("#fishing-molecules")?.value || 60),
          targets: Number($("#fishing-targets")?.value || 120),
          swiss_organisms: organisms.length ? organisms : ["Homo_sapiens"],
          external_genes: externalGenes,
        };
        await launchJob(
          "/api/fishing/run",
          { smiles, name: $("#fishing-name")?.value || "查询分子", options },
          "靶点垂钓 · 多源融合",
          "fishing"
        );
        break;
      }
      case "fishing-use-ligand": {
        const field = $("#fishing-smiles");
        const nameField = $("#fishing-name");
        if (field) field.value = el.dataset.smiles || "";
        if (nameField && (!nameField.value || nameField.value === "query")) {
          nameField.value = el.dataset.name || "查询分子";
        }
        S.fishingSmiles = el.dataset.smiles || "";
        S.fishingName = nameField?.value || "";
        S.fishingTouched = true;
        toast("已填入当前项目配体", "success");
        break;
      }
      case "fishing-methods": {
        const smiles = ($("#fishing-smiles")?.value || "").trim();
        if (!smiles) return toast("请先填入 SMILES", "warn");
        const depth = $("#fishing-depth")?.value || "deep";
        const similarity = $("#fishing-similarity")?.value || 50;
        const molecules = $("#fishing-molecules")?.value || 60;
        const targets = $("#fishing-targets")?.value || 60;
        const organism = $("#fishing-swiss-organism")?.value || "Homo_sapiens";
        const organismText = { Homo_sapiens: "人源", Mus_musculus: "小鼠", Rattus_norvegicus: "大鼠" }[organism] || "人源";
        const organismEn = { Homo_sapiens: "human", Mus_musculus: "mouse", Rattus_norvegicus: "rat" }[organism] || "human";
        const external = ($("#fishing-external-genes")?.value || "").trim();
        const zh = `潜在靶点通过多源反向垂钓获得。以化合物 SMILES（${smiles}）为查询，整合 ChEMBL 相似性检索（Tanimoto ≥ ${similarity}%，纳入前 ${molecules} 个相似分子的人源活性与作用机制记录）、SwissTargetPrediction（2D/3D 相似性，${organismText}）、BindingDB、PubChem、DGIdb、Open Targets 与 RCSB PDB 共晶结构证据；${external ? "并导入 PharmMapper/SEA/idTarget/COMET 等外部预测的基因列表；" : ""}靶点按综合证据评分排序，保留前 ${targets} 个（${depth === "deep" ? "深度模式" : "标准模式"}），基因符号经 UniProt 统一映射后用于 STRING 蛋白互作网络与 Enrichr 通路富集。上述结果仅为 in silico 靶点假设，需通过 SPR、CETSA、亲和捕获或细胞实验进一步验证。`;
        const en = `Potential targets were obtained through multi-source reverse fishing. Using the compound SMILES (${smiles}) as the query, we integrated ChEMBL similarity searching (Tanimoto ≥ ${similarity}%, top ${molecules} analogues with human activity and mechanism records), SwissTargetPrediction (2D/3D similarity, ${organismEn}), BindingDB, PubChem, DGIdb, Open Targets and RCSB PDB co-crystal evidence${external ? ", together with gene lists imported from PharmMapper/SEA/idTarget/COMET" : ""}. Targets were ranked by a composite evidence score and the top ${targets} were retained (${depth === "deep" ? "deep mode" : "standard mode"}). Gene symbols were unified through UniProt and used for STRING protein–protein interaction networks and Enrichr pathway enrichment. These results are in silico target hypotheses and require experimental validation by SPR, CETSA, affinity capture or cell-based assays.`;
        const text = `${zh}\n\n${en}`;
        try {
          await navigator.clipboard.writeText(text);
          toast("方法段落已复制到剪贴板", "success");
        } catch (_) {
          window.prompt("复制以下方法段落：", text);
        }
        break;
      }
      case "fishing-sdf": {
        const input = $("#fishing-sdf");
        if (!input?.files?.length) return toast("请选择 SDF/SMILES 文件", "warn");
        const form = new FormData();
        form.append("file", input.files[0]);
        const response = await fetch("/api/tools/structure-to-smiles", {
          method: "POST",
          body: form,
        });
        if (!response.ok) {
          let detail = `HTTP ${response.status}`;
          try { detail = (await response.json()).detail || detail; } catch (_) { /* ignore */ }
          return toast(`提取失败：${detail}`, "err");
        }
        const data = await response.json();
        const first = data.molecules?.[0];
        if (!first) return toast("文件里没有有效分子", "warn");
        const field = $("#fishing-smiles");
        if (field) field.value = first.smiles;
        toast(`已提取 ${data.count} 个分子，使用 ${first.name || "第一个分子"}`, "success");
        break;
      }
      case "fishing-to-sdf": {
        const smiles = ($("#fishing-smiles")?.value || "").trim();
        if (!smiles) return toast("请先输入或提取 SMILES", "warn");
        const response = await fetch("/api/tools/smiles-to-sdf", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles, name: "fishing_query" }),
        });
        if (!response.ok) {
          let detail = `HTTP ${response.status}`;
          try { detail = (await response.json()).detail || detail; } catch (_) { /* ignore */ }
          return toast(`转换失败：${detail}`, "err");
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "fishing_query.sdf";
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        toast("已导出 SDF", "success");
        break;
      }
      case "netpharm-start": {
        const fishingId = $("#netpharm-fishing")?.value || "";
        if (!fishingId) return toast("请先完成靶点垂钓，或选择一次垂钓结果", "warn");
        const options = {
          genes: Number($("#netpharm-genes")?.value || 80),
          string_score: Number($("#netpharm-score")?.value || 400),
          expand: Number($("#netpharm-expand")?.value || 15),
        };
        const extraGenes = ($("#netpharm-extra-genes")?.value || "")
          .split(/[\s,;]+/)
          .map((item) => item.trim())
          .filter(Boolean);
        await launchJob(
          "/api/netpharm/run",
          { options, fishing_id: fishingId, genes: extraGenes },
          "网络药理学 · 垂钓基因集 + STRING + 多库富集",
          "netpharm"
        );
        break;
      }
      case "goto-fishing": {
        S.step = "fishing";
        localStorage.setItem("dockforge_step", S.step);
        render();
        break;
      }
      case "goto-netpharm": {
        S.step = "netpharm";
        localStorage.setItem("dockforge_step", S.step);
        render();
        break;
      }
      case "fishing-load": {
        const response = await api(`/api/fishing?record_id=${encodeURIComponent(el.dataset.id)}`);
        S.fishingResult = response.record;
        S.fishingSmiles = response.record?.compound?.smiles || "";
        render();
        break;
      }
      case "netpharm-load": {
        const response = await api(`/api/netpharm?record_id=${encodeURIComponent(el.dataset.id)}`);
        S.netpharmResult = response.record;
        render();
        break;
      }
      case "synth-use-ligand": {
        const smilesField = $("#synth-smiles");
        const nameField = $("#synth-name");
        if (smilesField) smilesField.value = el.dataset.smiles || "";
        if (nameField && !nameField.value) nameField.value = el.dataset.name || "";
        toast("已填入当前结果配体", "success");
        break;
      }
      case "synthesis-start": {
        const smiles = ($("#synth-smiles")?.value || "").trim();
        if (!smiles) return toast("请输入候选分子 SMILES", "warn");
        const name = ($("#synth-name")?.value || "").trim() || "候选分子";
        toast("正在计算 SAscore 与逆合成路线 ...", "info");
        const response = await api("/api/synthesis/run", {
          method: "POST",
          body: { smiles, name },
        });
        S.synthesisResult = response.record;
        await refresh();
        break;
      }
      case "synthesis-load": {
        const response = await api(`/api/synthesis?record_id=${encodeURIComponent(el.dataset.id)}`);
        S.synthesisResult = response.record;
        S.step = "synthesis";
        localStorage.setItem("dockforge_step", S.step);
        render();
        break;
      }
      case "synthesis-result": {
        const response = await api(
          `/api/runs/${encodeURIComponent(el.dataset.run)}/result/${Number(el.dataset.index)}/synthesis`
        );
        S.synthesisResult = response.record;
        S.step = "synthesis";
        localStorage.setItem("dockforge_step", S.step);
        render();
        break;
      }
      case "admet-use-ligand": {
        const smilesField = $("#admet-single-smiles");
        const nameField = $("#admet-name");
        if (smilesField) smilesField.value = el.dataset.smiles || "";
        if (nameField && !nameField.value) nameField.value = el.dataset.name || "";
        toast("已填入当前结果配体", "success");
        break;
      }
      case "admet-run": {
        const smiles = ($("#admet-single-smiles")?.value || "").trim();
        if (!smiles) return toast("请输入 SMILES", "warn");
        const name = ($("#admet-name")?.value || "").trim() || "候选分子";
        const data = await api(`/api/admet?smiles=${encodeURIComponent(smiles)}&name=${encodeURIComponent(name)}`);
        S.admetResult = data;
        render();
        toast("ADMET 计算完成", "success");
        break;
      }
      case "admet-pdf-single": {
        const smiles = ($("#admet-single-smiles")?.value || "").trim();
        if (!smiles) return toast("请输入 SMILES", "warn");
        const name = ($("#admet-name")?.value || "").trim() || "候选分子";
        window.open(`/api/admet.pdf?smiles=${encodeURIComponent(smiles)}&name=${encodeURIComponent(name)}`, "_blank");
        break;
      }
      case "admet-batch": {
        const text = ($("#admet-batch-input")?.value || "").trim();
        if (!text) return toast("请粘贴至少一个 SMILES", "warn");
        const molecules = text
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter(Boolean)
          .map((line, index) => {
            const comma = line.indexOf(",");
            if (comma > 0) {
              return { name: line.slice(0, comma).trim() || `分子 ${index + 1}`, smiles: line.slice(comma + 1).trim() };
            }
            return { name: `分子 ${index + 1}`, smiles: line };
          })
          .filter((item) => item.smiles);
        if (!molecules.length) return toast("没有可解析的 SMILES", "warn");
        const response = await api("/api/admet/batch", { method: "POST", body: { molecules } });
        S.admetBatch = response.profiles || [];
        S.admetErrors = response.errors || [];
        render();
        toast(`已计算 ${S.admetBatch.length} 个分子`, "success");
        break;
      }
      case "pharm-use-ligand": {
        const ligandId = $("#pharm-ligand")?.value;
        const ligand = ligandById(ligandId);
        if (!ligand) return toast("请选择一个项目配体", "warn");
        const smiles = ligand.smiles || ligand.meta?.smiles || "";
        if (!smiles) return toast("该配体没有 SMILES", "warn");
        if ($("#pharm-smiles")) $("#pharm-smiles").value = smiles;
        if ($("#pharm-name")) $("#pharm-name").value = ligand.name || "";
        toast("已填入配体", "success");
        break;
      }
      case "pharm-run": {
        const smiles = ($("#pharm-smiles")?.value || "").trim();
        if (!smiles) return toast("请输入 SMILES", "warn");
        const params = new URLSearchParams({
          smiles,
          name: ($("#pharm-name")?.value || "").trim() || "候选分子",
          species: $("#pharm-species")?.value || S.pharmSpecies || "human",
          genes: ($("#pharm-genes")?.value || "").trim(),
          online: $("#pharm-online")?.checked ? "true" : "false",
        });
        S.pharmGenes = params.get("genes") || "";
        S.pharmSpecies = params.get("species") || "human";
        const data = await api(`/api/pharm?${params.toString()}`);
        S.pharmResult = data;
        render();
        toast("综合药理计算完成", "success");
        break;
      }
      case "pharm-pdf": {
        const smiles = ($("#pharm-smiles")?.value || "").trim();
        if (!smiles) return toast("请输入 SMILES", "warn");
        const params = new URLSearchParams({
          smiles,
          name: ($("#pharm-name")?.value || "").trim() || "候选分子",
          species: $("#pharm-species")?.value || S.pharmSpecies || "human",
          genes: ($("#pharm-genes")?.value || "").trim(),
          online: $("#pharm-online")?.checked ? "true" : "false",
          lang: el.dataset.lang || "zh",
        });
        window.open(`/api/pharm.pdf?${params.toString()}`, "_blank");
        break;
      }
      case "pharm-report": {
        const smiles = ($("#pharm-smiles")?.value || "").trim();
        if (!smiles) return toast("请输入 SMILES", "warn");
        const params = new URLSearchParams({
          smiles,
          name: ($("#pharm-name")?.value || "").trim() || "候选分子",
          species: $("#pharm-species")?.value || S.pharmSpecies || "human",
          genes: ($("#pharm-genes")?.value || "").trim(),
          online: $("#pharm-online")?.checked ? "true" : "false",
          lang: el.dataset.lang || "zh",
        });
        window.open(`/api/reports/compound.pdf?${params.toString()}`, "_blank");
        break;
      }
      case "experts-refresh": {
        S.expertsData = null;
        render();
        break;
      }
      case "experts-validation": {
        const run = activeRun();
        if (!run) break;
        try {
          toast("正在做重对接验证 ...", "info", 2500);
          S.validationData = await api(`/api/runs/${encodeURIComponent(run.id)}/validation.json`);
        } catch (error) {
          toast(`重对接验证失败：${error.message}`, "warn", 5000);
        }
        render();
        break;
      }
      case "goto-pharm": {
        S.step = "pharm";
        localStorage.setItem("dockforge_step", S.step);
        render();
        break;
      }
      case "ddi-use-ligand": {
        const smilesField = $("#ddi-smiles");
        const nameField = $("#ddi-name");
        if (smilesField) smilesField.value = el.dataset.smiles || "";
        if (nameField && !nameField.value) nameField.value = el.dataset.name || "";
        toast("已填入当前结果配体", "success");
        break;
      }
      case "ddi-default-drugs": {
        const field = $("#ddi-codrugs");
        if (field && !field.value.trim()) {
          field.value = [
            "Atorvastatin",
            "Warfarin",
            "Clopidogrel",
            "Omeprazole",
            "Ciprofloxacin",
            "Sertraline",
            "Simvastatin",
            "Losartan",
          ].join("\n");
        }
        break;
      }
      case "ddi-start": {
        const smiles = ($("#ddi-smiles")?.value || "").trim();
        if (!smiles) return toast("请输入候选分子 SMILES", "warn");
        const name = ($("#ddi-name")?.value || "").trim() || "候选分子";
        const coDrugs = ($("#ddi-codrugs")?.value || "")
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter(Boolean)
          .map((line) => {
            const comma = line.indexOf(",");
            if (comma > 0) {
              return { name: line.slice(0, comma).trim(), smiles: line.slice(comma + 1).trim() };
            }
            return { name: line };
          });
        if (!coDrugs.length && !$("#ddi-fda-all")?.checked) return toast("请至少填写一个联用药物，或勾选与全部 FDA 药物比对", "warn");
        const targets = ($("#ddi-targets")?.value || "")
          .split(/[\s,;]+/)
          .map((item) => item.trim())
          .filter(Boolean);
        const response = await api("/api/ddi/run", {
          method: "POST",
          body: {
            smiles,
            name,
            co_drugs: coDrugs,
            targets,
            use_fda_all: Boolean($("#ddi-fda-all")?.checked),
          },
        });
        S.ddiResult = response.record;
        await refresh();
        toast("药物互作分析完成", "success");
        break;
      }
      case "drugref-use-ligand": {
        const smilesField = $("#drugref-smiles");
        const nameField = $("#drugref-name");
        if (smilesField) smilesField.value = el.dataset.smiles || "";
        if (nameField && !nameField.value) nameField.value = el.dataset.name || "";
        toast("已填入当前结果配体", "success");
        break;
      }
      case "drugref-start": {
        const smiles = ($("#drugref-smiles")?.value || "").trim();
        if (!smiles) return toast("请输入候选分子 SMILES", "warn");
        const name = ($("#drugref-name")?.value || "").trim() || "候选分子";
        const targets = ($("#drugref-targets")?.value || "")
          .split(/[\s,;]+/)
          .map((item) => item.trim())
          .filter(Boolean);
        toast("正在计算结构、靶点与 ADMET 相似度 ...", "info");
        const response = await api("/api/drugref/run", {
          method: "POST",
          body: { smiles, name, targets },
        });
        S.drugrefResult = response.record;
        await refresh();
        toast("FDA 参考分析完成", "success");
        break;
      }
      case "design-save-db": {
        const text = $("#design-db")?.value || "";
        if (!text.trim()) return toast("数据库为空", "warn");
        const response = await api("/api/design/database", { method: "POST", body: { text } });
        toast(`已保存 ${response.count} 条数据库记录`, "success");
        break;
      }
      case "opt-refine": {
        const runId = el.dataset.run;
        const index = Number(el.dataset.index);
        S.runId = runId;
        applyResultSelection(index);
        S.optResult = null;
        render();
        const run = activeRun();
        const result = activeResult();
        if (!run || !result || result.status !== "ok") return toast("最佳候选不可用", "warn");
        await launchJob(
          "/api/optimize",
          {
            run_id: run.id,
            index,
            options: {
              max_candidates: 6,
              exhaustiveness: 8,
              num_modes: 3,
              engine: "cpu",
              parallel_jobs: 2,
              cpu: 0,
              seed: 42,
            },
          },
          "继续优化最佳候选",
          "optimize"
        );
        break;
      }
      case "figure-studio":
        await openFigureStudio();
        break;
      case "rmsd":
        await loadRmsd();
        break;
      case "posebusters": {
        if (S.poseBusy) return;
        const run = activeRun();
        const result = activeResult();
        if (!run || !result || result.status !== "ok") return;
        const mode = S.mode;
        const index = S.resultIndex;
        S.poseBusy = true;
        render();
        try {
          const data = await api(`/api/runs/${run.id}/result/${index}/posebusters`, {
            method: "POST",
            body: { mode },
          });
          if (S.runId !== run.id || S.resultIndex !== index || S.mode !== mode) return;
          S.poseCheck = data;
          S.poseCheckKey = `${run.id}:${index}:${mode}`;
          if (data.available === false) toast(data.error || "PoseBusters 不可用", "warn", 6000);
          else if (data.all_passed) toast(`物理合理性：${data.passed}/${data.total} 项全部通过`, "success");
          else toast(`未通过：${(data.failed_labels || []).join("、")}`, "warn", 7000);
        } catch (error) {
          toast(error.message, "error", 6000);
        } finally {
          S.poseBusy = false;
          if (S.step === "results") render();
        }
        break;
      }
      case "posebatch": {
        if (S.poseBusy) return;
        const run = activeRun();
        if (!run) return;
        S.poseBusy = true;
        render();
        try {
          const data = await api(`/api/runs/${run.id}/posebusters`, {
            method: "POST",
            body: { limit: 50 },
          });
          S.poseBatch = data;
          const rows = (data.rows || []).filter((row) => row.available !== false);
          toast(
            `物理校验完成：${rows.filter((row) => row.all_passed).length}/${rows.length} 个构象全部通过`,
            "success",
            6000
          );
          await refresh();
        } catch (error) {
          toast(error.message, "error", 6000);
        } finally {
          S.poseBusy = false;
          if (S.step === "results") render();
        }
        break;
      }
      case "view-log":
        await showLogModal(el.dataset.run, Number(el.dataset.index));
        break;
      case "delete-run":
        if (!confirm("删除这次对接记录及其构象文件？")) return;
        await api(`/api/runs/${el.dataset.id}`, { method: "DELETE" });
        S.runId = null;
        S.analysis = null;
        S.analysisKey = "";
        await refresh();
        toast("记录已删除", "success");
        break;
      case "viewer-reset":
        if (viewer) {
          viewer.zoomTo();
          viewer.render();
        }
        break;
      case "viewer-surface":
        S.viewerSurface = !S.viewerSurface;
        $("#surface-btn").style.color = S.viewerSurface ? "var(--accent)" : "";
        syncViewer();
        break;
      case "toggle-viewer-drawer": {
        const panel = $("#viewer-panel");
        panel.classList.toggle("drawer");
        if (panel.classList.contains("drawer")) {
          syncViewer();
          setTimeout(() => {
            if (!viewer) return;
            viewer.resize();
            viewer.render();
          }, 260);
        }
        break;
      }
      case "toggle-console":
        openConsole($("#console").classList.contains("collapsed"));
        break;
      case "clear-log":
        clearConsole();
        break;
      case "new-project":
        openNewProjectModal();
        break;
      case "open-project":
        openProjectModal();
        break;
      case "project-reveal":
        try {
          await api("/api/project/reveal", { method: "POST", body: {} });
        } catch (error) {
          toast(error.message, "error");
        }
        break;
      case "project-qc": {
        try {
          const qc = await api("/api/project/qc");
          const target = $("#project-qc-result");
          if (target) target.innerHTML = renderProjectQc(qc);
        } catch (error) {
          toast(error.message, "error", 9000);
        }
        break;
      }
      case "project-backup": {
        toast("正在打包当前项目，请稍候 ...", "info", 6000);
        try {
          const result = await api("/api/project/backup", { method: "POST", body: {} });
          toast(`备份完成：${result.path}（${result.size_mb} MB）`, "success", 10000);
        } catch (error) {
          toast(error.message, "error", 10000);
        }
        break;
      }
      case "create-project": {
        const name = $("#new-project-name")?.value.trim();
        if (!name) return toast("请输入项目名称", "warn");
        await api("/api/project/new", { method: "POST", body: { name } });
        closeModal();
        resetSelections();
        S.step = "receptor";
        await refresh();
        toast("项目已创建", "success");
        break;
      }
      case "open-project-path": {
        const path = $("#open-project-path")?.value.trim();
        if (!path) return toast("请输入项目目录", "warn");
        await openProject(path);
        break;
      }
      case "open-recent":
        await openProject(el.dataset.path);
        break;
      case "enumerate":
        await enumerateLibrary();
        break;
      case "import-enum": {
        if (!S.enumResults.length) return toast("没有可导入的分子", "warn");
        const text = S.enumResults.map((m) => `${m.smiles} ${m.name}`).join("\n");
        const response = await api("/api/ligands/smiles", { method: "POST", body: { text } });
        closeModal();
        await refresh();
        toast(`已导入 ${response.added} 个配体`, "success");
        break;
      }
      case "tool-standardize": {
        const input = $("#tool-standardize-file");
        if (!input?.files?.length) return toast("请选择 SDF / SMILES / CSV 文件", "warn");
        const form = new FormData();
        form.append("file", input.files[0]);
        const response = await fetch("/api/tools/standardize", { method: "POST", body: form });
        if (!response.ok) {
          let detail = `HTTP ${response.status}`;
          try { detail = (await response.json()).detail || detail; } catch (_) { /* ignore */ }
          return toast(`标准化失败：${detail}`, "err");
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = "standardized.sdf";
        anchor.click();
        URL.revokeObjectURL(url);
        toast("标准化完成，已下载 SDF", "success");
        break;
      }
      case "tool-diversity": {
        const count = Number($("#tool-diversity-count")?.value || 12);
        const response = await api("/api/tools/diversity", { method: "POST", body: { count } });
        S.selectedLigands = new Set(response.ids);
        S.autoSelect = false;
        closeModal();
        render();
        toast(`已选中 ${response.count} 个多样配体`, "success");
        break;
      }
      case "online-mode": {
        S.onlineMode = el.dataset.mode || "structure";
        render();
        break;
      }
      case "online-structure-run": {
        const mode = $("#online-structure-mode")?.value || "alphafold";
        const gene = ($("#online-gene")?.value || "").trim();
        const sequence = ($("#online-sequence")?.value || "").trim();
        if (mode === "alphafold" && !gene) return toast("请输入基因名或 UniProt 登录号", "warn");
        if (mode === "esmfold" && sequence.replace(/[^A-Za-z]/g, "").length < 20) return toast("请粘贴至少 20 个氨基酸的序列", "warn");
        S.onlineStructure = null;
        S.onlineBusy = mode === "esmfold" ? "ESMFold 结构预测" : "AlphaFold DB 模型下载";
        render();
        try {
          const result = await api("/api/online/structure", { method: "POST", body: { mode, gene, sequence } });
          S.onlineStructure = result;
          if (result.available) toast(`已获取 ${result.source} 结构`, "success");
          else toast(result.error || "结构获取失败", "warn", 6500);
        } finally {
          S.onlineBusy = "";
          render();
        }
        break;
      }
      case "online-structure-save": {
        const result = S.onlineStructure;
        if (!result?.pdb) return toast("当前没有可登记的结构", "warn");
        const base = String(result.gene || result.accession || "predicted").replace(/[^\w.-]+/g, "_");
        const source = String(result.source || "model").split(" ")[0];
        S.onlineBusy = "登记受体";
        render();
        try {
          const saved = await api("/api/online/structure/save", {
            method: "POST",
            body: { pdb: result.pdb, name: `${base}_${source}`, prepare: true },
          });
          if (saved.receptor_id) S.receptorId = saved.receptor_id;
          toast(
            saved.prepare_error
              ? `已登记为受体；自动准备未启动：${saved.prepare_error}`
              : "已登记为受体，后台正在准备",
            saved.prepare_error ? "warn" : "success",
            7000
          );
          await refresh();
        } finally {
          S.onlineBusy = "";
          render();
        }
        break;
      }
      case "online-target-run": {
        const query = ($("#online-target-query")?.value || "").trim();
        if (!query) return toast("请输入基因名或 UniProt 登录号", "warn");
        const species = $("#online-target-species")?.value || "human";
        S.onlineTarget = null;
        S.onlineBusy = "靶点情报";
        render();
        try {
          const result = await api("/api/online/target", { method: "POST", body: { query, species } });
          S.onlineTarget = result;
          if (result.available) toast("靶点情报已获取", "success");
          else toast(result.error || "未找到靶点", "warn", 6000);
        } finally {
          S.onlineBusy = "";
          render();
        }
        break;
      }
      case "online-compound-run": {
        const query = ($("#online-compound-query")?.value || "").trim();
        if (!query) return toast("请输入化合物名称或 SMILES", "warn");
        const kind = $("#online-compound-kind")?.value || "name";
        const threshold = Number($("#online-compound-threshold")?.value || 0.9);
        S.onlineCompound = null;
        S.onlineBusy = "PubChem 化合物查询";
        render();
        try {
          const result = await api("/api/online/compound", {
            method: "POST",
            body: { query, kind, threshold, limit: 10 },
          });
          S.onlineCompound = result;
          if (result.record?.available) toast("化合物记录已获取", "success");
          else toast(result.record?.error || "未找到化合物", "warn", 6000);
        } finally {
          S.onlineBusy = "";
          render();
        }
        break;
      }
      case "online-compound-add": {
        const smiles = el.dataset.smiles || "";
        const name = (el.dataset.name || "analog").replace(/\s+/g, "_");
        if (!smiles) return toast("该记录没有 SMILES", "warn");
        const result = await api("/api/ligands/smiles", {
          method: "POST",
          body: { text: `${smiles} ${name}` },
        });
        toast(`已加入配体库（${result.added} 个）`, "success");
        await refresh();
        break;
      }
      case "online-enrich-run": {
        const genes = ($("#online-enrich-genes")?.value || "").trim();
        if (!genes) return toast("请输入至少 3 个基因", "warn");
        const library = $("#online-enrich-library")?.value || "KEGG_2021_Human";
        S.onlineEnrich = null;
        S.onlineBusy = "Enrichr 通路富集";
        render();
        try {
          const result = await api("/api/online/enrichment", {
            method: "POST",
            body: { genes, library },
          });
          S.onlineEnrich = result;
          if (result.available) toast(`获得 ${result.rows?.length || 0} 条富集结果`, "success");
          else toast(result.error || "富集失败", "warn", 6000);
        } finally {
          S.onlineBusy = "";
          render();
        }
        break;
      }
      case "close-modal":
        closeModal();
        break;
      default:
        break;
    }
  }

  function setStep(step) {
    if (!STEP_META[step]) return;
    S.step = step;
    localStorage.setItem("dockforge_step", step);
    render();
    if (step === "results") loadAnalysis(true);
  }

  async function importBulkLigands(file) {
    const form = new FormData();
    form.append("file", file);
    toast("正在载入准备结果 ...");
    try {
      const response = await api("/api/ligands/bulk/import", { method: "POST", body: form });
      const skipped = Number(response.skipped || 0);
      toast(`已载入 ${response.imported} 个已准备配体${skipped ? `，跳过 ${skipped} 个` : ""}`, "success", 6000);
      S.fineLigands = null;
      await refresh();
      if (S.step === "ligands" && S.ligandMode === "fine") {
        await loadFineLigands(S.fineQuery, S.fineOffset);
      }
    } catch (error) {
      toast(error.message, "error", 7000);
    }
  }

  async function uploadFiles(url, files, label) {
    if (!files || !files.length) return;
    const form = new FormData();
    [...files].forEach((file) => form.append("files", file));
    toast(`正在导入 ${files.length} 个${label}文件 ...`);
    try {
      const response = await api(url, { method: "POST", body: form });
      toast(`已导入 ${response.added} 个${label}`, "success");
      if (label === "配体") {
        S.fineLigands = null;
        if (Number(response.added || 0) > FINE_LIMIT) S.ligandMode = "bulk";
      }
      await refresh();
      if (label === "配体" && S.step === "ligands" && S.ligandMode === "fine") {
        await loadFineLigands("", 0);
      }
    } catch (error) {
      toast(error.message, "error", 7000);
    }
  }

  async function loadAnalysis(silent) {
    const run = activeRun();
    const result = activeResult();
    if (!run || !result || result.status !== "ok") return;
    const runId = run.id;
    const index = S.resultIndex;
    const mode = S.mode;
    const key = `${runId}:${index}:${mode}`;
    if (S.analysisKey === key) return;
    try {
      const data = await api(`/api/runs/${runId}/result/${index}/analyze`, {
        method: "POST",
        body: { mode },
      });
      if (S.runId !== runId || S.resultIndex !== index || S.mode !== mode) return;
      S.analysis = data;
      S.analysisKey = key;
      if (S.step === "results") render();
    } catch (error) {
      if (!silent) toast(error.message, "error", 6000);
    }
  }

  async function loadRmsd() {
    const run = activeRun();
    const result = activeResult();
    if (!run || !result) return;
    const rec = S.data.receptors.find((r) => r.id === result.receptor_id);
    const hetero = rec?.hetero || [];
    if (!hetero.length) return toast("该受体没有共晶配体，无法计算 RMSD", "warn");
    const select = $("#rmsd-ref");
    const index = select ? Number(select.value) : 0;
    const reference = hetero[index] || hetero[0];
    try {
      const data = await api(`/api/runs/${run.id}/result/${S.resultIndex}/rmsd`, {
        method: "POST",
        body: { mode: S.mode, reference },
      });
      S.rmsd = data;
      S.rmsdKey = `${run.id}:${S.resultIndex}:${S.mode}`;
      if (data.available) toast(`RMSD = ${Number(data.rmsd).toFixed(2)} Å`, "success");
      else toast(data.error || "RMSD 不可用", "warn", 6000);
      render();
    } catch (error) {
      toast(error.message, "error", 6000);
    }
  }

  /* ------------------------------------------- professional add-on analyses */
  function scoreBar(score) {
    const value = Math.max(0, Math.min(100, Number(score) || 0));
    const color = value >= 70 ? "var(--green)" : value >= 45 ? "var(--amber)" : "var(--red)";
    return `<div class="score-bar"><span style="width:${Math.max(2, value)}%;background:${color}"></span></div><span class="mono" style="font-size:10px;color:${color}">${value.toFixed(0)}</span>`;
  }

  function admetBlocks(data) {
    const p = data.physchem;
    const d = data.druglike;
    const quick = [
      ["QED", Number(d.qed).toFixed(3), ""],
      ["Lipinski", d.rules.lipinski.pass ? "通过" : `${d.rules.lipinski.violations} 违反`, d.rules.lipinski.pass ? "good" : "bad"],
      ["ESOL logS", Number(p.logs).toFixed(2), ""],
      ["TPSA", `${Number(p.tpsa).toFixed(0)} Å²`, ""],
    ];
    const props = [
      ["分子量", Number(p.mw).toFixed(1)],
      ["cLogP", Number(p.logp).toFixed(2)],
      ["HBD / HBA", `${p.hbd} / ${p.hba}`],
      ["可旋转键", p.rotb],
      ["芳环 / 总环", `${p.aromatic_rings} / ${p.rings}`],
      ["Fsp³", Number(p.fsp3).toFixed(2)],
      ["摩尔折射率", p.mr],
      ["形式电荷", p.charge],
    ].map(([k, v]) => `<div class="opt-metric" style="border-left-color:var(--line)"><strong>${v}</strong><span>${k}</span></div>`).join("");
    const rows = data.admet.map((item) => `
      <tr>
        <td>${esc(item.name)}</td>
        <td class="mono">${item.value}${item.unit ? " " + esc(item.unit) : ""}</td>
        <td style="min-width:120px">${scoreBar(item.score)}</td>
        <td><span class="badge ${item.score >= 70 ? "ok" : item.score >= 45 ? "warn" : "err"}">${esc(item.verdict)}</span></td>
      </tr>`).join("");
    const eff = data.efficiency ? `
      <div class="count-grid" style="grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin-bottom:14px">
        <div><span class="eyebrow">配体效率 LE</span><div class="mono" style="font-size:16px;color:var(--accent)">${data.efficiency.le}</div></div>
        <div><span class="eyebrow">LipE</span><div class="mono" style="font-size:16px;color:var(--accent)">${data.efficiency.lipe}</div></div>
        <div><span class="eyebrow">BEI</span><div class="mono" style="font-size:16px;color:var(--accent)">${data.efficiency.bei}</div></div>
        <div><span class="eyebrow">估算 pKd</span><div class="mono" style="font-size:16px;color:var(--accent)">${data.efficiency.p_kd}</div></div>
      </div>` : "";
    const alerts = d.alerts.length
      ? `<div class="note warn" style="margin:0 18px 14px"><span>结构警示：${d.alerts.map(esc).join("、")}；建议结合实验 Ames / hERG 数据判断。</span></div>`
      : "";
    return `
      <div class="opt-hero">
        <div><span class="eyebrow">ADMET · 类药性</span><h2>${esc(data.name)}</h2><p>${esc(p.formula)} · ${esc(data.smiles)}</p></div>
        ${quick.map(([k, v, cls]) => `<div class="opt-metric ${cls}"><strong>${v}</strong><span>${k}</span></div>`).join("")}
      </div>
      ${eff}
      <div class="count-grid" style="grid-template-columns:repeat(4,1fr);border-bottom:1px solid var(--line)">${props}</div>
      <div class="table-scroll" style="padding:4px 18px 10px">
        <table class="result-table">
          <thead><tr><th>端点</th><th>预测值</th><th>质量分</th><th>评价</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      ${alerts}
      <div class="note" style="margin:0 18px 16px"><span>${esc(data.method)}。结果用于排序与趋势判断，不替代实验测定。</span></div>`;
  }

  async function openAdmetModal(runId, index) {
    openModal("ADMET 与类药性", "基于描述符规则的快速预测", `<div class="note">正在计算分子描述符 ...</div>`);
    try {
      const data = await api(`/api/runs/${encodeURIComponent(runId)}/result/${index}/admet`);
      openModal(
        "ADMET 与类药性",
        `${data.name} · 12 个端点预测`,
        admetBlocks(data),
        `<a class="btn primary sm" href="/api/runs/${encodeURIComponent(runId)}/result/${index}/admet.pdf" target="_blank" rel="noopener">下载 ADMET PDF</a>
         <button class="btn ghost sm" data-action="close-modal">关闭</button>`
      );
    } catch (error) {
      closeModal();
      toast(error.message, "error", 7000);
    }
  }

  async function openPharmacophoreModal(runId, index) {
    openModal("药效团分析", "几何特征与残基互补性", `<div class="note">正在识别药效团特征 ...</div>`);
    try {
      const data = await api(`/api/runs/${encodeURIComponent(runId)}/result/${index}/pharmacophore`);
      const rows = data.features.map((item, i) => {
        const contact = item.contacts[0];
        const residue = contact ? `${contact.residue} ${contact.chain}${contact.resseq}` : "无接触";
        return `<tr>
          <td class="mono">${i + 1}</td>
          <td><span class="badge" style="border-color:${item.color};color:${item.color}">${esc(item.label)}</span></td>
          <td class="mono">${item.atoms.slice(0, 5).map((a) => a + 1).join(", ")}</td>
          <td>${esc(residue)}</td>
          <td class="mono">${item.min_distance != null ? item.min_distance.toFixed(2) : "—"}</td>
          <td>${item.matched ? `<span class="good">✓ ${esc(contact?.complement || "互补")}</span>` : `<span class="bad">未满足</span>`}</td>
        </tr>`;
      }).join("");
      const body = `
        <div class="opt-hero">
          <div><span class="eyebrow">药效团</span><h2>${data.total} 个特征</h2><p>互补覆盖 ${data.coverage}% · 未满足 ${data.unsatisfied.length} 个</p></div>
          <div class="opt-metric"><strong>${data.matched}</strong><span>已满足</span></div>
          <div class="opt-metric"><strong>${data.coverage}%</strong><span>覆盖率</span></div>
          <div class="opt-metric"><strong>${data.counts["疏水中心"] || 0}</strong><span>疏水中心</span></div>
        </div>
        <div class="table-scroll" style="padding:14px 18px">
          <table class="result-table">
            <thead><tr><th>#</th><th>特征</th><th>配体原子</th><th>最近残基</th><th>距离 (Å)</th><th>互补性</th></tr></thead>
            <tbody>${rows || `<tr><td colspan="6" class="muted">未识别到药效团特征</td></tr>`}</tbody>
          </table>
        </div>
        ${data.unsatisfied.length ? `<div class="note warn" style="margin:0 18px 14px"><span>未满足：${data.unsatisfied.slice(0, 6).map((item) => `${esc(item.label)}(${esc(item.residue)})`).join("、")}。可在分子设计中针对这些方向补充极性或疏水基团。</span></div>` : ""}`;
      openModal(
        "药效团分析",
        `${data.total} 个特征 · 覆盖 ${data.coverage}%`,
        body,
        `<a class="btn primary sm" href="/api/runs/${encodeURIComponent(runId)}/result/${index}/pharmacophore.pdf" target="_blank" rel="noopener">下载药效团 PDF</a>
         <button class="btn ghost sm" data-action="close-modal">关闭</button>`
      );
    } catch (error) {
      closeModal();
      toast(error.message, "error", 7000);
    }
  }

  async function openMmgbsaModal(runId, index) {
    openModal("结合自由能分解", "Prime 风格残基贡献", `<div class="note">正在计算残基能量分解 ...</div>`);
    try {
      const data = await api(`/api/runs/${encodeURIComponent(runId)}/result/${index}/mmgbsa`);
      const terms = [["范德华", data.terms.vdw], ["静电", data.terms.elec], ["去溶剂化", data.terms.desolv], ["氢键", data.terms.hbond]]
        .map(([k, v]) => `<div class="opt-metric ${v < 0 ? "good" : ""}" style="border-left-color:var(--line)"><strong>${Number(v).toFixed(1)}</strong><span>${k}</span></div>`).join("");
      const rows = data.top.slice(0, 12).map((item) => `
        <tr>
          <td>${esc(item.residue)}${item.resseq} <span class="muted">${esc(item.chain)}</span></td>
          <td class="mono">${item.vdw.toFixed(1)}</td>
          <td class="mono">${item.elec.toFixed(1)}</td>
          <td class="mono">${item.desolv.toFixed(1)}</td>
          <td class="mono">${item.hbond.toFixed(1)}</td>
          <td class="mono ${item.total < 0 ? "good" : ""}"><strong>${item.total.toFixed(1)}</strong></td>
          <td class="mono">${item.min_distance.toFixed(2)}</td>
        </tr>`).join("");
      const body = `
        <div class="opt-hero">
          <div><span class="eyebrow">MM-GBSA 近似</span><h2>${Number(data.total).toFixed(1)} kcal/mol</h2><p>${data.residues.length} 个接触残基 · 埋藏 ${Math.round(data.buried_fraction * 100)}%</p></div>
          <div class="opt-metric"><strong>${data.vina != null ? Number(data.vina).toFixed(2) : "—"}</strong><span>Vina 打分</span></div>
          <div class="opt-metric"><strong>${data.ligand_atoms}</strong><span>配体重原子</span></div>
        </div>
        <div class="count-grid" style="grid-template-columns:repeat(4,1fr);border-bottom:1px solid var(--line)">${terms}</div>
        <div class="table-scroll" style="padding:14px 18px">
          <table class="result-table">
            <thead><tr><th>残基</th><th>范德华</th><th>静电</th><th>去溶剂化</th><th>氢键</th><th>合计</th><th>最近 (Å)</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        <div class="note" style="margin:0 18px 16px"><span>${esc(data.note)} 方法：${esc(data.method)}。</span></div>`;
      openModal(
        "结合自由能分解",
        `估算 ΔG ${Number(data.total).toFixed(1)} kcal/mol`,
        body,
        `<a class="btn primary sm" href="/api/runs/${encodeURIComponent(runId)}/result/${index}/mmgbsa.pdf" target="_blank" rel="noopener">下载自由能分解 PDF</a>
         <button class="btn ghost sm" data-action="close-modal">关闭</button>`
      );
    } catch (error) {
      closeModal();
      toast(error.message, "error", 7000);
    }
  }

  async function runAdmetCalculator() {
    const input = $("#admet-smiles");
    const output = $("#admet-calc-result");
    if (!input || !output) return;
    const smiles = input.value.trim();
    if (!smiles) return toast("请输入 SMILES", "warn");
    output.innerHTML = `<div class="note">正在计算 ...</div>`;
    try {
      const data = await api(`/api/admet?smiles=${encodeURIComponent(smiles)}`);
      output.innerHTML = `
        <div style="margin-top:12px;border-top:1px solid var(--line);padding-top:12px">
          ${admetBlocks(data)}
          <div class="row" style="padding:0 18px 8px">
            <a class="btn primary sm" href="/api/admet.pdf?smiles=${encodeURIComponent(smiles)}&name=${encodeURIComponent(data.name)}" target="_blank" rel="noopener">下载 ADMET PDF</a>
          </div>
        </div>`;
    } catch (error) {
      output.innerHTML = `<div class="note warn">${esc(error.message)}</div>`;
    }
  }

  /* ------------------------------------------------------------------ modals */
  function openModal(title, subtitle, body, footer = "") {
    const root = $("#modal-root");
    root.innerHTML = `
      <div class="modal-overlay">
        <div class="modal" role="dialog" aria-modal="true">
          <div class="modal-head">
            <div><h2>${esc(title)}</h2>${subtitle ? `<p>${esc(subtitle)}</p>` : ""}</div>
            <button class="modal-close" data-action="close-modal" aria-label="关闭">×</button>
          </div>
          <div class="modal-body">${body}</div>
          ${footer ? `<div class="modal-foot">${footer}</div>` : ""}
        </div>
      </div>`;
    root.querySelector(".modal-overlay").addEventListener("click", (event) => {
      if (event.target === event.currentTarget) closeModal();
    });
    S.modalOpen = true;
  }

  function closeModal() {
    $("#modal-root").innerHTML = "";
    S.modalOpen = false;
  }

  function renderProjectQc(qc) {
    const issues = qc.issues || [];
    const rows = issues.length
      ? issues.map((item) => {
          const cls = item.level === "error" ? "err" : item.level === "warn" ? "warn" : "";
          return `<div class="note ${cls}" style="margin:6px 0">${esc(item.text)}</div>`;
        }).join("")
      : `<div class="note" style="margin:6px 0">未发现明显问题。</div>`;
    const statuses = Object.entries(qc.ligands?.by_status || {})
      .map(([key, value]) => `${esc(key)} ${value}`)
      .join(" / ") || "无";
    return `<div class="note" style="margin-top:10px">
      <strong>项目体检</strong>：受体 ${qc.receptors?.ready || 0}/${qc.receptors?.total || 0} 就绪；
      配体 ${qc.ligands?.total || 0} 个（${statuses}）；
      对接记录 ${qc.runs?.total || 0} 次（失败 ${qc.runs?.failed || 0}，中断 ${qc.runs?.interrupted || 0}）。
    </div>${rows}
    <div class="muted" style="font-size:10px;margin-top:6px">日志目录：${esc(qc.logs_dir || "")}</div>`;
  }

  function openToolsModal() {
    const d = S.data;
    const system = d.system || {};
    const recent = d.recent || [];
    const recentHtml = recent.length
      ? recent.map((item) => `<div class="recent-item"><span class="path">${esc(item.path)}</span><button data-action="open-recent" data-path="${esc(item.path)}">打开</button></div>`).join("")
      : `<span class="muted" style="font-size:10.5px">暂无最近项目</span>`;
    const body = `
      <div class="tool-section">
        <h4>项目</h4>
        <p>当前项目：<span class="mono" style="color:var(--text-soft)">${esc(d.path)}</span></p>
        <div class="row">
          <button class="btn primary sm" data-action="new-project">新建项目</button>
          <button class="btn ghost sm" data-action="open-project">打开已有项目</button>
          <button class="btn ghost sm" data-action="project-reveal">打开项目文件夹</button>
          <button class="btn ghost sm" data-action="project-qc">项目体检</button>
          <button class="btn ghost sm" data-action="project-backup">备份项目</button>
        </div>
        <div id="project-qc-result"></div>
      </div>
      <div class="tool-section">
        <h4>参考文献</h4>
        <p>报告中引用的方法与数据库文献，可直接导入 EndNote / Zotero / NoteExpress。</p>
        <div class="row">
          <a class="btn ghost sm" href="/api/references.bib" download>导出 BibTeX</a>
          <a class="btn ghost sm" href="/api/references.ris" download>导出 RIS</a>
        </div>
      </div>
      <div class="tool-section">
        <h4>R 基团化合物库</h4>
        <p>输入带 R / [R] 占位符的 SMILES，按取代基列表批量生成类似物。结果可直接导入为配体。</p>
        <div class="field" style="margin-bottom:9px"><label>骨架 SMILES</label><input class="input mono" id="enum-smiles" placeholder="例如 c1ccc(cc1)R"></div>
        <div class="field" style="margin-bottom:9px"><label>取代基（逗号或换行分隔）</label><textarea class="textarea" id="enum-subs" style="min-height:64px">C, O, N, F, Cl, C(F)(F)F, C#N, OC</textarea></div>
        <div class="row"><button class="btn sm" data-action="enumerate">生成衍生物</button><button class="btn ghost sm" data-action="import-enum">全部导入为配体</button><span class="muted" id="enum-count" style="font-size:10px"></span></div>
        <div class="enum-list" id="enum-list"></div>
      </div>
      <div class="tool-section">
        <h4>分子性质 / ADMET 计算器</h4>
        <p>输入任意 SMILES，快速获得理化性质、类药规则、ADMET 端点与配体效率，并可导出 PDF 图。</p>
        <div class="field" style="margin-bottom:9px"><label>SMILES</label><input class="input mono" id="admet-smiles" placeholder="例如 CC(=O)Oc1ccccc1C(=O)O"></div>
        <div class="row"><button class="btn sm" data-action="admet-calc">计算性质与 ADMET</button><span class="muted" style="font-size:10px">基于 RDKit 描述符与公开经验模型</span></div>
        <div id="admet-calc-result"></div>
      </div>
      <div class="tool-section">
        <h4>批量工具箱</h4>
        <p>结构标准化、配体描述符导出与多样性挑选。</p>
        <div class="row" style="flex-wrap:wrap;margin-bottom:9px">
          <input type="file" id="tool-standardize-file" accept=".sdf,.sd,.mol,.smi,.txt,.csv,.tsv">
          <button class="btn sm" data-action="tool-standardize">标准化并下载 SDF</button>
        </div>
        <div class="row" style="flex-wrap:wrap;margin-bottom:9px">
          <input class="input" id="tool-diversity-count" type="number" min="1" max="200" value="12" style="max-width:90px">
          <button class="btn ghost sm" data-action="tool-diversity">挑选多样配体</button>
          <a class="btn ghost sm" href="/api/tools/ligands.csv">导出描述符 CSV</a>
        </div>
      </div>
      <div class="tool-section">
        <h4>系统信息</h4>
        <dl class="sys-grid">
          <dt>Vina</dt><dd>${esc(system.vina_version || "未找到")}</dd>
          <dt>GPU 引擎</dt><dd>${esc(system.gpu_available ? system.gpu_path : "未检测到 Vina-GPU")}</dd>
          <dt>Vina 路径</dt><dd>${esc(system.vina_path || "—")}</dd>
          <dt>Python</dt><dd>${esc(system.python || "—")}</dd>
          <dt>RDKit</dt><dd>${esc(system.rdkit || "—")}</dd>
          <dt>Meeko</dt><dd>${esc(system.meeko || "—")}</dd>
          <dt>项目根目录</dt><dd>${esc(system.projects_dir || "—")}</dd>
        </dl>
      </div>
      <div class="tool-section">
        <h4>最近项目</h4>
        <div class="recent-list">${recentHtml}</div>
      </div>
      <div class="tool-section">
        <h4>快速上手</h4>
        <p style="margin-bottom:0">① 受体：输入 PDB ID 或上传结构，点击准备。② 配体：上传 SDF/MOL2 或粘贴 SMILES，点击准备。③ 位点：使用共晶配体、预测口袋或手动盒子。④ 对接：勾选结构并开始。⑤ 结果：查看打分/构象，运行作用力分析与 RMSD，导出 CSV 或复合物。</p>
      </div>
      <div class="tool-section">
        <h4>说明</h4>
        <p style="margin-bottom:0">DrugHX 调用 AutoDock Vina 1.2.5 完成对接。口袋预测是快速启发式算法，建议结合共晶配体、文献或 fpocket 结果确认位点；对接打分用于优先级排序，不等同于实验结合自由能。</p>
      </div>`;
    openModal("项目与工具", "项目管理、化合物库生成与系统状态", body);
  }

  function openNewProjectModal() {
    openModal("新建项目", "会在项目根目录下创建独立文件夹，所有结构、构象和日志都会保存其中。",
      `<div class="field"><label>项目名称</label><input class="input" id="new-project-name" placeholder="例如 EGFR_抑制剂筛选"></div>`,
      `<button class="btn ghost" data-action="close-modal">取消</button><button class="btn primary" data-action="create-project">创建项目</button>`);
  }

  function openProjectModal() {
    openModal("打开项目", "输入包含 project.json 的项目目录。",
      `<div class="field"><label>项目目录</label><input class="input mono" id="open-project-path" placeholder="C:\\Users\\...\\DrugHXProjects\\My_Project"></div>`,
      `<button class="btn ghost" data-action="close-modal">取消</button><button class="btn primary" data-action="open-project-path">打开</button>`);
  }

  async function openProject(path) {
    try {
      await api("/api/project/open", { method: "POST", body: { path } });
      closeModal();
      resetSelections();
      S.step = "receptor";
      await refresh();
      toast("项目已打开", "success");
    } catch (error) {
      toast(error.message, "error", 6000);
    }
  }

  function resetSelections() {
    S.receptorId = null;
    S.ligandId = null;
    S.ligandCache = {};
    S.runId = null;
    S.resultIndex = 0;
    S.mode = 1;
    S.analysis = null;
    S.analysisKey = "";
    S.rmsd = null;
    S.rmsdKey = "";
    S.fishingLigandId = null;
    S.fishingName = "";
    S.fishingSmiles = "";
    S.fishingTouched = false;
    S.selectedReceptors = new Set();
    S.selectedLigands = new Set();
    S.knownReceptors = new Set();
    S.knownLigands = new Set();
    S.selectionInit = false;
    S.autoSelect = true;
    S.recOptions = null;
    S.ligOptions = null;
    S.optResult = null;
    S.designResult = null;
    S.designReceptorId = null;
  }

  async function openFigureStudio() {
    const run = activeRun();
    const result = activeResult();
    if (!run || !result || result.status !== "ok") return toast("请先选择一个成功的结果", "warn");
    const body = `
      <div class="note" style="margin-bottom:16px">
        <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M12 8h.01M11 12h1v5h1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
        <span>所有图均以 PDF 矢量图输出：2D 结构、作用力连线、打分排序与热图。可直接用于投稿排版，也可在 Illustrator/Inkscape 中编辑。</span>
      </div>
      <div class="card" style="box-shadow:none;margin-bottom:12px">
        <div class="card-head" style="padding-bottom:8px"><div><h3 style="font-size:13px">当前构象</h3><p>${esc(result.ligand_name)} · 构象 ${S.mode} · ${esc(result.receptor_name)}</p></div></div>
        <div class="card-body" style="padding-top:8px">
          <div class="parent-strip">
            <span>Vina <strong>${result.affinity != null ? Number(result.affinity).toFixed(2) : "—"}</strong> kcal/mol</span>
            <span>Vinardo <strong>${result.vinardo != null ? Number(result.vinardo).toFixed(2) : "—"}</strong></span>
            <span>引擎 <strong>${esc(result.engine || "cpu")}</strong></span>
          </div>
        </div>
      </div>
      <div class="row" style="flex-wrap:wrap">
        <a class="btn primary sm" href="/api/runs/${encodeURIComponent(run.id)}/result/${S.resultIndex}/figures.pdf?mode=${S.mode}" target="_blank" rel="noopener">打开当前构象 PDF</a>
        <a class="btn sm" href="/api/runs/${encodeURIComponent(run.id)}/figures" target="_blank" rel="noopener">下载完整论文图表包</a>
      </div>
      <div class="note" style="margin-top:14px">
        <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M12 8h.01M11 12h1v5h1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
        <span>当前构象 PDF 包含：第 1 页 2D 结构与分子性质，第 2 页蛋白–配体作用力示意图。完整图表包包含打分排序、作用力热图、双面板汇总图和 PyMOL 场景。</span>
      </div>`;
    openModal(
      "论文级矢量图 · PDF",
      `${result.ligand_name} · 构象 ${S.mode} · ${result.receptor_name}`,
      body,
      `<button class="btn ghost" data-action="close-modal">关闭</button>
       <a class="btn primary" href="/api/runs/${encodeURIComponent(run.id)}/result/${S.resultIndex}/figures.pdf?mode=${S.mode}" target="_blank" rel="noopener">打开 PDF</a>`,
    );
  }

  async function downloadPublicationPng() {
    const run = activeRun();
    const result = activeResult();
    if (!run || !result || result.status !== "ok") return toast("没有可用的构象", "warn");
    const dpi = Number($("#fig-dpi")?.value || 300);
    const showSurface = Boolean($("#fig-surface")?.checked);
    const showLabels = Boolean($("#fig-labels")?.checked);
    const key = `${run.id}:${S.resultIndex}:${S.mode}`;
    let analysis = S.analysisKey === key ? S.analysis : null;
    if (!analysis) {
      try {
        analysis = await api(`/api/runs/${run.id}/result/${S.resultIndex}/analyze`, {
          method: "POST",
          body: { mode: S.mode },
        });
        S.analysis = analysis;
        S.analysisKey = key;
      } catch (error) {
        return toast(`作用力信息不可用：${error.message}`, "warn", 5000);
      }
    }

    const width = Math.round(8 * dpi);
    const height = Math.round(6 * dpi);
    const host = document.createElement("div");
    host.style.cssText = `position:fixed;left:-100000px;top:0;width:${width}px;height:${height}px;background:#fff;`;
    document.body.appendChild(host);
    let pub = null;
    try {
      toast(`正在渲染 ${dpi} dpi 图像 ...`, "info", 3000);
      const [pdb, pose] = await Promise.all([
        fetchText(`/api/runs/${run.id}/result/${S.resultIndex}/receptor.pdb`),
        fetchText(`/api/runs/${run.id}/result/${S.resultIndex}/pose?mode=${S.mode}`),
      ]);
      pub = window.$3Dmol.createViewer(host, {
        backgroundColor: "white",
        width,
        height,
        antialias: true,
      });
      pub.setBackgroundColor("#ffffff");
      pub.addModel(pdb, "pdb");
      pub.setStyle({ model: 0 }, { cartoon: { color: "#94a3b8", opacity: 0.92 } });
      pub.addModel(pose, "pdbqt");
      const ligandScheme = {
        carbon: "#0f766e",
        oxygen: "#dc2626",
        nitrogen: "#1d4ed8",
        sulfur: "#b45309",
        hydrogen: "#f8fafc",
        fluorine: "#65a30d",
        chlorine: "#15803d",
        bromine: "#92400e",
      };
      pub.setStyle({ model: 1 }, { stick: { colorscheme: ligandScheme, radius: 0.2 } });
      if (showSurface) {
        pub.addSurface(window.$3Dmol.SurfaceType.VDW, { opacity: 0.1, color: "#94a3b8" }, { model: 0 });
      }

      const residues = analysis?.residues || [];
      if (residues.length) {
        const resi = residues.map((r) => r.resseq);
        pub.setStyle(
          { model: 0, resi },
          {
            cartoon: { color: "#cbd5e1", opacity: 0.35 },
            stick: { colorscheme: { carbon: "#94a3b8", oxygen: "#dc2626", nitrogen: "#1d4ed8", sulfur: "#b45309" }, radius: 0.14 },
          },
        );
      }
      if (showLabels && residues.length) {
        const ca = parseCaCoords(pdb);
        residues.slice(0, 14).forEach((item) => {
          const coord = ca[`${item.chain}:${item.resseq}`];
          if (!coord) return;
          pub.addLabel(`${item.residue}${item.resseq}`, {
            position: coord,
            fontSize: Math.max(10, Math.round(dpi / 34)),
            fontColor: "#0f172a",
            backgroundColor: "#ffffff",
            backgroundOpacity: 0.9,
            borderThickness: 1,
            borderColor: "#cbd5e1",
            inFront: true,
          });
        });
      }
      const line = (item, color) => {
        if (!item.protein_coord || !item.ligand_coord) return;
        pub.addLine({
          start: { x: item.protein_coord[0], y: item.protein_coord[1], z: item.protein_coord[2] },
          end: { x: item.ligand_coord[0], y: item.ligand_coord[1], z: item.ligand_coord[2] },
          color,
          dashed: true,
          linewidth: Math.max(1, dpi / 220),
        });
      };
      (analysis?.hbonds || []).forEach((item) => line(item, "#b45309"));
      (analysis?.salt_bridges || []).forEach((item) => line(item, "#b91c1c"));

      const bounds = poseBounds(pose);
      if (bounds) {
        const center = [
          (bounds.min[0] + bounds.max[0]) / 2,
          (bounds.min[1] + bounds.max[1]) / 2,
          (bounds.min[2] + bounds.max[2]) / 2,
        ];
        const size = [
          bounds.max[0] - bounds.min[0] + 16,
          bounds.max[1] - bounds.min[1] + 16,
          bounds.max[2] - bounds.min[2] + 16,
        ];
        pub.addModel(frameXyz(center, size), "xyz");
        pub.setStyle({ model: 2 }, { sphere: { radius: 0.01, opacity: 0 } });
        pub.zoomTo({ model: 2 });
      } else {
        pub.zoomTo({ model: 1 });
      }
      pub.render();
      const uri = pub.pngURI();
      if (!uri) throw new Error("渲染结果为空");
      const a = document.createElement("a");
      a.href = uri;
      a.download = `DrugHX_${safeFile(result.ligand_name)}_mode${S.mode}_${dpi}dpi.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      toast("高分辨率图像已生成", "success");
    } catch (error) {
      toast(`图像生成失败：${error.message}`, "error", 7000);
    } finally {
      try { if (pub) pub.clear(); } catch (_) { /* ignore */ }
      host.remove();
    }
  }

  function frameXyz(center, size) {
    const [cx, cy, cz] = center;
    const [sx, sy, sz] = size;
    const corners = [];
    [cx - sx / 2, cx + sx / 2].forEach((x) =>
      [cy - sy / 2, cy + sy / 2].forEach((y) =>
        [cz - sz / 2, cz + sz / 2].forEach((z) => corners.push({ x, y, z })),
      ),
    );
    return `${corners.length}\nframe\n${corners
      .map((p) => `${p.x.toFixed(3)} ${p.y.toFixed(3)} ${p.z.toFixed(3)}`)
      .join("\n")}`;
  }

  function parseCaCoords(pdb) {
    const map = {};
    pdb.splitlines().forEach((line) => {
      if (!line.startsWith("ATOM") || line.slice(12, 16).trim() !== "CA") return;
      const chain = line[21] || "";
      const resseq = Number(line.slice(22, 26));
      if (!Number.isFinite(resseq)) return;
      map[`${chain}:${resseq}`] = {
        x: Number(line.slice(30, 38)),
        y: Number(line.slice(38, 46)),
        z: Number(line.slice(46, 54)),
      };
    });
    return map;
  }

  function safeFile(name) {
    return String(name || "ligand").replace(/[^\w\u4e00-\u9fa5.-]+/g, "_");
  }

  function showReceptorInfo(receptorId) {
    const rec = S.data.receptors.find((r) => r.id === receptorId);
    if (!rec) return;
    const m = rec.meta || {};
    const box = rec.box;
    const hetero = rec.hetero || [];
    const rows = [
      ["名称", rec.name],
      ["PDB ID", rec.pdb_id || "—"],
      ["文件标题", m.title || "—"],
      ["分类", m.header || "—"],
      ["分辨率", m.resolution ? `${Number(m.resolution).toFixed(2)} Å` : "—"],
      ["链", (m.chains || []).join(", ") || "—"],
      ["残基 / 原子", `${m.residues ?? "—"} / ${m.atoms ?? "—"}`],
      ["PDBQT 原子", m.pdbqt_atoms ?? "—"],
      ["保留离子", m.ions ?? 0],
      ["准备状态", (STATUS[rec.status] || [rec.status])[0]],
      ["对接盒", box ? `${box.label || "已设置"} · ${box.size.map((v) => Number(v).toFixed(1)).join(" × ")} Å` : "未设置"],
    ];
    const heteroHtml = hetero.length
      ? `<div style="margin-top:14px"><div class="eyebrow" style="margin-bottom:7px">共晶配体</div>${hetero.map((h) => `<div class="recent-item"><span>${esc(h.name)} <span class="mono">${esc(h.key)}</span> · ${h.heavy} 重原子</span></div>`).join("")}</div>`
      : "";
    openModal(
      `受体信息 · ${rec.name}`,
      "来自 PDB 文件头与准备结果",
      `<dl class="sys-grid">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>${heteroHtml}
       <div class="row" style="margin-top:18px"><a class="btn ghost sm" href="/api/receptors/${rec.id}/original" download>下载原始结构</a></div>`,
    );
  }

  async function showLogModal(runId, index) {
    try {
      const text = await fetchText(`/api/runs/${runId}/result/${index}/log`);
      openModal("Vina 日志", "完整标准输出与随机种子信息",
        `<pre class="console-body" style="max-height:55vh;background:#050a12;border:1px solid var(--line);border-radius:9px;padding:12px">${esc(text)}</pre>`);
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function enumerateLibrary() {
    const smiles = $("#enum-smiles")?.value.trim();
    const subs = ($("#enum-subs")?.value || "").split(/[,\n;]+/).map((s) => s.trim()).filter(Boolean);
    if (!smiles) return toast("请输入骨架 SMILES", "warn");
    try {
      const response = await api("/api/tools/enumerate", { method: "POST", body: { smiles, substituents: subs, limit: 300 } });
      S.enumResults = response.molecules || [];
      $("#enum-count").textContent = `生成 ${response.count} 个`;
      $("#enum-list").innerHTML = S.enumResults.map((m) => `<div class="enum-item"><span>${esc(m.name)}</span><span>${esc(m.smiles)}</span></div>`).join("");
      if (!response.count) toast("没有生成有效分子，请检查占位符与取代基", "warn");
    } catch (error) {
      toast(error.message, "error", 6000);
    }
  }

  /* ------------------------------------------------------------------ misc */
  window.addEventListener("resize", () => viewer && viewer.resize());
})();
