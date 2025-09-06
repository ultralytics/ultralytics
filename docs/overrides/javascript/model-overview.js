// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

(function () {
  let DATA = [];
  const TASK_ORDER = [
    "Detect",
    "Segment",
    "Classify",
    "Pose",
    "OBB",
    "Open Vocabulary",
  ];

  // Function to load model data from JSON
  async function loadModelData() {
    try {
      const response = await fetch("/assets/model_data.json");
      const modelData = await response.json();

      DATA = Object.entries(modelData)
        .filter(([, info]) => info.show_card)
        .map(([name, info]) => ({
          id: name
            .toLowerCase()
            .replace(/[-\s]/g, "-")
            .replace(/[^a-z0-9-]/g, ""),
          name,
          org: info.org,
          docs: info.docs,
          latest: info.latest,
          official: info.official,
          tasks: info.tasks,
          performance: Object.fromEntries(
            Object.entries(info.performance).map(([size, perf]) => [
              size,
              {
                map: perf.map,
                t4: perf.t4,
                params: perf.params,
                file_size: perf.file_size,
                model_full_name: perf.model_full_name,
              },
            ]),
          ),
          weightBase: info.weightBase,
          shortDescription: info.short_description,
        }));

      return true;
    } catch (error) {
      console.error("Error loading model data:", error);
      return false;
    }
  }

  async function initModelOverview() {
    // Prevent duplicate initialization
    if (document.getElementById("mo-content").dataset.moReady) return;
    document.getElementById("mo-content").dataset.moReady = "true";

    // Load the data first
    const dataLoaded = await loadModelData();
    if (!dataLoaded) {
      const container = document.getElementById("mo-content");
      if (container) {
        container.innerHTML =
          "<p>Error loading model data. Please try refreshing the page.</p>";
      }
      return;
    }

    let activeTask = "all";
    let query = "";

    const $ = (sel, el = document) => el.querySelector(sel);
    const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));

    const container = document.getElementById("mo-content");
    const searchInput = document.getElementById("mo-search");

    function groupByTask(task) {
      const groups = Object.fromEntries(TASK_ORDER.map((t) => [t, []]));

      DATA.filter(
        (m) => !query || m.name.toLowerCase().includes(query),
      ).forEach((m) => m.tasks.forEach((t) => groups[t]?.push(m)));

      if (task !== "all") {
        Object.keys(groups).forEach((k) => k !== task && delete groups[k]);
      }

      Object.keys(groups).forEach((k) => !groups[k].length && delete groups[k]);
      return groups;
    }

    function render() {
      const groups = groupByTask(activeTask);
      container.innerHTML = "";
      const visible = Object.values(groups).flat();
      if (!visible.length) {
        container.innerHTML = "<p>No models found.</p>";
        return;
      }

      const maxMap = Math.max(
        60,
        ...visible.flatMap((m) =>
          Object.values(m.performance).map((v) => v.map || 0),
        ),
      );
      const maxT4 = Math.max(
        16,
        ...visible.flatMap((m) =>
          Object.values(m.performance).map((v) => v.t4 || 0),
        ),
      );
      const modelRank = (m) =>
        m.official && m.latest ? 0 : m.official ? 1 : 2;

      Object.entries(groups).forEach(([task, models]) => {
        models.sort((a, b) => {
          const ra = modelRank(a),
            rb = modelRank(b);
          return ra !== rb ? ra - rb : a.name.localeCompare(b.name);
        });

        const section = document.createElement("section");
        section.className = "mo-section";
        section.innerHTML = `<h2>${task} models</h2>`;

        const grid = document.createElement("div");
        grid.className = "mo-grid";
        models.forEach((model) =>
          grid.appendChild(renderCard(model, { maxMap, maxT4, task })),
        );

        section.appendChild(grid);
        container.appendChild(section);
      });
    }

    function brandIcon(model) {
      return model.official
        ? '<img class="mo-brand" src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics-logomark-color.png" alt="Ultralytics" width="24" height="24" />'
        : "";
    }

    function getModelWeight(model, size) {
      return model.weightBase
        ? `${model.weightBase}${size}.pt`
        : `${model.name.toLowerCase().replace(/[-]/g, "")}${size}.pt`;
    }

    function getModelClass(model) {
      if (model.name.includes("YOLO-World"))
        return {
          class: "YOLOWorld",
          import: "from ultralytics import YOLOWorld",
        };
      if (model.name.includes("RT-DETR"))
        return { class: "RTDETR", import: "from ultralytics import RTDETR" };
      return { class: "YOLO", import: "from ultralytics import YOLO" };
    }

    function renderCard(model, scales) {
      const card = document.createElement("article");
      card.className = "mo-card";
      const sizes = Object.keys(model.performance);
      const defaultSize = sizes.includes("m") ? "m" : sizes[0];
      const titleHTML = `<div class="mo-title">${brandIcon(model)}<span>${model.name} <span class="mo-subtle">${scales.task}</span></span></div>`;
      card.innerHTML = `
        <div class="mo-card__body">
          <div class="mo-card__title">
            ${titleHTML}
            ${model.latest ? '<span class="mo-badge mo-badge--latest">Latest</span>' : ""}
          </div>
          <div>
            <div class="mo-subtle">Choose size</div>
            <div class="mo-size-row" role="tablist" aria-label="Choose size">
              ${sizes.map((s) => `<button class=\"mo-size ${s === defaultSize ? "is-active" : ""}\" data-size=\"${s}\"><span class=\"mo-size-label\">${s.toUpperCase()}</span><span class=\"mo-size-sub\">${model.performance[s].params || ""}${model.performance[s].params ? "M" : ""}</span></button>`).join("")}
            </div>
          </div>
          <div class="mo-bars">
            <div class="mo-bar" data-kind="map">
              <div class="mo-subtle">mAP</div>
              <div class="mo-bar__track"><div class="mo-bar__fill" style="width:0"></div></div>
              <div class="mo-subtle mo-val"></div>
            </div>
            <div class="mo-bar" data-kind="speed">
              <div class="mo-subtle">Speed</div>
              <div class="mo-bar__track"><div class="mo-bar__fill" style="width:0"></div></div>
              <div class="mo-subtle mo-val"></div>
            </div>
          </div>
          <div class="mo-actions">
            <div class="mo-meta"><span title="Organization">${model.org}</span></div>
            <button class="md-button md-button--primary mo-quick" type="button" data-id="${model.id}" data-task="${scales.task}">Quick view</button>
          </div>
        </div>`;
      function update(size) {
        const perf = model.performance[size];
        const map = perf.map || 0;
        const t4 = perf.t4 || 0;
        const mapPct = Math.min(100, (map / scales.maxMap) * 100);
        let speedPct = Math.min(
          100,
          Math.max(0, (1 - t4 / scales.maxT4) * 100),
        );
        if (t4 > 0 && speedPct < 5) {
          speedPct = 5;
        }

        const bars = $$(".mo-bar", card);
        bars[0].querySelector(".mo-bar__fill").style.width =
          mapPct.toFixed(1) + "%";
        bars[0].querySelector(".mo-val").textContent = perf.map ?? "—";
        bars[1].querySelector(".mo-bar__fill").style.width =
          speedPct.toFixed(1) + "%";
        bars[1].querySelector(".mo-val").textContent = perf.t4
          ? perf.t4 + " ms (T4)"
          : "—";
        $$(".mo-size", card).forEach((b) =>
          b.classList.toggle("is-active", b.dataset.size === size),
        );
      }
      $$(".mo-size", card).forEach((b) =>
        b.addEventListener("click", () => update(b.dataset.size)),
      );
      update(defaultSize);
      return card;
    }

    $$(".mo-chip").forEach((btn) =>
      btn.addEventListener("click", () => {
        $$(".mo-chip").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        activeTask = btn.dataset.task;
        render();
      }),
    );
    searchInput.addEventListener("input", (e) => {
      query = (e.target.value || "").trim().toLowerCase();
      render();
    });

    const backdrop = document.getElementById("mo-modal-backdrop");
    const modalTitle = document.getElementById("mo-modal-title");
    const tabButtons = ["overview", "quick", "deploy"].map((id) =>
      document.getElementById("mo-tab-" + id),
    );
    const panels = {
      overview: document.getElementById("mo-panel-overview"),
      quick: document.getElementById("mo-panel-quick"),
      deploy: document.getElementById("mo-panel-deploy"),
    };
    let lastFocus = null;
    let activeModel = null;
    let activeSize = null;

    function highlightInline(root) {
      root.querySelectorAll("pre.mo-code code").forEach((code) => {
        if (code.querySelector("span")) {
          return;
        }

        const text = code.textContent;
        code.innerHTML = "";
        let currentIndex = 0;
        const regex =
          /(\b(?:from|import|def|class|if|else|elif|for|while|try|except|finally|with|as|return|yield|break|continue|pass|raise|assert|global|nonlocal|lambda|and|or|not|in|is|True|False|None)\b|\b(?:YOLO|YOLOWorld|RTDETR)\b|\b(?:model|results)\s*=|'[^']*'|"[^"]*")/g;
        let match;
        while ((match = regex.exec(text)) !== null) {
          if (match.index > currentIndex) {
            code.appendChild(
              document.createTextNode(
                text.substring(currentIndex, match.index),
              ),
            );
          }

          // Create and append the span
          const span = document.createElement("span");
          let className = "";
          if (
            /\b(from|import|def|class|if|else|elif|for|while|try|except|finally|with|as|return|yield|break|continue|pass|raise|assert|global|nonlocal|lambda|and|or|not|in|is|True|False|None)\b/.test(
              match[0],
            )
          ) {
            className = "mo-token-kw";
          } else if (/\b(YOLO|YOLOWorld|RTDETR)\b/.test(match[0])) {
            className = "mo-token-fn";
          } else if (/\b(model|results)\s*=\s*/.test(match[0])) {
            className = "mo-token-kw";
            span.className = className;
            span.textContent = match[0].replace(/\s*=/, "");
            code.appendChild(span);
            code.appendChild(document.createTextNode("="));
            currentIndex = regex.lastIndex;
            continue;
          } else if (/'[^']*'/.test(match[0]) || /"[^"]*"/.test(match[0])) {
            className = "mo-token-str";
          }
          span.className = className;
          span.textContent = match[0];
          code.appendChild(span);
          currentIndex = regex.lastIndex;
        }
        if (currentIndex < text.length) {
          code.appendChild(
            document.createTextNode(text.substring(currentIndex)),
          );
        }
      });
    }

    function buildOverviewHTML(model, size) {
      const perf = model.performance[size];
      const chips = [...model.tasks]
        .map((t) => `<span class="mo-mini-chip">${t}</span>`)
        .join("");
      const fileSize = perf.file_size
        ? typeof perf.file_size === "number"
          ? `${perf.file_size} MB`
          : perf.file_size
        : "—";
      return `<h3>Key Features</h3><div class=\"mo-chipline\">${chips}<span class=\"mo-mini-chip\">Real-time</span><span class=\"mo-mini-chip\">ONNX Runtime</span></div>
      <h3>Model Architecture</h3>
      <div class=\"mo-infobar\">\n        <div class=\"mo-infobox\"><span>Model Size</span><strong>${perf.model_full_name || size.toUpperCase()} (${fileSize || "?"})</strong></div>\n        <div class=\"mo-infobox\"><span>Parameters</span><strong>${perf.params || "?"}M</strong></div>\n        <div class=\"mo-infobox\"><span>Input Size</span><strong>640x640</strong></div>\n      </div>`;
    }

    function buildQuickHTML(model, size) {
      const weight = getModelWeight(model, size);
      const { class: importClass, import: importStatement } =
        getModelClass(model);

      return `<h3>CLI</h3><pre class=\"mo-code\"><code class=\"lang-bash\">yolo detect predict model=${weight} source=path/to/images</code></pre>\n<h3>Python</h3><pre class=\"mo-code\"><code class=\"lang-python\">${importStatement}\nmodel = ${importClass}('${weight}')\nresults = model('image.jpg')</code></pre>`;
    }

    function buildDeployHTML(model, size) {
      const deployPlatforms = [
        {
          name: "Torch",
          icon: "pytorch.svg",
          format: "torch",
          desc: "Torch Native",
        },
        {
          name: "TorchScript",
          icon: "pytorch.svg",
          format: "torchscript",
          desc: "TorchScript format",
        },
        {
          name: "ONNX",
          icon: "onnx.svg",
          format: "onnx",
          desc: "Cross-platform inference",
        },
        {
          name: "OpenVINO",
          icon: "openvino.svg",
          format: "openvino",
          desc: "Intel optimization",
        },
        {
          name: "TensorRT",
          icon: "nvidia.svg",
          format: "engine",
          desc: "NVIDIA GPU acceleration",
        },
        {
          name: "CoreML",
          icon: "coreml.svg",
          format: "coreml",
          desc: "Apple devices",
        },
        {
          name: "Sony's IMX500",
          icon: "aitrios.svg",
          format: "imx",
          desc: "Sony's IMX500",
        },
        {
          name: "TensorFlow SavedModel",
          icon: "tensorflow.svg",
          format: "tf",
          desc: "TensorFlow SavedModel",
        },
        {
          name: "TensorFlow GraphDef",
          icon: "tensorflow.svg",
          format: "tf",
          desc: "TensorFlow GraphDef",
        },
        {
          name: "TensorFlow Lite",
          icon: "tensorflow-lite.svg",
          format: "tflite",
          desc: "Mobile & edge devices",
        },
        {
          name: "Edge TPU",
          icon: "tensorflow-edge-tpu.svg",
          format: "edgetpu",
          desc: "Google Coral devices",
        },
        {
          name: "TensorFlow.js",
          icon: "tensorflow-js.svg",
          format: "tfjs",
          desc: "Browser & Node.js",
        },
        {
          name: "PaddlePaddle",
          icon: "paddle-paddle.svg",
          format: "paddle",
          desc: "Baidu framework",
        },
        {
          name: "NCNN",
          icon: "ncnn.svg",
          format: "ncnn",
          desc: "Mobile optimization",
        },
      ];

      const weight = getModelWeight(model, size);
      const deployGrid = deployPlatforms
        .map(
          (platform, index) =>
            `\n        <div class="mo-deploy-item ${index === 0 ? "is-active" : ""}" title="${platform.desc}" data-format="${platform.format}" data-name="${platform.name}">\n          <img class="mo-deploy-icon" src="https://raw.githubusercontent.com/ultralytics/assets/main/mkdocs/logos/${platform.icon}" alt="${platform.name}" />\n          <div class="mo-deploy-content">\n            <div class="mo-deploy-name">${platform.name}</div>\n            <div class="mo-deploy-desc"></div>\n          </div>\n        </div>\n      `,
        )
        .join("");

      const initialCommand =
        deployPlatforms[0].format === "torch"
          ? ""
          : `yolo export model=${weight} format=${deployPlatforms[0].format}`;

      return `<h3>Deployment Platforms</h3>\n        <p>Choose your target platform for optimized deployment:</p>\n        <div class="mo-deploy-grid">${deployGrid}</div>\n        <h3 id="mo-deploy-format-title">Export Command (${deployPlatforms[0].name})</h3>\n        <p>Use the following command to export your model:</p>\n        <pre class="mo-code"><code class="lang-bash" id="mo-deploy-command">${initialCommand}</code></pre>`;
    }

    function setTab(tab) {
      tabButtons.forEach((btn) => {
        const sel = btn.dataset.tab === tab;
        btn.setAttribute("aria-selected", sel);
        document
          .getElementById("mo-panel-" + btn.dataset.tab)
          .classList.toggle("is-active", sel);
      });
      highlightInline(backdrop);
    }

    function openModal(model, size, taskName) {
      activeModel = model;
      activeSize = size;
      const taskLabel = taskName
        ? ` <span class=\"mo-subtle\">${taskName}</span>`
        : "";
      modalTitle.innerHTML = `${brandIcon(model)}<span>${model.name}${taskLabel}</span>`; // removed Latest badge in modal title
      const detailsLink = document.getElementById("mo-modal-details");
      if (detailsLink) {
        detailsLink.href = model.docs;
        detailsLink.setAttribute(
          "aria-label",
          "Full details for " + model.name,
        );
      }

      // Set the description
      const descriptionEl = document.getElementById("mo-modal-description");
      if (descriptionEl && model.shortDescription) {
        descriptionEl.textContent = model.shortDescription;
        descriptionEl.style.display = "block";
      } else if (descriptionEl) {
        descriptionEl.style.display = "none";
      }

      panels.overview.innerHTML = buildOverviewHTML(model, size);
      panels.quick.innerHTML = buildQuickHTML(model, size);
      panels.deploy.innerHTML = buildDeployHTML(model, size);
      // Store current model data for deploy platform interaction
      backdrop.setAttribute("data-current-model", model.name);
      backdrop.setAttribute("data-current-size", size);
      setTab("overview");
      backdrop.classList.add("is-open");
      backdrop.removeAttribute("aria-hidden");
      lastFocus = document.activeElement;
      backdrop.querySelector(".mo-modal-close").focus();
      document.addEventListener("keydown", escListener);
      highlightInline(backdrop);
    }
    function closeModal() {
      backdrop.classList.remove("is-open");
      backdrop.setAttribute("aria-hidden", "true");
      document.removeEventListener("keydown", escListener);
      if (lastFocus) lastFocus.focus();
    }
    function escListener(e) {
      if (e.key === "Escape") closeModal();
    }

    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) closeModal();
    });
    backdrop
      .querySelector(".mo-modal-close")
      .addEventListener("click", closeModal);
    backdrop.querySelectorAll(".mo-tab").forEach((btn) =>
      btn.addEventListener("click", () => {
        setTab(btn.dataset.tab);
      }),
    );

    document.getElementById("mo-content").addEventListener("click", (e) => {
      const btn = e.target.closest(".mo-quick");
      if (!btn) return;
      const id = btn.dataset.id;
      const model = DATA.find((m) => m.id === id);
      if (!model) return;
      const card = btn.closest(".mo-card");
      const active = card.querySelector(".mo-size.is-active");
      const size = active ? active.dataset.size : "n";
      const taskName =
        btn.dataset.task ||
        (activeTask !== "all" ? activeTask : model.tasks[0] || "");
      openModal(model, size, taskName);
    });

    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) closeModal();

      // Handle deploy platform clicks
      const deployItem = e.target.closest(".mo-deploy-item");
      if (deployItem) {
        const format = deployItem.dataset.format;
        const platformName = deployItem.dataset.name;

        // Update active state
        const container = deployItem.parentNode;
        container
          .querySelectorAll(".mo-deploy-item")
          .forEach((item) => item.classList.remove("is-active"));
        deployItem.classList.add("is-active");

        // Update export command
        const commandEl = document.getElementById("mo-deploy-command");
        const titleEl = document.getElementById("mo-deploy-format-title");

        if (commandEl && activeModel && activeSize) {
          const weight = getModelWeight(activeModel, activeSize);

          commandEl.textContent =
            format === "torch"
              ? ""
              : `yolo export model=${weight} format=${format}`;
          commandEl.dataset.hl = "";
          commandEl.dataset.needsUpdate = "1";

          highlightInline(commandEl.closest("pre"));

          if (titleEl) {
            titleEl.textContent = `Export Command (${platformName})`;
          }
        }
      }
    });

    render();
  }

  if (window.document$) {
    window.document$.subscribe(() => {
      try {
        initModelOverview();
      } catch (e) {
        console.error("Model Overview init failed", e);
      }
    });
  } else if (document.readyState !== "loading") {
    initModelOverview();
  } else document.addEventListener("DOMContentLoaded", initModelOverview);
})();
