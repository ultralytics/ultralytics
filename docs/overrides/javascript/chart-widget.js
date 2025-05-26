// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//Reusable Chart.js toolbar widget with download functionality
class ChartWidget {
  constructor(chart, options = {}) {
    this.chart = chart;
    this.options = { position: "top-right", ...options };
    this.init();
  }

  init() {
    const canvas = this.chart.canvas;
    const container = canvas.parentElement;
    container.style.position = "relative";

    this.toolbar = document.createElement("div");
    this.toolbar.innerHTML = this.getHTML();
    container.appendChild(this.toolbar);

    // Wait for chart to settle before positioning
    requestAnimationFrame(() => {
      this.toolbar.style.cssText = this.getCSS(canvas);
      this.attachEvents();
      this.setupHover(canvas);
    });
  }

  getHTML() {
    return `
      <button data-action="png" data-tip="Download chart as PNG image">📷</button>
      <button data-action="csv" data-tip="Download chart data as CSV file">📊</button>
      <button data-action="ultralytics" data-tip="Made with Chart.js and Ultralytics">
        <img src="https://github.com/ultralytics/assets/raw/main/logo/Ultralytics-logomark-color.png" width="18" height="18">
      </button>
      <div class="tip" style="position:absolute;bottom:100%;left:50%;transform:translateX(-50%);background:CanvasText;color:Canvas;padding:4px 8px;border-radius:3px;font-size:11px;white-space:nowrap;display:none;margin-bottom:5px;z-index:1001;pointer-events:none;"></div>
    `;
  }

  getCSS(canvas) {
    const rect = canvas.getBoundingClientRect();
    const containerRect = canvas.parentElement.getBoundingClientRect();
    const top = rect.top - containerRect.top - 50;
    const left = rect.right - containerRect.left - 110;

    return `
      position: absolute; top: ${top}px; left: ${left}px;
      background: Canvas; border: 1px solid rgba(128,128,128,0.3);
      border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      backdrop-filter: blur(8px); display: flex; gap: 2px; padding: 4px;
      opacity: 0; pointer-events: none; z-index: 1000;
      transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
      transform: translateY(-5px) scale(0.95);
    `;
  }

  attachEvents() {
    const tip = this.toolbar.querySelector(".tip");

    this.toolbar.addEventListener("click", (e) => {
      const action = e.target.closest("button").dataset.action;
      if (action === "png") this.downloadPNG();
      if (action === "csv") this.downloadCSV();
      if (action === "ultralytics")
        window.open("https://ultralytics.com", "_blank");
    });

    this.toolbar.querySelectorAll("button").forEach((btn) => {
      Object.assign(btn.style, {
        border: "none",
        background: "none",
        padding: "8px",
        cursor: "pointer",
        borderRadius: "4px",
        fontSize: "18px",
        minWidth: "32px",
        minHeight: "32px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        transition: "all 0.15s ease",
        position: "relative",
      });

      btn.onmouseenter = () => {
        btn.style.background = "ButtonFace";
        btn.style.transform = "scale(1.20)";
        tip.textContent = btn.dataset.tip;
        tip.style.display = "block";
      };
      btn.onmouseleave = () => {
        btn.style.background = "none";
        btn.style.transform = "scale(1)";
        tip.style.display = "none";
      };
    });
  }

  setupHover(canvas) {
    let timeout;
    const show = () => {
      clearTimeout(timeout);
      Object.assign(this.toolbar.style, {
        opacity: "1",
        pointerEvents: "auto",
        transform: "translateY(0) scale(1)",
      });
    };
    const hide = () => {
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        Object.assign(this.toolbar.style, {
          opacity: "0",
          pointerEvents: "none",
          transform: "translateY(-5px) scale(0.95)",
        });
      }, 1000);
    };

    canvas.addEventListener("mouseenter", show);
    canvas.addEventListener("mouseleave", hide);
    this.toolbar.addEventListener("mouseenter", show);
    this.toolbar.addEventListener("mouseleave", hide);
  }

  downloadPNG() {
    const a = document.createElement("a");
    a.download = `chart-${Date.now()}.png`;
    a.href = this.chart.toBase64Image("image/png", 1);
    a.click();
  }

  downloadCSV() {
    const xTitle = this.chart.options?.scales?.x?.title?.text || "x";
    const yTitle = this.chart.options?.scales?.y?.title?.text || "y";

    const data = [];
    this.chart.data.datasets.forEach((dataset, i) => {
      if (this.chart.getDatasetMeta(i).hidden) return; // Skip unselected models

      dataset.data.forEach((point) => {
        data.push({
          model: dataset.label,
          version: point.version || "",
          [xTitle]: point.x,
          [yTitle]: point.y,
        });
      });
    });

    const headers = Object.keys(data[0]);
    const csv = [
      headers.join(","),
      ...data.map((row) => headers.map((h) => `"${row[h] || ""}"`).join(",")),
    ].join("\n");

    const a = document.createElement("a");
    a.download = `chart-data-${Date.now()}.csv`;
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.click();
  }

  destroy() {
    this.toolbar?.remove();
  }
}

window.ChartWidget = ChartWidget;
