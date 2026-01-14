// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// Auto-load chart-widget.js if not already loaded
const loadChartWidget = () =>
  new Promise((resolve) => {
    if (window.ChartWidget) {
      return resolve();
    }
    const s = document.createElement("script");
    const base =
      (document.currentScript || document.querySelector('script[src*="benchmark.js"]'))?.src.replace(/[^/]*$/, "") ||
      "./";
    s.src = `${base}chart-widget.js`;
    s.onload = s.onerror = resolve;
    document.head.appendChild(s);
  });

// YOLO models chart ---------------------------------------------------------------------------------------------------
const data = {
  YOLO26: {
    n: { speed: 1.7, mAP: 40.9 },
    s: { speed: 2.5, mAP: 48.6 },
    m: { speed: 4.7, mAP: 53.1 },
    l: { speed: 6.2, mAP: 55.0 },
    x: { speed: 11.8, mAP: 57.5 },
  },
  YOLO11: {
    n: { speed: 1.5, mAP: 39.5 },
    s: { speed: 2.5, mAP: 47.0 },
    m: { speed: 4.7, mAP: 51.5 },
    l: { speed: 6.2, mAP: 53.4 },
    x: { speed: 11.3, mAP: 54.7 },
  },
  YOLOv10: {
    n: { speed: 1.56, mAP: 39.5 },
    s: { speed: 2.66, mAP: 46.7 },
    m: { speed: 5.48, mAP: 51.3 },
    b: { speed: 6.54, mAP: 52.7 },
    l: { speed: 8.33, mAP: 53.3 },
    x: { speed: 12.2, mAP: 54.4 },
  },
  YOLOv9: {
    t: { speed: 2.3, mAP: 38.3 },
    s: { speed: 3.54, mAP: 46.8 },
    m: { speed: 6.43, mAP: 51.4 },
    c: { speed: 7.16, mAP: 53.0 },
    e: { speed: 16.77, mAP: 55.6 },
  },
  YOLOv8: {
    n: { speed: 1.47, mAP: 37.3 },
    s: { speed: 2.66, mAP: 44.9 },
    m: { speed: 5.86, mAP: 50.2 },
    l: { speed: 9.06, mAP: 52.9 },
    x: { speed: 14.37, mAP: 53.9 },
  },
  YOLOv7: { l: { speed: 6.84, mAP: 51.4 }, x: { speed: 11.57, mAP: 53.1 } },
  "YOLOv6-3.0": {
    n: { speed: 1.17, mAP: 37.5 },
    s: { speed: 2.66, mAP: 45.0 },
    m: { speed: 5.28, mAP: 50.0 },
    l: { speed: 8.95, mAP: 52.8 },
  },
  YOLOv5: {
    n: { speed: 1.12, mAP: 28.0 },
    s: { speed: 1.92, mAP: 37.4 },
    m: { speed: 4.03, mAP: 45.4 },
    l: { speed: 6.61, mAP: 49.0 },
    x: { speed: 11.89, mAP: 50.7 },
  },
  "PP-YOLOE+": {
    t: { speed: 2.84, mAP: 39.9 },
    s: { speed: 2.62, mAP: 43.7 },
    m: { speed: 5.56, mAP: 49.8 },
    l: { speed: 8.36, mAP: 52.9 },
    x: { speed: 14.3, mAP: 54.7 },
  },
  "DAMO-YOLO": {
    t: { speed: 2.32, mAP: 42.0 },
    s: { speed: 3.45, mAP: 46.0 },
    m: { speed: 5.09, mAP: 49.2 },
    l: { speed: 7.18, mAP: 50.8 },
  },
  YOLOX: {
    s: { speed: 2.56, mAP: 40.5 },
    m: { speed: 5.43, mAP: 46.9 },
    l: { speed: 9.04, mAP: 49.7 },
    x: { speed: 16.1, mAP: 51.1 },
  },
  RTDETRv2: {
    s: { speed: 5.03, mAP: 48.1 },
    m: { speed: 7.51, mAP: 51.9 },
    l: { speed: 9.76, mAP: 53.4 },
    x: { speed: 15.03, mAP: 54.3 },
  },
  EfficientDet: {
    d0: { speed: 3.92, mAP: 34.6 },
    d1: { speed: 7.31, mAP: 40.5 },
    d2: { speed: 10.92, mAP: 43.0 },
    d3: { speed: 19.59, mAP: 47.5 },
    // d4: { speed: 33.55, mAP: 49.4 },
    // d5: { speed: 67.86, mAP: 50.7 },
    // d6: { speed: 89.29, mAP: 51.7 },
    // d7: { speed: 128.07, mAP: 53.7 },
    // d8: { speed: 157.57, mAP: 55.1 }
  },
};

// Color overrides for specific models
const colorOverrides = {
  YOLO26: "#0b23a9",
  YOLO11: "#1e90ff",
  YOLOv10: "#ff7f0e",
  YOLOv9: "#2ca02c",
  YOLOv8: "#d62728",
  YOLOv7: "#9467bd",
  "YOLOv6-3.0": "#8c564b",
  YOLOv5: "#e377c2",
  "PP-YOLOE+": "#7f7f7f",
  "DAMO-YOLO": "#bcbd22",
  YOLOX: "#17becf",
  RTDETRv2: "#eccd22",
  EfficientDet: "#000000",
};

let chart = null;
let chartWidget = null;

const lighten = (hex, amt = 0.6) => {
  const [r, g, b] = [1, 3, 5].map((i) => Number.parseInt(hex.slice(i, i + 2), 16));
  return `#${[r, g, b]
    .map((c) =>
      Math.min(255, Math.round(c + (255 - c) * amt))
        .toString(16)
        .padStart(2, "0"),
    )
    .join("")}`;
};

const createDataset = (algo, i, activeModels) => {
  const baseColor = colorOverrides[algo] || `hsl(${(i * 137) % 360}, 70%, 50%)`;
  const isFirst = i === 0;
  return {
    label: algo,
    data: Object.entries(data[algo]).map(([ver, pt]) => ({
      x: pt.speed,
      y: pt.mAP,
      version: ver.toUpperCase(),
    })),
    fill: false,
    borderColor: isFirst ? baseColor : lighten(baseColor),
    tension: 0.2,
    pointRadius: isFirst ? 7 : 4,
    pointHoverRadius: isFirst ? 9 : 6,
    pointBackgroundColor: isFirst ? baseColor : lighten(baseColor),
    pointBorderColor: "#ffffff",
    borderWidth: isFirst ? 3 : 1.5,
    hidden: activeModels.length > 0 && !activeModels.includes(algo),
  };
};

const chartConfig = {
  type: "line",
  data: { datasets: Object.keys(data).map(createDataset) },
  options: {
    plugins: {
      legend: {
        display: true,
        position: "right",
        align: "start",
        labels: { color: "#808080" },
      },
      tooltip: {
        callbacks: {
          label: ({ dataset, dataIndex }) => {
            const pt = dataset.data[dataIndex];
            return `${dataset.label}${pt.version.toLowerCase()}: Speed = ${pt.x}ms/img, mAP50-95 = ${pt.y}`;
          },
        },
        mode: "nearest",
        intersect: false,
      },
    },
    interaction: { mode: "nearest", axis: "x", intersect: false },
    scales: {
      x: {
        type: "linear",
        position: "bottom",
        title: {
          display: true,
          text: "Latency T4 TensorRT10 FP16 (ms/img)",
          color: "#808080",
        },
        grid: { color: "#e0e0e0" },
        ticks: { color: "#808080" },
        min: 0,
        max: 18,
      },
      y: {
        title: { display: true, text: "COCO mAP 50-95", color: "#808080" },
        grid: { color: "#e0e0e0" },
        ticks: { color: "#808080" },
        min: 36,
        max: 58,
      },
    },
  },
};

const updateChart = async (activeModels = []) => {
  chart?.destroy();
  chartWidget?.destroy();

  chartConfig.data.datasets = Object.keys(data).map((algo, i) => createDataset(algo, i, activeModels));

  chart = new Chart(document.getElementById("modelComparisonChart").getContext("2d"), chartConfig);

  // Load widget and add to chart
  await loadChartWidget();
  if (window.ChartWidget) {
    chartWidget = new ChartWidget(chart, { position: "top-right" });
  }
};

// Get active models from page config or use default
// e.g. <canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLO11"]'></canvas>
const initChart = () => {
  const activeModels = JSON.parse(
    document.getElementById("modelComparisonChart").getAttribute("active-models") || "[]",
  );
  updateChart(activeModels);
};

document$.subscribe(() => {
  const init = () => (typeof Chart !== "undefined" ? initChart() : setTimeout(init, 50));
  init();
});
