// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// Auto-load chart-widget.js if not already loaded
const loadChartWidget = () =>
  new Promise((resolve) => {
    if (window.ChartWidget) return resolve();
    const s = document.createElement("script");
    const base =
      (
        document.currentScript ||
        document.querySelector('script[src*="benchmark.js"]')
      )?.src.replace(/[^/]*$/, "") || "./";
    s.src = base + "chart-widget.js";
    s.onload = s.onerror = resolve;
    document.head.appendChild(s);
  });

// YOLO models chart
// Load and process data from model_data.json
const loadModelData = async () => {
  try {
    const response = await fetch("/assets/model_data.json");

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const modelData = await response.json();
    const processedData = {};
    const colors = {};

    // Only include models where show_graph is true
    Object.entries(modelData).forEach(([modelName, modelInfo]) => {
      if (modelInfo.show_graph === true && modelInfo.performance) {
        const modelPerformance = {};

        // Process each model variant (n, s, m, l, x, etc.)
        Object.entries(modelInfo.performance).forEach(([variant, perfData]) => {
          // Only include variants that have both t4 and map values
          if (perfData.t4 && perfData.map) {
            modelPerformance[variant] = {
              speed: perfData.t4,
              mAP: perfData.map,
            };
          }
        });

        // Only add the model if it has at least one valid variant
        if (Object.keys(modelPerformance).length > 0) {
          processedData[modelName] = modelPerformance;
          // Store color from JSON data if available
          if (modelInfo.graph_color_override) {
            colors[modelName] = modelInfo.graph_color_override;
          }
        }
      }
    });

    return { data: processedData, colors };
  } catch (error) {
    console.error("Error loading model data:", error);
    return { data: {}, colors: {} };
  }
};

// Initialize with empty data - will be populated after loading
let data = {};
let colorOverrides = {};

let chart = null;
let chartWidget = null;

const lighten = (hex, amt = 0.6) => {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
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
        max: 56,
      },
    },
  },
};

const updateChart = async (activeModels = []) => {
  chart?.destroy();
  chartWidget?.destroy();

  chartConfig.data.datasets = Object.keys(data).map((algo, i) =>
    createDataset(algo, i, activeModels),
  );

  chart = new Chart(
    document.getElementById("modelComparisonChart").getContext("2d"),
    chartConfig,
  );

  // Load widget and add to chart
  await loadChartWidget();
  if (window.ChartWidget) {
    chartWidget = new ChartWidget(chart, { position: "top-right" });
  }
};

// Get active models from page config or use default
// e.g. <canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8", "YOLO11"]'></canvas>
const initChart = async () => {
  // Load data from JSON file first
  const result = await loadModelData();
  data = result.data;
  colorOverrides = result.colors;

  const activeModels = JSON.parse(
    document
      .getElementById("modelComparisonChart")
      .getAttribute("active-models") || "[]",
  );
  updateChart(activeModels);
};

document$.subscribe(() => {
  const init = async () => {
    if (typeof Chart !== "undefined") {
      await initChart();
    } else {
      setTimeout(init, 50);
    }
  };
  init();
});
