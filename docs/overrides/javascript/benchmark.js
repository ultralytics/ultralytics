// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// YOLO models chart ---------------------------------------------------------------------------------------------------
const data = {
  //  YOLO12: {
  //    n: { speed: 1.64, mAP: 40.6 },
  //    s: { speed: 2.61, mAP: 48.0 },
  //    m: { speed: 4.86, mAP: 52.5 },
  //    l: { speed: 6.77, mAP: 53.7 },
  //    x: { speed: 11.79, mAP: 55.2 },
  //  },
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

let modelComparisonChart = null; // chart variable will hold the reference to the current chart instance.

// Function to lighten a hex color by a specified amount.
function lightenHexColor(color, amount = 0.5) {
  const r = parseInt(color.slice(1, 3), 16);
  const g = parseInt(color.slice(3, 5), 16);
  const b = parseInt(color.slice(5, 7), 16);
  const newR = Math.min(255, Math.round(r + (255 - r) * amount));
  const newG = Math.min(255, Math.round(g + (255 - g) * amount));
  const newB = Math.min(255, Math.round(b + (255 - b) * amount));
  return `#${newR.toString(16).padStart(2, "0")}${newG.toString(16).padStart(2, "0")}${newB.toString(16).padStart(2, "0")}`;
}

// Function to update the benchmarks chart.
function updateChart(initialDatasets = []) {
  if (modelComparisonChart) {
    modelComparisonChart.destroy();
  } // If a chart instance already exists, destroy it.

  // Define a specific color map for models.
  const colorMap = {
    YOLO11: "#0b23a9",
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

  // Always include all models in the dataset creation
  const datasets = Object.keys(data).map((algorithm, i) => {
    const baseColor =
      colorMap[algorithm] || `hsl(${Math.random() * 360}, 70%, 50%)`;
    const lineColor =
      Object.keys(data).indexOf(algorithm) === 0
        ? baseColor
        : lightenHexColor(baseColor, 0.6);

    return {
      label: algorithm,
      data: Object.entries(data[algorithm]).map(([version, point]) => ({
        x: point.speed,
        y: point.mAP,
        version: version.toUpperCase(),
      })),
      fill: false,
      borderColor: lineColor,
      tension: 0.2,
      pointRadius: Object.keys(data).indexOf(algorithm) === 0 ? 7 : 4,
      pointHoverRadius: Object.keys(data).indexOf(algorithm) === 0 ? 9 : 6,
      pointBackgroundColor: lineColor,
      pointBorderColor: "#ffffff",
      borderWidth: i === 0 ? 3 : 1.5,
      hidden:
        initialDatasets.length > 0 && !initialDatasets.includes(algorithm),
    };
  });

  // Create a new chart instance.
  modelComparisonChart = new Chart(
    document.getElementById("modelComparisonChart").getContext("2d"),
    {
      type: "line",
      data: { datasets },
      options: {
        //aspectRatio: 2.5,  // higher is wider
        plugins: {
          legend: {
            display: true,
            position: "right",
            align: "start", // start, end, center
            labels: { color: "#808080" },
            onClick: (e, legendItem, legend) => {
              const index = legendItem.datasetIndex;
              const ci = legend.chart;
              const meta = ci.getDatasetMeta(index);
              meta.hidden =
                meta.hidden === null ? !ci.data.datasets[index].hidden : null;
              ci.update();
            },
          }, // Configure the legend.
          tooltip: {
            callbacks: {
              label: (tooltipItem) => {
                const { dataset, dataIndex } = tooltipItem;
                const point = dataset.data[dataIndex];
                return `${dataset.label}${point.version.toLowerCase()}: Speed = ${point.x}ms/img, mAP50-95 = ${point.y}`; // Custom tooltip label.
              },
            },
            mode: "nearest",
            intersect: false,
          }, // Configure the tooltip.
        },
        interaction: { mode: "nearest", axis: "x", intersect: false }, // Configure the interaction mode.
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
    },
  );
}

function initChart(activeModels) {
  updateChart(activeModels);
}

document$.subscribe(function () {
  (function initializeApp() {
    if (typeof Chart !== "undefined") {
      // Get active models from page config or use default
      // e.g. <canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5", "YOLOv8"]'></canvas>
      const pageConfig = document
        .getElementById("modelComparisonChart")
        .getAttribute("active-models");
      const activeModels = pageConfig ? JSON.parse(pageConfig) : [];
      initChart(activeModels);
    } else {
      setTimeout(initializeApp, 50); // Retry every 50 ms
    }
  })();
});

// Export chart as PNG file
document.getElementById("btn-download").addEventListener("click", () => {
  const canvas = document.getElementById("modelComparisonChart");
  const tempCanvas = document.createElement("canvas");
  const ctx = tempCanvas.getContext("2d");

  tempCanvas.width = canvas.width;
  tempCanvas.height = canvas.height;

  const bgColor = isDark ? "#121212" : "#ffffff";  // Use dark or light background based on current theme

  ctx.fillStyle = bgColor;
  ctx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
  ctx.drawImage(canvas, 0, 0);

  const link = document.createElement("a");
  link.download = "chart.png";
  link.href = tempCanvas.toDataURL("image/png");
  link.click();
});

// Export Chart data as CSV file
document.getElementById("btn-download-data").addEventListener("click", () => {
  let csv = "Model,mAP50-95,Speed (ms/img)\n";

  Object.entries(data).forEach(([model, versions]) => {
    Object.entries(versions).forEach(([ver, point]) => {
      csv += `${model}${ver},${point.mAP},${point.speed}\n`;
    });
  });

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "model_benchmark_data.csv";
  link.click();
});


// Chart theme Default: light mode
let isDark = false;
document.getElementById("btn-toggle-theme").addEventListener("click", () => {
  isDark = !isDark;
  updateChartTheme();
  document.getElementById("btn-toggle-theme").textContent = isDark ? "🌞" : "🌙";
});

function updateChartTheme() {
  const options = modelComparisonChart.options;
  const fg = isDark ? "#ffffff" : "#333333";
  const bg = isDark ? "#121212" : "#ffffff";
  const grid = isDark ? "#444444" : "#e0e0e0";

  // Update global chart styles
  options.scales.x.title.color = fg;
  options.scales.y.title.color = fg;
  options.scales.x.ticks.color = fg;
  options.scales.y.ticks.color = fg;
  options.scales.x.grid.color = grid;
  options.scales.y.grid.color = grid;
  options.plugins.legend.labels.color = fg;
  options.plugins.tooltip.backgroundColor = isDark ? "#333" : "rgba(0,0,0,0.8)";
  options.plugins.tooltip.titleColor = fg;
  options.plugins.tooltip.bodyColor = fg;

  modelComparisonChart.update();
  document.getElementById("chart-container").style.background = bg;  // Optional: change container background
}

// Bar chart display
let currentChartType = 'line';
document.getElementById("btn-toggle-type").addEventListener("click", () => {
  currentChartType = currentChartType === 'line' ? 'bar' : 'line';
  updateChartType();
});

function updateChartType() {
  if (!modelComparisonChart) return;

  const newType = currentChartType;

  const cleanData = {
    datasets: modelComparisonChart.data.datasets.map((ds) => ({
      label: ds.label,
      data: ds.data.map(d => ({
        x: d.x,  // Speed
        y: d.y,  // mAP
        version: d.version,
      })),
      backgroundColor: ds.borderColor,
      borderColor: ds.borderColor,
      borderWidth: 2,
      barThickness: 10,  // You can adjust this
    })),
  };

  modelComparisonChart.destroy();
  modelComparisonChart = new Chart(
    document.getElementById("modelComparisonChart").getContext("2d"),
    {
      type: newType,  // will be 'bar'
      data: cleanData,
      options: {
        indexAxis: 'x', // keep vertical bars
        plugins: {
          legend: {
            display: true,
            position: "right",
            labels: { color: isDark ? "#fff" : "#333" },
          },
          tooltip: {
            callbacks: {
              label: (tooltipItem) => {
                const { raw } = tooltipItem;
                return `${tooltipItem.dataset.label}${raw.version.toLowerCase()}: Speed = ${raw.x}ms/img, mAP50-95 = ${raw.y}`;
              },
            }
          },
        },
        interaction: { mode: "nearest", axis: "x", intersect: false },
        scales: {
          x: {
            type: 'linear',
            title: {
              display: true,
              text: 'Latency T4 TensorRT10 FP16 (ms/img)',
              color: isDark ? "#fff" : "#333",
            },
            grid: { color: isDark ? "#444" : "#ccc" },
            ticks: { color: isDark ? "#fff" : "#333" },
            min: 0,
            max: 18
          },
          y: {
            title: {
              display: true,
              text: 'COCO mAP 50-95',
              color: isDark ? "#fff" : "#333",
            },
            grid: { color: isDark ? "#444" : "#ccc" },
            ticks: { color: isDark ? "#fff" : "#333" },
            min: 36,
            max: 56
          }
        }
      }
    }
  );
}
