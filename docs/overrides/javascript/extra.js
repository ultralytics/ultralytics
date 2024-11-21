// Light/Dark Mode -----------------------------------------------------------------------------------------------------
const applyTheme = (isDark) => {
  document.body.setAttribute(
    "data-md-color-scheme",
    isDark ? "slate" : "default",
  );
  document.body.setAttribute(
    "data-md-color-primary",
    isDark ? "black" : "indigo",
  );
};

// Check and apply auto theme
const checkAutoTheme = () => {
  const palette = JSON.parse(localStorage.getItem(".__palette") || "{}");

  if (palette.index === 0) {
    applyTheme(window.matchMedia("(prefers-color-scheme: dark)").matches);
  }
};

// Event listeners for theme changes
const mediaQueryList = window.matchMedia("(prefers-color-scheme: dark)");
mediaQueryList.addListener(checkAutoTheme);

// Initial theme check
checkAutoTheme();

// Auto theme input listener
document.addEventListener("DOMContentLoaded", () => {
  const autoThemeInput = document.getElementById("__palette_1");
  autoThemeInput?.addEventListener("click", () => {
    if (autoThemeInput.checked) {
      setTimeout(checkAutoTheme);
    }
  });
});

// Inkeep --------------------------------------------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  const enableSearchBar = true;

  const inkeepScript = document.createElement("script");
  inkeepScript.src = "https://unpkg.com/@inkeep/uikit-js@0.3.18/dist/embed.js";
  inkeepScript.type = "module";
  inkeepScript.defer = true;
  document.head.appendChild(inkeepScript);

  if (enableSearchBar) {
    const containerDiv = document.createElement("div");
    containerDiv.style.transform = "scale(0.7)";
    containerDiv.style.transformOrigin = "left center";

    const inkeepDiv = document.createElement("div");
    inkeepDiv.id = "inkeepSearchBar";
    containerDiv.appendChild(inkeepDiv);

    const headerElement = document.querySelector(".md-header__inner");
    const searchContainer = headerElement.querySelector(".md-header__source");

    if (headerElement && searchContainer) {
      headerElement.insertBefore(containerDiv, searchContainer);
    }
  }

  // configure and initialize the widget
  const addInkeepWidget = (componentType, targetElementId) => {
    const inkeepWidget = Inkeep().embed({
      componentType,
      ...(componentType !== "ChatButton"
        ? { targetElement: targetElementId }
        : {}),
      colorModeSync: {
        observedElement: document.documentElement,
        isDarkModeCallback: (el) => {
          const currentTheme = el.getAttribute("data-color-mode");
          return currentTheme === "dark";
        },
        colorModeAttribute: "data-color-mode-scheme",
      },
      properties: {
        chatButtonType: "PILL",
        fixedPositionXOffset: "1rem",
        fixedPositionYOffset: "3rem",
        chatButtonBgColor: "#E1FF25",
        baseSettings: {
          apiKey: "13dfec2e75982bc9bae3199a08e13b86b5fbacd64e9b2f89",
          integrationId: "cm1shscmm00y26sj83lgxzvkw",
          organizationId: "org_e3869az6hQZ0mXdF",
          primaryBrandColor: "#E1FF25",
          organizationDisplayName: "Ultralytics",
          theme: {
            stylesheetUrls: ["/stylesheets/style.css"],
          },
        },
        modalSettings: {
          // optional settings
        },
        searchSettings: {
          placeholder: "Search",
        },
        aiChatSettings: {
          chatSubjectName: "Ultralytics",
          botAvatarSrcUrl:
            "https://storage.googleapis.com/organization-image-assets/ultralytics-botAvatarSrcUrl-1729379860806.svg",
          quickQuestions: [
            "What's new in Ultralytics YOLO11?",
            "How can I get started with Ultralytics HUB?",
            "How does Ultralytics Enterprise Licensing work?",
          ],
          getHelpCallToActions: [
            {
              name: "Ask on Ultralytics GitHub",
              url: "https://github.com/ultralytics/ultralytics",
              icon: {
                builtIn: "FaGithub",
              },
            },
            {
              name: "Ask on Ultralytics Discourse",
              url: "https://community.ultralytics.com/",
              icon: {
                builtIn: "FaDiscourse",
              },
            },
            {
              name: "Ask on Ultralytics Discord",
              url: "https://discord.com/invite/ultralytics",
              icon: {
                builtIn: "FaDiscord",
              },
            },
          ],
        },
      },
    });
  };
  inkeepScript.addEventListener("load", () => {
    const widgetContainer = document.getElementById("inkeepSearchBar");

    addInkeepWidget("ChatButton");
    widgetContainer && addInkeepWidget("SearchBar", "#inkeepSearchBar");
  });
});

// YOLO models chart ---------------------------------------------------------------------------------------------------
const data = {
  YOLO11: {
    n: { speed: 1.55, mAP: 39.5 },
    s: { speed: 2.63, mAP: 47.0 },
    m: { speed: 5.27, mAP: 51.4 },
    l: { speed: 6.84, mAP: 53.2 },
    x: { speed: 12.49, mAP: 54.7 },
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
    t: { speed: 2.3, mAP: 37.8 },
    s: { speed: 3.54, mAP: 46.5 },
    m: { speed: 6.43, mAP: 51.5 },
    c: { speed: 7.16, mAP: 52.8 },
    e: { speed: 16.77, mAP: 55.1 },
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
};

let chart = null; // chart variable will hold the reference to the current chart instance.

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
function updateChart() {
  if (chart) {
    chart.destroy();
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
  };

  // Get the selected algorithms from the checkboxes.
  const selectedAlgorithms = [
    ...document.querySelectorAll('input[name="algorithm"]:checked'),
  ].map((e) => e.value);

  // Create the datasets for the selected algorithms.
  const datasets = selectedAlgorithms.map((algorithm, i) => {
    const baseColor =
      colorMap[algorithm] || `hsl(${Math.random() * 360}, 70%, 50%)`;
    const lineColor = i === 0 ? baseColor : lightenHexColor(baseColor, 0.6); // Lighten non-primary lines.

    return {
      label: algorithm, // Label for the data points in the legend.
      data: Object.entries(data[algorithm]).map(([version, point]) => ({
        x: point.speed, // Speed data points on the x-axis.
        y: point.mAP, // mAP data points on the y-axis.
        version: version.toUpperCase(), // Store the version as additional data.
      })),
      fill: false, // Don't fill the chart.
      borderColor: lineColor, // Use the lightened color for the line.
      tension: 0.3, // Smooth the line.
      pointRadius: i === 0 ? 7 : 4, // Highlight primary dataset points.
      pointHoverRadius: i === 0 ? 9 : 6, // Highlight hover for primary dataset.
      pointBackgroundColor: lineColor, // Fill points with the line color.
      pointBorderColor: "#ffffff", // Add a border around points for contrast.
      borderWidth: i === 0 ? 3 : 1.5, // Slightly increase line size for the primary dataset.
    };
  });

  if (datasets.length === 0) {
    return;
  } // If there are no selected algorithms, return without creating a new chart.

  // Create a new chart instance.
  chart = new Chart(document.getElementById("chart").getContext("2d"), {
    type: "line", // Set the chart type to line.
    data: { datasets },
    options: {
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: { color: "#808080" },
        }, // Configure the legend.
        tooltip: {
          callbacks: {
            label: (tooltipItem) => {
              const { dataset, dataIndex } = tooltipItem;
              const point = dataset.data[dataIndex];
              return `${dataset.label}${point.version.toLowerCase()}: Speed = ${point.x}, mAP = ${point.y}`; // Custom tooltip label.
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
          }, // X-axis title.
          grid: { color: "#e0e0e0" }, // Grid line color.
          ticks: { color: "#808080" }, // Tick label color.
        },
        y: {
          title: { display: true, text: "mAP", color: "#808080" }, // Y-axis title.
          grid: { color: "#e0e0e0" }, // Grid line color.
          ticks: { color: "#808080" }, // Tick label color.
        },
      },
    },
  });
}

// Poll for Chart.js to load, then initialize checkboxes and chart
function initializeApp() {
  if (typeof Chart !== "undefined") {
    document
      .querySelectorAll('input[name="algorithm"]')
      .forEach((checkbox) => checkbox.addEventListener("change", updateChart));
    updateChart();
  } else {
    setTimeout(initializeApp, 100); // Retry every 100ms
  }
}
document.addEventListener("DOMContentLoaded", initializeApp); // Initial chart rendering on page load
