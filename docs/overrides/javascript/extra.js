// Apply theme based on user preference
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
  const supportedLangCodes = [
    "en",
    "zh",
    "ko",
    "ja",
    "ru",
    "de",
    "fr",
    "es",
    "pt",
    "it",
    "tr",
    "vi",
    "ar",
  ];
  const langCode = window.location.pathname.split("/")[1];
  const localStorageKey = `${supportedLangCodes.includes(langCode) ? `/${langCode}` : ""}/.__palette`;
  const palette = JSON.parse(localStorage.getItem(localStorageKey) || "{}");

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

// Iframe navigation
window.onhashchange = () => {
  window.parent.postMessage(
    {
      type: "navigation",
      hash:
        window.location.pathname +
        window.location.search +
        window.location.hash,
    },
    "*",
  );
};

// Add Inkeep button
document.addEventListener("DOMContentLoaded", () => {
  const inkeepScript = document.createElement("script");
  inkeepScript.src = "https://unpkg.com/@inkeep/uikit-js@0.3.11/dist/embed.js";
  inkeepScript.type = "module";
  inkeepScript.defer = true;
  document.head.appendChild(inkeepScript);

  // Configure and initialize the widget
  const addInkeepWidget = () => {
    const inkeepWidget = Inkeep().embed({
      componentType: "ChatButton",
      colorModeSync: {
        observedElement: document.documentElement,
        isDarkModeCallback: (el) => {
          const currentTheme = el.getAttribute("data-color-mode");
          return currentTheme === "dark";
        },
        colorModeAttribute: "data-color-mode",
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
          // ...optional settings
        },
        modalSettings: {
          // optional settings
        },
        searchSettings: {
          // optional settings
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
    addInkeepWidget(); // initialize the widget
  });
});

// This object contains the benchmark data for various object detection models.
const data = {
    'YOLOv5':  {s: {speed: 1.92, mAP: 37.4}, m: {speed: 4.03, mAP: 45.4}, l: {speed: 6.61, mAP: 49.0}, x: {speed: 11.89, mAP: 50.7}},
    'YOLOv6':  {n: {speed: 1.17, mAP: 37.5}, s: {speed: 2.66, mAP: 45.0}, m: {speed: 5.28, mAP: 50.0}, l: {speed: 8.95, mAP: 52.8}},
    'YOLOv7':  {l: {speed: 6.84, mAP: 51.4}, x: {speed: 11.57, mAP: 53.1}},
    'YOLOv8':  {n: {speed: 1.47, mAP: 37.3}, s: {speed: 2.66, mAP: 44.9}, m: {speed: 5.86, mAP: 50.2}, l: {speed: 9.06, mAP: 52.9}, x: {speed: 14.37, mAP: 53.9}},
    'YOLOv9':  {t: {speed: 2.30, mAP: 37.8}, s: {speed: 3.54, mAP: 46.5}, m: {speed: 6.43, mAP: 51.5}, c: {speed: 7.16, mAP: 52.8}, e: {speed: 16.77, mAP: 55.1}},
    'YOLOv10': {n: {speed: 1.56, mAP: 39.5}, s: {speed: 2.66, mAP: 46.7}, m: {speed: 5.48, mAP: 51.3}, b: {speed: 6.54, mAP: 52.7}, l: {speed: 8.33, mAP: 53.3}, x: {speed: 12.2, mAP: 54.4}},
    'PPYOLOE': {t: {speed: 2.84, mAP: 39.9}, s: {speed: 2.62, mAP: 43.7}, m: {speed: 5.56, mAP: 49.8}, l: {speed: 8.36, mAP: 52.9}, x: {speed: 14.3, mAP: 54.7}},
    'YOLO11':  {n: {speed: 1.55, mAP: 39.5}, s: {speed: 2.63, mAP: 47.0}, m: {speed: 5.27, mAP: 51.4}, l: {speed: 6.84, mAP: 53.2}, x: {speed: 12.49, mAP: 54.7}}
};

let chart = null;  // chart variable will hold the reference to the current chart instance.

// This function is responsible for updating the benchmarks chart.
function updateChart() {
    // If a chart instance already exists, destroy it.
    if (chart) {
      chart.destroy();
    }

    // Get the selected algorithms from the checkboxes.
    const selectedAlgorithms = [...document.querySelectorAll('input[name="algorithm"]:checked')].map(e => e.value);

    // Create the datasets for the selected algorithms.
    const datasets = selectedAlgorithms.map((algorithm, index) => ({
        label: algorithm,  // Label for the data points in the legend.
        data: Object.entries(data[algorithm]).map(([version, point]) => ({
            x: point.speed,     // Speed data points on the x-axis.
            y: point.mAP,       // mAP data points on the y-axis.
            version: version.toUpperCase() // Store the version as additional data.
        })),
        fill: false,    // Don't fill the chart.
        borderColor: `hsl(${index * 90}, 70%, 50%)`,  // Assign a unique color to each dataset.
        tension: 0.3, // Smooth the line.
        pointRadius: 5, // Increase the dot size.
        pointHoverRadius: 10, // Increase the dot size on hover.
        borderWidth: 2 // Set the line thickness.
    }));

    // If there are no selected algorithms, return without creating a new chart.
    if (datasets.length === 0) {
      return;
    }

    // Create a new chart instance.
    chart = new Chart(document.getElementById('chart').getContext('2d'), {
        type: 'line', // Set the chart type to line.
        data: { datasets },
        options: {
            plugins: {
                legend: { display: true, position: 'top', labels: {color: '#808080'} }, // Configure the legend.
                tooltip: {
                    callbacks: {
                        label: (tooltipItem) => {
                            const { dataset, dataIndex } = tooltipItem;
                            const point = dataset.data[dataIndex];
                            return `${dataset.label}${point.version.toLowerCase()}: Speed = ${point.x}, mAP = ${point.y}`; // Custom tooltip label.
                        }
                    },
                    mode: 'nearest',
                    intersect: false
                } // Configure the tooltip.
            },
            interaction: { mode: 'nearest', axis: 'x', intersect: false }, // Configure the interaction mode.
            scales: {
                x: {
                    type: 'linear', position: 'bottom',
                    title: { display: true, text: 'Latency T4 TensorRT10 FP16 (ms/img)', color: '#808080'}, // X-axis title.
                    grid: { color: '#e0e0e0' }, // Grid line color.
                    ticks: { color: '#808080' } // Tick label color.
                },
                y: {
                    title: { display: true, text: 'mAP', color: '#808080'}, // Y-axis title.
                    grid: { color: '#e0e0e0' }, // Grid line color.
                    ticks: { color: '#808080' } // Tick label color.
                }
            }
        }
    });
}

// Poll for Chart.js to load, then initialize checkboxes and chart
function initializeApp() {
    if (typeof Chart !== 'undefined') {
        document.querySelectorAll('input[name="algorithm"]').forEach(checkbox =>
            checkbox.addEventListener('change', updateChart)
        );
        updateChart();
    } else {
        setTimeout(initializeApp, 100);  // Retry every 100ms
    }
}
document.addEventListener("DOMContentLoaded", initializeApp); // Initial chart rendering on page load
