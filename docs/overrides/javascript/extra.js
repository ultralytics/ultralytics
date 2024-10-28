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
    if (autoThemeInput.checked) setTimeout(checkAutoTheme);
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

const data = {
    'YOLO11': {s: { speed: 1.55, mAP: 39.5 }, m: { speed: 2.63, mAP: 47.0 }, x: { speed: 5.27, mAP: 51.4 }},
    'YOLOv10': {s: { speed: 1.56, mAP: 39.5 }, m: { speed: 2.66, mAP: 46.7 }, x: { speed: 5.48, mAP: 51.3 }},
    'YOLOv9': {s: { speed: 2.30, mAP: 37.8 }, m: { speed: 3.54, mAP: 46.5 }, x: { speed: 6.43, mAP: 51.5 }},
    'YOLOv5': {s: { speed: 1.92, mAP: 37.4 }, m: { speed: 4.03, mAP: 45.4 }, x: { speed: 6.61, mAP: 49.0 }}};

let chart = null;
function updateChart() {if (chart) { chart.destroy(); }

    const selectedAlgorithms = [...document.querySelectorAll('input[name="algorithm"]:checked')].map(e => e.value);
    const datasets = selectedAlgorithms.map((algorithm, index) => ({
        label: algorithm,
        data: Object.entries(data[algorithm]).map(([version, point]) => ({
            x: point.speed, y: point.mAP,
            version: version.toUpperCase() // Store version as additional data
        })),
        fill: false,
        borderColor: `hsl(${index * 90}, 70%, 50%)`,
        tension: 0.3, // Smooth line
        pointRadius: 5, // Increased dot size
        pointHoverRadius: 10,
        borderWidth: 2 // Line thickness
    }));

    if (datasets.length === 0) return;
    chart = new Chart(document.getElementById('chart').getContext('2d'), {
        type: 'line', data: { datasets },
        options: {
            plugins: {
                legend: { display: true, position: 'top', labels: { color: '#111e68' } },
                tooltip: {
                    callbacks: {
                        label: (tooltipItem) => {
                            const { dataset, dataIndex } = tooltipItem;
                            const point = dataset.data[dataIndex];
                            return `${dataset.label}${point.version.toLowerCase()}: Speed = ${point.x}, mAP = ${point.y}`;
                        }
                    },
                    mode: 'nearest',
                    intersect: false
                }
            },
            interaction: { mode: 'nearest', axis: 'x', intersect: false },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: { display: true, text: 'Latency T4 TensorRT10 FP16 (ms/img)', color: '#111e68' },
                    grid: { color: '#e0e0e0' },
                    ticks: { color: '#111e68' }
                },
                y: {
                    title: { display: true, text: 'mAP', color: '#111e68' },
                    grid: { color: '#e0e0e0' },
                    ticks: { color: '#111e68' }
                }
            }
        }
    });
}

// Attach event listeners to checkboxes
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll('input[name="algorithm"]').forEach(checkbox =>
        checkbox.addEventListener('change', updateChart)
    );
});
