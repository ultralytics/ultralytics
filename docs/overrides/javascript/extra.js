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
