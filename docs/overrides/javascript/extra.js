// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// Apply theme colors based on dark/light mode
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

// Check and apply appropriate theme based on system/user preference
const checkTheme = () => {
  const palette = JSON.parse(localStorage.getItem(".__palette") || "{}");
  if (palette.index === 0) {
    // Auto mode is selected
    applyTheme(window.matchMedia("(prefers-color-scheme: dark)").matches);
  }
};

// Watch for system theme changes
window
  .matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", checkTheme);

// Initialize theme handling on page load
document.addEventListener("DOMContentLoaded", () => {
  // Watch for theme toggle changes
  document
    .getElementById("__palette_1")
    ?.addEventListener(
      "change",
      (e) => e.target.checked && setTimeout(checkTheme),
    );
  // Initial theme check
  checkTheme();
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
