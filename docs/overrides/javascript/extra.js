// Function that applies light/dark theme based on the user's preference
const applyAutoTheme = () => {
  // Determine the user's preferred color scheme
  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

  // Apply the appropriate attributes based on the user's preference
  if (prefersLight) {
    document.body.setAttribute("data-md-color-scheme", "default");
    document.body.setAttribute("data-md-color-primary", "indigo");
  } else if (prefersDark) {
    document.body.setAttribute("data-md-color-scheme", "slate");
    document.body.setAttribute("data-md-color-primary", "black");
  }
};

// Function that checks and applies light/dark theme based on the user's preference (if auto theme is enabled)
function checkAutoTheme() {
  // Array of supported language codes -> each language has its own palette (stored in local storage)
  const supportedLangCodes = ["en", "zh", "ko", "ja", "ru", "de", "fr", "es", "pt", "it", "tr", "vi", "nl"];
  // Get the URL path
  const path = window.location.pathname;
  // Extract the language code from the URL (assuming it's in the format /xx/...)
  const langCode = path.split("/")[1];
  // Check if the extracted language code is in the supported languages
  const isValidLangCode = supportedLangCodes.includes(langCode);
  // Construct the local storage key based on the language code if valid, otherwise default to the root key
  const localStorageKey = isValidLangCode ? `/${langCode}/.__palette` : "/.__palette";
  // Retrieve the palette from local storage using the constructed key
  const palette = localStorage.getItem(localStorageKey);
  if (palette) {
    // Check if the palette's index is 0 (auto theme)
    const paletteObj = JSON.parse(palette);
    if (paletteObj && paletteObj.index === 0) {
      applyAutoTheme();
    }
  }
}

// Run function when the script loads
checkAutoTheme();

// Re-run the function when the user's preference changes (when the user changes their system theme)
window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", checkAutoTheme);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", checkAutoTheme);

// Re-run the function when the palette changes (e.g. user switched from dark theme to auto theme)
// ! We can't use window.addEventListener("storage", checkAutoTheme) because it will NOT be triggered on the current tab
// ! So we have to use the following workaround:
// Get the palette input for auto theme
var autoThemeInput = document.getElementById("__palette_1");
if (autoThemeInput) {
  // Add a click event listener to the input
  autoThemeInput.addEventListener("click", function () {
    // Check if the auto theme is selected
    if (autoThemeInput.checked) {
      // Re-run the function after a short delay (to ensure that the palette has been updated)
      setTimeout(applyAutoTheme);
    }
  });
}

// Add iframe navigation
window.onhashchange = function() {
    window.parent.postMessage({
        type: 'navigation',
        hash: window.location.pathname + window.location.search + window.location.hash
    }, '*');
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
          botAvatarSrcUrl: "https://storage.googleapis.com/organization-image-assets/ultralytics-botAvatarSrcUrl-1727908259285.png",
          botAvatarDarkSrcUrl: "https://storage.googleapis.com/organization-image-assets/ultralytics-botAvatarDarkSrcUrl-1727908258478.png",
          quickQuestions: [
            "What's new in Ultralytics YOLO11?",
            "How can I get started with Ultralytics HUB?",
            "How does Ultralytics Enterprise Licensing work?"
          ],
          getHelpCallToActions: [
            {
              name: "Ask on Ultralytics GitHub",
              url: "https://github.com/ultralytics/ultralytics",
              icon: {
                builtIn: "FaGithub"
              }
            },
            {
              name: "Ask on Ultralytics Discourse",
              url: "https://community.ultralytics.com/",
              icon: {
                builtIn: "FaDiscourse"
              }
            },
            {
              name: "Ask on Ultralytics Discord",
              url: "https://discord.com/invite/ultralytics",
              icon: {
                builtIn: "FaDiscord"
              }
            }
          ],
        },
      },
    });
  };
  inkeepScript.addEventListener("load", () => {
    addInkeepWidget(); // initialize the widget
  });
});
