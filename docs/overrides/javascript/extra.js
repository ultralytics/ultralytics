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

document.addEventListener("DOMContentLoaded", () => {
   // Load the Inkeep script
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
        baseSettings: {
          apiKey: "INKEEP_API_KEY", // required
          integrationId: "INKEEP_INTEGRATION_ID", // required
          organizationId: "INKEEP_ORGANIZATION_ID", // required
          primaryBrandColor: "#26D6FF", // your brand color, widget color scheme is derived from this
          organizationDisplayName: "Inkeep",
          // ...optional settings
        },
        modalSettings: {
          // optional settings
        },
        searchSettings: {
          // optional settings
        },
        aiChatSettings: {
          // optional settings
          botAvatarSrcUrl: "https://mydomain.com/mylogo", // use your own bot avatar
          quickQuestions: [
            "Example question 1?",
            "Example question 2?",
            "Example question 3?",
          ],
        },
      },
    });
  };

  inkeepScript.addEventListener("load", () => {
    addInkeepWidget(); // initialize the widget
  });
});

