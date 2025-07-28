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
  inkeepScript.src =
    "https://cdn.jsdelivr.net/npm/@inkeep/cxkit-js@0.5/dist/embed.js";
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

  // Configuration object for Inkeep
  const config = {
    baseSettings: {
      apiKey: "13dfec2e75982bc9bae3199a08e13b86b5fbacd64e9b2f89",
      primaryBrandColor: "#E1FF25",
      organizationDisplayName: "Ultralytics",
      colorMode: {
        enableSystem: true,
      },
      theme: {
        styles: [
          {
            key: "main",
            type: "link",
            value: "/stylesheets/style.css",
          },
          {
            key: "chat-button",
            type: "style",
            value: `
              /* Light mode styling */
              .ikp-chat-button__button {
                background-color: #E1FF25;
                color: #111F68;
              }
              /* Dark mode styling */
              [data-theme="dark"] .ikp-chat-button__button {
                background-color: #40434f;
                color: #ffffff;
              }
              .ikp-chat-button__container {
                position: fixed;
                right: 1rem;
                bottom: 3rem;
              }
            `,
          },
        ],
      },
    },
    searchSettings: {
      placeholder: "Search",
    },
    aiChatSettings: {
      chatSubjectName: "Ultralytics",
      aiAssistantAvatar:
        "https://storage.googleapis.com/organization-image-assets/ultralytics-botAvatarSrcUrl-1729379860806.svg",
      exampleQuestions: [
        "What's new in Ultralytics YOLO11?",
        "How can I get started with Ultralytics HUB?",
        "How does Ultralytics Enterprise Licensing work?",
      ],
      getHelpOptions: [
        {
          name: "Ask on Ultralytics GitHub",
          icon: {
            builtIn: "FaGithub",
          },
          action: {
            type: "open_link",
            url: "https://github.com/ultralytics/ultralytics",
          },
        },
        {
          name: "Ask on Ultralytics Discourse",
          icon: {
            builtIn: "FaDiscourse",
          },
          action: {
            type: "open_link",
            url: "https://community.ultralytics.com/",
          },
        },
        {
          name: "Ask on Ultralytics Discord",
          icon: {
            builtIn: "FaDiscord",
          },
          action: {
            type: "open_link",
            url: "https://discord.com/invite/ultralytics",
          },
        },
      ],
    },
  };

  // Initialize Inkeep widgets when script loads
  inkeepScript.addEventListener("load", () => {
    const widgetContainer = document.getElementById("inkeepSearchBar");

    Inkeep.ChatButton(config);
    widgetContainer && Inkeep.SearchBar("#inkeepSearchBar", config);
  });
});

// Fix language switcher links
(function () {
  function fixLanguageLinks() {
    const path = location.pathname;
    const links = document.querySelectorAll(".md-select__link");
    if (!links.length) return;

    const langs = [];
    let defaultLink = null;

    // Extract language codes
    links.forEach((link) => {
      const href = link.getAttribute("href");
      if (!href) return;

      const url = new URL(href, location.origin);
      const match = url.pathname.match(/^\/([a-z]{2})\/?$/);

      if (match) langs.push({ code: match[1], link });
      else if (url.pathname === "/" || url.pathname === "") defaultLink = link;
    });

    // Find current language and base path
    let basePath = path;
    for (const lang of langs) {
      if (path.startsWith("/" + lang.code + "/")) {
        basePath = path.substring(lang.code.length + 1);
        break;
      }
    }

    // Update links
    langs.forEach(
      (lang) => (lang.link.href = location.origin + "/" + lang.code + basePath),
    );
    if (defaultLink) defaultLink.href = location.origin + basePath;
  }

  // Run immediately
  fixLanguageLinks();

  // Handle SPA navigation
  if (typeof document$ !== "undefined") {
    document$.subscribe(() => setTimeout(fixLanguageLinks, 50));
  } else {
    let lastPath = location.pathname;
    setInterval(() => {
      if (location.pathname !== lastPath) {
        lastPath = location.pathname;
        setTimeout(fixLanguageLinks, 50);
      }
    }, 200);
  }
})();
