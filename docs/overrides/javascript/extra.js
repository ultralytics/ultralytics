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

// Sync widget theme with Material theme
const syncWidgetTheme = () => {
  const isDark = document.body.getAttribute("data-md-color-scheme") === "slate";
  document.documentElement.setAttribute(
    "data-theme",
    isDark ? "dark" : "light",
  );
};

// Check and apply appropriate theme based on system/user preference
const checkTheme = () => {
  const palette = JSON.parse(localStorage.getItem(".__palette") || "{}");
  if (palette.index === 0) {
    applyTheme(window.matchMedia("(prefers-color-scheme: dark)").matches);
    syncWidgetTheme();
  }
};

// Initialize theme handling on page load
document.addEventListener("DOMContentLoaded", () => {
  checkTheme();
  syncWidgetTheme();

  // Watch for system theme changes
  window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", checkTheme);

  // Watch for theme toggle changes
  document.getElementById("__palette_1")?.addEventListener("change", (e) => {
    if (e.target.checked) setTimeout(checkTheme);
  });

  // Watch for Material theme changes and sync to widget
  new MutationObserver(syncWidgetTheme).observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"],
  });
});

// Ultralytics Chat Widget ---------------------------------------------------------------------------------------------
let ultralyticsChat = null;

document.addEventListener("DOMContentLoaded", () => {
  ultralyticsChat = new UltralyticsChat({
    branding: {
      name: "Ultralytics AI",
      tagline: "Ask anything about Ultralytics, YOLO, and more",
      logo: "https://cdn.prod.website-files.com/680a070c3b99253410dd3dcf/680a070c3b99253410dd3e13_logo.svg",
      logomark: "https://storage.googleapis.com/organization-image-assets/ultralytics-botAvatarSrcUrl-1729379860806.svg",
      pillText: "Ask AI",
    },
    theme: {
      primary: "#042AFF",
      dark: "#111F68",
      yellow: "#E1FF25",
      text: "#0b0b0f",
    },
    welcome: {
      title: "Hi!",
      message:
        "I'm an AI assistant trained on documentation, guides, and other content.<br>Ask me anything about Ultralytics.",
      examples: [
        "What's new in SAM3?",
        "How can I get started with YOLO?",
        "How does Enterprise Licensing work?",
      ],
    },
    ui: {
      placeholder: "Ask anything…",
      copyText: "Copy thread",
      downloadText: "Download thread",
      clearText: "New chat",
    },
  });

  // Add search bar to header
  const headerElement = document.querySelector(".md-header__inner");
  const searchContainer = headerElement?.querySelector(".md-header__source");

  if (headerElement && searchContainer) {
    const searchBar = document.createElement("div");
    searchBar.className = "ult-header-search";
    searchBar.innerHTML = `
      <button class="ult-search-button" title="Search documentation (⌘K)">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="8"/>
          <path d="m21 21-4.35-4.35"/>
        </svg>
        <span>Search</span>
      </button>
    `;
    headerElement.insertBefore(searchBar, searchContainer);

    searchBar
      .querySelector(".ult-search-button")
      .addEventListener("click", () => {
        ultralyticsChat?.toggle(true, "search");
      });
  }

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (
      (e.metaKey || e.ctrlKey) &&
      e.key === "k" &&
      !/input|textarea/i.test(e.target.tagName) &&
      !e.target.isContentEditable
    ) {
      e.preventDefault();
      ultralyticsChat?.toggle(true, "search");
    }
  });
});

// Fix language switcher links
(function () {
  function fixLanguageLinks() {
    const path = location.pathname;
    const links = document.querySelectorAll(".md-select__link");
    if (!links.length) {
      return;
    }

    const langs = [];
    let defaultLink = null;

    // Extract language codes
    links.forEach((link) => {
      const href = link.getAttribute("href");
      if (!href) {
        return;
      }

      const url = new URL(href, location.origin);
      const match = url.pathname.match(/^\/([a-z]{2})\/?$/);

      if (match) {
        langs.push({ code: match[1], link });
      } else if (url.pathname === "/" || url.pathname === "") {
        defaultLink = link;
      }
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
    if (defaultLink) {
      defaultLink.href = location.origin + basePath;
    }
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
