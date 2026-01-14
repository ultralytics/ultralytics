// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// Block sitemap.xml fetches triggered by Weglot's hreflang tags detected by MkDocs Material
(() => {
  const EMPTY_SITEMAP = `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>`;

  const originalFetch = window.fetch;
  window.fetch = function (url, options) {
    if (typeof url === "string" && url.includes("/sitemap.xml")) {
      return Promise.resolve(
        new Response(EMPTY_SITEMAP, { status: 200, headers: { "Content-Type": "application/xml" } }),
      );
    }
    return originalFetch.apply(this, arguments);
  };

  const originalXHROpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (method, url) {
    if (typeof url === "string" && url.includes("/sitemap.xml")) {
      this._blockRequest = true;
    }
    return originalXHROpen.apply(this, arguments);
  };

  const originalXHRSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.send = function () {
    if (this._blockRequest) {
      Object.defineProperty(this, "status", { value: 200 });
      Object.defineProperty(this, "responseText", { value: EMPTY_SITEMAP });
      Object.defineProperty(this, "response", { value: EMPTY_SITEMAP });
      Object.defineProperty(this, "responseXML", {
        value: new DOMParser().parseFromString(EMPTY_SITEMAP, "application/xml"),
      });
      this.dispatchEvent(new Event("load"));
      return;
    }
    return originalXHRSend.apply(this, arguments);
  };
})();

// Apply theme colors based on dark/light mode
const applyTheme = (isDark) => {
  document.body.setAttribute("data-md-color-scheme", isDark ? "slate" : "default");
  document.body.setAttribute("data-md-color-primary", isDark ? "black" : "indigo");
};

// Sync widget theme with Material theme
const syncWidgetTheme = () => {
  const isDark = document.body.getAttribute("data-md-color-scheme") === "slate";
  document.documentElement.setAttribute("data-theme", isDark ? "dark" : "light");
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
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", checkTheme);

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
document.addEventListener("DOMContentLoaded", () => {
  const ultralyticsChat = new UltralyticsChat({
    welcome: {
      title: "Hello 👋",
      message: "Ask about YOLO, tutorials, training, export, deployment, or troubleshooting.",
      chatExamples: [
        "What's new in SAM 3?",
        "How can I get started with YOLO26?",
        "How does Enterprise Licensing work?",
      ],
      searchExamples: [
        "YOLO26 quickstart",
        "custom dataset training",
        "model export formats",
        "object detection tutorial",
        "hyperparameter tuning",
      ],
    },
  });

  const headerElement = document.querySelector(".md-header__inner");
  const searchContainer = headerElement?.querySelector(".md-header__source");

  if (headerElement && searchContainer) {
    const searchBar = document.createElement("div");
    searchBar.className = "ult-header-search";
    const hotkey = /Mac|iPod|iPhone|iPad/.test(navigator.platform) ? "⌘K" : "Ctrl+K";
    searchBar.innerHTML = `
      <button class="ult-search-button" title="Search documentation (${hotkey})" aria-label="Search documentation">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <circle cx="11" cy="11" r="8"></circle>
          <path d="m21 21-4.35-4.35"></path>
        </svg>
        <span>Search</span>
        <span class="ult-search-hotkey" aria-hidden="true">${hotkey}</span>
      </button>
    `;
    headerElement.insertBefore(searchBar, searchContainer);

    const defaultSearchToggle = headerElement.querySelector('label[for="__search"]');
    const defaultSearchInput = document.getElementById("__search");
    const defaultSearchDialog = document.querySelector(".md-search");
    if (defaultSearchToggle) {
      defaultSearchToggle.setAttribute("aria-hidden", "true");
      defaultSearchToggle.style.display = "none";
    }
    if (defaultSearchInput) {
      defaultSearchInput.setAttribute("tabindex", "-1");
      defaultSearchInput.setAttribute("aria-hidden", "true");
    }
    if (defaultSearchDialog) defaultSearchDialog.style.display = "none";

    searchBar.querySelector(".ult-search-button").addEventListener("click", () => {
      ultralyticsChat?.toggle(true, "search");
    });
  }
});

// Fix language switcher links to preserve current page path, query string, and hash
(() => {
  function fixLanguageLinks() {
    const path = location.pathname;
    const links = document.querySelectorAll(".md-select__link[hreflang]");
    if (!links.length) return;

    // Derive language codes from the actual links (config-driven)
    const langCodes = Array.from(links)
      .map((link) => link.getAttribute("hreflang"))
      .filter(Boolean);
    const defaultLang =
      Array.from(links)
        .find((link) => link.getAttribute("href") === "/")
        ?.getAttribute("hreflang") || "en";

    // Extract base path (without leading slash and language prefix)
    let basePath = path.startsWith("/") ? path.slice(1) : path;
    for (const code of langCodes) {
      if (code === defaultLang) continue;
      const prefix = `${code}/`;
      if (basePath === code || basePath === prefix) {
        basePath = "";
        break;
      }
      if (basePath.startsWith(prefix)) {
        basePath = basePath.slice(prefix.length);
        break;
      }
    }

    // Preserve query string and hash
    const suffix = location.search + location.hash;

    // Update all language links
    links.forEach((link) => {
      const lang = link.getAttribute("hreflang");
      link.href =
        lang === defaultLang
          ? `${location.origin}/${basePath}${suffix}`
          : `${location.origin}/${lang}/${basePath}${suffix}`;
    });
  }

  // Run on load and navigation
  fixLanguageLinks();

  if (typeof document$ !== "undefined") {
    document$.subscribe(() => setTimeout(fixLanguageLinks, 50));
  }
})();
