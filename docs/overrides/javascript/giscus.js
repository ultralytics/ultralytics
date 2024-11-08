// Giscus functionality
function loadGiscus() {
  const giscusContainer = document.getElementById("giscus-container");
  if (!giscusContainer || giscusContainer.querySelector("script")) {
    return;
  }

  const script = document.createElement("script");
  script.src = "https://giscus.app/client.js";
  script.setAttribute("data-repo", "ultralytics/ultralytics");
  script.setAttribute("data-repo-id", "R_kgDOH-jzvQ");
  script.setAttribute("data-category", "Docs");
  script.setAttribute("data-category-id", "DIC_kwDOH-jzvc4CWLkL");
  script.setAttribute("data-mapping", "pathname");
  script.setAttribute("data-strict", "1");
  script.setAttribute("data-reactions-enabled", "1");
  script.setAttribute("data-emit-metadata", "0");
  script.setAttribute("data-input-position", "top");
  script.setAttribute("data-theme", "preferred_color_scheme");
  script.setAttribute("data-lang", "en");
  script.setAttribute("data-loading", "lazy");
  script.setAttribute("crossorigin", "anonymous");
  script.setAttribute("async", "");

  giscusContainer.appendChild(script);

  // Synchronize Giscus theme with palette
  var palette = __md_get("__palette");
  if (palette && typeof palette.color === "object") {
    var theme = palette.color.scheme === "slate" ? "dark" : "light";
    script.setAttribute("data-theme", theme);
  }

  // Register event handlers for theme changes
  var ref = document.querySelector("[data-md-component=palette]");
  if (ref) {
    ref.addEventListener("change", function () {
      var palette = __md_get("__palette");
      if (palette && typeof palette.color === "object") {
        var theme = palette.color.scheme === "slate" ? "dark" : "light";

        // Instruct Giscus to change theme
        var frame = document.querySelector(".giscus-frame");
        if (frame) {
          frame.contentWindow.postMessage(
            { giscus: { setConfig: { theme } } },
            "https://giscus.app",
          );
        }
      }
    });
  }
}

// Use Intersection Observer to load Giscus when the container is visible
function setupGiscusLoader() {
  const giscusContainer = document.getElementById("giscus-container");

  if (giscusContainer) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          loadGiscus();
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 }); // Trigger when 10% of the element is visible

    observer.observe(giscusContainer);
  }
}

// Hook into MkDocs' navigation system
if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    // This function is called on every page load/change
    setupGiscusLoader();
  });
} else {
  console.warn("MkDocs document$ not found. Falling back to DOMContentLoaded.");
  document.addEventListener("DOMContentLoaded", setupGiscusLoader);
}
