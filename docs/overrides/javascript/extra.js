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
  // Retrieve the palette from local storage
  const palette = localStorage.getItem("/.__palette");
  if (palette) {
    // Check if the palette's index is 0 (auto theme)
    const paletteObj = JSON.parse(palette);
    if (paletteObj && paletteObj.index === 0) {
      applyAutoTheme();
    }
  }
}
// ! No need to run the function when the script loads as by default the theme is determined by the user's preference (if auto theme is enabled)
// checkAutoTheme();
// Run the function when the user's preference changes (when the user changes their system theme)
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
