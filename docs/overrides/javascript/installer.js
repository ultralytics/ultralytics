const options = {
  build: "latest",
  package: "Pip",
};

function updateOption(key, value) {
  // Update the selected option
  options[key] = value;

  // Use a single selector to handle active state efficiently
  document.querySelectorAll(`[data-key="${key}"]`).forEach((btn) =>
    btn.classList.toggle("active", btn.dataset.value === value)
  );

  updateCommand();
}

function updateCommand() {
  // Generate and update the command dynamically
  const { build, package } = options;
  const baseCommand =
    package === "Pip"
      ? "pip3 install ultralytics"
      : "conda install -c conda-forge ultralytics";
  const versionSuffix = build !== "latest" ? (package === "Pip" ? `==${build}` : `=${build}`) : "";

  document.getElementById("command").innerText = `${baseCommand}${versionSuffix}`;
}

// Initialize default selections on DOMContentLoaded
document.addEventListener("DOMContentLoaded", () => {
  updateOption("build", "latest");
  updateOption("package", "Pip");
});
