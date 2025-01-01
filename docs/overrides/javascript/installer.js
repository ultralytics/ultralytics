const options = { build: "latest", package: "Pip" };

function updateOption(key, value) {
  options[key] = value;
  document.querySelectorAll(`[data-key="${key}"]`).forEach((btn) =>
    btn.classList.toggle("active", btn.dataset.value === value)
  );
  updateCommand();
}

function updateCommand() {
  const { build, package } = options;
  const version = build !== "latest" ? (package === "Pip" ? `==${build}` : `=${build}`) : "";
  document.getElementById("command").innerText =
    `${package === "Pip" ? "pip3 install ultralytics" : "conda install -c conda-forge ultralytics"}${version}`;
}

document.addEventListener("DOMContentLoaded", () => {
  ["build", "package"].forEach((key) => updateOption(key, options[key]));
});
