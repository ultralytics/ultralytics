const options = { build: "latest", package: "Pip" };

const updateOption = (key, value) => {
  options[key] = value;
  document
    .querySelectorAll(`[data-key="${key}"]`)
    .forEach((btn) =>
      btn.classList.toggle("active", btn.dataset.value === value),
    );
  updateCommand();
};

const updateCommand = () => {
  const { build, package } = options;
  document.getElementById("command").innerText =
    `${package === "Pip" ? "pip3 install ultralytics" : "conda install -c conda-forge ultralytics"}${build !== "latest" ? (package === "Pip" ? `==${build}` : `=${build}`) : ""}`;
};

document.addEventListener("DOMContentLoaded", () =>
  Object.keys(options).forEach((key) => updateOption(key, options[key])),
);
