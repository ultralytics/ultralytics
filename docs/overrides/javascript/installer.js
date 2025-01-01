const options = {
  build: "latest",
  package: "Pip",
};

function updateOption(key, value) {
  // Update the selection and apply the active class
  options[key] = value;

  // Remove active class from all buttons in the current group
  const buttons = document.querySelectorAll(`[data-key="${key}"]`);
  buttons.forEach((btn) => btn.classList.remove("active"));

  // Add active class to the selected button
  const selectedButton = document.querySelector(
    `[data-key="${key}"][data-value="${value}"]`,
  );
  if (selectedButton) {
    selectedButton.classList.add("active");
  }
  updateCommand();
}

function updateCommand() {
  // Generate and update the command dynamically
  const { build, package } = options;
  let command = "";

  if (package === "Pip") {
    command = `pip3 install ultralytics`;
    if (build !== "latest") {
      command += `==${build}`;
    }
  } else if (package === "Conda") {
    command = `conda install -c conda-forge ultralytics`;
    if (build !== "latest") {
      command += `=${build}`;
    }
  }

  document.getElementById("command").innerText = command; // Update the displayed command
}

document.addEventListener("DOMContentLoaded", () => {
  // Initialize default selections
  updateOption("build", "latest");
  updateOption("package", "Pip");
});
