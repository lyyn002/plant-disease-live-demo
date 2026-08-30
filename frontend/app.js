const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const predictBtn = document.getElementById("predict-btn");
const preview = document.getElementById("preview");
const results = document.getElementById("results");
const primaryResult = document.getElementById("primary-result");
const topKList = document.getElementById("top-k");
const latencyEl = document.getElementById("latency");
const errorEl = document.getElementById("error");

let selectedFile = null;

function setError(message) {
  if (!message) {
    errorEl.classList.add("hidden");
    errorEl.textContent = "";
    return;
  }
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function formatConfidence(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function renderPrediction(item, isPrimary = false) {
  const confidencePct = item.confidence * 100;
  if (isPrimary) {
    primaryResult.innerHTML = `
      <div class="disease">${item.disease}</div>
      <div class="meta">${item.crop} · ${item.label.replaceAll("_", " ")}</div>
      <div class="meta">Confidence: ${formatConfidence(item.confidence)}</div>
      <div class="confidence-bar"><div class="confidence-fill" style="width:${confidencePct}%"></div></div>
    `;
    return;
  }

  const li = document.createElement("li");
  li.innerHTML = `<span>${item.disease} <small>(${item.crop})</small></span><strong>${formatConfidence(item.confidence)}</strong>`;
  topKList.appendChild(li);
}

function handleFile(file) {
  if (!file) {
    return;
  }
  selectedFile = file;
  predictBtn.disabled = false;
  preview.src = URL.createObjectURL(file);
  results.classList.remove("hidden");
  setError("");
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const file = event.dataTransfer.files[0];
  handleFile(file);
});

fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = "Analyzing...";
  setError("");

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/api/v1/predict", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    topKList.innerHTML = "";
    renderPrediction(payload.primary, true);
    payload.top_k.forEach((item) => renderPrediction(item, false));
    latencyEl.textContent = `Server inference: ${payload.inference_ms} ms`;
  } catch (error) {
    setError(error.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Analyze leaf";
  }
});
