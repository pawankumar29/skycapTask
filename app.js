const input = document.querySelector("#noteInput");
const dropZone = document.querySelector("#dropZone");
const canvas = document.querySelector("#previewCanvas");
const ctx = canvas.getContext("2d");
const emptyState = document.querySelector("#emptyState");
const statusCard = document.querySelector("#statusCard");
const statusPill = document.querySelector("#statusPill");
const resultTitle = document.querySelector("#resultTitle");
const resultText = document.querySelector("#resultText");
const denominationEl = document.querySelector("#denomination");
const confidenceEl = document.querySelector("#confidence");
const authScoreEl = document.querySelector("#authScore");
const checkList = document.querySelector("#checkList");

input.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) analyzeFile(file);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragging");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("is-dragging"));

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  const [file] = event.dataTransfer.files;
  if (file && file.type.startsWith("image/")) analyzeFile(file);
});

async function analyzeFile(file) {
  setWorkingState();
  await drawPreview(file);

  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || "The image could not be analyzed.");
    }

    renderResult(await response.json());
  } catch (error) {
    renderError(error.message);
  }
}

function drawPreview(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
        const width = img.width * scale;
        const height = img.height * scale;
        const x = (canvas.width - width) / 2;
        const y = (canvas.height - height) / 2;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#e8eee9";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, x, y, width, height);
        emptyState.classList.add("is-hidden");
        resolve();
      };
      img.onerror = () => reject(new Error("This file could not be read as an image."));
      img.src = reader.result;
    };
    reader.onerror = () => reject(new Error("The selected file could not be loaded."));
    reader.readAsDataURL(file);
  });
}

function renderResult(analysis) {
  statusCard.className = "status-card";
  if (analysis.accepted) statusCard.classList.add("pass");
  else statusCard.classList.add("fail");

  statusPill.textContent = analysis.accepted ? "Accepted" : "Rejected";
  resultTitle.textContent = analysis.accepted
    ? `${analysis.denomination} note detected`
    : analysis.isKnownFake
      ? "Fake note rejected"
      : "Image rejected";
  resultText.textContent = analysis.message;
  denominationEl.textContent = analysis.accepted ? analysis.denomination : "Not valid";
  confidenceEl.textContent = `${analysis.confidence}%`;
  authScoreEl.textContent = `${analysis.authenticity}%`;

  checkList.innerHTML = "";
  for (const item of analysis.checks) {
    const li = document.createElement("li");
    li.className = item.passed ? "pass" : item.warning ? "warn" : "fail";
    li.textContent = `${item.title}: ${item.detail}`;
    checkList.appendChild(li);
  }
}

function setWorkingState() {
  statusCard.className = "status-card";
  statusPill.textContent = "Scanning";
  resultTitle.textContent = "Analyzing image";
  resultText.textContent = "Processing with local Python/OpenCV backend.";
  denominationEl.textContent = "-";
  confidenceEl.textContent = "-";
  authScoreEl.textContent = "-";
  checkList.innerHTML = "<li>Reading note features...</li>";
}

function renderError(message) {
  statusCard.className = "status-card fail";
  statusPill.textContent = "Error";
  resultTitle.textContent = "Image not analyzed";
  resultText.textContent = message;
  denominationEl.textContent = "-";
  confidenceEl.textContent = "-";
  authScoreEl.textContent = "-";
  checkList.innerHTML = "<li class=\"fail\">Please upload a valid image file.</li>";
}
