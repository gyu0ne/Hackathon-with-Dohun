const state = { file: null, result: null, previewUrl: null };

const $ = (id) => document.getElementById(id);
const form = $("uploadForm");
const fileInput = $("fileInput");
const dropzone = $("dropzone");

function selectFile(file) {
  if (!file || !["image/jpeg", "image/png"].includes(file.type)) {
    showError("JPEG 또는 PNG 이미지만 선택할 수 있습니다.");
    return;
  }
  state.file = file;
  $("fileName").textContent = file.name;
  $("analyzeButton").disabled = false;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  $("previewImage").src = state.previewUrl;
  hideError();
}

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
}));
dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

function setBusy(busy) {
  $("loading").classList.toggle("hidden", !busy);
  $("analyzeButton").disabled = busy || !state.file;
  if (busy) $("results").classList.add("hidden");
}

function showError(message) {
  $("errorBox").textContent = message;
  $("errorBox").classList.remove("hidden");
}

function hideError() {
  $("errorBox").classList.add("hidden");
}

async function apiError(response) {
  try {
    const body = await response.json();
    return body.detail || "요청을 처리하지 못했습니다.";
  } catch {
    return "서버와 통신하지 못했습니다.";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.file) return;
  setBusy(true);
  hideError();
  const data = new FormData();
  data.append("file", state.file);
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: data });
    if (!response.ok) throw new Error(await apiError(response));
    state.result = await response.json();
    renderResult();
  } catch (error) {
    showError(error.message || "문서 분석 중 오류가 발생했습니다.");
  } finally {
    setBusy(false);
  }
});

function citationButtons(ids) {
  const wrapper = document.createElement("span");
  wrapper.className = "citation-list";
  ids.forEach((id) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "citation";
    button.textContent = `근거 ${id}`;
    button.addEventListener("click", () => highlightLines(ids));
    wrapper.appendChild(button);
  });
  return wrapper;
}

function citedText(text, ids, className = "") {
  const node = document.createElement("div");
  if (className) node.className = className;
  const content = document.createElement("span");
  content.textContent = text;
  node.append(content, citationButtons(ids));
  return node;
}

function renderFacts(container, facts) {
  container.replaceChildren();
  facts.forEach((fact) => container.appendChild(citedText(fact.text, fact.source_line_ids, "fact")));
}

function highlightLines(ids) {
  const result = state.result;
  if (!result?.ocr) return;
  $("overlay").replaceChildren();
  document.querySelectorAll(".ocr-line").forEach((line) => line.classList.remove("active"));
  ids.forEach((id) => {
    const line = result.ocr.lines.find((candidate) => candidate.id === id);
    if (!line) return;
    const box = document.createElement("div");
    box.className = "highlight-box";
    box.style.left = `${100 * line.bbox.x / result.image.width}%`;
    box.style.top = `${100 * line.bbox.y / result.image.height}%`;
    box.style.width = `${100 * line.bbox.width / result.image.width}%`;
    box.style.height = `${100 * line.bbox.height / result.image.height}%`;
    $("overlay").appendChild(box);
    document.querySelector(`[data-line-id="${id}"]`)?.classList.add("active");
  });
  document.querySelector(`[data-line-id="${ids[0]}"]`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderOcr(ocr) {
  const list = $("ocrList");
  list.replaceChildren();
  ocr.lines.forEach((line) => {
    const row = document.createElement("div");
    row.className = "ocr-line";
    row.dataset.lineId = line.id;
    row.innerHTML = `<span class="line-id">${line.id}</span><span></span><span class="confidence">${line.confidence.toFixed(0)}%</span>`;
    row.children[1].textContent = line.text;
    row.addEventListener("click", () => highlightLines([line.id]));
    list.appendChild(row);
  });
  $("ocrCount").textContent = `${ocr.lines.length}개 줄`;
}

function renderSummary(summary) {
  $("summaryTitle").textContent = summary.title;
  $("overview").replaceChildren(citedText(summary.overview, summary.overview_source_line_ids));
  renderFacts($("keyPoints"), summary.key_points);
  renderFacts($("actions"), summary.actions);
  $("keyPointSection").classList.toggle("hidden", !summary.key_points.length);
  $("actionSection").classList.toggle("hidden", !summary.actions.length);
  const generated = summary.status === "generated";
  $("aiBadge").textContent = generated ? `Gemini · ${summary.model}` : "안전 대체 요약";
  $("aiBadge").className = `badge ${generated ? "info" : "warning"}`;
}

function renderResult() {
  const result = state.result;
  const passed = result.quality.status === "pass";
  $("results").classList.remove("hidden");
  $("qualityBadge").textContent = passed ? "읽기 적합" : "재촬영 권장";
  $("qualityBadge").className = `badge ${passed ? "success" : "warning"}`;
  $("blurMetric").textContent = result.quality.metrics.focus_score.toFixed(1);
  $("ssimMetric").textContent = `${result.quality.metrics.text_contrast.toFixed(1)}%`;
  $("totalMetric").textContent = `${(result.timings.total_ms / 1000).toFixed(1)}초`;
  $("retakePanel").classList.toggle("hidden", passed);
  $("ocrPanel").classList.toggle("hidden", !result.ocr);
  $("summaryPanel").classList.toggle("hidden", !result.summary);
  $("feedbackPanel").classList.toggle("hidden", !result.summary);
  if (result.ocr) renderOcr(result.ocr);
  if (result.summary) renderSummary(result.summary);
  $("overlay").replaceChildren();
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function sendFeedback(accepted, correctedText = null) {
  if (!state.result) return;
  const response = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis_id: state.result.analysis_id, accepted, corrected_text: correctedText }),
  });
  if (!response.ok) throw new Error(await apiError(response));
  $("feedbackMessage").textContent = "의견이 저장되었습니다. 감사합니다.";
  $("feedbackMessage").classList.remove("hidden");
  $("correctionBox").classList.add("hidden");
}

$("acceptButton").addEventListener("click", () => sendFeedback(true).catch((error) => showError(error.message)));
$("rejectButton").addEventListener("click", () => $("correctionBox").classList.remove("hidden"));
$("sendCorrection").addEventListener("click", () => {
  const text = $("correctedText").value.trim();
  if (!text) return showError("수정할 내용을 입력해 주세요.");
  sendFeedback(false, text).catch((error) => showError(error.message));
});
