const state = { file: null, result: null, previewUrl: null };
const $ = (id) => document.getElementById(id);

function selectFile(file) {
  if (!file || !["image/jpeg", "image/png"].includes(file.type)) return showError("JPEG 또는 PNG 이미지를 선택해 주세요.");
  state.file = file;
  $("fileName").textContent = file.name;
  $("compareButton").disabled = false;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  $("previewImage").src = state.previewUrl;
  $("errorBox").classList.add("hidden");
}

const dropzone = $("dropzone");
$("fileInput").addEventListener("change", (event) => selectFile(event.target.files[0]));
["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault(); dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault(); dropzone.classList.remove("dragging");
}));
dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

function showError(message) {
  $("errorBox").textContent = message;
  $("errorBox").classList.remove("hidden");
}

async function apiError(response) {
  try { return (await response.json()).detail || "비교를 실행하지 못했습니다."; }
  catch { return "서버와 통신하지 못했습니다."; }
}

$("compareForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.file) return;
  $("loading").classList.remove("hidden");
  $("comparisonResults").classList.add("hidden");
  $("compareButton").disabled = true;
  const data = new FormData(); data.append("file", state.file);
  try {
    const response = await fetch("/api/compare", { method: "POST", body: data });
    if (!response.ok) throw new Error(await apiError(response));
    state.result = await response.json();
    render();
  } catch (error) { showError(error.message || "비교 중 오류가 발생했습니다."); }
  finally { $("loading").classList.add("hidden"); $("compareButton").disabled = false; }
});

function row(label, ours, photoFilter, baseline) {
  const tr = document.createElement("tr");
  [label, ours, photoFilter, baseline].forEach((value) => { const td = document.createElement("td"); td.textContent = value; tr.appendChild(td); });
  return tr;
}

function renderMetrics() {
  const { ours, photo_filter: photoFilter, baseline } = state.result;
  const filterStrategy = photoFilter.ocr?.strategy || "실행 안 함";
  const body = $("metricsBody"); body.replaceChildren();
  body.append(
    row("선택 OCR 전략", ours.ocr?.strategy || "실행 안 함", filterStrategy, baseline.ocr.strategy),
    row("OCR 글자 수", `${ours.metrics.character_count}자`, `${photoFilter.metrics.character_count}자`, `${baseline.metrics.character_count}자`),
    row("모델 내부 확신도", `${ours.metrics.average_confidence.toFixed(1)}%`, `${photoFilter.metrics.average_confidence.toFixed(1)}%`, `${baseline.metrics.average_confidence.toFixed(1)}%`),
    row("OCR 시간", `${ours.metrics.ocr_ms.toFixed(0)}ms`, `${photoFilter.metrics.ocr_ms.toFixed(0)}ms`, `${baseline.metrics.ocr_ms.toFixed(0)}ms`),
    row("AI 시간", `${(ours.metrics.ai_ms / 1000).toFixed(1)}초`, "실행 안 함", `${(baseline.metrics.ai_ms / 1000).toFixed(1)}초`),
    row("전체 시간", `${(ours.metrics.total_ms / 1000).toFixed(1)}초`, `${(photoFilter.metrics.total_ms / 1000).toFixed(1)}초`, `${(baseline.metrics.total_ms / 1000).toFixed(1)}초`),
    row("근거 인용률", ours.metrics.citation_coverage == null ? "-" : `${ours.metrics.citation_coverage.toFixed(1)}%`, "OCR만 비교", "제공 안 함"),
  );
}

function highlight(ids) {
  const result = state.result; if (!result?.ours.ocr) return;
  $("overlay").replaceChildren();
  ids.forEach((id) => {
    const line = result.ours.ocr.lines.find((item) => item.id === id); if (!line) return;
    const box = document.createElement("div"); box.className = "highlight-box";
    box.style.left = `${100 * line.bbox.x / result.image.width}%`;
    box.style.top = `${100 * line.bbox.y / result.image.height}%`;
    box.style.width = `${100 * line.bbox.width / result.image.width}%`;
    box.style.height = `${100 * line.bbox.height / result.image.height}%`;
    $("overlay").appendChild(box);
  });
}

function citedFact(text, ids, className = "fact") {
  const node = document.createElement("div"); node.className = className;
  const content = document.createElement("span"); content.textContent = text; node.appendChild(content);
  const citations = document.createElement("span"); citations.className = "citation-list";
  ids.forEach((id) => { const button = document.createElement("button"); button.className = "citation"; button.type = "button"; button.textContent = `근거 ${id}`; button.onclick = () => highlight(ids); citations.appendChild(button); });
  node.appendChild(citations); return node;
}

function renderOcr(container, ocr, clickable = false) {
  container.replaceChildren();
  if (!ocr) { container.textContent = "이 경로를 실행하지 못했습니다."; return; }
  if (!ocr.lines.length) { container.textContent = "인식된 텍스트가 없습니다."; return; }
  ocr.lines.forEach((line) => {
    const item = document.createElement("div"); item.className = "ocr-line";
    item.innerHTML = `<span class="line-id">${line.id}</span><span></span><span class="confidence">${line.confidence.toFixed(0)}%</span>`;
    item.children[1].textContent = line.text;
    if (clickable) item.onclick = () => highlight([line.id]);
    container.appendChild(item);
  });
}

function renderOurs() {
  const ours = state.result.ours; const container = $("oursSummary"); container.replaceChildren();
  if (!ours.cited_summary) {
    const note = document.createElement("div"); note.className = "notice"; note.textContent = "품질 기준 미달로 잘못된 요약 생성을 중단했습니다."; container.appendChild(note);
    $("oursBadge").textContent = "품질 보호 작동"; $("oursBadge").className = "badge warning"; return;
  }
  const summary = ours.cited_summary;
  const title = document.createElement("h3"); title.className = "summary-title"; title.textContent = summary.title;
  container.append(title, citedFact(summary.overview, summary.overview_source_line_ids, "overview"));
  summary.key_points.forEach((fact) => container.appendChild(citedFact(fact.text, fact.source_line_ids)));
  $("oursBadge").textContent = summary.status === "generated" ? "Gemini + 근거 검증" : "안전 대체 요약";
}

function render() {
  const result = state.result; const passed = result.quality.status === "pass";
  $("qualityBadge").textContent = passed ? "읽기 적합" : "재촬영 권장";
  $("qualityBadge").className = `badge ${passed ? "success" : "warning"}`;
  renderMetrics(); renderOurs();
  const filterReady = result.photo_filter.status === "completed";
  $("filterBadge").textContent = filterReady ? "언워핑 적용" : "실행 불가";
  $("filterBadge").className = `badge ${filterReady ? "info" : "warning"}`;
  $("baselineSummary").textContent = result.baseline.plain_summary || "요약이 없습니다.";
  renderOcr($("oursOcr"), result.ours.ocr, true);
  renderOcr($("filterOcr"), result.photo_filter.ocr, false);
  renderOcr($("baselineOcr"), result.baseline.ocr, false);
  $("comparisonResults").classList.remove("hidden");
  $("comparisonResults").scrollIntoView({ behavior: "smooth", block: "start" });
}
