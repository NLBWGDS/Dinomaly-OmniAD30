const state = { config: null, category: null, samples: [], index: 0, view: "overlay", renderToken: 0 };
const $ = (id) => document.getElementById(id);

async function getJSON(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function metric(value) {
  return value === null || value === undefined ? "--" : Number(value).toFixed(4);
}

function showError(error) {
  const toast = $("errorToast");
  toast.textContent = error.message || String(error);
  toast.classList.add("visible");
  setTimeout(() => toast.classList.remove("visible"), 5000);
}

function activeCategory() {
  return state.config.categories.find((item) => item.name === state.category);
}

function renderCategories(filter = "") {
  const needle = filter.trim().toLowerCase();
  const list = $("categoryList");
  list.innerHTML = "";
  state.config.categories.filter((item) => item.name.toLowerCase().includes(needle)).forEach((item) => {
    const button = document.createElement("button");
    button.className = `category-item${item.name === state.category ? " active" : ""}`;
    button.innerHTML = `<span>${item.name}</span><em>${metric(item.metrics.pixel_f1)}</em><small>${item.images} 张 · Pixel F1</small>`;
    button.addEventListener("click", () => selectCategory(item.name));
    list.appendChild(button);
  });
}

async function selectCategory(category) {
  state.category = category;
  state.index = 0;
  state.samples = await getJSON(`/api/samples?category=${encodeURIComponent(category)}`);
  const info = activeCategory();
  $("categoryTitle").textContent = category;
  $("sampleMeta").textContent = `${info.route} · ${info.images} 张评估图像 · ${info.anomalies} 张异常样本`;
  $("pixelF1").textContent = metric(info.metrics.pixel_f1);
  $("imageF1").textContent = metric(info.metrics.image_f1);
  $("pixelAupro").textContent = metric(info.metrics.pixel_aupro);
  $("sampleRange").max = Math.max(state.samples.length - 1, 0);
  renderCategories($("categorySearch").value);
  renderSample();
}

function renderSample() {
  const sample = state.samples[state.index];
  if (!sample) return;
  $("sampleName").textContent = sample.name;
  $("sampleIndex").textContent = state.index + 1;
  $("sampleTotal").textContent = state.samples.length;
  $("sampleRange").value = state.index;
  $("imageScore").textContent = sample.score.toFixed(4);
  $("sampleLabel").textContent = sample.is_anomaly ? "异常样本" : "正常样本";
  $("sampleLabel").classList.toggle("good", !sample.is_anomaly);
  const image = $("resultImage");
  const loading = $("loading");
  const token = ++state.renderToken;
  loading.classList.remove("hidden");
  image.onload = () => {
    if (token !== state.renderToken) return;
    loading.classList.add("hidden");
    if (state.samples.length > 1) {
      const nextIndex = (state.index + 1) % state.samples.length;
      const preload = new Image();
      preload.src = `/api/render?category=${encodeURIComponent(state.category)}&index=${nextIndex}&view=${state.view}`;
    }
  };
  image.onerror = () => {
    if (token !== state.renderToken) return;
    loading.classList.add("hidden");
    showError(new Error("图像生成失败"));
  };
  image.src = `/api/render?category=${encodeURIComponent(state.category)}&index=${state.index}&view=${state.view}`;
}

function move(delta) {
  if (!state.samples.length) return;
  state.index = (state.index + delta + state.samples.length) % state.samples.length;
  renderSample();
}

async function initialize() {
  try {
    state.config = await getJSON("/api/state");
    $("categoryCount").textContent = state.config.categories.length;
    const mean = state.config.mean;
    $("scheme").textContent = `${state.config.scheme} · ${state.config.images} 张 · 全局 PF1 ${metric(mean.pixel_f1)} / IF1 ${metric(mean.image_f1)} / AUPRO ${metric(mean.pixel_aupro)}`;
    await selectCategory(state.config.categories[0].name);
  } catch (error) {
    showError(error);
    $("categoryTitle").textContent = "结果载入失败";
    $("sampleMeta").textContent = error.message;
  }
}

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-view]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.view = button.dataset.view;
    renderSample();
  });
});
$("previous").addEventListener("click", () => move(-1));
$("next").addEventListener("click", () => move(1));
$("sampleRange").addEventListener("input", (event) => { state.index = Number(event.target.value); renderSample(); });
$("categorySearch").addEventListener("input", (event) => renderCategories(event.target.value));
document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "INPUT" && event.target.type === "search") return;
  if (event.key === "ArrowLeft") move(-1);
  if (event.key === "ArrowRight") move(1);
  const views = { "1": "original", "2": "mask", "3": "heatmap", "4": "overlay" };
  if (views[event.key]) document.querySelector(`[data-view="${views[event.key]}"]`).click();
});

initialize();
