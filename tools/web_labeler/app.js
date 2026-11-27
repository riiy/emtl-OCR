(() => {
  "use strict";
  const token = document.querySelector('meta[name="labeler-token"]').content;
  const el = (id) => document.getElementById(id);
  const ui = {
    count: el("progress-count"), detail: el("progress-detail"), bar: el("progress-bar"),
    index: el("image-index"), status: el("image-status"), image: el("captcha-image"),
    placeholder: el("image-placeholder"), name: el("image-name"), input: el("label-input"),
    submit: el("submit-button"), skip: el("skip-button"), previous: el("previous-button"),
    next: el("next-button"), capacity: el("capacity-note"), message: el("message"),
    saveState: el("save-state")
  };
  let state = null;
  let index = 0;
  let dirty = false;
  let savedInput = "";
  let busy = false;

  async function request(path, options = {}) {
    const headers = { "X-Labeler-Token": token, ...(options.headers || {}) };
    const response = await fetch(path, { cache: "no-store", ...options, headers });
    const type = response.headers.get("content-type") || "";
    const body = type.includes("application/json") ? await response.json() : await response.blob();
    if (!response.ok) throw new Error(body.error || ("HTTP " + String(response.status)));
    return body;
  }

  function current() { return state.items[index]; }
  function setMessage(message, kind = "") {
    ui.message.textContent = message;
    ui.message.className = "message " + kind;
  }
  function updateSummary() {
    const s = state.summary;
    ui.count.textContent = String(s.labeled_count) + " / " + String(s.target_labels) + " 已标注";
    ui.detail.textContent = String(s.skipped_count) + " 已跳过 · " + String(s.processed_count) + " 张已处理 · " + String(s.event_count) + " 条操作记录";
    ui.bar.style.width = String(Math.min(100, s.labeled_count / s.target_labels * 100)) + "%";
    ui.capacity.hidden = !s.at_capacity;
  }
  function updateDirty() {
    dirty = ui.input.value !== savedInput;
    ui.saveState.textContent = dirty ? "有未提交输入" : "所有更改实时保存到本地 JSONL";
    ui.saveState.classList.toggle("unsaved", dirty);
  }
  function render() {
    const item = current();
    if (!item) return;
    ui.index.textContent = String(index + 1) + " / " + String(state.items.length);
    ui.name.textContent = item.path;
    ui.status.textContent = item.status === "labeled" ? "已标注 · 可修改" : item.status === "skipped" ? "已跳过 · 可重新标注" : "未标注";
    ui.status.className = "status-pill " + item.status;
    ui.image.hidden = true;
    ui.placeholder.hidden = false;
    ui.placeholder.textContent = "正在读取图片…";
    ui.image.onload = () => { ui.image.hidden = false; ui.placeholder.hidden = true; };
    ui.image.onerror = () => { ui.placeholder.textContent = "图片读取失败，请检查本地归档"; };
    ui.image.src = "/api/image/" + encodeURIComponent(item.id) + "?v=" + String(Date.now());
    ui.input.value = item.status === "labeled" ? item.label : "";
    savedInput = ui.input.value;
    updateDirty();
    const atCapacity = state.summary.at_capacity;
    const canCreateLabel = item.status === "labeled" || !atCapacity;
    ui.submit.disabled = busy || !canCreateLabel;
    ui.skip.disabled = busy || (atCapacity && item.status !== "labeled");
    ui.input.disabled = busy || !canCreateLabel;
    ui.previous.disabled = busy || index === 0;
    ui.next.disabled = busy || index === state.items.length - 1;
    updateSummary();
    setMessage(item.status === "labeled" ? "可修改当前标签；提交后会追加一条修订记录。" : item.status === "skipped" ? "该图片之前已跳过，可以重新输入并提交。" : "请仅根据图片本身进行人工识别。");
  }
  function mayLeave() {
    if (!dirty) return true;
    return window.confirm("当前输入尚未提交。离开将丢弃这次输入，是否继续？");
  }
  function navigate(nextIndex) {
    if (nextIndex < 0 || nextIndex >= state.items.length || !mayLeave()) return;
    index = nextIndex;
    render();
  }
  async function submit(status) {
    if (busy) return;
    if (status === "skipped" && dirty && !window.confirm("当前输入尚未提交。跳过将丢弃这次输入，是否继续？")) return;
    const item = current();
    const label = status === "labeled" ? ui.input.value : null;
    if (status === "labeled" && !/^[0-9]{4}$/.test(label)) {
      setMessage("请输入恰好四位 ASCII 数字，例如 0042。", "error");
      ui.input.focus();
      return;
    }
    busy = true;
    ui.submit.disabled = ui.skip.disabled = true;
    let feedback = null;
    let feedbackKind = "";
    try {
      const nextState = await request("/api/event", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_id: item.id, status, label })
      });
      state = nextState;
      savedInput = status === "labeled" ? label : "";
      dirty = false;
      ui.saveState.textContent = "已保存并同步到本地 JSONL";
      feedback = status === "labeled" ? "标注已保存。" : "已记录跳过。";
      feedbackKind = "success";
      if (status === "labeled" && index < state.items.length - 1 && !state.summary.at_capacity) {
        index += 1;
      } else if (status === "skipped" && index < state.items.length - 1) {
        index += 1;
      }
      render();
    } catch (error) {
      feedback = error.message;
      feedbackKind = "error";
    } finally {
      busy = false;
      render();
      if (feedback) setMessage(feedback, feedbackKind);
    }
  }

  ui.input.addEventListener("input", () => {
    const cleaned = ui.input.value.replace(/[^0-9]/g, "").slice(0, 4);
    if (cleaned !== ui.input.value) ui.input.value = cleaned;
    updateDirty();
  });
  ui.submit.addEventListener("click", () => submit("labeled"));
  ui.skip.addEventListener("click", () => submit("skipped"));
  ui.previous.addEventListener("click", () => navigate(index - 1));
  ui.next.addEventListener("click", () => navigate(index + 1));
  ui.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); submit("labeled"); }
  });
  window.addEventListener("beforeunload", (event) => {
    if (dirty) { event.preventDefault(); event.returnValue = ""; }
  });

  (async () => {
    try {
      state = await request("/api/state");
      index = state.resume_index;
      render();
      ui.input.focus();
    } catch (error) {
      ui.placeholder.textContent = "无法加载标注状态：" + error.message;
      setMessage(error.message, "error");
    }
  })();
})();
