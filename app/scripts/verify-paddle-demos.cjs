// Browser-level checks using a dedicated headless Chrome CDP endpoint.
// node app/scripts/verify-paddle-demos.cjs http://127.0.0.1:9234
// Requires Node with the built-in WebSocket client (the demo machine uses 24).
const fs = require("node:fs");
const path = require("node:path");

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const origin = process.argv[2] || "http://127.0.0.1:9234";
const appUrl = process.argv[3] || "http://127.0.0.1:3200";
const output = path.resolve(__dirname, "../../artifacts");

class Cdp {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.seq = 0;
    this.pending = new Map();
    this.errors = [];
    this.consoleErrors = [];
    this.opened = new Promise((resolve, reject) => {
      this.ws.addEventListener("open", resolve, { once: true });
      this.ws.addEventListener("error", reject, { once: true });
    });
    this.ws.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.method === "Runtime.exceptionThrown") this.errors.push(message.params.exceptionDetails);
      if (message.method === "Runtime.consoleAPICalled" && message.params.type === "error") {
        this.consoleErrors.push(message.params.args.map((arg) => arg.value || arg.description).join(" "));
      }
      if (message.id && this.pending.has(message.id)) {
        const { resolve, reject, timer } = this.pending.get(message.id);
        clearTimeout(timer);
        this.pending.delete(message.id);
        message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result);
      }
    });
  }
  call(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = ++this.seq;
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 10000);
      this.pending.set(id, { resolve, reject, timer });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  async evaluate(expression) {
    const result = await this.call("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  }
  async wait(expression, timeout = 45000) {
    const end = Date.now() + timeout;
    while (Date.now() < end) {
      // Do not serialize a matching DOM node (React attaches deep objects).
      if (await this.evaluate(`Boolean(${expression})`)) return;
      await sleep(200);
    }
    throw new Error(`UI wait timeout: ${expression}`);
  }
  async click(label) {
    const expr = `(() => {
      const button = [...document.querySelectorAll('button')].find(b => b.innerText.includes(${JSON.stringify(label)}) && !b.disabled);
      if (!button) return false;
      button.click(); return true;
    })()`;
    await this.wait(expr, 15000);
  }
  async fill(selector, text) {
    await this.call("Page.bringToFront");
    const found = await this.evaluate(`(() => {
      const input = document.querySelector(${JSON.stringify(selector)});
      if (!input) return false;
      input.focus(); input.select(); return true;
    })()`);
    if (!found) throw new Error(`Missing input ${selector}`);
    await this.call("Input.insertText", { text });
  }
  async screenshot(filename) {
    const shot = await this.call("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
    fs.writeFileSync(path.join(output, filename), Buffer.from(shot.data, "base64"));
  }
}

async function main() {
  fs.mkdirSync(output, { recursive: true });
  const target = await (await fetch(`${origin}/json/new?about:blank`, { method: "PUT" })).json();
  const cdp = new Cdp(target.webSocketDebuggerUrl);
  await cdp.opened;
  const rows = [];
  try {
    await cdp.call("Page.enable");
    await cdp.call("Runtime.enable");
    await cdp.call("Page.bringToFront");
    await cdp.call("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1300, deviceScaleFactor: 1, mobile: false });
    for (const [name, label, helper] of [
      ["densenet", "DenseNet-121", "modified_KaimingNormal.py"],
      ["inception", "Inception-v3", "modified_TruncatedNormal.py"],
    ]) {
      await cdp.call("Page.navigate", { url: appUrl });
      await cdp.click("无人机模型部署适配");
      await cdp.click("PyTorch → PaddlePaddle");
      await cdp.click(label);
      const sampleBadge = await cdp.evaluate(`(() => {
        const button = [...document.querySelectorAll('button')].find(b => b.innerText.includes(${JSON.stringify(label)}));
        return button?.innerText.includes('底层库改写');
      })()`);
      if (!sampleBadge) throw new Error(`Missing underlying-library sample badge: ${name}`);
      // Observe the actual UI request without changing request/response data.
      await cdp.evaluate(`(() => {
        const originalFetch = window.fetch;
        window.__paddleTaskId = null;
        window.fetch = async (...args) => {
          const response = await originalFetch(...args);
          if (String(args[0]).endsWith('/generate') && response.ok) {
            window.__paddleTaskId = (await response.clone().json()).task_id;
          }
          return response;
        };
      })()`);
      const started = Date.now();
      await cdp.click("开始生成");
      await cdp.wait(`document.querySelector('[data-testid="paddle-artifact"]')?.innerText.includes(${JSON.stringify(helper)})`);
      const result = await cdp.evaluate("fetch('/api/result/' + window.__paddleTaskId).then(r => r.json())");
      const runtime = result.evaluation?.paddle_runtime;
      if (result.pre_generated || !runtime?.forward_ok || !runtime.output_shape_ok) throw new Error(`Bad live result for ${name}`);
      if (!result.files.some((f) => f.name === helper)) throw new Error(`Missing helper ${helper}`);
      await cdp.click(helper);
      await cdp.wait(`document.querySelector('pre code')?.textContent.includes(${JSON.stringify(helper.includes("Kaiming") ? "class Modified_KaimingNormal" : "class Modified_TruncatedNormal")})`);
      const text = await cdp.evaluate("document.body.innerText");
      for (const expected of ["Paddle 评估过程", "ModelX CMC 源码转换", "Paddle 导入与 dummy 前向",
        "语法/迁移/结构静态评测", "ModelX 底层 Python 库改写", "伴随 Python 模块语法合法",
        "Paddle 模型构造成功", "Paddle dummy 前向成功"]) {
        if (!text.includes(expected)) throw new Error(`Missing UI label ${expected}`);
      }
      const task = await cdp.evaluate("fetch('/api/task/' + window.__paddleTaskId).then(r => r.json())");
      const steps = task.intermediate?.evaluation_steps;
      if (!steps || steps.length !== 3 || steps.some((step) => step.status !== "passed")) {
        throw new Error(`Incomplete evaluation stages for ${name}: ${JSON.stringify(steps)}`);
      }
      const screenshot = `paddle-${name}-library-validation.png`;
      await cdp.screenshot(screenshot);
      rows.push({ name, label, elapsed_ms: Date.now() - started, pre_generated: result.pre_generated,
        class_name: runtime.class_name, output_shape: runtime.output_shape,
        evaluation_steps: steps, helpers: result.evaluation.library_modifications, screenshot });
    }
    await cdp.call("Page.navigate", { url: appUrl + "/knowledge" });
    // The tab buttons exist in SSR HTML before React attaches handlers.
    // The fetched rule count confirms hydration before a programmatic click.
    await cdp.wait("[...document.querySelectorAll('button')].some(b => b.innerText.includes('Paddle 转换规则') && b.innerText.includes('541'))");
    await cdp.click("Paddle 转换规则");
    await cdp.wait("document.querySelector('input')?.placeholder.includes('Paddle') && document.querySelector('svg[aria-label^=\"全部 \"]')?.querySelectorAll('circle').length === 541");
    await cdp.fill("input", "no_such_modelx_rule_20261024");
    await cdp.wait("document.body.innerText.includes('无匹配记录')");
    const emptyView = await cdp.evaluate(`(() => {
      const svg = document.querySelector('svg[aria-label^="全部 "]');
      return { viewBox: svg?.getAttribute('viewBox'), circles: svg?.querySelectorAll('circle').length };
    })()`);
    if (!emptyView.viewBox || /NaN|Infinity/.test(emptyView.viewBox) || emptyView.circles !== 0) {
      throw new Error(`Invalid empty search visualization: ${JSON.stringify(emptyView)}`);
    }
    await cdp.fill("input", "torch.nn.Conv2d");
    await cdp.wait("document.body.innerText.includes('torch.nn.Conv2d') && document.querySelector('svg[aria-label^=\"全部 \"] circle')");
    if (cdp.errors.length) throw new Error(JSON.stringify(cdp.errors));
    if (cdp.consoleErrors.length) throw new Error(JSON.stringify(cdp.consoleErrors));
    const report = { passed: rows.length, cases: rows, knowledge_search: { passed: true, empty_view: emptyView },
      browser_errors: cdp.errors, console_errors: cdp.consoleErrors };
    fs.writeFileSync(path.join(output, "paddle-browser-validation.json"), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally {
    cdp.ws.close();
    await fetch(`${origin}/json/close/${target.id}`).catch(() => {});
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
