const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const statusEl = document.getElementById("status");

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function addUser(text) {
  const m = el("div", "msg user");
  m.dir = "auto";
  m.textContent = text;
  chat.appendChild(m);
  chat.scrollTop = chat.scrollHeight;
}

function addBot(data) {
  const m = el("div", "msg bot");
  m.dir = "auto";
  const tag = el("span", "tag",
    data.mode === "search" ? "بحث حديث / fresh search" :
    data.mode === "analytical" ? "تحليل معرفي مدعوم بالأدلة / evidence-backed analysis" :
    "معرفة مدرّبة / trained knowledge");
  m.appendChild(tag);
  m.appendChild(el("div", null, data.reply));

  if (data.sources && data.sources.length) {
    const box = el("div", "sources");
    data.sources.forEach((s) => {
      const c = el("div", "source");
      const a = el("a", null, s.title);
      a.href = s.url; a.target = "_blank"; a.rel = "noopener";
      c.appendChild(a);
      c.appendChild(el("div", "prov", s.provider + " · " + s.url));
      if (s.snippet) c.appendChild(el("div", "snip", s.snippet));
      box.appendChild(c);
    });
    m.appendChild(box);
  }

  const d = data.diagnostics;
  if (d) {
    if (d.conflict) m.appendChild(el("div", "conflict", "⚠ تعارض: " + d.conflict));
    const det = el("details", "diag");
    det.appendChild(el("summary", null, "تشخيص الانتقالات / transition diagnostics"));
    det.appendChild(el("div", "reason", "السبب: " + d.reason));
    if (d.events && d.events.length)
      det.appendChild(el("div", "reason", "أحداث: " + d.events.join(" | ")));
    const tbl = el("table");
    tbl.innerHTML = "<tr><th>unit</th><th>prob</th><th>conf</th><th>verified</th><th>score</th></tr>";
    d.paths.forEach((p) => {
      const tr = el("tr");
      [p.unit, p.model_prob, p.confidence, p.verified, p.final_score].forEach((v) =>
        tr.appendChild(el("td", null, String(v))));
      tbl.appendChild(tr);
    });
    det.appendChild(tbl);
    det.appendChild(el("div", "reason", "المصدر: " + d.source));
    if (d.evidence && d.evidence.length) {
      const ev = el("div", "reason", "الأدلة المستخدمة / evidence:");
      d.evidence.forEach((item) => {
        ev.appendChild(el("div", "prov", item.concept + " · " + item.verification + " · " + item.source));
      });
      det.appendChild(ev);
    }
    m.appendChild(det);
  }
  chat.appendChild(m);
  chat.scrollTop = chat.scrollHeight;
}

function setStatus(text) {
  if (!text) { statusEl.hidden = true; statusEl.textContent = ""; return; }
  statusEl.hidden = false; statusEl.textContent = text;
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  addUser(text);
  sendBtn.disabled = true;
  const isSearch = /^(ابحث|بحث|search|look up|google)/i.test(text);
  setStatus(isSearch ? "جارٍ البحث… / searching…" : "يفكّر… / thinking…");
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    addBot(data);
  } catch (e) {
    addBot({ mode: "knowledge", reply: "خطأ في الاتصال بالخادم: " + e, sources: [] });
  } finally {
    setStatus("");
    sendBtn.disabled = false;
    input.focus();
  }
}

sendBtn.addEventListener("click", send);
input.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
clearBtn.addEventListener("click", async () => {
  await fetch("/api/clear", { method: "POST" });
  chat.innerHTML = "";
  const w = el("div", "welcome");
  w.appendChild(el("p", null, "تم مسح المحادثة. بيانات التدريب سليمة."));
  chat.appendChild(w);
});

(async () => {
  try {
    const info = await (await fetch("/api/info")).json();
    document.getElementById("model-name").textContent = info.model_name;
    document.getElementById("model-meta").textContent =
      `v${info.version} · ${info.variant} · ${info.units} units · vocab ${info.vocab} · ${info.analytical_concepts || 0} analytical concepts`;
  } catch (e) { /* header stays default */ }
  input.focus();
})();