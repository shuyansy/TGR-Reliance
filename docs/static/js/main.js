(function () {
  "use strict";
  var D = window.PAGE_DATA || {};
  var $ = function (s, el) { return (el || document).querySelector(s); };
  var $$ = function (s, el) { return Array.prototype.slice.call((el || document).querySelectorAll(s)); };
  var h = function (tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  var fmt = function (x, d) { return (x == null || isNaN(x)) ? "–" : Number(x).toFixed(d == null ? 2 : d); };
  var timeOf = function (path) { var m = /_t(\d+(?:\.\d+)?)s\.jpg$/.exec(path); return m ? parseFloat(m[1]) : null; };
  var inSpans = function (t, spans) { return spans.some(function (s) { return t >= s[0] && t <= s[1]; }); };
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };

  /* ---------------- nav + reveal ---------------- */
  var nav = $("#nav");
  window.addEventListener("scroll", function () { nav.classList.toggle("scrolled", window.scrollY > 30); }, { passive: true });
  var onReveal = {};
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (!en.isIntersecting) return;
      en.target.classList.add("visible");
      var id = en.target.id;
      if (id && onReveal[id] && !en.target.dataset.done) { en.target.dataset.done = "1"; onReveal[id](); }
      io.unobserve(en.target);
    });
  }, { threshold: 0.18 });
  function watch() { $$(".reveal").forEach(function (el) { io.observe(el); }); }

  /* ---------------- counters ---------------- */
  function countUp(el) {
    var target = parseInt(el.dataset.count, 10), t0 = null, dur = 1600;
    function step(ts) {
      if (!t0) t0 = ts;
      var p = Math.min(1, (ts - t0) / dur), v = Math.round(target * (1 - Math.pow(1 - p, 3)));
      el.textContent = v.toLocaleString("en-US");
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  var stats = $(".stats");
  if (stats) { stats.id = "statsStrip"; onReveal.statsStrip = function () { $$("[data-count]", stats).forEach(countUp); }; }

  /* ---------------- hero ---------------- */
  (function hero() {
    var H = D.hero; if (!H) return;
    var v = $("#heroVideo"), head = $("#heroHead"), needle = $("#heroNeedle"), badge = $("#heroBadge"), badgeT = $("#heroBadgeText"), speed = $("#heroSpeed");
    var T = H.duration, s = H.needle_span[0], e = H.needle_span[1];
    needle.style.left = (100 * s / T) + "%"; needle.style.width = (100 * (e - s) / T) + "%";
    var ticks = $("#heroTicks");
    [0, 30, 60, 90].concat([Math.round(T)]).forEach(function (t) { var sp = h("span", null, t + "s"); sp.style.left = (100 * Math.min(t, T) / T) + "%"; ticks.appendChild(sp); });
    $("#heroQuestion").textContent = H.question;
    var opts = $("#heroOptions");
    H.options.forEach(function (o, i) { var L = String.fromCharCode(65 + i); var d = h("span", "opt" + (L === H.answer ? " correct" : ""), "<b>" + L + ".</b> " + esc(o)); opts.appendChild(d); });
    var correct = $(".opt.correct", opts);
    function srcTime(tOut) {
      for (var i = 0; i < H.segments.length; i++) {
        var g = H.segments[i];
        if (tOut <= g.out1 || i === H.segments.length - 1) return { t: Math.min(g.src1, g.src0 + Math.max(0, tOut - g.out0) * g.speed), sp: g.speed };
      }
      return { t: 0, sp: 1 };
    }
    function tick() {
      var r = srcTime(v.currentTime || 0), inside = r.t >= s && r.t <= e;
      head.style.left = (100 * r.t / T) + "%";
      badge.classList.toggle("in", inside);
      badgeT.textContent = inside ? "Needle · STAR / Charades" : "Haystack · Video-MME";
      speed.textContent = (r.sp % 1 === 0 ? r.sp.toFixed(0) : r.sp.toFixed(1)) + "×";
      if (correct) correct.classList.toggle("show", r.t > e);
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
    // the example plays when it scrolls into view and pauses when it leaves
    var vio = new IntersectionObserver(function (es) { es.forEach(function (en) { if (en.isIntersecting) { var p = v.play(); if (p && p.catch) p.catch(function () {}); } else v.pause(); }); }, { threshold: 0.3 });
    vio.observe(v);
    var iv = document.getElementById("introVideo"); if (iv) { var q = iv.play && iv.play(); if (q && q.catch) q.catch(function () {}); }
  })();

  /* ---------------- math (KaTeX, with a plain-text fallback) ---------------- */
  function tex(el, src, display) {
    if (window.katex) { try { window.katex.render(src, el, { displayMode: !!display, throwOnError: false }); return; } catch (e) {} }
    el.textContent = src.replace(/\\(text|operatorname)\{([^}]*)\}/g, "$2").replace(/\\bar p/g, "p̄").replace(/[{}\\]/g, "");
  }
  $$(".tex").forEach(function (el) { tex(el, el.textContent, false); });
  $$(".tex-block").forEach(function (el) { tex(el, el.textContent, true); });

  /* ---------------- ReGAP demo ---------------- */
  (function regap() {
    var R = D.regap_section; if (!R) return;
    var T = R.duration, names = Object.keys(R.models), sel = names[0], act = "full";
    var COLORS = ["#60a5fa", "#a78bfa", "#f59e0b", "#f472b6"];
    var color = {}; names.forEach(function (n, i) { color[n] = COLORS[i % COLORS.length]; });
    var strip = $("#regapStrip");
    R.frames.forEach(function (src, i) {
      var f = h("div", "fr"); var im = h("img"); im.src = src; im.alt = "frame at " + R.times[i] + " s"; im.loading = "lazy"; f.appendChild(im);
      if (R.times[i] >= R.needle_span[0] && R.times[i] <= R.needle_span[1]) f.classList.add("in-needle");
      strip.appendChild(f);
    });
    var mseg = $("#regapModel");
    names.forEach(function (n, i) {
      var b = h("button", i === 0 ? "on" : "", '<span class="dotc" style="background:' + color[n] + '"></span>' + esc(n));
      b.dataset.m = n; mseg.appendChild(b);
    });
    $("#regapQ").textContent = R.question;
    var ai = R.answer.charCodeAt(0) - 65;
    $("#regapA").textContent = "Answer: " + R.answer + ". " + R.options[ai];
    function spanEl(track, a, b, col) { var sp = h("div", "sp"); sp.style.left = (100 * a / T) + "%"; sp.style.width = (100 * (b - a) / T) + "%"; if (col) sp.style.background = col; track.appendChild(sp); }
    spanEl($("#laneNeedle"), R.needle_span[0], R.needle_span[1]);
    function randBlocks(pred) {   // illustrative positions: same total length as the span, outside it
      var len = pred.reduce(function (a, s) { return a + (s[1] - s[0]); }, 0), out = [], tries = [0.3, 0.55, 0.8, 0.42, 0.68, 0.92];
      for (var i = 0; i < tries.length && out.length < 3; i++) {
        var a = tries[i] * (T - len), b = a + len;
        var clash = pred.concat(out).some(function (s) { return a < s[1] + 1 && b > s[0] - 1; });
        if (!clash && b <= T) out.push([a, b]);
      }
      return out;
    }
    var bars = $("#regapBars");
    names.forEach(function (n) {
      var row = h("div", "bar");
      row.innerHTML = '<span class="bn"><span class="dotc" style="background:' + color[n] + '"></span>' + esc(n) + '</span><div class="bt"><div class="bf"></div></div><span class="bv">–</span>';
      $(".bf", row).style.background = color[n];
      row.dataset.m = n; bars.appendChild(row);
    });
    var CLIP = R.clip || 1e-4;
    var logit = function (p) { p = Math.min(1 - CLIP, Math.max(CLIP, p)); return Math.log(p / (1 - p)); };
    var f2 = function (x) { return (x < 0 ? "−" : "") + Math.abs(x).toFixed(2); };
    function live() {
      var m = R.models[sel], lr = logit(m.p_random_deleted), ls = logit(m.p_span_deleted), c = lr - ls;
      var box = $("#regapLive");
      box.innerHTML = '<div class="lv-h"><span class="dotc" style="background:' + color[sel] + '"></span>Worked example: <b>' + esc(sel) + '</b> on this question</div>' +
        '<div class="lv-row"><span class="lv-a"></span></div><div class="lv-row"><span class="lv-b"></span></div>' +
        '<p class="lv-p">Over all 725 ReliaNeedle questions, ' + esc(sel) + "'s ReGAP is <b>" + f2(m.regap_all) + "</b> (" + names.map(function (n) { return esc(n) + " " + f2(R.models[n].regap_all); }).join(", ") +
        "). One question only illustrates the computation; ReGAP averages over the benchmark.</p>";
      tex($(".lv-a", box), "p_{\\text{span}} = " + m.p_span_deleted.toFixed(3) + ",\\quad \\bar p_{\\text{rand}} = " + m.p_random_deleted.toFixed(3) +
          "\\quad\\Rightarrow\\quad c = " + lr.toFixed(2) + " - (" + ls.toFixed(2) + ") = \\mathbf{" + c.toFixed(2) + "}", true);
      tex($(".lv-b", box), "c\\,/\\,\\tau = " + c.toFixed(2) + " / " + m.tau.toFixed(2) + " = " + (c / m.tau).toFixed(2), true);
    }
    function render() {
      var m = R.models[sel], pred = m.pred_spans || [], rb = randBlocks(pred);
      var lp = $("#lanePred"), lrn = $("#laneRand"); lp.innerHTML = ""; lrn.innerHTML = "";
      pred.forEach(function (s) { spanEl(lp, s[0], s[1], color[sel]); });
      rb.forEach(function (s) { spanEl(lrn, s[0], s[1]); });
      lrn.classList.toggle("active", act === "rand");
      $$(".fr", strip).forEach(function (f, i) {
        var t = R.times[i];
        var del = (act === "span" && inSpans(t, pred.map(function (s) { return [s[0] - 1, s[1] + 1]; }))) || (act === "rand" && inSpans(t, rb));
        f.classList.toggle("del", del);
      });
      $("#regapBarsTitle").textContent = act === "full" ? "p(answer) with the full video" : act === "span" ? "p(answer) after deleting each model's predicted span" : "p(answer) after deleting random blocks (average of 3)";
      $$(".bar", bars).forEach(function (row) {
        var mm = R.models[row.dataset.m], p = act === "full" ? mm.p_full : act === "span" ? mm.p_span_deleted : mm.p_random_deleted;
        $(".bf", row).style.width = Math.max(1.5, 100 * (p || 0)) + "%";
        $(".bv", row).textContent = fmt(p, 2);
        row.classList.toggle("sel", row.dataset.m === sel);
      });
      var note = act === "full" ? "With the full video all three models are confident and correct."
        : act === "span" ? names.map(function (n) { var x = R.models[n]; return n + " " + fmt(x.p_full) + " → " + fmt(x.p_span_deleted); }).join(" · ") + ". Same evidence found, very different reliance."
        : "Deleting random blocks barely matters for any of them (" + names.map(function (n) { return fmt(R.models[n].p_random_deleted); }).join(", ") + "). ReGAP compares the two deletions.";
      $("#regapNote").textContent = note;
      live();
    }
    mseg.addEventListener("click", function (ev) { var b = ev.target.closest("button"); if (!b) return; sel = b.dataset.m; $$("button", mseg).forEach(function (x) { x.classList.toggle("on", x === b); }); render(); });
    $("#regapAction").addEventListener("click", function (ev) { var b = ev.target.closest("button"); if (!b) return; act = b.dataset.act; $$("#regapAction button").forEach(function (x) { x.classList.toggle("on", x === b); }); render(); });
    render();
    onReveal.regapDemo = function () {
      var seq = ["span", "rand", "full"], k = 0;
      function go() { if (k >= seq.length) return; $('#regapAction button[data-act="' + seq[k] + '"]').click(); k++; setTimeout(go, 3200); }
      setTimeout(go, 1200);
    };
  })();

  /* ---------------- ReliaNeedle build ---------------- */
  (function build() {
    var N = D.relianeedle; if (!N) return;
    var hay = $("#buildHay"), nd = $("#buildNeedle"), mg = $("#buildMerged");
    var add = function (box, arr, cls) { return arr.map(function (src) { var im = h("img", cls || ""); im.src = src; im.alt = ""; im.loading = "lazy"; box.appendChild(im); return im; }); };
    add(hay, N.before); var gap = h("span", "", ""); gap.style.width = "24px"; hay.appendChild(gap); add(hay, N.after);
    var nIms = add(nd, N.needle);
    var mIms = add(mg, N.before).concat(add(mg, N.needle, "n")).concat(add(mg, N.after));
    function play() {
      mIms.forEach(function (im) { im.classList.remove("show"); }); nIms.forEach(function (im) { im.classList.remove("fly"); });
      var nb = N.before.length, nn = N.needle.length;
      mIms.forEach(function (im, i) {
        var delay = i < nb ? i * 120 : i < nb + nn ? 700 + (i - nb) * 160 : 1800 + (i - nb - nn) * 120;
        setTimeout(function () { im.classList.add("show"); if (i >= nb && i < nb + nn) nIms[i - nb].classList.add("fly"); }, 300 + delay);
      });
    }
    onReveal.needleBuild = play;
    $("#buildReplay").addEventListener("click", play);
  })();

  /* ---------------- leaderboard ---------------- */
  (function board() {
    var L = (D.leaderboard || []).slice(); if (!L.length) return;
    var tb = $("#leaderboard tbody"), key = "regap", desc = true, max = Math.max.apply(null, L.map(function (x) { return x.regap; }));
    function draw(anim) {
      L.sort(function (a, b) { var x = a[key], y = b[key]; if (key === "name") return desc ? (y > x ? 1 : -1) : (x > y ? 1 : -1); return desc ? y - x : x - y; });
      tb.innerHTML = "";
      L.forEach(function (r) {
        var tr = h("tr", r.kind);
        var neg = function (v) { return v < 0 ? ' neg' : ''; };
        tr.innerHTML = "<td>" + esc(r.name) + "</td>" +
          '<td><div class="rg"><div class="t"><div class="f" data-w="' + Math.max(0, 100 * r.regap / max) + '"></div></div><span class="v' + neg(r.regap) + '">' + fmt(r.regap, 3) + "</span></div></td>" +
          '<td class="num' + neg(r.hit) + '">' + fmt(r.hit, 3) + '</td><td class="num' + neg(r.miss) + '">' + fmt(r.miss, 3) + "</td>" +
          '<td class="num">' + fmt(r.acc, 1) + '</td><td class="num">' + fmt(r.r03, 1) + "</td>";
        tb.appendChild(tr);
      });
      var setW = function () { $$(".f", tb).forEach(function (f) { f.style.width = f.dataset.w + "%"; }); };
      if (anim) setTimeout(setW, 60); else setW();
    }
    var lbCard = $(".table-card"); lbCard.id = "lbCard";
    onReveal.lbCard = function () { draw(true); };
    draw(false); $$(".f", tb).forEach(function (f) { f.style.width = "0"; });
    $$("#leaderboard th").forEach(function (th) {
      th.addEventListener("click", function () {
        var k = th.dataset.k; if (k === key) desc = !desc; else { key = k; desc = k !== "name"; }
        $$("#leaderboard th").forEach(function (x) { x.classList.toggle("sorted", x === th); x.textContent = x.textContent.replace(/ [▾▴]$/, ""); });
        th.textContent += desc ? " ▾" : " ▴"; draw(true);
      });
    });
  })();

  /* ---------------- bars (audit, fix) ---------------- */
  function fillBars(sel, max) { $$(sel).forEach(function (f, i) { setTimeout(function () { f.style.width = (100 * parseFloat(f.dataset.v) / max) + "%"; }, 120 * i); }); }
  var audit = $(".audit"); if (audit) { audit.id = "auditCard"; onReveal.auditCard = function () { fillBars("#auditBars .ab-fill", 100); }; }
  onReveal.fixBars = function () { fillBars("#fixBars .fb-fill", 25); };

  /* ---------------- EviCe certification cards ---------------- */
  (function cert() {
    var E = D.evice || [], grid = $("#certGrid"); if (!E.length) return;
    var MODES = [["full", "Full video"], ["span_only", "Evidence only"], ["span_deleted", "Evidence deleted"]];
    E.forEach(function (x, ci) {
      var c = h("div", "card cert reveal"); c.id = "cert" + ci;
      var frames = x.uniform.map(function (p) { return { p: p, t: timeOf(p) }; });
      x.span_frames.forEach(function (arr) { arr.forEach(function (p) { frames.push({ p: p, t: timeOf(p) }); }); });
      frames.sort(function (a, b) { return a.t - b.t; });
      var title = (x.tag === "groot" ? "Two moments, one answer" : "Long video (" + Math.round(x.duration) + " s), two distant moments");
      c.innerHTML = "<h4>" + title + ' <span class="c-needle" style="font-weight:500;font-size:12px">' + esc(x.id) + "</span></h4>" +
        '<p class="cq"><span class="q-tag">Q</span>' + esc(x.question) + ' <b class="c-needle">→ ' + esc(x.answer) + "</b></p>";
      var st = h("div", "cstrip"); st.style.gridTemplateColumns = "repeat(" + frames.length + ", 1fr)";
      frames.forEach(function (f) { var d = h("div", "fr" + (inSpans(f.t, x.spans) ? " ev" : "")); var im = h("img"); im.src = f.p; im.alt = ""; im.loading = "lazy"; d.appendChild(im); st.appendChild(d); });
      c.appendChild(st);
      var tl = h("div", "ctl"); x.spans.forEach(function (s) { var sp = h("div", "sp"); sp.style.left = (100 * s[0] / x.duration) + "%"; sp.style.width = Math.max(0.8, 100 * (s[1] - s[0]) / x.duration) + "%"; tl.appendChild(sp); });
      c.appendChild(tl);
      c.appendChild(h("div", "ctl-lbl", "<span>0 s</span><span>evidence: " + x.spans.map(function (s) { return "[" + s[0] + ", " + s[1] + "]"; }).join(" ") + "</span><span>" + Math.round(x.duration) + " s</span>"));
      var modes = h("div", "modes");
      MODES.forEach(function (m) {
        var md = h("div", "mode"); md.dataset.m = m[0];
        var lines = Object.keys(x.judges).map(function (j) {
          var a = x.judges[j][m[0]], ok = a === x.answer, want = m[0] !== "span_deleted";
          return esc(j.replace("-Thinking", "")) + ": <span class='" + (ok ? "ok" : "bad") + "'>" + esc(a) + " " + (ok ? "✓" : "✗") + "</span>" + (ok === want ? "" : "");
        }).join("<br>");
        md.innerHTML = '<div class="mt">' + m[1] + '</div><div class="jr">' + lines + "</div>";
        modes.appendChild(md);
      });
      c.appendChild(modes);
      c.appendChild(h("div", "verdict", "✓ Sufficient and necessary for both judges → certified"));
      grid.appendChild(c);
      var k = 0, timer = null;
      function show(mode) {
        $$(".mode", modes).forEach(function (md) { md.classList.toggle("on", md.dataset.m === mode); });
        $$(".fr", st).forEach(function (f) {
          var ev = f.classList.contains("ev");
          f.classList.toggle("dim", mode === "span_only" && !ev);
          f.classList.toggle("cut", mode === "span_deleted" && ev);
        });
      }
      function cycle() { show(MODES[k % 3][0]); k++; timer = setTimeout(cycle, 2600); }
      modes.addEventListener("click", function (ev) { var md = ev.target.closest(".mode"); if (!md) return; clearTimeout(timer); show(md.dataset.m); });
      show("full");
      onReveal[c.id] = function () { setTimeout(cycle, 400 + 500 * ci); };
    });
  })();

  /* ---------------- RelyGen stepper (kale) ---------------- */
  (function kale() {
    var K = D.kale; if (!K) return;
    var T = K.duration, svg = $("#kaleSvg"), panel = $("#kalePanel"), nav = $("#kaleNav"), strip = $("#kaleStrip");
    $("#kaleQ").innerHTML = esc(K.question) + ' <span class="c-needle">(' + K.choices.map(function (c, i) { return String.fromCharCode(65 + i) + ". " + esc(c); }).join(" · ") + ")</span>";
    var frames = K.uniform.map(function (p) { return { p: p, t: timeOf(p) }; });
    frames.forEach(function (f) { var d = h("div", "fr"); var im = h("img"); im.src = f.p; im.alt = ""; im.loading = "lazy"; d.appendChild(im); strip.appendChild(d); f.el = d; });
    var X = function (t) { return 1000 * t / T; };
    var NS = "http://www.w3.org/2000/svg";
    function rect(x0, x1, y, hh, fill, op, cls) { var r = document.createElementNS(NS, "rect"); r.setAttribute("x", X(x0)); r.setAttribute("y", y); r.setAttribute("width", Math.max(2, X(x1) - X(x0))); r.setAttribute("height", hh); r.setAttribute("rx", 3); r.setAttribute("fill", fill); r.setAttribute("opacity", op == null ? 1 : op); if (cls) r.setAttribute("class", cls); svg.appendChild(r); return r; }
    function label(x, y, s, fill) { var t = document.createElementNS(NS, "text"); t.setAttribute("x", x); t.setAttribute("y", y); t.setAttribute("font-size", "11"); t.setAttribute("fill", fill || "#98a2b8"); t.textContent = s; svg.appendChild(t); return t; }
    var STEPS = [
      { n: "Question", focus: null },
      { n: "1 · Visual check", focus: null },
      { n: "2 · Grounding query", focus: null },
      { n: "3 · Caption retrieval", focus: K.proposals },
      { n: "4 · Two grounders", focus: K.spans },
      { n: "5 · Two judges", focus: K.spans }
    ];
    function drawSvg(k) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      rect(0, T, 2, 6, "rgba(255,255,255,.08)");
      if (k >= 3) {
        label(0, 24, "captions (30 s windows)");
        K.captions.forEach(function (w, i) { var hit = K.proposals.some(function (p) { return w.t0 === p[0] && w.t1 === p[1]; }); rect(w.t0 + 0.6, w.t1 - 0.6, 30 + (i % 2) * 7, 6, hit ? "#34d399" : "#64748b", hit ? 1 : 0.55); });
      }
      if (k >= 4) {
        label(0, 66, "TimeLens2-8B", "#fbbf24"); K.timelens.forEach(function (s) { rect(s[0], s[1], 72, 9, "#fbbf24"); });
        label(0, 98, "Keye-VL-2.0", "#a78bfa"); K.keye.forEach(function (s) { rect(s[0], s[1], 104, 9, "#a78bfa"); });
        label(1000 - 150, 98, "agreement IoU " + K.iou.toFixed(3), "#e2e8f0");
      }
      if (k >= 5) { label(0, 132, "certified evidence", "#34d399"); K.spans.forEach(function (s) { rect(s[0], s[1], 138, 9, "#34d399"); }); }
      [0, 60, 120, 180, 240, 300].forEach(function (t) { label(X(t) + 2, 16, t + "s", "#5b6478"); });
    }
    function judgeTable() {
      var rows = Object.keys(K.judges).map(function (j) {
        var r = K.judges[j], c = function (a, want) { var ok = a === K.answer; return "<td><span style='color:" + (ok === want ? "#34d399" : "#f87171") + "'>" + esc(a) + " " + (ok ? "✓" : "✗") + "</span></td>"; };
        return "<tr><td>" + esc(j) + "</td>" + c(r.full, true) + c(r.span_only, true) + c(r.span_deleted, false) + "</tr>";
      }).join("");
      return "<table class='jt'><tr><th>Judge</th><th>Full video</th><th>Evidence only</th><th>Evidence deleted</th></tr>" + rows + "</table>";
    }
    function panelHtml(k) {
      if (k === 0) return "<h5>A VideoQA pair without evidence annotation</h5><p>LongVideo-Reason, " + Math.round(T) + " s. Answer: <b class='c-needle'>" + esc(K.answer) + "</b>. Where in the video is the evidence?</p>";
      if (k === 1) return "<h5>Is the video needed at all?</h5><p>A strong LLM answers from the text alone, in two option orders. It fails, so the question needs the video and is kept.</p>";
      if (k === 2) return "<h5>Rewrite the question into what should be visible</h5><p class='mono'>“" + esc(K.query) + "”</p>";
      if (k === 3) return "<h5>Long video: retrieve candidate windows from captions</h5><div class='caps'>" + K.captions.map(function (w) {
        var hit = K.proposals.some(function (p) { return w.t0 === p[0] && w.t1 === p[1]; });
        return "<div class='cap" + (hit ? " hit" : "") + "'><b>[" + w.t0 + "–" + w.t1 + "]</b>" + esc(w.cap.length > 150 ? w.cap.slice(0, 150) + "…" : w.cap) + "</div>";
      }).join("") + "</div>";
      if (k === 4) return "<h5>Two temporal grounders must agree</h5><p class='mono'>TimeLens2-8B: " + JSON.stringify(K.timelens) + "<br>Keye-VL-2.0: " + JSON.stringify(K.keye) + "<br>IoU " + K.iou.toFixed(3) + " ≥ 0.3 → keep " + JSON.stringify(K.spans) + "</p>";
      var imgs = (K.span1 || []).concat(K.span2 || []).map(function (p) { return "<img src='" + p + "' alt=''>"; }).join("");
      return "<h5>Two judges from different model families certify the evidence</h5>" + judgeTable() + "<div class='frames'>" + imgs + "</div>" +
        "<p class='mono' style='margin-top:8px'>EviCe target: &lt;think&gt;[27.00, 50.00], [113.00, 130.00]&lt;/think&gt;&lt;answer&gt;" + esc(K.answer) + "&lt;/answer&gt;</p>";
    }
    var cur = 0, timer = null;
    STEPS.forEach(function (s, i) { var b = h("button", "", s.n); b.addEventListener("click", function () { stop(); go(i); }); nav.appendChild(b); });
    function go(k) {
      cur = k;
      $$("button", nav).forEach(function (b, i) { b.classList.toggle("on", i === k); b.classList.toggle("done", i < k); });
      var f = STEPS[k].focus;
      frames.forEach(function (fr) { fr.el.classList.toggle("dim", !!f && !inSpans(fr.t, f)); });
      drawSvg(k); panel.innerHTML = panelHtml(k);
    }
    function stop() { clearTimeout(timer); timer = null; $("#kalePlay").textContent = "Play"; }
    function play() { $("#kalePlay").textContent = "Pause"; timer = setTimeout(function nextStep() { if (cur >= STEPS.length - 1) { stop(); return; } go(cur + 1); timer = setTimeout(nextStep, 3600); }, 1800); }
    $("#kalePrev").addEventListener("click", function () { stop(); go(Math.max(0, cur - 1)); });
    $("#kaleNext").addEventListener("click", function () { stop(); go(Math.min(STEPS.length - 1, cur + 1)); });
    $("#kalePlay").addEventListener("click", function () { if (timer) stop(); else { if (cur >= STEPS.length - 1) go(0); play(); } });
    go(0);
    onReveal.kale = play;
  })();

  /* ---------------- agent intervention ---------------- */
  (function agent() {
    var A = D.intervention; if (!A) return;
    var arm = A.arms["s-1"], T = A.duration, svg = $("#agentSvg"), NS = "http://www.w3.org/2000/svg";
    $("#agentQ").innerHTML = esc(A.question) + " <span class='c-needle'>(answer " + A.answer + ". " + esc(A.options[A.answer.charCodeAt(0) - 65]) + ")</span>";
    var f = arm.ours.forced, r = arm.ours.recrops[0];
    $("#agentForced").textContent = "[" + f[0].toFixed(1) + ", " + f[1].toFixed(1) + "] s";
    $("#agentRecrop").textContent = "[" + r[0].toFixed(1) + ", " + r[1].toFixed(1) + "] s";
    $("#agentQuote").textContent = "“" + A.official_quote + "…”";
    var addImgs = function (box, arr) { arr.forEach(function (p) { var im = h("img"); im.src = p; im.alt = ""; im.loading = "lazy"; box.appendChild(im); }); };
    addImgs($("#agentForcedFrames"), A.forced_frames); addImgs($("#agentRecropFrames"), A.recrop_frames);
    var X0 = 120, W = 860, X = function (t) { return X0 + W * t / T; };
    function el(tag, attrs, parent) { var e = document.createElementNS(NS, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).appendChild(e); return e; }
    function txt(x, y, s, fill, size, weight) { var t = el("text", { x: x, y: y, fill: fill || "#98a2b8", "font-size": size || 13, "font-weight": weight || 500 }); t.textContent = s; return t; }
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var defs = el("defs", {}); var m = el("marker", { id: "arr", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
      el("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#34d399" }, m);
      var s = A.needle_span;
      el("rect", { x: X(s[0]), y: 18, width: X(s[1]) - X(s[0]), height: 190, fill: "rgba(52,211,153,.10)", stroke: "rgba(52,211,153,.5)", "stroke-dasharray": "4 4", rx: 8 });
      txt(X(s[0]) + 6, 34, "needle (evidence)", "#34d399", 12, 700);
      [0, 30, 60, 90, 120].forEach(function (t) { txt(X(t) - 6, 214, t + "s", "#5b6478", 11); });
      var rows = [["Ours", 78, arm.ours], ["Video-o3", 150, arm.official_sft_rl]];
      rows.forEach(function (row, ri) {
        var y = row[1], a = row[2];
        txt(16, y + 5, row[0], ri === 0 ? "#dbeafe" : "#e2e8f0", 14, 700);
        el("rect", { x: X0, y: y - 4, width: W, height: 8, rx: 4, fill: "rgba(255,255,255,.06)" });
        var g1 = el("g", { class: "step s1" });
        el("rect", { x: X(a.forced[0]), y: y - 13, width: X(a.forced[1]) - X(a.forced[0]), height: 26, rx: 6, fill: "rgba(248,113,113,.25)", stroke: "#f87171" }, g1);
        var t1 = el("text", { x: X(a.forced[0]) + 8, y: y + 4, fill: "#fecaca", "font-size": 12, "font-weight": 600 }, g1); t1.textContent = "crop 1 (forced)";
        if (a.recrops.length) {
          var rc = a.recrops[0], g2 = el("g", { class: "step s2" });
          el("path", { d: "M " + (X(a.forced[1]) - 20) + " " + (y - 14) + " C " + X(a.forced[1]) + " " + (y - 44) + ", " + X(rc[0]) + " " + (y - 44) + ", " + (X(rc[0]) + 30) + " " + (y - 15), fill: "none", stroke: "#34d399", "stroke-width": 2, "marker-end": "url(#arr)" }, g2);
          el("rect", { x: X(rc[0]), y: y - 13, width: X(rc[1]) - X(rc[0]), height: 26, rx: 6, fill: "rgba(52,211,153,.28)", stroke: "#34d399" }, g2);
          var t2 = el("text", { x: X(rc[0]) + 8, y: y + 4, fill: "#d1fae5", "font-size": 12, "font-weight": 600 }, g2); t2.textContent = "crop 2 → needle";
        }
        var g3 = el("g", { class: "step s3" });
        var ok = a.correct;
        var t3 = el("text", { x: X0 + W - 4, y: y + 34, fill: ok ? "#34d399" : "#f87171", "font-size": 13, "font-weight": 700, "text-anchor": "end" }, g3);
        t3.textContent = (a.recrops.length ? "" : "no new crop · ") + "answers " + a.letter + (ok ? " ✓" : " ✗");
      });
    }
    var style = document.createElement("style");
    style.textContent = "#agentSvg .step{opacity:0;transition:opacity .7s}#agentSvg.p1 .s1,#agentSvg.p2 .s1,#agentSvg.p2 .s2,#agentSvg.p3 .s1,#agentSvg.p3 .s2,#agentSvg.p3 .s3{opacity:1}";
    document.head.appendChild(style);
    function play() {
      draw(); svg.setAttribute("class", "agent-svg");
      setTimeout(function () { svg.setAttribute("class", "agent-svg p1"); }, 400);
      setTimeout(function () { svg.setAttribute("class", "agent-svg p2"); }, 1700);
      setTimeout(function () { svg.setAttribute("class", "agent-svg p3"); }, 3000);
    }
    draw();
    onReveal.agentDemo = play;
    $("#agentReplay").addEventListener("click", play);
  })();

  /* ---------------- intro sound toggle ---------------- */
  (function sound() {
    var v = document.getElementById("introVideo"), b = document.getElementById("soundBtn"); if (!v || !b) return;
    var waves = document.getElementById("soundWaves"), lab = document.getElementById("soundLabel");
    function sync() { var on = !v.muted && v.volume > 0; b.classList.toggle("on", on); lab.textContent = on ? "Sound off" : "Sound on"; waves.setAttribute("opacity", on ? "1" : ".35"); b.setAttribute("aria-label", on ? "Turn sound off" : "Turn sound on"); }
    b.addEventListener("click", function () {
      v.muted = !v.muted; if (!v.muted) { v.volume = 0.8; var p = v.play(); if (p && p.catch) p.catch(function () {}); }
      sync();
    });
    v.addEventListener("volumechange", sync); sync();
  })();

  /* ---------------- BibTeX ---------------- */
  var cb = $("#copyBib");
  if (cb) cb.addEventListener("click", function () {
    var t = $("#bibText").textContent;
    var done = function () { cb.textContent = "Copied"; setTimeout(function () { cb.textContent = "Copy"; }, 1500); };
    if (navigator.clipboard) navigator.clipboard.writeText(t).then(done, done); else done();
  });

  watch();
})();
