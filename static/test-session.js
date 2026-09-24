/* Real-time gaze-typing test session.
 *
 * Shows the participant a target sentence, they type it by gaze, and every
 * committed keystroke is scored the instant it happens. Nothing is simulated.
 *
 * Activate with a query string; absent it, this file defines a global and does
 * nothing else, so the keyboard is untouched in normal use:
 *
 *     /?test=1                 classical rule drives the commits (deployed)
 *     /?test=1&rule=quantum    quantum conditional selection drives them
 *     /?test=1&rule=quantum&n=20&trial=2
 *
 * The two modes are not symmetric, and the asymmetry is the honest part.
 * smartCommit() computes log-linear fusion locally, so in classical mode the
 * session is a passive observer: it reports the decision the keyboard already
 * made and returns false. The quantum rule is not implemented in the browser —
 * it lives in quantum/conditional_selection.py — so in quantum mode the session
 * must ask the server for the decision and apply it. That round trip is part of
 * the measurement, which is why the server reports its own model-only time
 * separately from the round trip recorded here.
 *
 * Target normalisation. The keyboard has a-z and space. A target containing a
 * character with no key would leave the participant with nothing to gaze at, so
 * targets are normalised before display: "we'll cook" is shown as "well cook".
 * The participant reads exactly what they are scored against.
 *
 * Scoring. Letters only. Space is typed to end a word — the language model
 * needs the boundary — but is not scored, because the space bar is a much
 * larger target than a letter key and mixing the two inflates the number. This
 * matches the offline protocol in eval/selection_rule_bench.py, which also
 * scores letters only.
 *
 * Alignment. Ground truth is positional: the pointer advances on every commit,
 * including a wrong one. Participants must not correct mid-sentence —
 * backspacing destroys the alignment that per-keystroke ground truth needs.
 *
 * Layout. The bar is a flex SIBLING of #keyboard, not a fixed overlay. body is
 * a flex column and the keyboard is flex:1, so a sibling makes the keyboard
 * shrink to fit and no key is ever covered — an overlay hid the bottom two
 * rows, which are exactly what the participant has to gaze at. Its height is
 * fixed and its content never wraps, because a reflow mid-session would move
 * every key and change the task being measured. The live chart is therefore
 * not drawn here at all; run eval/live_dual_chart.py in its own window.
 *
 * Typing speed. Entry rate is characters actually in the text field over
 * elapsed time, clocked from the first decision to the latest one -- real
 * throughput, including characters the participant resolved from a prompt,
 * because those cost time and produce text. WPM uses the text-entry
 * convention of a word being five characters. Commit-only rate is reported
 * alongside it, since that is the part the selection rule produced.
 */

(function () {
  "use strict";

  const q = new URLSearchParams(location.search);
  const ACTIVE = q.get("test") === "1";
  const RULE = q.get("rule") === "quantum" ? "quantum" : "classical";
  const N = Math.max(1, parseInt(q.get("n") || "20", 10));
  const TRIAL = Math.max(1, parseInt(q.get("trial") || "1", 10));

  /* Only a-z survives; everything else is dropped rather than substituted, so
   * the displayed target and the scored target are the same string. */
  const normWord = (w) => w.toLowerCase().replace(/[^a-z]/g, "");
  const normSentence = (s) => s.split(/\s+/).map(normWord).filter(Boolean).join(" ");

  const S = {
    active: false, rule: RULE, trial: TRIAL,
    sentences: [], si: 0, ci: 0,
    stats: { quantum: null, classical: null },
    busy: false, roundTrips: [], t0: 0, tLast: 0, charsAtLast: 0,
  };

  /* Freeze an element's height for the duration of a session.
   * body is a flex column, so anything above #keyboard that changes height
   * pushes every key. Gaze typing against a target that moves under you is not
   * the task we mean to measure, and these shifts land mid-sentence. Restored
   * on finish(), so normal use keeps its responsive layout. */
  function pin(elm) {
    if (!elm) return;
    const h = Math.ceil(elm.getBoundingClientRect().height);
    elm.style.height = h + "px";
    elm.style.flex = "0 0 " + h + "px";
    elm.style.overflow = "hidden";
    elm.dataset.tsPinned = "1";
  }

  // ---------------------------------------------------------------- UI

  let el = {};

  function buildUI() {
    const box = document.createElement("div");
    box.id = "test-session";
    box.innerHTML = `
      <span class="ts-badge">TEST</span>
      <span class="ts-target"></span>
      <span class="ts-speed"></span>
      <span class="ts-progress"></span>
      <button class="ts-skip" type="button">skip</button>
      <button class="ts-end" type="button">end &amp; save</button>`;
    // Insert as a flex sibling ABOVE the keyboard so the keyboard shrinks
    // rather than being covered. Falls back to the old overlay only if the
    // expected structure is not there.
    const kb = document.getElementById("keyboard");
    if (kb && kb.parentNode) kb.parentNode.insertBefore(box, kb);
    else { box.classList.add("ts-floating"); document.body.appendChild(box); }
    el = {
      box,
      target: box.querySelector(".ts-target"),
      speed: box.querySelector(".ts-speed"),
      progress: box.querySelector(".ts-progress"),
      skip: box.querySelector(".ts-skip"),
      end: box.querySelector(".ts-end"),
    };
    el.skip.addEventListener("click", nextSentence);
    el.end.addEventListener("click", finish);
  }

  function renderTarget() {
    const sent = S.sentences[S.si] || "";
    let html = "";
    for (let i = 0; i < sent.length; i++) {
      const c = sent[i];
      const cls = i < S.ci ? "done" : (i === S.ci ? "now" : "");
      html += `<span class="${cls}">${c === " " ? "&nbsp;" : c}</span>`;
    }
    el.target.innerHTML = html || "<em>done</em>";
    el.progress.textContent = `${Math.min(S.si + 1, S.sentences.length)}/${S.sentences.length}`;
  }

  /* Entry rate. The clock runs from the first decision to the LAST one, not to
   * "now": a pause after the final keystroke -- reading the result, reaching
   * for the mouse -- is not typing time, and measuring to now made the figure
   * fall the longer the session sat idle. */
  function speed() {
    if (!S.t0 || !S.tLast) return null;
    const mins = (S.tLast - S.t0) / 60000;
    if (mins <= 0) return null;
    // Snapshot taken at the last decision, not read live: the word-correction
    // pass rewrites the text field after a word boundary, so a live read makes
    // the figure drift while nobody is typing.
    const chars = S.charsAtLast;
    const committed = S.stats.quantum?.committed_count
                   ?? S.stats.classical?.committed_count ?? 0;
    return { mins, chars, wpm: (chars / 5) / mins, cpm: chars / mins,
             commit_wpm: (committed / 5) / mins, committed };
  }

  function renderSpeed() {
    const s2 = speed();
    el.speed.textContent = s2 ? `${s2.wpm.toFixed(1)} wpm` : "";
  }

  // Cumulative accuracy per arm, kept for the summary and for
  // eval/live_dual_chart.py. Deliberately not plotted in this bar: a chart
  // here would either cover keys or resize them mid-session.
  let series = { quantum: [], classical: [] };

  // ------------------------------------------------------- session flow

  function expectedChar() {
    const sent = S.sentences[S.si];
    if (sent === undefined) return null;
    return sent[S.ci] ?? null;
  }

  function advance() {
    const sent = S.sentences[S.si] || "";
    S.ci++;
    if (S.ci >= sent.length) nextSentence(); else renderTarget();
  }

  function nextSentence() {
    S.si++; S.ci = 0;
    if (S.si >= S.sentences.length) { finish(); return; }
    renderTarget();
  }

  /* Called by smartCommit once per keystroke decision.
   * Returns true when this session has taken over the commit (quantum mode). */
  function onDecision(ctx) {
    const truth = expectedChar();
    if (truth === null) return false;

    // A space is typed to close the word but is not scored; let the keyboard
    // handle it and step the pointer past it.
    if (truth === " ") {
      if (ctx.top.p >= 0 && !ctx.top.letter) advance();
      return false;
    }
    if (S.busy) return false;             // a decision is already in flight

    const p_gaze = {};
    for (const c of ctx.cands) {
      const k = c.letter || (c.action === "space" ? "$" : null);
      if (k && c.pGaze > 0) p_gaze[k] = c.pGaze;
    }
    if (!Object.keys(p_gaze).length) return false;

    if (!S.t0) S.t0 = performance.now();
    S.tLast = performance.now();
    S.charsAtLast = (window.__gaze?.state?.().text || "").length;
    const sp = speed();
    const body = {
      p_gaze, prefix: ctx.prefix, truth,
      sentence: S.sentences[S.si],
      elapsed_s: sp ? sp.mins * 60 : 0,
      chars_entered: (window.__gaze?.state?.().text || "").length,
      prediction: ctx.top.letter || (ctx.top.action === "space" ? "$" : null),
      top_p: ctx.top.p,
    };

    if (S.rule === "classical") {
      // passive: the keyboard has already decided, this only records it
      post("/api/eval/keystroke", body).then(onScored).catch(() => {});
      advance();
      return false;
    }

    // quantum: the server decides, we apply it
    S.busy = true;
    const t0 = performance.now();
    post("/api/eval/decide", body)
      .then((d) => {
        S.roundTrips.push(performance.now() - t0);
        if (!d || !d.ok) { ctx.apply(ctx.top); return; }
        onScored(d);
        const want = d.quantum.prediction;
        const hit = ctx.fused.find(
          (c) => (c.letter && c.letter === want) ||
                 (want === "$" && c.action === "space"));
        // Apply the quantum verdict with the quantum confidence, so THETA is
        // applied to the rule under test rather than to the local one.
        ctx.apply(hit ? { ...hit, p: d.quantum.top_p } : { ...ctx.top, p: 0 });
      })
      .catch(() => ctx.apply(ctx.top))
      .finally(() => { S.busy = false; advance(); });
    return true;
  }

  function onScored(d) {
    if (!d || !d.ok) return;
    for (const k of ["quantum", "classical"]) {
      if (!d[k]) continue;
      S.stats[k] = d[k];
      if (d[k].committed && typeof d[k].cumulative_accuracy === "number") {
        series[k].push(d[k].cumulative_accuracy);
      }
    }
    renderSpeed();
  }

  function post(url, body) {
    return fetch(url, { method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(body) }).then((r) => r.json());
  }

  async function finish() {
    const sp = speed();
    S.active = false;
    window.__evalSession.active = false;
    document.querySelectorAll("[data-ts-pinned]").forEach((e) => {
      e.style.height = ""; e.style.flex = ""; e.style.overflow = "";
      e.style.flexWrap = "";
      delete e.dataset.tsPinned;
    });
    // Typing is over, so the bar may now grow without disturbing key geometry.
    el.box.classList.add("ts-done");
    // Push the final entry rate before reading the summary, so the number on
    // screen and the number in the JSON are the same measurement.
    if (sp) {
      await post("/api/eval/finish",
                 { elapsed_s: sp.mins * 60, chars_entered: sp.chars }).catch(() => {});
    }
    const s2 = await fetch("/api/eval/summary").then((r) => r.json()).catch(() => null);
    if (!s2 || !s2.ok) { el.target.textContent = "ended (no summary)"; return; }
    const line = (k) => {
      const a2 = s2.arms[k];
      if (!a2 || !a2.total_committed) return `${k}: no commits`;
      return `${k} — ${a2.final_commit_accuracy_pct.toFixed(2)}% · ` +
             `${a2.total_committed} committed, ${a2.correct} correct, ${a2.incorrect} incorrect · ` +
             `${a2.prompts} prompts (${a2.prompt_rate_pct.toFixed(1)}%) · ` +
             `model ${a2.mean_latency_ms.toFixed(3)} ms`;
    };
    const rt = S.roundTrips.length
      ? ` · round trip ${(S.roundTrips.reduce((x, y) => x + y, 0) / S.roundTrips.length).toFixed(1)} ms`
      : "";
    const spd = sp
      ? `speed — ${sp.wpm.toFixed(1)} wpm · ${sp.cpm.toFixed(0)} cpm · ` +
        `${sp.chars} chars in ${(sp.mins * 60).toFixed(0)}s · ` +
        `commit-only ${sp.commit_wpm.toFixed(1)} wpm`
      : "speed — not measured";
    el.box.innerHTML =
      `<div class="ts-result"><b>TRIAL ${s2.trial} COMPLETE</b> — commits driven by ` +
      `<b>${s2.rule}</b>${rt}<br>${spd}<br>${line("quantum")}<br>${line("classical")}` +
      `<br>saved: eval/live_commit_log_{quantum,classical}.csv · live_commit_summary.json</div>`;
    console.log("[test-session] summary", s2, "speed", sp);
  }

  async function start() {
    buildUI();
    el.target.textContent = "starting…";
    const d = await post("/api/eval/start",
                         { trial: TRIAL, rule: RULE, n_sentences: N })
      .catch(() => null);
    if (!d || !d.ok) {
      el.target.textContent = "no server";
      return;
    }
    S.sentences = (d.sentences || []).map(normSentence).filter(Boolean);
    if (!S.sentences.length) {
      el.target.textContent = "no targets";
      return;
    }
    /* Freeze the header's height for the session.
     * The text preview grows as typed text wraps -- measured at 300px -> 354px
     * on a 768px viewport -- and because body is a flex column that pushes
     * every key up by ~18px mid-sentence. Gaze typing against a target that
     * moves under you is not the task we mean to measure, and the shift lands
     * in the middle of a trial. Pinned only while a session runs, so normal
     * use keeps its responsive header. */
    pin(document.querySelector("header"));

    /* The suggestion row is the worse offender: word chips wrap to a second
     * line when the words are long, which moved every key down 44px in the
     * middle of a sentence and back up again on the next one.
     *
     * Reserving two rows fixes the shift but costs 44px of keyboard, taking
     * key height from 126px to 87px -- and key size against gaze error is the
     * binding constraint on this whole system, so paying for stability in key
     * area is the wrong trade. Forbidding the wrap instead keeps the row one
     * line tall and the keys full size; a long chip is clipped horizontally,
     * which costs a peripheral convenience rather than target area. Choice
     * chips, the ones that actually resolve a prompt, are single letters and
     * never come close to overflowing. */
    const sug = document.getElementById("suggestions");
    if (sug) {
      sug.style.flexWrap = "nowrap";
      pin(sug);
    }

    S.active = true;
    window.__evalSession.active = true;
    console.log("[test-session] %d sentences · rule=%s · THETA=%s · phase_scale=%s · targets from %s",
                S.sentences.length, RULE, d.theta, d.phase_scale, d.source);
    console.log("[test-session] type each sentence exactly; do NOT correct mistakes");
    renderTarget(); renderSpeed();
  }

  // The keyboard checks for this object on every keystroke; defining it with
  // active=false keeps that check cheap and inert when no session is running.
  window.__evalSession = {
    active: false,
    onDecision,
    state: () => ({ ...S, series }),
    finish,
  };

  if (ACTIVE) window.addEventListener("load", start);
})();
