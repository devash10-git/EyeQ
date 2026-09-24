# Actions for the Classical-model Claude Code chat

Paste the fenced block below into the other chat. It is self-contained and
assumes no knowledge of this project.

What it mirrors on this side:

| piece | file here |
|---|---|
| session server + scoring | `eval/live_eval.py` |
| session UI, target display, live chart | `static/test-session.js` |
| commit hook | `static/app.js::smartCommit` → `applyDecision()` |
| targets | `tune.load_sentences(60, seed=4, skip=2060)[:20]`, normalised |

Run order: classical chat runs its session, sends back the three files; this
side runs `/?test=1&rule=quantum&n=20`; then the two logs are overlaid.

---

```
I have an existing Classical Eye-Gaze Typing implementation.

Build a REAL-TIME TRANSCRIPTION TEST SESSION for it, and an evaluation layer
that scores it live. This is an evaluation layer only.

=================================================================
1. HARD CONSTRAINTS
=================================================================

1.1  Do NOT modify, retune, or "improve" the model. Not the gaze estimation,
     not the calibration, not the language model, not the thresholds, not the
     selection/decision logic, not the dwell timing.
1.2  The session code must be strictly additive and OFF by default — gate it
     behind a URL flag or an explicit command, so normal use is untouched.
1.3  All metrics must come from REAL-TIME gaze typing: a live webcam session
     with a human actually gazing at keys. Do NOT simulate gaze with synthetic
     noise, do NOT replay a recording, and do NOT compute anything that needs
     the run to finish first.
1.4  Do NOT fabricate, smooth, cherry-pick or post-hoc adjust any number. If a
     result is bad, report it as it is.
1.5  Add a test that asserts the instrumented commit path produces the SAME
     per-keystroke decision as the uninstrumented model — exact equality, not a
     tolerance. Report whether it passes.

=================================================================
2. HOW THE TEST SESSION WORKS
=================================================================

2.1  The session shows the participant ONE target sentence at a time, in large
     text, with the character they are currently expected to type visually
     highlighted.
2.2  The participant gaze-types that sentence exactly, using the normal dwell
     mechanism. Nothing about dwell, calibration or thresholds changes.
2.3  Ground truth is POSITIONAL. Keep a pointer into the target string. The
     pointer advances by one on every keystroke DECISION — including a wrong
     commit, and including a keystroke the system declined to commit.
2.4  The participant must NOT correct mistakes mid-sentence. Per-keystroke
     ground truth depends on a fixed alignment and backspacing destroys it.
     Show this instruction on screen.
2.5  Provide only "skip" and "end & save" controls. Keep the session bar
     MINIMAL: the target word, the typing speed, progress, and those two
     buttons. Nothing else.

2.5a LAYOUT IS PART OF THE MEASUREMENT. The bar must not cover a single key,
     and no key may move or resize once typing has begun.
       - Do NOT use a fixed/absolute overlay. Make the bar a normal sibling in
         the layout flow so the keyboard shrinks to make room instead of being
         covered. Verify programmatically that ZERO keys intersect the bar.
       - Give the bar a fixed height and stop its contents wrapping. It may
         only grow after the session has ENDED.
       - Freeze anything else above the keyboard that can change height. In my
         implementation two things moved keys mid-sentence and both had to be
         pinned: the text preview grew as typed text wrapped (54px), and the
         word-suggestion row gained a second line when the words were long
         (44px). Prefer stopping the wrap over reserving a second row --
         reserving cost 31% of key height, and key size against gaze error is
         the binding constraint on accuracy.
       - Verify: record the key rectangles on every keystroke of a run and
         assert there is exactly ONE distinct key top and ONE distinct key
         height across the whole session. Report those counts.
2.6  When all sentences are done, stop the session and write the files.

=================================================================
3. TARGET SENTENCES — use exactly these 20, in this order
=================================================================

 1. i feel unwell
 2. well cook
 3. take a bath
 4. he asked me to
 5. are they gone
 6. he stutters
 7. he is crazy
 8. never mind tom
 9. it isnt true
10. youre unkind
11. i could help
12. its a pipe
13. whos with me
14. i know her
15. have we met
16. stop laughing
17. i like you
18. toms scared
19. im a minister
20. there it is

197 letters to type, excluding spaces.

These are ALREADY NORMALISED to the a-z + space alphabet — that is why they
read as "well cook" and "it isnt true". Display them exactly as written above.
Do not re-add apostrophes and do not re-normalise. The participant must read
precisely the string they are scored against, or they will gaze at a key that
does not exist and stall.

=================================================================
4. WHAT COUNTS AS A COMMIT
=================================================================

A COMMIT is a character the system entered ON ITS OWN, from its own confidence.

If the system declines and instead prompts / asks / shows candidate choices for
the user to pick from, that is NOT a commit:
  - exclude it from the main CSV entirely
  - still count it, so you can report the prompt rate
  - let the participant resolve the prompt normally; a character inserted by
    picking a suggestion is NOT a model commit and must NOT be scored as one

This distinction is the whole metric. A system that refuses to decide can
otherwise score 100% by never committing.

SCORE LETTERS ONLY. The space is typed to close a word — the language model
needs the boundary — but do NOT score it. The space bar is a much larger target
than a letter key and mixing the two inflates the number.

=================================================================
5. METRICS, per committed keystroke
=================================================================

  keystroke_index      committed keystrokes only, counting from 1
  ground_truth         expected character at the pointer
  prediction           character the model actually committed
  correct              1 if prediction == ground_truth else 0
  cumulative_accuracy  (total correct / total committed) * 100
  top_posterior        the model's confidence in the committed character
  decision_latency_ms  wall-clock time for the model to decide THIS keystroke,
                       measured around the decision itself

Track but keep out of the main CSV: keystrokes_presented, prompts, prompt_rate.

=================================================================
6. OUTPUT FILES
=================================================================

6.1  live_commit_log_classical.csv — EXACTLY these columns, in this order:

trial,sentence,keystroke_index,ground_truth,prediction,correct,cumulative_accuracy

6.2  live_commit_log_classical_raw.csv — the same rows plus:

top_posterior,decision_latency_ms,prefix

     ("prefix" = the word typed so far when this keystroke was decided.)
     The raw confidence is what makes it possible to re-score both systems at
     MATCHED PROMPT RATE later. If the two systems use different commit
     thresholds, comparing accuracy directly is meaningless — a higher
     threshold buys accuracy by prompting more. Without top_posterior that
     correction is impossible and the run is not reusable.

6.3  live_commit_summary_classical.json — containing:

     - final_commit_accuracy_pct, total_committed, correct, incorrect
     - keystrokes_presented, prompts, prompt_rate_pct
     - mean_latency_ms, median_latency_ms
     - every decision constant BY NAME AND VALUE, read from the source, with
       the file and line it came from (commit threshold, language-model
       exponent / weight, any prior floor, dwell duration)
     - the per-keystroke series, so the chart can be rebuilt elsewhere:

       {"series": {"x": [1,2,3,...],
                   "cumulative_accuracy_pct": [...],
                   "cumulative_mean_latency_ms": [...]}}

     Write all three files after EVERY keystroke, not at the end, so an aborted
     session keeps its data.

=================================================================
6a. TYPING SPEED
=================================================================

Measure entry rate live and show it in the session bar.

  elapsed        from the FIRST keystroke decision to the LAST one. Not to
                 "now" -- a pause after the final keystroke is not typing time,
                 and clocking to now makes the figure decay while the session
                 sits idle. Not from session start either, so reading the first
                 target does not count against the participant.
  chars_entered  length of the text actually produced, snapshotted AT the last
                 decision. Do not re-read it afterwards: a word-correction pass
                 can rewrite the field after a word boundary and make the
                 number drift when nobody is typing.
  wpm            (chars_entered / 5) / minutes     [5 chars = 1 word]
  cpm            chars_entered / minutes
  commit_wpm     (committed_chars / 5) / minutes   [the part the rule produced]

Count characters the participant resolved from a prompt. They cost time and
they produce text, so excluding them would overstate throughput. commit_wpm is
reported alongside for the part the selection rule produced on its own.

Whichever side owns the clock must own the final number. If the UI computes the
speed and the storage layer also computes one by sampling, they WILL disagree
-- they sample at different instants. Push the final figure once at the end and
have the saved file and the on-screen report be the same measurement. Verify
they match to 0.1 wpm.

Put wpm, cpm, commit_wpm, elapsed_s and chars_entered in the summary JSON.

=================================================================
7. LIVE VISUALIZATION
=================================================================

Draw the chart in a SEPARATE WINDOW or process, never on the typing page.
Anything drawn next to the keyboard either covers keys or shrinks them, and key
geometry is what is being measured. Have the page expose the running series and
let a separate viewer poll it.

The chart updates after EVERY committed character:

  X-axis: Keystroke Number (committed)
  Y-axis: Cumulative Commit Accuracy (%)

Scale the Y-axis to the data, NOT a fixed 0-100 — the whole signal lives in the
top few percent and a 0-100 axis renders it as a flat line at the top.

Second chart, same X-axis:  Y-axis = Cumulative Mean Decision Latency (ms)

If no display is available, fall back to writing a PNG periodically and SAY SO
explicitly rather than silently producing a non-live chart.

=================================================================
8. END-OF-TRIAL REPORT (print on screen and in the console)
=================================================================

  - Final commit accuracy (%)
  - Total committed keystrokes
  - Correct keystrokes
  - Incorrect keystrokes
  - Keystrokes presented, prompts, prompt rate (%)
  - Mean and median decision latency (ms)
  - wpm, cpm, commit_wpm, elapsed_s, chars_entered

=================================================================
9. SESSION CONDITIONS — record these, they decide comparability
=================================================================

Report in the summary JSON: participant id, screen resolution, physical
distance from screen, lighting, whether glasses were worn, calibration
procedure and how many points, dwell duration, and the wall-clock start time.

Run the whole 20-sentence set in ONE sitting without recalibrating mid-way.

=================================================================
10. WHAT TO REPORT BACK TO ME
=================================================================

10.1  The three output files.
10.2  Whether the non-interference test passed.
10.3  The decision constants you found, with file and line.
10.4  Anything in this spec you could NOT match, and why. Do not paper over a
      mismatch. I am pairing this against a second system's log from a
      different codebase, and a silent difference in protocol — a different
      commit definition, a different alphabet, scoring the space bar, allowing
      corrections — would invalidate the comparison without showing up as an
      error anywhere.
```

---

## Caveat to carry into the write-up

Two live sessions in two codebases are **not statistically paired** — different
fixations, different calibration, different fatigue. Keep them as close as
possible: same participant, same sitting, same lighting, back to back, and
alternate which runs first across trials.

The strictly paired alternative already exists on this side:
`eval/live_eval.py` scores **both** rules on identical gaze evidence within a
single session, because the keyboard types once and both rules decide on the
same posterior. If the classical implementation is close enough to port its
rule into that harness, prefer it — it removes the between-session variance
entirely.
