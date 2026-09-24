# Actions for the Classical chat — offline σ-sweep replication

This is the test that produced `eval/commit_log_quantum.csv`,
`eval/commit_log_classical.csv` and the best-trial extracts. It is
deterministic: same seed, same sentences, same noise, no webcam, no human.

Companion to `eval/CLASSICAL_RUN_PROMPT.md`, which covers the live
gaze-typing session. Run this one first — it is reproducible and takes
seconds, so any protocol mismatch shows up before anyone sits in front of a
camera.

**The harness is fixed; the model is theirs.** Everything in the fenced block
below — key geometry, noise draws, sentences, threshold, scoring, CSV — must be
byte-identical. The gaze→letter decision rule and the language model are the
classical implementation's own. That is the only thing the comparison is
allowed to vary.

---

```
I have an existing Classical Eye-Gaze Typing implementation.

Run a deterministic offline benchmark of it against a FIXED test harness, and
write the results to CSV. This is an evaluation layer only.

=================================================================
0. WHAT IS FIXED AND WHAT IS YOURS
=================================================================

FIXED (must match this spec exactly, or the results are not comparable):
  key geometry, gaze noise model, RNG seed and draw order, the sentence list,
  the commit threshold, what counts as a commit, the scoring arithmetic, and
  the CSV schema.

YOURS (this is what is being measured):
  the gaze -> letter decision rule, the language model / prior, and any
  smoothing or fusion your system applies.

Do NOT modify the model to fit the harness. If something cannot be matched,
say so explicitly rather than approximating it silently.

=================================================================
1. HARD CONSTRAINTS
=================================================================

1.1  Do NOT modify, retune or "improve" the model in any way.
1.2  Do NOT fabricate, smooth, cherry-pick or post-hoc adjust any number.
1.3  Report all five trials. Do not report only the best one.
1.4  Add a test asserting the benchmark harness calls your model through the
     SAME code path normal typing uses. Report whether it passes.

=================================================================
2. KEYBOARD GEOMETRY (exact)
=================================================================

Three rows, in this order:
    row 0: qwertyuiop
    row 1: asdfghjkl
    row 2: zxcvbnm

KEY_W = 100.0
KEY_H = 110.0

For row index r and position i within that row:
    indent = (10 - len(row)) * KEY_W / 2.0
    x0 = indent + i * KEY_W
    y0 = r * KEY_H
    x1 = x0 + KEY_W
    y1 = y0 + KEY_H
    centre = ((x0 + x1) / 2, (y0 + y1) / 2)

There is no space key and no backspace key in this harness. Only a-z.

=================================================================
3. GAZE NOISE (exact)
=================================================================

For each intended character c, with centre (cx, cy) and noise level sigma:

    gx = cx + normal(0, sigma)
    gy = cy + normal(0, sigma)

RNG: numpy default_rng(7). A FRESH generator per trial, seeded 7 every time.
Draw order is exactly two draws per keystroke, x first then y. Do not
vectorise the draws or reorder them — the sequence is part of the protocol.

SIGMAS = (30.0, 40.0, 50.0, 60.0, 75.0)   -> trials 1..5, in this order.

=================================================================
4. GAZE POSTERIOR P(key | gaze) (exact)
=================================================================

The exact 2-D Gaussian integral over each key rectangle, axis-independent:

    Phi(t, mu, s) = 0.5 * (1 + erf((t - mu) / (s * sqrt(2))))

    p(key) = (Phi(x1, gx, sigma) - Phi(x0, gx, sigma))
           * (Phi(y1, gy, sigma) - Phi(y0, gy, sigma))

Keep keys with p > 1e-9, then normalise so they sum to 1. Feed THIS
distribution to your model as its gaze evidence. If your model takes a point
rather than a distribution, feed it (gx, gy) and say so in your report.

=================================================================
5. THE TYPING LOOP (exact)
=================================================================

for trial, sigma in enumerate(SIGMAS, start=1):
    rng = fresh generator seeded 7
    for each sentence in the list in section 7:
        for each word in sentence.split(" "):
            typed = ""                      # prefix resets at each word
            for each character ch in word:
                draw gx, gy                 # section 3
                build p_gaze                # section 4
                ask YOUR model for a decision given (p_gaze, typed)
                -> it returns (prediction, confidence in [0,1])
                record the keystroke        # section 6
                # prefix advances with the TRUTH when the system did not
                # commit, because a prompt yields the right character
                if confidence < THETA:
                    typed += ch
                else:
                    typed += prediction

THETA = 0.80

Spaces are word boundaries only. They are NOT typed and NOT scored: the space
bar is a far larger target than a letter key and including it inflates the
number. Every scored keystroke has an a-z ground truth.

=================================================================
6. WHAT COUNTS, AND THE METRICS
=================================================================

A COMMIT is a keystroke where confidence >= THETA. Below THETA the system
would prompt the user, who supplies the right character — so a prompt is NOT a
commit and must NOT be scored. Counting prompts as correct would let any
system reach 100% by refusing to decide.

Per committed keystroke:
    keystroke_index      committed keystrokes only, from 1, restarting each trial
    ground_truth         ch
    prediction           what the model returned
    correct              1 if prediction == ground_truth else 0
    cumulative_accuracy  (correct so far / committed so far) * 100

Also track per trial: keystrokes_presented, prompts, prompt_rate,
and mean/median decision latency in ms.

=================================================================
7. TEST SENTENCES — all 60, in exactly this order
=================================================================

 1. i feel unwell        2. well cook            3. take a bath
 4. he asked me to       5. are they gone        6. he stutters
 7. he is crazy          8. never mind tom       9. it isnt true
10. youre unkind        11. i could help        12. its a pipe
13. whos with me        14. i know her          15. have we met
16. stop laughing       17. i like you          18. toms scared
19. im a minister       20. there it is         21. i was warned
22. are you coming      23. cool down           24. im thrilled
25. i was mistaken      26. its bogus           27. leave tomorrow
28. save yourself       29. you stay here       30. i went twice
31. its bedtime         32. what can i do       33. she looks sad
34. open the safe       35. tom shot mary       36. i got nothing
37. go see tom          38. i was hurt          39. youre grumpy
40. i saw nothing       41. i have proof        42. he is lazy
43. i felt guilty       44. i eat beans         45. ill take tom
46. he is well off      47. can you read        48. head north
49. hes broke           50. im broke            51. im a bit shy
52. tom joined us       53. we made it          54. that was me
55. tom will talk       56. he hugged her       57. lets watch tv
58. i want a pony       59. tom listened        60. just go away

Already normalised to a-z plus single spaces — that is why they read "well
cook" and "it isnt true". Use them verbatim. Do not re-add apostrophes.

SELF-CHECK: 60 sentences, 599 letters excluding spaces. Each trial must
present exactly 599 keystrokes. If your count differs, your character handling
differs and nothing downstream is comparable — stop and report it.

=================================================================
8. OUTPUT FILES
=================================================================

8.1  commit_log_classical.csv — EXACTLY these columns, in this order,
     with all five trials concatenated:

trial,sentence,keystroke_index,ground_truth,prediction,correct,cumulative_accuracy

8.2  commit_log_classical_raw.csv — same rows plus:

top_posterior,decision_latency_ms,sigma_px,prefix

     top_posterior is your model's confidence in the committed character.
     Keep it. If your threshold differs from 0.80, this is the ONLY thing that
     makes a fair comparison still possible, by re-scoring both systems at
     MATCHED PROMPT RATE afterwards. Without it the run is not reusable.

8.3  commit_log_classical_summary.json:
     - per trial: sigma_px, final_commit_accuracy_pct, total_committed,
       correct, incorrect, keystrokes_presented, prompts, prompt_rate_pct,
       mean_latency_ms, median_latency_ms
     - pooled over all five trials: the same fields
     - every decision constant BY NAME AND VALUE, read from the source, with
       the file and line it came from

=================================================================
9. REPORT (print this table)
=================================================================

  trial  sigma   commit_acc%   committed  correct  incorrect  prompt_rate%
  1      30px    ...
  ...
  POOLED         ...

=================================================================
10. WHAT TO REPORT BACK TO ME
=================================================================

10.1  The three output files.
10.2  The keystrokes_presented count per trial. It must be 599.
10.3  Your model's own commit threshold and decision constants, with file and
      line. If it is not 0.80, say so loudly — I will re-score at matched
      prompt rate rather than compare the accuracy columns directly.
10.4  Whether your model consumes a distribution over keys or a single point.
10.5  Anything in this spec you could NOT match, and why. Do not paper over a
      mismatch. I am pairing this against a second implementation, and a silent
      difference — a different key size, a different noise draw order, scoring
      the space bar, a different prefix-advance rule — would invalidate the
      comparison without surfacing as an error anywhere.
```
