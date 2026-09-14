"""Cluster-constrained hybrid decoder.

Translating cluster digits back to English is ambiguous by nature: every
letter in a cluster shares the same digit, so distinct words can share a code
("fee" and "gee" are both 511 in the 9-cluster layout). A raw seq2seq model
can therefore hallucinate words that don't even match the gazed clusters.

This decoder removes that failure mode with three tiers:

1. Exact lookup   — word codes with a single known candidate decode
                    deterministically (100% accurate on unambiguous vocabulary).
2. Lattice search — ambiguous codes expand to their candidate words; a beam
                    search over the lattice picks the path with the best
                    bigram/unigram language-model score. Every path re-encodes
                    to the input by construction.
3. Seq2seq        — if a fine-tuned model is available, its beams are accepted
                    only when they re-encode exactly to the input clusters;
                    otherwise the model rescores the lattice paths. The model
                    disambiguates, but can never contradict the gaze input.

Only when a word code was never seen in training AND the model has no
consistent hypothesis does the decoder fall back to the model's raw guess,
and the result is flagged `consistent=False` so the UI can show it.
"""

import json
import math
from dataclasses import dataclass, field

from clusters import PREFIX, encode_word, is_consistent

UNK = "?"


@dataclass
class DecodeResult:
    text: str
    consistent: bool
    method: str
    word_candidates: list = field(default_factory=list)

    def to_dict(self):
        return {
            "text": self.text,
            "consistent": self.consistent,
            "method": self.method,
            "word_candidates": self.word_candidates,
        }


class HybridDecoder:
    def __init__(self, vocab_path="vocab-9.json", model_dir=None, num_clusters=9,
                 max_candidates=8, beam_width=8, rescore_top=16,
                 bigram_weight=0.99):
        self.num_clusters = num_clusters
        self.max_candidates = max_candidates
        self.beam_width = beam_width
        self.rescore_top = rescore_top
        #: Interpolation weight on the bigram estimate vs the unigram fallback
        #: in `_log_prob`, tuned against held-out sentences by `tune.py`.
        #:
        #: Measured: raising this from 0.7 to 0.99 lifts held-out sentence
        #: exact-match by ~4 percentage points (0.910 -> 0.950) and word
        #: accuracy by ~1.4 pp. The reason is that the bigram estimate is
        #: *already* add-one smoothed, so it can never be zero; the unigram
        #: term was duplicating that backoff and diluting the context signal.
        #: A small unigram share is retained for robustness on corpora with
        #: sparser bigram coverage than this one.
        self.bigram_weight = bigram_weight

        with open(vocab_path) as f:
            vocab = json.load(f)
        if vocab.get("num_clusters") != num_clusters:
            raise ValueError(
                f"vocab file was built for {vocab.get('num_clusters')} clusters, "
                f"decoder configured for {num_clusters}"
            )
        self.unigrams = vocab["unigrams"]
        self.total_unigrams = sum(self.unigrams.values())
        self.bigrams = vocab["bigrams"]
        self.prev_totals = {}
        for key, count in self.bigrams.items():
            prev = key.split(" ", 1)[0]
            self.prev_totals[prev] = self.prev_totals.get(prev, 0) + count

        # cluster code -> [(word, count), ...] most frequent first
        self.code_to_words = {}
        for word, count in self.unigrams.items():
            code = encode_word(word, num_clusters)
            self.code_to_words.setdefault(code, []).append((word, count))
        for code in self.code_to_words:
            self.code_to_words[code].sort(key=lambda wc: -wc[1])

        self.model = None
        self.tokenizer = None
        if model_dir:
            self._load_model(model_dir)

    def _load_model(self, model_dir):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.model.eval()

    # ---------------- n-gram scoring ----------------

    def _log_prob(self, word, prev):
        """Interpolated bigram/unigram log-probability with add-one smoothing."""
        uni = (self.unigrams.get(word, 0) + 1) / (self.total_unigrams + len(self.unigrams) + 1)
        big_count = self.bigrams.get(f"{prev} {word}", 0)
        prev_total = self.prev_totals.get(prev, 0)
        big = (big_count + 1) / (prev_total + len(self.unigrams) + 1)
        w = self.bigram_weight
        return math.log(w * big + (1.0 - w) * uni)

    def _lattice_paths(self, candidates_per_word):
        """Beam search over the word lattice; returns [(sentence_words, score)] best first."""
        beams = [([], 0.0)]
        for candidates in candidates_per_word:
            next_beams = []
            for words, score in beams:
                prev = words[-1] if words else "<s>"
                for word, _count in candidates:
                    next_beams.append((words + [word], score + self._log_prob(word, prev)))
            next_beams.sort(key=lambda b: -b[1])
            beams = next_beams[: max(self.beam_width, self.rescore_top)]
        return beams

    def _viterbi_best(self, candidates_per_word):
        """Exact best path through the word lattice (Viterbi).

        Under a bigram model the score of a path depends on the past only
        through its last word, so keeping the single best path per last-word
        state is lossless. Beam search keeps the top-K *paths*, which can
        drop a state entirely when several paths share a last word — this
        cannot. Returns (words, score).
        """
        # state: last word -> (best score, best path)
        states = {None: (0.0, [])}
        for candidates in candidates_per_word:
            nxt = {}
            for word, _count in candidates:
                best = None
                for prev_word, (score, path) in states.items():
                    prev = prev_word if prev_word is not None else "<s>"
                    s = score + self._log_prob(word, prev)
                    if best is None or s > best[0]:
                        best = (s, path)
                nxt[word] = (best[0], best[1] + [word])
            states = nxt
        score, path = max(states.values(), key=lambda sp: sp[0])
        return path, score

    # ---------------- seq2seq scoring ----------------

    def _model_generate_consistent(self, cluster_sequence):
        """Generate beams from the seq2seq model, keep the best cluster-consistent one."""
        torch = self.torch
        inputs = self.tokenizer(PREFIX + cluster_sequence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=self.beam_width,
                num_return_sequences=self.beam_width,
                early_stopping=True,
            )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for hyp in decoded:  # beams come back best-first
            if is_consistent(hyp, cluster_sequence, self.num_clusters):
                return hyp.strip(), decoded
        return None, decoded

    def _model_rescore(self, cluster_sequence, sentences):
        """Return the sentence with the lowest seq2seq loss for this input."""
        torch = self.torch
        src = [PREFIX + cluster_sequence] * len(sentences)
        enc = self.tokenizer(src, text_target=sentences, return_tensors="pt", padding=True)
        labels = enc["labels"].masked_fill(enc["labels"] == self.tokenizer.pad_token_id, -100)
        with torch.no_grad():
            logits = self.model(input_ids=enc["input_ids"],
                                attention_mask=enc["attention_mask"],
                                labels=labels).logits
        # per-sentence mean token loss (batch loss would average across sentences)
        losses = []
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        flat = loss_fn(logits.transpose(1, 2), labels)  # (batch, seq)
        for i in range(len(sentences)):
            mask = labels[i] != -100
            losses.append((flat[i][mask].mean().item(), sentences[i]))
        losses.sort()
        return losses[0][1]

    # ---------------- public API ----------------

    def candidates_for_code(self, code):
        return self.code_to_words.get(code, [])[: self.max_candidates]

    def suggest(self, prefix, n=6):
        """Top-n vocabulary words starting with `prefix`, most frequent first.

        Used by the individual-letter QWERTY keyboard for word autocomplete:
        the letters are exact (no cluster ambiguity), so this is a plain
        frequency-ranked prefix lookup over the unigram vocabulary.
        """
        prefix = prefix.lower()
        if not prefix:
            return []
        matches = [(w, c) for w, c in self.unigrams.items() if w.startswith(prefix)]
        matches.sort(key=lambda wc: -wc[1])
        return [w for w, _ in matches[:n]]

    def decode(self, cluster_sequence):
        cluster_sequence = " ".join(cluster_sequence.split())
        if not cluster_sequence:
            return DecodeResult("", True, "empty")

        codes = cluster_sequence.split()
        candidates_per_word = [self.candidates_for_code(code) for code in codes]
        word_candidates = [
            {"code": code, "candidates": [w for w, _ in cands]}
            for code, cands in zip(codes, candidates_per_word)
        ]
        all_known = all(candidates_per_word)

        # Tier 1: every code resolves to exactly one known word — deterministic.
        if all_known and all(len(c) == 1 for c in candidates_per_word):
            text = " ".join(c[0][0] for c in candidates_per_word)
            return DecodeResult(text, True, "exact-lookup", word_candidates)

        # Tier 3a: model beams, accepted only if they re-encode to the input.
        model_beams = []
        if self.model is not None:
            hyp, model_beams = self._model_generate_consistent(cluster_sequence)
            if hyp is not None:
                return DecodeResult(hyp, True, "seq2seq-constrained", word_candidates)

        if all_known:
            # Exact best path (Viterbi) — beam search only supplies the
            # additional candidates the model may rescore.
            best_words, _ = self._viterbi_best(candidates_per_word)
            best_text = " ".join(best_words)
            if self.model is not None:
                paths = self._lattice_paths(candidates_per_word)
                top, seen = [best_text], {best_text}
                for words, _score in paths[: self.rescore_top]:
                    t = " ".join(words)
                    if t not in seen:
                        seen.add(t)
                        top.append(t)
                # Tier 3b: the model picks among cluster-consistent lattice paths.
                if len(top) > 1:
                    best = self._model_rescore(cluster_sequence, top)
                    return DecodeResult(best, True, "seq2seq+lattice-rescored", word_candidates)
            # Tier 2: exact n-gram decoding (no model loaded, or single path).
            return DecodeResult(best_text, True, "lattice-viterbi", word_candidates)

        # Unknown word code(s): patch known words via lattice, leave the rest to
        # the model's raw beam if available, else mark with UNK.
        patched = []
        for code, cands in zip(codes, candidates_per_word):
            patched.append(cands[0][0] if cands else UNK)
        if model_beams:
            return DecodeResult(model_beams[0].strip(), False,
                                "seq2seq-unconstrained-fallback", word_candidates)
        return DecodeResult(" ".join(patched), False, "lookup-with-unknowns", word_candidates)
