"""Letter-level language evidence for the QWERTY gaze keyboard.

Two questions the keyboard asks at commit time, answered from the same
16k-word frequency vocabulary the decoder already uses:

1. letter_prior(prefix)  — P(next letter | word typed so far), plus "$" for
   "the word ends here" (which the keyboard maps to SPACE). Computed from a
   frequency-weighted prefix trie, backed off to a character 4-gram when the
   prefix leaves the vocabulary.

2. correct(word)         — gaze-aware noisy-channel correction for a word
   that is not in the vocabulary: argmax P(word) * P(typed | word), where the
   channel model is a weighted edit distance whose substitution cost grows
   with physical key distance on the QWERTY layout. Confusing H with J is
   cheap; confusing H with Q is nearly impossible — exactly the error
   profile a gaze keyboard produces.
"""

import math

END = "$"  # end-of-word symbol in priors (keyboard maps it to SPACE)

# QWERTY key centres in key-pitch units (row stagger matches a standard board)
_ROWS = [("qwertyuiop", 0.0, 0.0), ("asdfghjkl", 0.25, 1.0), ("zxcvbnm", 0.75, 2.0)]
KEY_POS = {}
for row, xoff, yrow in _ROWS:
    for i, ch in enumerate(row):
        KEY_POS[ch] = (xoff + i, yrow)


def key_distance(a, b):
    """Physical distance between two letter keys, in key pitches."""
    if a not in KEY_POS or b not in KEY_POS:
        return 3.0
    (ax, ay), (bx, by) = KEY_POS[a], KEY_POS[b]
    return math.hypot(ax - bx, ay - by)


def _sub_cost(a, b):
    """Substitution cost from key geometry: adjacent ~0.55, far ~1.5."""
    if a == b:
        return 0.0
    return 0.3 + 1.2 * min(1.0, key_distance(a, b) / 3.0)


class LetterModel:
    def __init__(self, unigrams, char_order=3, trie_mix=0.85, smoothing_k=0.1):
        """Build the letter-prediction model.

        Args:
            unigrams: ``{word: count}`` — the decoder's existing vocabulary.
            char_order: context length of the character n-gram (number of
                preceding characters). Higher order is sharper but sparser;
                tuned empirically in ``tune.py``.
            trie_mix: weight on the exact vocabulary trie versus the smoothed
                character n-gram. The trie is exact where it has mass, so this
                should be high; the n-gram covers the out-of-vocabulary tail.
            smoothing_k: add-k smoothing constant for the character n-gram.
        """
        self.unigrams = unigrams
        self.char_order = int(char_order)
        self.trie_mix = float(trie_mix)
        self.smoothing_k = float(smoothing_k)
        self._by_len = {}
        for w, c in unigrams.items():
            self._by_len.setdefault(len(w), []).append((w, c))

        # frequency-weighted prefix trie: node = [total, end, {ch: node}]
        self.trie = [0, 0, {}]
        for w, c in unigrams.items():
            node = self.trie
            node[0] += c
            for ch in w:
                node = node[2].setdefault(ch, [0, 0, {}])
                node[0] += c
            node[1] += c

        # character n-gram (context = previous `char_order` chars incl. "^"
        # padding), frequency-weighted, with add-k smoothing at query time
        pad = "^" * self.char_order
        self.chargrams = {}
        for w, c in unigrams.items():
            s = pad + w + END
            for i in range(self.char_order, len(s)):
                ctx, ch = s[i - self.char_order:i], s[i]
                d = self.chargrams.setdefault(ctx, {})
                d[ch] = d.get(ch, 0) + c

        self.alphabet = [chr(o) for o in range(ord("a"), ord("z") + 1)] + [END]
        self._alpha_set = set(self.alphabet)

    # ---------------- letter prior ----------------

    def _char_dist(self, prefix):
        """Char-gram distribution, conditioned on the keyboard's alphabet.

        The vocabulary contains apostrophes ("don't", "it's"), but the
        keyboard can only produce a-z and a word boundary. Probabilities are
        therefore normalized over the alphabet alone — otherwise the mass on
        unreachable continuations silently disappears from the result.
        """
        ctx = ("^" * self.char_order + prefix)[-self.char_order:]
        counts = self.chargrams.get(ctx, {})
        total = sum(c for ch, c in counts.items() if ch in self._alpha_set)
        k = self.smoothing_k
        denom = total + k * len(self.alphabet)
        return {ch: (counts.get(ch, 0) + k) / denom for ch in self.alphabet}

    def letter_prior(self, prefix):
        """P(next symbol | prefix) over a-z plus END. Sums to 1."""
        prefix = prefix.lower()
        char_d = self._char_dist(prefix)

        node = self.trie
        for ch in prefix:
            node = node[2].get(ch)
            if node is None:
                return char_d  # left the vocabulary: char-gram alone

        # condition on reachable continuations only (see _char_dist)
        total = node[1] + sum(c[0] for ch, c in node[2].items() if ch in self._alpha_set)
        if total <= 0:
            return char_d  # only unreachable continuations here
        trie_d = {ch: 0.0 for ch in self.alphabet}
        trie_d[END] = node[1] / total
        for ch, child in node[2].items():
            if ch in self._alpha_set:
                trie_d[ch] = child[0] / total
        # trie is exact where it has mass; char-gram smooths the tail
        m = self.trie_mix
        return {ch: m * trie_d[ch] + (1.0 - m) * char_d[ch] for ch in self.alphabet}

    def completions(self, prefix, n=32):
        """Vocabulary continuations of ``prefix``, as ``(word, weight)`` pairs.

        The hypothesis set behind ``letter_prior``. That method sums this list
        into a per-letter marginal; callers that want to know *how* the mass was
        distributed — one dominant word, or a diffuse spray that happens to
        share a next character — need the list itself.

        Read straight off the existing frequency-weighted trie, so the weights
        are exactly the ones ``letter_prior`` aggregates: for any letter ``c``,
        ``sum(w for w, word in completions if word[len(prefix)] == c)`` is that
        letter's trie branch mass.

        Args:
            prefix: the word typed so far.
            n: keep only the ``n`` heaviest completions. The tail is long and
                individually negligible, and every consumer normalises.

        Returns:
            ``[(word, weight)]`` sorted by descending weight. Empty when the
            prefix has left the vocabulary.
        """
        prefix = prefix.lower()
        node = self.trie
        for ch in prefix:
            node = node[2].get(ch)
            if node is None:
                return []

        out = []
        stack = [(prefix, node)]
        while stack:
            word, nd = stack.pop()
            if nd[1] > 0:
                out.append((word, float(nd[1])))
            for ch, child in nd[2].items():
                if ch in self._alpha_set:
                    stack.append((word + ch, child))
        out.sort(key=lambda t: -t[1])
        return out[:n]

    # ---------------- noisy-channel correction ----------------

    def _edit_cost(self, typed, word, band=2):
        """Banded weighted edit distance with geometric substitution costs."""
        n, m = len(typed), len(word)
        INF = 1e9
        prev = [j * 0.9 if j <= band else INF for j in range(m + 1)]
        for i in range(1, n + 1):
            cur = [INF] * (m + 1)
            if i <= band:
                cur[0] = i * 0.9
            lo, hi = max(1, i - band), min(m, i + band)
            for j in range(lo, hi + 1):
                cur[j] = min(
                    prev[j] + 0.9,                                   # delete
                    cur[j - 1] + 0.9,                                # insert
                    prev[j - 1] + _sub_cost(typed[i - 1], word[j - 1]),
                )
            prev = cur
        return prev[m]

    def correct(self, typed, n=3, max_cost=2.6):
        """Top-n vocabulary corrections for an out-of-vocab word."""
        typed = typed.lower()
        if not typed:
            return {"in_vocab": False, "suggestions": []}
        if typed in self.unigrams:
            return {"in_vocab": True, "suggestions": []}
        scored = []
        for L in range(max(1, len(typed) - 2), len(typed) + 3):
            for w, c in self._by_len.get(L, []):
                cost = self._edit_cost(typed, w)
                if cost <= max_cost:
                    scored.append((math.log(c + 1) - 2.2 * cost, w))
        scored.sort(reverse=True)
        return {"in_vocab": False, "suggestions": [w for _, w in scored[:n]]}
