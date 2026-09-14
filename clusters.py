"""Shared cluster definitions for the eye-gaze keyboard.

The QWERTY keyboard is divided into spatial regions ("clusters"). Each key
maps to a single cluster digit, so a typed word becomes a digit string:
"hello" -> 51663 (9-cluster layout). These maps are the single source of
truth used by dataset preparation, training, decoding, and the web app.
"""

# Keyboard regions grouped by spatial proximity on a QWERTY layout
CLUSTER_MAPS = {
    9: {"qwe": "1", "rtyu": "2", "iop": "3", "asd": "4", "fgh": "5", "jkl": "6", "zxcv": "7", "bnm": "9"},
    7: {"qwert": "1", "yuiop": "2", "asdfg": "3", "hjkl": "4", "zxc": "5", "vbnm": "6"},
    5: {"qweasdz": "1", "rtfgxc": "2", "yuhjvb": "3", "iopklnm": "4"},
    3: {"qwertasdfgzxc": "1", "yuiophjklvbnm": "2"},
}

PREFIX = "translate Cluster to English: "


def build_char_map(num_clusters):
    """letter -> cluster digit, e.g. {'q': '1', 'w': '1', ...}"""
    char_map = {}
    for keys, digit in CLUSTER_MAPS[num_clusters].items():
        for ch in keys:
            char_map[ch] = digit
    return char_map


def encode_word(word, num_clusters):
    """Encode a single word into its cluster-digit code. Non-letters are dropped."""
    char_map = build_char_map(num_clusters)
    return "".join(char_map[c] for c in word.lower() if c in char_map)


def encode_text(text, num_clusters):
    """Encode a sentence: letters -> digits, spaces preserved, everything else dropped."""
    char_map = build_char_map(num_clusters)
    result = []
    for c in text.lower():
        if c.isalpha():
            result.append(char_map.get(c, ""))
        elif c == " ":
            result.append(" ")
    return "".join(result)


def is_consistent(sentence, cluster_sequence, num_clusters):
    """True iff `sentence` re-encodes to exactly `cluster_sequence`.

    This is the correctness guarantee used by the constrained decoder: any
    accepted hypothesis must map back to precisely what the user gazed.
    """
    return encode_text(sentence, num_clusters).split() == cluster_sequence.split()
