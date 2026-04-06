"""
Adversarial Attack Module for Spam Detection Research
=====================================================
Implements multiple perturbation strategies at different linguistic levels:
  - Character-level: leetspeak, homoglyph, insertion, deletion
  - Word-level: synonym injection, benign word stuffing
  - Sentence-level: structure shuffling
"""

import random
import re
import string

# ─── CHARACTER-LEVEL ATTACKS ──────────────────────────────────────────────────

# Leetspeak substitution mapping
LEET_MAP = {
    'a': '@', 'e': '3', 'i': '1', 'o': '0',
    's': '$', 'l': '1', 't': '7', 'g': '9',
    'b': '8', 'z': '2',
}

# Homoglyph substitution (visually identical unicode chars)
HOMOGLYPH_MAP = {
    'a': 'а',   # Cyrillic а
    'e': 'е',   # Cyrillic е
    'o': 'о',   # Cyrillic о
    'p': 'р',   # Cyrillic р
    'c': 'с',   # Cyrillic с
    'x': 'х',   # Cyrillic х
    'i': 'і',   # Ukrainian і
    's': 'ѕ',   # Cyrillic ѕ
}


def attack_leetspeak(text, intensity=0.5):
    """Replace characters with leetspeak equivalents."""
    words = text.split()
    result = []
    for word in words:
        if random.random() < intensity:
            new_word = "".join(LEET_MAP.get(c.lower(), c) for c in word)
            result.append(new_word)
        else:
            result.append(word)
    return " ".join(result)


def attack_homoglyph(text, intensity=0.3):
    """Replace characters with visually identical Unicode homoglyphs."""
    result = []
    for char in text:
        if random.random() < intensity and char.lower() in HOMOGLYPH_MAP:
            result.append(HOMOGLYPH_MAP[char.lower()])
        else:
            result.append(char)
    return "".join(result)


def attack_char_insert(text, intensity=0.15):
    """Insert random characters into words to break tokenization."""
    words = text.split()
    result = []
    for word in words:
        if len(word) > 3 and random.random() < intensity:
            pos = random.randint(1, len(word) - 1)
            char = random.choice(string.ascii_lowercase)
            word = word[:pos] + char + word[pos:]
        result.append(word)
    return " ".join(result)


def attack_char_delete(text, intensity=0.15):
    """Delete random characters from words."""
    words = text.split()
    result = []
    for word in words:
        if len(word) > 4 and random.random() < intensity:
            pos = random.randint(1, len(word) - 2)
            word = word[:pos] + word[pos + 1:]
        result.append(word)
    return " ".join(result)


def attack_char_swap(text, intensity=0.2):
    """Swap adjacent characters within words (typo simulation)."""
    words = text.split()
    result = []
    for word in words:
        if len(word) > 3 and random.random() < intensity:
            chars = list(word)
            pos = random.randint(0, len(chars) - 2)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            word = "".join(chars)
        result.append(word)
    return " ".join(result)


# ─── WORD-LEVEL ATTACKS ──────────────────────────────────────────────────────

BENIGN_WORDS = [
    "hello", "thanks", "meeting", "schedule", "project",
    "report", "update", "please", "kindly", "regards",
    "team", "office", "document", "review", "calendar",
    "lunch", "tomorrow", "morning", "afternoon", "sincerely",
]

SPAM_SYNONYMS = {
    "free": ["complimentary", "no-cost", "gratis", "costless"],
    "win": ["earn", "receive", "claim", "collect"],
    "money": ["cash", "funds", "prize", "reward"],
    "click": ["tap", "press", "visit", "go to"],
    "urgent": ["immediate", "critical", "time-sensitive", "asap"],
    "offer": ["deal", "promotion", "discount", "bargain"],
    "buy": ["purchase", "order", "acquire", "get"],
    "prize": ["award", "jackpot", "bonus", "gift"],
    "call": ["contact", "ring", "dial", "reach"],
    "now": ["immediately", "today", "instantly", "right away"],
}


def attack_synonym_replace(text, intensity=0.5):
    """Replace spam-indicative words with synonyms to evade keyword detection."""
    words = text.split()
    result = []
    for word in words:
        clean = word.lower().strip(string.punctuation)
        if clean in SPAM_SYNONYMS and random.random() < intensity:
            replacement = random.choice(SPAM_SYNONYMS[clean])
            # Preserve original casing roughly
            if word[0].isupper():
                replacement = replacement.capitalize()
            result.append(replacement)
        else:
            result.append(word)
    return " ".join(result)


def attack_benign_inject(text, num_words=3):
    """Inject benign words into a spam message to dilute spam signals."""
    words = text.split()
    for _ in range(num_words):
        pos = random.randint(0, len(words))
        benign = random.choice(BENIGN_WORDS)
        words.insert(pos, benign)
    return " ".join(words)


# ─── SENTENCE-LEVEL ATTACKS ──────────────────────────────────────────────────

def attack_word_shuffle(text, intensity=0.3):
    """Shuffle word order within the message (preserving some structure)."""
    words = text.split()
    n_swap = max(1, int(len(words) * intensity))
    for _ in range(n_swap):
        if len(words) > 2:
            i = random.randint(0, len(words) - 2)
            words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def attack_invisible_chars(text, intensity=0.2):
    """Insert zero-width characters between letters to break tokenization."""
    zwsp = '\u200b'  # Zero-width space
    result = []
    for char in text:
        result.append(char)
        if char.isalpha() and random.random() < intensity:
            result.append(zwsp)
    return "".join(result)


# ─── ATTACK REGISTRY ─────────────────────────────────────────────────────────

ATTACK_REGISTRY = {
    "leetspeak": {
        "fn": attack_leetspeak,
        "level": "character",
        "description": "Substitutes characters with leetspeak equivalents (e→3, a→@)",
    },
    "homoglyph": {
        "fn": attack_homoglyph,
        "level": "character",
        "description": "Replaces Latin characters with visually identical Cyrillic/Unicode characters",
    },
    "char_insert": {
        "fn": attack_char_insert,
        "level": "character",
        "description": "Inserts random characters into words to break tokenization",
    },
    "char_delete": {
        "fn": attack_char_delete,
        "level": "character",
        "description": "Deletes random characters from words",
    },
    "char_swap": {
        "fn": attack_char_swap,
        "level": "character",
        "description": "Swaps adjacent characters within words (simulates typos)",
    },
    "synonym_replace": {
        "fn": attack_synonym_replace,
        "level": "word",
        "description": "Replaces spam-indicative keywords with synonyms",
    },
    "benign_inject": {
        "fn": attack_benign_inject,
        "level": "word",
        "description": "Injects benign words to dilute spam signal density",
    },
    "word_shuffle": {
        "fn": attack_word_shuffle,
        "level": "sentence",
        "description": "Shuffles word ordering within the message",
    },
    "invisible_chars": {
        "fn": attack_invisible_chars,
        "level": "sentence",
        "description": "Inserts zero-width Unicode characters between letters",
    },
}


def apply_attack(text, attack_name, **kwargs):
    """Apply a named attack to text."""
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {attack_name}. Available: {list(ATTACK_REGISTRY.keys())}")
    return ATTACK_REGISTRY[attack_name]["fn"](text, **kwargs)


def apply_all_attacks(text):
    """Apply every attack strategy and return results dict."""
    results = {}
    for name, info in ATTACK_REGISTRY.items():
        try:
            results[name] = {
                "perturbed_text": info["fn"](text),
                "level": info["level"],
                "description": info["description"],
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return results
