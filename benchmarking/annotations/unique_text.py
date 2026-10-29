"""Finite, single-pass source of legal Biblioteca abbreviation variants."""

from __future__ import annotations

import random
import re
import unicodedata
from collections.abc import Generator

from .bibliotecaitaliana import ABBREVIATIONS, get_line_generator


_PATTERNS = tuple(
    (word, re.compile(r"\b" + re.escape(word) + r"\b"), tuple(symbols))
    for word, symbols in ABBREVIATIONS.items()
)

# Legal variant spaces are tiny for almost every line (mean rule occurrences is
# below one), so exhaustive enumeration is cheap. A handful of very long
# paragraphs expand combinatorially; they are also far too wide to render as a
# single OCR line, so they are skipped instead of monopolizing generation.
MAX_VARIANTS_PER_LINE = 4096


def _key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _occurrences(text: str) -> list[int]:
    return [len(pattern.findall(text)) for _word, pattern, _reps in _PATTERNS]


def _variant_count(occurrences: list[int]) -> int:
    count = 1
    for (_word, _pattern, replacements), hits in zip(_PATTERNS, occurrences):
        if hits:
            count *= (len(replacements) + 1) ** hits
    return count


def _variants(raw: str, rng: random.Random) -> Generator[str, None, None]:
    """Enumerate every replacement combination, using the source's rule order."""
    def apply_rule(index: int, text: str) -> Generator[str, None, None]:
        while index < len(_PATTERNS):
            _word, pattern, replacements = _PATTERNS[index]
            matches = list(pattern.finditer(text))
            if matches:
                break
            index += 1
        else:
            yield text
            return
        radix = len(replacements) + 1
        count = radix ** len(matches)
        # Rotate the combined choice space by a seeded offset, so the first
        # variants of a line are random rather than "no replacement".
        offset = rng.randrange(count)
        for step in range(count):
            choices = (step + offset) % count
            result = text
            for match in reversed(matches):
                choice = choices % radix
                choices //= radix
                if choice:
                    start, end = match.span()
                    result = result[:start] + replacements[choice - 1] + result[end:]
            yield from apply_rule(index + 1, result)

    yield from apply_rule(0, raw)


def iter_unique_text(seed: int, existing: set[str]) -> Generator[str, None, None]:
    """Yield legal source strings whose normalized key is not in `existing`.

    The Biblioteca corpus is fetched exactly once with abbreviations disabled;
    raw lines are kept distinct by exact spelling because case-sensitive rules
    can produce different legal variants. `existing` is only read: this function
    keeps its own set of keys it has already handed out, so the caller may keep
    mutating `existing` with confirmed lines without affecting enumeration.
    """
    rng = random.Random(seed)
    candidates: list[tuple[int, int, str]] = []
    skipped_long = 0
    raw_count = 0
    raw_seen: set[str] = set()
    for line in get_line_generator(
        allowed_languages=["ita"],
        periods=["Origini", "200", "300", "400"],
        abbreviations={},
        verbose=False,
    ):
        if not line.strip() or line in raw_seen:
            continue
        raw_seen.add(line)
        raw_count += 1
        occurrences = _occurrences(line)
        if not any(occurrences):
            # Exactly one possible spelling: nothing new can come from this line.
            continue
        count = _variant_count(occurrences)
        if count > MAX_VARIANTS_PER_LINE:
            skipped_long += 1
            continue
        candidates.append((count, len(line), line))
    raw_seen.clear()
    # Small spaces first: every variant of those lines is reachable, which keeps
    # the yield rate high and never lets one line monopolize the stream.
    rng.shuffle(candidates)
    candidates.sort(key=lambda item: (item[0], item[1]))
    print(
        f"Biblioteca source: {raw_count:,} distinct raw lines; "
        f"{len(candidates):,} expandable lines; {skipped_long:,} skipped as too long",
        flush=True,
    )

    reserved: set[str] = set()
    for _count, _length, line in candidates:
        for text in _variants(line, rng):
            key = _key(text)
            if not key or key in existing or key in reserved:
                continue
            reserved.add(key)
            yield text
