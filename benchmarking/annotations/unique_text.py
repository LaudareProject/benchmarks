"""Finite, single-pass source of Biblioteca text, variants first, windows next."""

from __future__ import annotations

import random
import re
import unicodedata
from bisect import bisect_right
from collections.abc import Generator

from .bibliotecaitaliana import ABBREVIATIONS, get_line_generator


# Every period exposed by the Biblioteca Italiana catalog, in consumption
# priority order. Earlier periods are preferred for diplomatic OCR. The four
# earliest periods alone exhaust well below the production OCR target, so the
# full catalog keeps the finite source above that target.
PERIODS = ("Origini", "200", "300", "400", "500", "600", "700", "800", "900")
LANGUAGES = ("ita",)
EXCLUDED_LANGUAGES = ("lat",)


_PATTERNS = tuple(
    (word, re.compile(r"\b" + re.escape(word) + r"\b"), tuple(symbols))
    for word, symbols in ABBREVIATIONS.items()
)

# Legal variant spaces are tiny for almost every line (mean rule occurrences is
# below one), so exhaustive enumeration is cheap. A handful of very long
# paragraphs expand combinatorially; they are also far too wide to render as a
# single OCR line, so they are skipped instead of monopolizing generation.
MAX_VARIANTS_PER_LINE = 4096

# Long source lines cannot render as one OCR line, and lines without
# abbreviation hits are skipped by the variant phase. After each period's
# variants, such lines are cut into word-aligned windows that overlap by a
# quarter of their length. Every window is yielded plain first. One extra
# spelling per window then replaces between one and four occurrences. The
# 50-character cap covers most font and size combinations; the renderer still
# rejects windows that are too wide.
WINDOW_CHARS = 50
WINDOW_OVERLAP = 0.25
MAX_WINDOW_ABBREVIATIONS = 4


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


def _windows(
    text: str,
    width: int = WINDOW_CHARS,
    overlap: float = WINDOW_OVERLAP,
) -> Generator[str, None, None]:
    """Yield word-aligned windows that overlap by a fraction of their length."""
    if width < 1 or not 0.0 <= overlap < 1.0:
        raise ValueError("width must be positive and overlap must be in [0, 1)")
    words = text.split()
    if len(words) < 2:
        return
    joined = " ".join(words)
    if len(joined) <= width:
        return
    starts: list[int] = []
    position = 0
    for word in words:
        starts.append(position)
        position += len(word) + 1
    end_of_text = len(joined)
    index = 0
    while index < len(words):
        start = starts[index]
        limit = start + width
        if limit >= end_of_text:
            end = end_of_text
        else:
            boundary = bisect_right(starts, limit) - 1
            if boundary <= index:
                boundary = index
            end = starts[boundary] + len(words[boundary])
            if boundary > index and end > limit:
                boundary -= 1
                end = starts[boundary] + len(words[boundary])
        yield joined[start:end]
        if end >= end_of_text:
            break
        overlap_chars = int(round((end - start) * overlap))
        target = max(start, end - overlap_chars)
        next_index = bisect_right(starts, target) - 1
        if next_index <= index:
            next_index = index + 1
        index = next_index


def _one_spelling(
    text: str,
    rng: random.Random,
    limit: int = MAX_WINDOW_ABBREVIATIONS,
) -> str | None:
    """Return one spelling with one to `limit` abbreviations, or None."""
    hits: list[tuple[int, int, tuple[str, ...]]] = []
    for _word, pattern, replacements in _PATTERNS:
        for match in pattern.finditer(text):
            hits.append((match.start(), match.end(), replacements))
    if not hits:
        return None
    chosen = rng.sample(hits, min(limit, len(hits)))
    result = text
    for start, end, replacements in sorted(
        chosen, key=lambda item: item[0], reverse=True
    ):
        result = result[:start] + rng.choice(replacements) + result[end:]
    return result


def iter_unique_text(seed: int, existing: set[str]) -> Generator[str, None, None]:
    """Yield source strings whose normalized key is not in `existing`.

    Periods are consumed in `PERIODS` order. Within a period, all legal
    abbreviation variants of whole lines are exhausted first. Long raw lines are
    then cut into overlapping windows. Every plain window is yielded before one
    extra spelling per window, and each extra spelling uses between one and four
    abbreviations. Both window steps finish before the next period is fetched.
    The corpus is fetched with abbreviations disabled and raw lines are kept
    distinct by exact spelling because case-sensitive rules can produce
    different legal variants. `existing` is only read: this function keeps its
    own set of keys it has already handed out, so the caller may keep mutating
    `existing` with confirmed lines without affecting enumeration.
    """
    rng = random.Random(seed)
    reserved: set[str] = set()
    window_seen: set[str] = set()
    for period in PERIODS:
        candidates: list[tuple[int, int, str]] = []
        raw_lines: list[str] = []
        raw_seen: set[str] = set()
        skipped_long = 0
        raw_count = 0
        for line in get_line_generator(
            allowed_languages=list(LANGUAGES),
            periods=[period],
            abbreviations={},
            verbose=False,
            excluded_languages=list(EXCLUDED_LANGUAGES),
        ):
            if not line.strip() or line in raw_seen:
                continue
            raw_seen.add(line)
            raw_lines.append(line)
            raw_count += 1
            occurrences = _occurrences(line)
            if not any(occurrences):
                # Exactly one possible spelling: nothing new can come from this
                # line.
                continue
            count = _variant_count(occurrences)
            if count > MAX_VARIANTS_PER_LINE:
                skipped_long += 1
                continue
            candidates.append((count, len(line), line))
        # Release the raw dedupe set before yielding. Cross-period duplicates
        # stay filtered because the reserved key set below is global.
        raw_seen.clear()
        # Small spaces first: every variant of those lines is reachable, which
        # keeps the yield rate high and never lets one line monopolize the
        # stream.
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: (item[0], item[1]))
        print(
            f"Biblioteca source [{period}]: {raw_count:,} distinct raw lines; "
            f"{len(candidates):,} expandable lines; "
            f"{skipped_long:,} skipped as too long",
            flush=True,
        )

        for _count, _length, line in candidates:
            for text in _variants(line, rng):
                key = _key(text)
                if not key or key in existing or key in reserved:
                    continue
                reserved.add(key)
                yield text

        period_windows: list[str] = []
        plain_count = 0
        for line in raw_lines:
            for window in _windows(line):
                plain_key = _key(window)
                # An identical window was handled in an earlier visit, so its
                # plain form and its extra spelling already had their chance.
                if not plain_key or plain_key in window_seen:
                    continue
                window_seen.add(plain_key)
                period_windows.append(window)
                if plain_key in existing or plain_key in reserved:
                    continue
                reserved.add(plain_key)
                plain_count += 1
                yield window

        spelling_count = 0
        for window in period_windows:
            spelling = _one_spelling(window, rng)
            if spelling is None:
                continue
            key = _key(spelling)
            if not key or key in existing or key in reserved:
                continue
            reserved.add(key)
            spelling_count += 1
            yield spelling
        print(
            f"Biblioteca windows [{period}]: {plain_count:,} plain, "
            f"{spelling_count:,} spellings",
            flush=True,
        )
