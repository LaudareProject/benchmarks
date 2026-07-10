from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

CLEF_RE = re.compile(r"K([CF])\s*([0-9])", re.IGNORECASE)
NOTE_RE = re.compile(r"([A-G])([b#]?)([0-9])", re.IGNORECASE)
DELIM_RE = re.compile(r"/+", re.IGNORECASE)
DEFAULT_CLEF = "KC3"


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char.isspace() or char == ")":
            i += 1
            continue
        if char == "(":
            end = text.find(")", i + 1)
            if end == -1:
                end = len(text)
                tokens.append(text[i:end])
                break
            tokens.append(text[i : end + 1])
            i = end + 1
            continue
        if char == "/":
            j = i
            while j < len(text) and text[j] == "/":
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
        j = i
        while j < len(text) and not text[j].isspace() and text[j] not in "()/":
            j += 1
        if j == i:
            i += 1
            continue
        tokens.append(text[i:j])
        i = j
    return tokens


def _normalize_note(match: tuple[str, str, str]) -> str:
    letter, accidental, octave = match
    return f"{letter.upper()}{accidental.lower()}{octave}"


def _extract_notes(text: str) -> list[str]:
    return [_normalize_note(match) for match in NOTE_RE.findall(text)]


def _normalize_clef(text: str) -> str | None:
    match = CLEF_RE.search(text)
    if not match:
        return None
    clef, octave = match.groups()
    return f"K{clef.upper()}{octave}"


def _normalize_delimiter(text: str) -> str | None:
    match = DELIM_RE.fullmatch(text.strip())
    if not match:
        return None
    return "//" if len(match.group(0)) >= 2 else "/"


def _normalize_group(token: str) -> str | None:
    notes = _extract_notes(token)
    if not notes:
        return None
    return f"({' '.join(notes)})"


def _normalize_free_token(token: str) -> str | None:
    delimiter = _normalize_delimiter(token)
    if delimiter is not None:
        return delimiter

    clef = _normalize_clef(token)
    if clef is not None:
        return clef

    notes = _extract_notes(token)
    if not notes:
        return None
    if len(notes) == 1:
        return notes[0]
    return f"({' '.join(notes)})"


def normalize_omr_tokens(text: str) -> list[str]:
    normalized: list[str] = []
    for token in _tokenize(text):
        value = _normalize_group(token) if token.startswith("(") else _normalize_free_token(token)
        if value:
            normalized.append(value)
    return normalized


def _is_clef(token: str) -> bool:
    return _normalize_clef(token) is not None


def _majority_clef(tokens: list[str]) -> str:
    clefs = [_normalize_clef(token) for token in tokens if _is_clef(token)]
    clefs = [clef for clef in clefs if clef]
    if not clefs:
        return DEFAULT_CLEF
    return Counter(clefs).most_common(1)[0][0]


def _split_segments(tokens: list[str]) -> tuple[list[list[str]], list[str]]:
    segments: list[list[str]] = []
    delimiters: list[str] = []
    current: list[str] = []
    for token in tokens:
        if token in {"/", "//"}:
            if current:
                segments.append(current)
                current = []
            elif segments:
                delimiters[-1] = "//"
            delimiters.append(token)
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments, delimiters[: max(0, len(segments) - 1)]


def _normalize_segment(segment: list[str], fallback_clef: str) -> tuple[list[str], str]:
    tokens = [token for token in segment if token]
    if not tokens:
        return [], fallback_clef

    first_clef_idx = next((idx for idx, token in enumerate(tokens) if _is_clef(token)), None)
    if first_clef_idx is None:
        tokens.insert(0, fallback_clef)
        first_clef_idx = 0
    elif first_clef_idx > 0:
        clef = _normalize_clef(tokens[first_clef_idx]) or fallback_clef
        tokens = [clef] + tokens[:first_clef_idx] + tokens[first_clef_idx + 1 :]
    else:
        tokens[0] = _normalize_clef(tokens[0]) or fallback_clef

    leading = [tokens[0]]
    idx = 1
    while idx < len(tokens) and _is_clef(tokens[idx]):
        idx += 1
    tokens = leading + tokens[idx:]

    return tokens, tokens[0]


def postprocess_omr_text(text: str) -> str:
    tokens = normalize_omr_tokens(text)
    if not tokens:
        return ""

    segments, delimiters = _split_segments(tokens)
    if not segments:
        return ""

    fallback_clef = _majority_clef(tokens)
    normalized_segments: list[list[str]] = []
    for segment in segments:
        normalized, fallback_clef = _normalize_segment(segment, fallback_clef)
        if normalized:
            normalized_segments.append(normalized)

    if not normalized_segments:
        return ""

    rebuilt: list[str] = []
    for idx, segment in enumerate(normalized_segments):
        if idx > 0 and idx - 1 < len(delimiters):
            rebuilt.append(delimiters[idx - 1])
        rebuilt.extend(segment)
    return " ".join(rebuilt)


def postprocess_prediction_dir(predictions_dir: Path) -> tuple[int, int]:
    total = 0
    changed = 0
    for path in sorted(predictions_dir.glob("*.pred.txt")):
        original = path.read_text(encoding="utf-8").strip()
        updated = postprocess_omr_text(original)
        total += 1
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return total, changed
