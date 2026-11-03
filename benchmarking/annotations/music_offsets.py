"""Pitch-labelled, offset GABC excerpts sourced from the pinned Gregobase dump."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from typing import Iterator

from benchmarking.annotations import gregobase
import warnings


_SQL_URL = gregobase.DEFAULT_CONFIG["gregobase_online_url"]
_LABEL_SPACE = re.compile(r"\s+")
_CLEF = re.compile(r"(?<![A-Za-z0-9])([cf])(b?)([1-4])(?![0-9])", re.IGNORECASE)
_ANNOTATION = re.compile(r"\[[^\]]*\]")
_NOTE_GROUP = re.compile(r"\(([^()]*)\)")


def _normalized_label(label: str) -> str:
    return _LABEL_SPACE.sub(" ", unicodedata.normalize("NFKC", label).strip()).casefold()


def _sql_rows(sql: str, column: str = "gabc") -> Iterator[str]:
    """Yield one SQL column from INSERT rows, honoring SQL quote escaping."""
    insert = re.compile(
        r"INSERT\s+INTO\s+`?gregobase_chants`?\s*\((.*?)\)\s*VALUES\s*",
        re.IGNORECASE | re.DOTALL,
    )
    pos = 0
    while match := insert.search(sql, pos):
        columns = [item.strip().strip("`") for item in match.group(1).split(",")]
        try:
            target = columns.index(column)
        except ValueError:
            pos = match.end()
            continue
        i = match.end()
        n = len(sql)
        while i < n:
            while i < n and (sql[i].isspace() or sql[i] == ","):
                i += 1
            if i >= n or sql[i] == ";":
                break
            if sql[i] != "(":
                break
            i += 1
            values: list[str | None] = []
            while i < n:
                while i < n and sql[i].isspace():
                    i += 1
                if i < n and sql[i] == ")":
                    i += 1
                    break
                if i < n and sql[i] == "'":
                    i += 1
                    chars: list[str] = []
                    while i < n:
                        if sql[i] == "'":
                            if i + 1 < n and sql[i + 1] == "'":
                                chars.append("'")
                                i += 2
                                continue
                            i += 1
                            break
                        # Decode the MySQL layer first; JSON escapes are then
                        # interpreted by json.loads in _gabc_payload.
                        if sql[i] == "\\" and i + 1 < n:
                            escaped = sql[i + 1]
                            chars.append({
                                "0": "\0", "b": "\b", "n": "\n",
                                "r": "\r", "t": "\t", "Z": "\x1a",
                                "\\": "\\", "'": "'", '"': '"',
                            }.get(escaped, escaped))
                            i += 2
                        else:
                            chars.append(sql[i])
                            i += 1
                    values.append("".join(chars))
                else:
                    start = i
                    while i < n and sql[i] not in ",)":
                        i += 1
                    value = sql[start:i].strip()
                    values.append(None if value.upper() == "NULL" else value)
                while i < n and sql[i].isspace():
                    i += 1
                if i < n and sql[i] == ",":
                    i += 1
                    continue
                if i < n and sql[i] == ")":
                    i += 1
                    break
                break
            if target < len(values) and values[target] is not None:
                yield values[target] or ""
            while i < n and sql[i].isspace():
                i += 1
            if i < n and sql[i] == ";":
                i += 1
                break
        pos = max(i, match.end())


def _gabc_payloads(serialized: str) -> list[str]:
    """Extract all GABC sections from a serialized editor blob."""
    try:
        editor_data = json.loads(serialized)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(editor_data, str):
        return [editor_data] if editor_data.strip() else []
    if not isinstance(editor_data, list):
        return []
    return [
        item[1]
        for item in editor_data
        if isinstance(item, list) and len(item) >= 2
        and item[0] == "gabc" and isinstance(item[1], str) and item[1].strip()
    ]


def _pitch_name(letter: str, clef: str) -> str:
    key = ("KC" if clef[0].lower() == "c" else "KF") + clef[1]
    return gregobase.PITCH_MAPPING[key][letter.lower()]


def _parse_groups(gabc: str) -> list[tuple[str, list[str], str]]:
    """Return note-bearing groups with pitches and the clef active at their start."""
    groups: list[tuple[str, list[str], str]] = []
    clef = "c4"
    for match in _NOTE_GROUP.finditer(gabc):
        body = match.group(1)
        start_clef = clef
        pitches: list[str] = []
        cursor = 0
        while cursor < len(body):
            annotation = _ANNOTATION.match(body, cursor)
            if annotation:
                cursor = annotation.end()
                continue
            clef_match = _CLEF.match(body, cursor)
            if clef_match:
                if clef_match.group(2):
                    warnings.warn(
                        "Skipping chant with a flat-marked clef unsupported by pitch labels",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return []
                clef = clef_match.group(1).lower() + clef_match.group(3)
                cursor = clef_match.end()
                continue
            char = body[cursor]
            if char.lower() in "xy":
                warnings.warn(
                    "Skipping chant with chromatic alteration unsupported by pitch labels",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return []
            if char.lower() in "abcdefghijklm":
                pitches.append(_pitch_name(char, clef))
            cursor += 1
        if pitches:
            unsupported_text = _ANNOTATION.sub(" ", _CLEF.sub(" ", body))
            unsupported = sorted({
                char for char in unsupported_text
                if char.isalpha() and char.lower() not in "abcdefghijklm"
                and char.lower() not in "opqrsuvwz"
            })
            if unsupported:
                warnings.warn(
                    f"Skipping chant with unhandled GABC notation letters {''.join(unsupported)!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return []
            groups.append((body, pitches, start_clef))
    return groups


def _chant_windows(gabc: str, source_id: str, split: str, section: int):
    groups = _parse_groups(gabc)
    # Offset and length count complete note-bearing neume groups, not pitches.
    for start in range(len(groups)):
        active_clef = groups[start][2]
        previous_clef = active_clef
        pitches: list[str] = []
        selected: list[str] = []
        for length in range(1, 16):
            index = start + length - 1
            if index >= len(groups):
                break
            body, group_pitches, group_clef = groups[index]
            if length > 1 and group_clef != previous_clef:
                selected.append(f"({group_clef})")
            selected.append(f"({body})")
            previous_clef = group_clef
            pitches.extend(group_pitches)
            if length < 7:
                continue
            # Embed each original neume intact, with the clef at the beginning
            # and all source clef-change neumes preserved as GABC groups.
            snippet = f"({active_clef}) " + " ".join(selected)
            yield {
                "gabc": snippet,
                "label": " ".join(pitches),
                "source_id": source_id,
                "start": start,
                "length": length,
                "split": split,
                "section": section,
            }


def _split_for(source_id: str) -> str:
    bucket = int(source_id[:16], 16) % 10
    return "val" if bucket < 2 else "train"


def _source_chunks(sql: str, size: int = 256):
    rng = random.Random(1675)
    chunk: list[tuple[str, str, int]] = []
    for serialized in _sql_rows(sql):
        sections = _gabc_payloads(serialized)
        if not sections:
            continue
        source_id = hashlib.sha256("\0".join(sections).encode("utf-8")).hexdigest()
        for section, gabc in enumerate(sections):
            chunk.append((source_id, gabc, section))
            if len(chunk) == size:
                rng.shuffle(chunk)
                yield from chunk
                chunk.clear()
    if chunk:
        rng.shuffle(chunk)
        yield from chunk


def _load_sql() -> str:
    # Use the repository's configured requests cache and pinned Gregobase URL.
    gregobase.setup_cache(gregobase.DEFAULT_CONFIG["cache_name"], gregobase.DEFAULT_CONFIG["cache_days"])
    response = gregobase.http_get(_SQL_URL, timeout=60)
    response.raise_for_status()
    content = response.content
    if "application/zip" in response.headers.get("Content-Type", ""):
        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            sql_name = next(name for name in archive.namelist() if name.endswith(".sql"))
            content = archive.read(sql_name)
    return content.decode("utf-8")


def iter_music_offsets(existing: set[str]) -> Iterator[dict[str, str | int]]:
    """Yield unique, pitch-labelled excerpts from cached Gregobase GABC.

    `existing` contains labels already present in the target corpus. It is read
    only; the iterator tracks labels it emits separately to prevent duplicates.
    """
    reserved = {_normalized_label(label) for label in existing}
    sql = _load_sql()
    for source_id, gabc, section in _source_chunks(sql):
        split = _split_for(source_id)
        for annotation in _chant_windows(gabc, source_id, split, section):
            key = _normalized_label(str(annotation["label"]))
            if not key or key in reserved:
                continue
            reserved.add(key)
            yield annotation
