"""
Gregobase music pages generator module.
Generates Gregorian music pages one by one in a similar way to bibliotecaitaliana.py.
"""

import datetime
import io
import os
import re
import shutil
import subprocess
import random
import zipfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, Optional, List, Tuple
from collections import deque

import cv2
import numpy as np
import requests
from fitz import Document
from jinja2 import Environment, BaseLoader

# Default configuration
DEFAULT_CONFIG = {
    "dpi": 300,
    "font_path": "gregoriofonts",
    "font_url":
    "https://github.com/gregorio-project/gregorio/releases/download/v<gregorioversion>/supp_fonts-<gregorioversion_underscore>.zip",
    "gregobase_online_url":
    "https://raw.githubusercontent.com/gregorio-project/GregoBase/2ebcda3f523f9b19933d59fa32bb4215cd8e7675/gregobase_online.sql",
    "template": "templates/gregorio_template.tex",
    "cache_name": "gregobase_cache",
    "cache_days": 30,
}

random.seed(1675)
# Matches patterns like "(abc def)" and captures "abc def"
GREGORIO_REGEX = re.compile(r"\(([^)]*)\)")

# Regex for parsing GABC notation - captures clefs and notes
# CLEF_REGEX: Matches clef notations (c1-4 or f1-4) - e.g., "c3", "f4"
# NOTE_REGEX: Matches individual note letters (a-m, case insensitive) - e.g., "a", "B", "g"
CLEF_REGEX = re.compile(r"([cf][1-4])")
NOTE_REGEX = re.compile(r"([a-mA-M])")

# Define the pitch mapping for each clef
# Each clef defines what note 'a' corresponds to
PITCH_MAPPING = {
    "KF1": {
        "a": "C4",
        "b": "D4",
        "c": "E4",
        "d": "F4",
        "e": "G4",
        "f": "A4",
        "g": "B4",
        "h": "C5",
        "i": "D5",
        "j": "E5",
        "k": "F5",
        "l": "G5",
        "m": "A5",
    },
    "KF2": {
        "a": "A3",
        "b": "B3",
        "c": "C4",
        "d": "D4",
        "e": "E4",
        "f": "F4",
        "g": "G4",
        "h": "A4",
        "i": "B4",
        "j": "C5",
        "k": "D5",
        "l": "E5",
        "m": "F5",
    },
    "KF3": {
        "a": "F3",
        "b": "G3",
        "c": "A3",
        "d": "B3",
        "e": "C4",
        "f": "D4",
        "g": "E4",
        "h": "F4",
        "i": "G4",
        "j": "A4",
        "k": "B4",
        "l": "C5",
        "m": "D5",
    },
    "KF4": {
        "a": "D3",
        "b": "E3",
        "c": "F3",
        "d": "G3",
        "e": "A3",
        "f": "B3",
        "g": "C4",
        "h": "D4",
        "i": "E4",
        "j": "F4",
        "k": "G4",
        "l": "A4",
        "m": "B4",
    },
    "KC1": {
        "a": "G3",
        "b": "A3",
        "c": "B3",
        "d": "C4",
        "e": "D4",
        "f": "E4",
        "g": "F4",
        "h": "G4",
        "i": "A4",
        "j": "B4",
        "k": "C5",
        "l": "D5",
        "m": "E5",
    },
    "KC2": {
        "a": "E3",
        "b": "F3",
        "c": "G3",
        "d": "A3",
        "e": "B3",
        "f": "C4",
        "g": "D4",
        "h": "E4",
        "i": "F4",
        "j": "G4",
        "k": "A4",
        "l": "B4",
        "m": "C5",
    },
    "KC3": {
        "a": "C3",
        "b": "D3",
        "c": "E3",
        "d": "F3",
        "e": "G3",
        "f": "A3",
        "g": "B3",
        "h": "C4",
        "i": "D4",
        "j": "E4",
        "k": "F4",
        "l": "G4",
        "m": "A4",
    },
    "KC4": {
        "a": "A2",
        "b": "B2",
        "c": "C3",
        "d": "D3",
        "e": "E3",
        "f": "F3",
        "g": "G3",
        "h": "A3",
        "i": "B3",
        "j": "C4",
        "k": "D4",
        "l": "E4",
        "m": "F4",
    },
}


def http_get(url: str, timeout: int = 10) -> requests.Response:
    """Simple HTTP GET request with timeout."""
    return requests.get(url, timeout=timeout)


def parse_sql(sql_string: str) -> Generator[str, None, None]:
    """
    Given a SQL string, generate Gregorian Chant scores in gabc format one by one.
    Only the column `gabc` of table `gregobase_chants` is parsed.
    """
    # Regex to find the gabc column index and the values
    header_regex = re.compile(
        r"INSERT INTO `gregobase_chants` \((.*?)\) VALUES\s*")
    # This regex will find a single value, handling quoted strings with escaped quotes.
    # It matches: a quoted string, NULL, or a number.
    value_regex = re.compile(r"'((?:[^']|'')*)'|NULL|\d+")

    gabc_index = -1
    lines = sql_string.split("\n")

    yielded_count = 0
    gabc_index = -1  # Initialize gabc_index once outside the loop

    for i, line in enumerate(lines):
        if "INSERT INTO `gregobase_chants`" in line:
            # Only parse header and set gabc_index if we haven't found it yet
            if gabc_index == -1:
                header_match = header_regex.search(line)
                if header_match:
                    columns_str = header_match.group(1)
                    columns = [
                        col.strip().strip("`")
                        for col in columns_str.split(",")
                    ]
                    if "gabc" in columns:
                        gabc_index = columns.index("gabc")
                    else:
                        print("'gabc' column not found in header.")
                        gabc_index = -1
                else:
                    print(
                        f"Line {i + 1}: Header regex did not match for INSERT statement."
                    )
                    gabc_index = -1
            continue

        if gabc_index != -1 and line.strip().startswith("("):
            # Extract the content within the parentheses of the VALUES clause
            values_content_match = re.search(r"^\s*\((.*)\)[,;]?\s*$", line)
            if not values_content_match:
                continue

            values_content = values_content_match.group(1)
            values = [
                match.group(1)
                if match.group(1) is not None else match.group(0)
                for match in value_regex.finditer(values_content)
            ]

            if len(values) > gabc_index:
                gabc_value = values[gabc_index]
                if gabc_value != "NULL":
                    yielded_count += 1

                    # Unescape double single-quotes ('') to a single quote (')
                    yield gabc_value.replace("''", "'")
            else:
                # This line doesn't have enough values, so we skip it.
                continue


def gabc_to_custom(gabc: str, key="KC3") -> str:
    """
    Convert GABC notation to custom pitch notation.
    """
    custom_notation = ""
    matches = GREGORIO_REGEX.findall(gabc)
    if not matches:
        return custom_notation

    notation = " ".join(matches)

    # Process tokens sequentially to handle clef changes mid-chant
    tokens = notation.split()
    current_key = key

    for token in tokens:
        # Check if token is a clef
        clef_matches = CLEF_REGEX.findall(token)
        for clef in clef_matches:
            if clef:
                clef_type = clef[0]
                clef_line = clef[1]

                # Update current key based on clef
                if clef_type == "c":
                    current_key = f"KC{clef_line}"
                elif clef_type == "f":
                    current_key = f"KF{clef_line}"
                custom_notation += current_key + " "

        # Find all note letters within this token (equivalent to findall approach)
        note_matches = NOTE_REGEX.findall(token)
        for note_char in note_matches:
            if note_char.lower() in "abcdefghijklm":
                pitch = PITCH_MAPPING[current_key][note_char.lower()]
                custom_notation += pitch + " "

    return custom_notation.strip()


def setup_gregoriofont(font_path: str, font_url: str):
    """
    Download and install the gregorio fonts. Return the list of available fonts.
    """
    gregoriofontdir = Path(font_path)
    gregoriofontdir.mkdir(exist_ok=True, parents=True)

    if not any(gregoriofontdir.iterdir()):
        print("   Installing gregorio fonts...")
        # get the installed gregorio version
        if shutil.which("gregorio") is None:
            raise RuntimeError("Please, install `texlive-music`")
        cmd = "gregorio", "--version"
        completed_process = subprocess.run(cmd, capture_output=True, text=True)
        gregorioversion = completed_process.stdout.splitlines()[0].split()[1]
        if gregorioversion.endswith("."):
            gregorioversion = gregorioversion[:-1]
        print("   Detected gregorio version:", gregorioversion)
        font_url = font_url.replace("<gregorioversion>", f"{gregorioversion}")
        font_url = font_url.replace("<gregorioversion_underscore>",
                                    gregorioversion.replace(".", "_"))
        print("   Downloading gregorio fonts from:", font_url)
        # if the font directory is empty
        # Download the font zip file
        response = requests.get(font_url)
        # Extract the font files to the font directory
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            zip_file.extractall(gregoriofontdir)
        # Install the fonts
        if shutil.which("texlua") is None:
            raise RuntimeError("Please, install `texlive-luatex`")
        else:
            cmd = "texlua", "install_supp_fonts.lua", "user"
            completed_process = subprocess.run(cmd, cwd=gregoriofontdir)
            if completed_process.returncode != 0:
                raise RuntimeError(
                    f"Command failed with exit code {completed_process.returncode}, using only the default gregorian font"
                )
    print("   Installed gregorio fonts!")


setup_gregoriofont(DEFAULT_CONFIG["font_path"], DEFAULT_CONFIG["font_url"])


def _truncate_gabc(
        gabc: str,
        debug: bool = False) -> Tuple[Optional[str], Optional[List[str]], int]:
    """
    Truncates a gabc string to a random length.

    - Ensures the chant has a minimum number of notes.
    - Truncates longer chants to a random maximum length to avoid processing errors.
    - Returns the processed gabc string, remaining notes (if they meet minimum threshold), and number of truncated notes.
    - Each chant will be used for one or more lines of a page
    """
    match = GREGORIO_REGEX.search(gabc)
    if not match:
        if debug:
            print("No valid gabc content found.")
        return None, None, 0

    content = match.group(1)
    if not content.strip():  # Empty gabc like "()"
        if debug:
            print("Skipping empty chant.")
        return None, None, 0

    notes = content.split()
    min_notes = 4

    if len(notes) <= min_notes:
        if debug:
            print(
                f"Skipping chant with only {len(notes)} notes (minimum: {min_notes})"
            )
        return None, None, 0

    # Truncate if longer than min_notes
    max_notes = random.randint(7, 15)
    truncated_notes = notes[:max_notes]
    remaining_notes = notes[max_notes:]

    # Check if remaining notes meet minimum threshold for reuse
    remaining_notes_qualified = remaining_notes if len(
        remaining_notes) >= min_notes else None

    if debug and remaining_notes_qualified:
        print(
            f"Remaining {len(remaining_notes)} notes will be queued for reuse")

    return f"({' '.join(truncated_notes)})", remaining_notes_qualified, max_notes


def generate(template,
             gabc,
             num_notes,
             width,
             height,
             dpi=300,
             debug=False) -> list[np.ndarray]:
    """
    Generate images from a gabc string using the provided template.

    Args:
        template: Jinja template to render
        gabc: The gabc chant notation
        width: Width of the output image
        height: Height of the output image
        dpi: Resolution of the output image

    Returns:
        List of numpy arrays containing images
    """
    # Limit the size of gabc to prevent long rendering times
    gregoriofonts = ["greciliae", "granapadano", "gregorio"]
    gregoriofont = random.choice(gregoriofonts)

    # Sanitize gabc to remove characters that might cause issues with gregoriotex
    gabc = gabc.replace(";", "")

    # Add margins to prevent content from going outside borders
    margin = 20  # pixels
    # safe_width = width - 2 * margin
    safe_height = height - 2 * margin

    # Calculate appropriate sizes
    fontsize = round(70 + random.random() * 20)
    # Scale staff size based on available height to prevent overflow
    max_staffsize = int(safe_height * 0.8)  # Leave 20% margin
    staffsize = min(int(fontsize * 0.8), max_staffsize)
    # rescale initial size based on number of notes, avoiding division by zero
    denominator = max(num_notes - 9, 1)  # Ensure we never divide by zero
    initialsize = int(staffsize * 2 / denominator)

    source = template.render(
        fontsize=fontsize,  # this is in pixel!
        dpi=dpi,
        width=width,
        height=height,
        margin=margin,
        initialline=0,
        initialsize=initialsize,  # this is in pixel!
        staffsize=staffsize,  # this is in pixel!
        gregoriofont=gregoriofont,
        gabc=gabc,
    )

    # Use a unique directory for each LaTeX build to avoid concurrent runs colliding
    Path("temp").mkdir(exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="gregobase_build_", dir="temp/")
    build_dir = Path(tmpdir)
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / "chant.tex"
    source_path.write_text(source, encoding="utf-8")

    pdf_path = build_dir / "chant.pdf"

    # Use lualatex directly instead of latexmk
    lualatex_path = shutil.which("lualatex")
    if not lualatex_path:
        print("Error: lualatex executable not found.")
        # cleanup unless debugging
        if not debug:
            try:
                shutil.rmtree(build_dir)
            except Exception:
                pass
        return []

    cmd = [
        lualatex_path,
        "-shell-escape",
        "-interaction=batchmode",
        "-halt-on-error",
        source_path.name,
    ]
    if debug:
        print(f"Running command: {' '.join(cmd)}")
        print(f"Build directory: {build_dir}")

    env = os.environ.copy()
    font_path_abs = Path(DEFAULT_CONFIG["font_path"]).resolve()
    texinputs_path = f".:{str(font_path_abs)}:"
    env["TEXINPUTS"] = texinputs_path
    env["LANG"] = "en_US.UTF-8"
    env["LC_ALL"] = "en_US.UTF-8"

    proc = subprocess.run(cmd,
                          capture_output=True,
                          text=True,
                          cwd=build_dir,
                          env=env,
                          timeout=120)

    if proc.returncode != 0 or not pdf_path.exists():
        timed_log_name = f"logs/error_{os.getpid()}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        Path("logs").mkdir(exist_ok=True)
        import traceback

        # Write full Python traceback and lualatex log to the error log file
        python_traceback = traceback.format_exc()
        log_path = build_dir / "chant.log"
        latex_log = ""
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as logf:
                latex_log = logf.read()
        with open(timed_log_name, "w") as f:
            f.write(
                f"Exception traceback:\n{python_traceback}\n\nlatex_log:\n{latex_log}"
            )
        # If not in debug mode, cleanup the temporary build directory
        if not debug:
            try:
                shutil.rmtree(build_dir)
            except Exception:
                pass
        if debug:
            print(f"Exception logged to {timed_log_name}")
        return []

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # cleanup build directory unless debugging
    if not debug:
        try:
            shutil.rmtree(build_dir)
        except Exception:
            pass
    else:
        print(
            f"Debug mode: keeping build directory {build_dir} for inspection")

    pdf = Document(stream=pdf_bytes)

    # create raster images from pdf
    images = []
    for pagenum in range(len(pdf)):
        page = pdf.load_page(pagenum)
        pix = page.get_pixmap(dpi=dpi, clip=False,
                              annots=False)  # type: ignore
        img = np.frombuffer(pix.samples_mv, dtype=np.uint8).reshape(
            (pix.height, pix.width, -1))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        # where the image is white, set alpha to 0
        img[np.all(img == 255, axis=-1)] = (200, 200, 200, 0)
        images.append(img)
    return images


def setup_cache(cache_name: str, cache_days: int) -> None:
    """Setup requests cache."""
    try:
        import requests_cache

        requests_cache.install_cache(
            cache_name=cache_name,
            expire_after=datetime.timedelta(days=cache_days),
        )
    except ImportError:
        print("requests_cache not installed, proceeding without caching")


def get_music_page_generator(
    width_range: Tuple[int, int] = (400, 800),
    height_range: Tuple[int, int] = (200, 600),
    cache_name: str = DEFAULT_CONFIG["cache_name"],
    cache_days: int = DEFAULT_CONFIG["cache_days"],
    latex_jobs: int = 4,
    max_pages: Optional[int] = None,
    debug: bool = False,
) -> Generator[Tuple[np.ndarray, str, str], None, None]:
    """
    This function is a wrapper that sets up a ThreadPoolExecutor and yields from
    the concurrent chant generator. This ensures proper cleanup of the executor.
    """
    with ThreadPoolExecutor(max_workers=latex_jobs) as executor:
        yield from generate_music_page_concurrently(
            executor,
            width_range,
            height_range,
            cache_name,
            cache_days,
            max_pages,
            debug,
        )


def generate_music_page_concurrently(
    executor: ThreadPoolExecutor,
    width_range: Tuple[int, int],
    height_range: Tuple[int, int],
    cache_name: str,
    cache_days: int,
    max_pages: Optional[int] = None,
    debug: bool = False,
) -> Generator[Tuple[np.ndarray, str, str], None, None]:
    """
    Generate music pages one by one as they are retrieved from the online database.
    This function uses a provided thread pool to parallelize LaTeX compilation.

    Args:
        executor: A ThreadPoolExecutor instance.
        width_range: Range of widths for generated images (min, max)
        height_range: Range of heights for generated images (min, max)
        cache_name: Name for the requests cache
        cache_days: Number of days to keep the cache
        max_pages: Maximum number of pages to generate (None for unlimited)

    Yields:
        Tuple[np.ndarray, str, str]: Image of the chant, the original gabc notation, and the custom notation
    """
    # Setup cache
    setup_cache(cache_name, cache_days)

    # Download SQL data dump from Gregobase

    print("Downloading SQL data dump from Gregobase...")

    try:
        response = http_get(DEFAULT_CONFIG["gregobase_online_url"])
        response.raise_for_status()

        # Check if the content is a zip file
        if "application/zip" in response.headers.get("Content-Type", ""):
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Find the .sql file in the zip archive
                sql_filename = next(
                    (name for name in z.namelist() if name.endswith(".sql")),
                    None)
                if sql_filename:
                    with z.open(sql_filename) as sql_file:
                        sql_content = sql_file.read().decode("utf-8")
                else:
                    print(
                        "Error: No .sql file found in the downloaded zip archive."
                    )
                    return
        else:
            # Fallback for non-zipped content, if the server changes back
            sql_content = response.content.decode("utf-8")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading Gregobase data: {e}")
        return
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data fetching: {e}")
        return

    # Setup template
    template_string = r"""% 1 in is 2.54 cm
% 1 pt is 1/72 in
% 1 px is 1 in at 1 dpi
% set dpi to D
% X px is our input
% it is X/D in
% X/D is 72X/D pt

\documentclass{article}

\usepackage{geometry}
\geometry{paperheight=\VAR{height / dpi}in,paperwidth=\VAR{width / dpi}in}
\geometry{hmargin={\VAR{margin / dpi}in,\VAR{margin / dpi}in},vmargin={\VAR{margin / dpi}in,\VAR{margin / dpi}in}}

\usepackage{fontspec}

%- if gabc is not none
\usepackage[autocompile]{gregoriotex}
%- endif

\begin{document}

\fontsize{\VAR{72 * fontsize / dpi}}{\VAR{1.2 *
 72 * fontsize / dpi}}

%- if gabc is not none
% the gabc code
% size of the initial line
\gresetinitiallines{\VAR{initialline}}
\grechangestyle{initial}{\VAR{72 * initialsize / dpi}pt}
% use red lines
\gresetlinecolor{gregoriocolor}
\grechangestaffsize{\VAR{72 * staffsize // dpi}}
\gresetgregoriofont{\VAR{gregoriofont}}
\gabcsnippet{\VAR{gabc}}
%- endif

\end{document}
"""
    env = Environment(
        loader=BaseLoader(),
        line_statement_prefix="%-",
        variable_start_string=r"\VAR{",
        variable_end_string="}",
    )
    template = env.from_string(template_string)

    # Parse SQL and stream segments with bounded futures window
    chant_generator = parse_sql(sql_content)
    remaining_notes_queue = deque()

    chants_processed = 0
    chants_skipped = 0
    segments_from_queue = 0
    segments_yielded = 0

    if debug:
        print("🎵 DEBUG: Note reuse system activated! Will show statistics...")

    # Bounded futures window (2x number of workers to keep pool busy)
    max_pending_futures = executor._max_workers * 2
    futures = {}
    
    def process_notes_to_chant(
            notes_list: List[str],
            source: str = "new") -> Optional[Tuple[str, str, int, int]]:
        """Helper function to process a list of notes into a chant."""
        gabc = f"({' '.join(notes_list)})"
        truncated_chant, remaining_notes, num_notes = _truncate_gabc(
            gabc, debug=debug)

        if truncated_chant is None:
            return None

        # Add remaining notes to queue if they qualify
        if remaining_notes:
            remaining_notes_queue.append(remaining_notes)
            if debug:
                print(f"Adding {len(remaining_notes)} to the note queue")

        custom_notation = gabc_to_custom(truncated_chant)
        width = random.randint(*width_range)
        height = random.randint(*height_range)

        return truncated_chant, custom_notation, width, height

    def submit_segment(truncated_chant, num_notes, custom_notation, width, height):
        """Submit a single segment to the thread pool."""
        nonlocal segments_yielded
        
        future = executor.submit(
            generate,
            template,
            truncated_chant,
            num_notes,
            width=width,
            height=height,
            dpi=DEFAULT_CONFIG["dpi"],
        )
        futures[future] = (truncated_chant, custom_notation)
        
        if debug:
            print(f"Submitted segment (pending: {len(futures)})")

    # Create iterators for streaming
    chant_iter = iter(chant_generator)
    current_chant = None
    
    def get_next_segment():
        """Get next segment from either queue or new chant."""
        nonlocal current_chant, chants_processed, chants_skipped, segments_from_queue
        
        # First try queued remaining notes
        if remaining_notes_queue:
            queued_notes = remaining_notes_queue.popleft()
            result = process_notes_to_chant(queued_notes, "queue")
            if result:
                truncated_chant, custom_notation, width, height = result
                segments_from_queue += 1
                return truncated_chant, len(queued_notes), custom_notation, width, height
        
        # Then try new chants
        while True:
            try:
                if current_chant is None:
                    current_chant = next(chant_iter)
                
                chants_processed += 1
                chant = current_chant
                current_chant = None  # Mark as consumed
                
                # Keep only text inside round brackets from chant
                filtered_chant = "(" + " ".join(GREGORIO_REGEX.findall(chant)) + ")"
                
                # Get notes from the filtered chant
                match = GREGORIO_REGEX.search(filtered_chant)
                if not match or not match.group(1).strip():
                    continue
                
                notes = match.group(1).split()
                result = process_notes_to_chant(notes, "new")
                if result is None:
                    chants_skipped += 1
                    continue
                
                truncated_chant, custom_notation, width, height = result
                return truncated_chant, len(notes), custom_notation, width, height
                
            except StopIteration:
                return None  # No more segments available

    # Fill initial futures window
    for _ in range(max_pending_futures):
        segment = get_next_segment()
        if segment is None:
            break
        if segments_yielded >= max_pages:
            break
        truncated_chant, num_notes, custom_notation, width, height = segment
        submit_segment(truncated_chant, num_notes, custom_notation, width, height)

    # Process results and maintain bounded window
    while futures:
        # Wait for next completion
        completed_future = next(as_completed(futures))
        truncated_chant, custom_notation = futures[completed_future]
        
        # Remove completed future immediately (KEY MEMORY FIX)
        del futures[completed_future]
        
        try:
            images = completed_future.result()
            if images:
                for image in images:
                    segments_yielded += 1
                    yield image, truncated_chant, custom_notation
                    
                    # Check if we've reached the limit
                    if segments_yielded >= max_pages:
                        # Cancel remaining futures and exit
                        for future in futures:
                            future.cancel()
                        print(f"Generated {segments_yielded} segments from {chants_processed} processed ({chants_skipped} skipped)")
                        print(f"  - {segments_yielded - segments_from_queue} from new segments")
                        print(f"  - {segments_from_queue} from reused remaining notes")
                        return
            
            # Refill the window with new segment (if available and not at limit)
            if segments_yielded < max_pages:
                segment = get_next_segment()
                if segment is not None:
                    truncated_chant, num_notes, custom_notation, width, height = segment
                    submit_segment(truncated_chant, num_notes, custom_notation, width, height)
                
        except Exception as exc:
            print(f"Chant generation created an exception: {exc}")

    print(f"Generated {segments_yielded} segments from {chants_processed} processed ({chants_skipped} skipped)")
    print(f"  - {segments_yielded - segments_from_queue} from new segments")
    print(f"  - {segments_from_queue} from reused remaining notes")
