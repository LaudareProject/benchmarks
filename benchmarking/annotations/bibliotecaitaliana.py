import json
import re
import random
import datetime
from typing import Generator, Optional, List, Dict

from lxml import etree  # type: ignore
import requests
import requests_cache

# These constants are kept as they don't belong to configuration
FORBIDDEN_CHARS = re.compile(r"[^a-zA-Z\sàáèéìíòóùú]")
A_TO_A = re.compile(r"[àá]")
E_TO_E = re.compile(r"[èé]")
I_TO_I = re.compile(r"[ìí]")
O_TO_O = re.compile(r"[òó]")
U_TO_U = re.compile(r"[ùú]")

ABBREVIATIONS = {
    "per": ["ꝑ"],
    "or": ["õ"],
    "ver": ["ꝟ", "ṽ"],
    "vir": ["ṽ"],
    "om": ["ō"],
    "on": ["ō"],
    "am": ["ā"],
    "an": ["ā"],
    "em": ["ē"],
    "en": ["ē"],
    "um": ["ū"],
    "un": ["ū"],
    "non": ["ṅ"],
    "ter": ["ť"],
    "tar": ["ť"],
    "tur": ["ť"],
    "erù": ["ẽ"],
    "et": ["ꭋ"],
    "ar": ["ã"],
    "con": ["Ꝯ"],
    "com": ["Ꝯ"],
    "men": ["ṁ"],
    "man": ["ṁ"],
    "mon": ["ṁ"],
    "de": ["ð"],
    "pre": ["ṕ"],
    "quar": ["Ꝗ"],
    "ser": ["ᶋ"],
    "que": ["ɋ"],
    "qua": ["ꬶ"],
    "sancti": ["sci"],
    "sancto": ["sco"],
    "christ": ["χρ"],
    "nostra": ["nra"],
    "Maria": ["ma"],
    "homo": ["ho"],
    "omni": ["oi"],
    "omne": ["oe"],
    "Iesu": ["ihu"],
}


def http_get(url: str, timeout: int = 10) -> requests.Response:
    """Simple HTTP GET request with timeout."""
    return requests.get(url, timeout=timeout)


def abbreviate_text(
    text: str, abbreviations: Dict[str, List[str]], probability: float = 0.5
) -> str:
    """
    Replace strings in text with corresponding symbols according to a given probability.

    Args:
        text: The input text to process
        abbreviations: Dictionary mapping strings to symbols (single symbol or list of symbols)
        probability: Probability (0.0 to 1.0) of replacing each occurrence

    Returns:
        str: Text with abbreviations applied
    """
    result = text
    for word, symbols in abbreviations.items():
        # Create a pattern that matches the word with word boundaries
        pattern = r"\b" + re.escape(word) + r"\b"

        # Find all occurrences of the word
        matches = list(re.finditer(pattern, result))

        # Process matches from end to beginning to avoid offset issues
        for match in reversed(matches):
            # Apply replacement with the given probability
            if random.random() < probability:
                # Get the replacement symbol (randomly select one if multiple are provided)
                replacement = random.choice(symbols)

                # Replace this occurrence
                start, end = match.span()
                result = result[:start] + replacement + result[end:]

    return result


def extract_text(xml_string: bytes) -> str:
    """
    Extract text from a TEI XML string and process it.

    Args:
        xml_string (bytes): A TEI XML content in bytes

    Returns:
        str: Processed extracted text from the XML
    """
    # Load the TEI XML string
    content = etree.fromstring(xml_string)[1]

    # Initialize an empty string to store the extracted text
    extracted_text = ""

    # Collect all the text
    for node in content.iter():
        if node.tag != "head":
            if node.tag == "l" or node.tag == "p":
                extracted_text += (node.text or "") + "\n"
            else:
                extracted_text += node.text or ""

    # Remove every punctuation and number (characters that are not alphabetical nor whitespaces)
    extracted_text = re.sub(FORBIDDEN_CHARS, "", extracted_text)

    # Replace accented characters àáèéùúìíòó with aaeiuuio
    extracted_text = re.sub(A_TO_A, "a", extracted_text)
    extracted_text = re.sub(E_TO_E, "e", extracted_text)
    extracted_text = re.sub(I_TO_I, "i", extracted_text)
    extracted_text = re.sub(O_TO_O, "o", extracted_text)
    extracted_text = re.sub(U_TO_U, "u", extracted_text)

    return extracted_text


def setup_cache(cache_name: str, cache_days: int) -> None:
    """Setup requests cache."""
    requests_cache.install_cache(
        cache_name=cache_name,
        expire_after=datetime.timedelta(days=cache_days),
    )


def get_line_generator(
    allowed_languages: List[str],
    periods: List[str],
    abbreviations: Dict[str, list[str]] = ABBREVIATIONS,
    abbreviation_probability: float = 0.5,
    cache_name: str = "biblio_cache",
    cache_days: int = 30,
    verbose: bool = False,
) -> Generator[str, None, None]:
    """
    Generate lines one by one as they are retrieved from the online database.

    Args:
        allowed_languages: List of languages to filter by
        periods: List of historical periods to filter by
        abbreviations: Dictionary mapping strings to replacement symbols
        abbreviation_probability: Probability of applying abbreviations
        cache_name: Name for the requests cache
        cache_days: Number of days to keep the cache
        verbose: Whether to print status messages

    Yields:
        str: One line of text at a time
    """
    # Setup cache
    setup_cache(cache_name, cache_days)

    # Construct URL for catalog
    url_catalogo = "http://backend.bibliotecaitaliana.it/wp-json/muruca/v1/solr/select?facet=false&rows=100000&start=0&q=*:* AND post_type:muruca_resource AND post_type:muruca_resource&sort=author_sort asc, post_title asc&&fq={!tag%3Dresource_period_str}"

    # Add periods filter
    for i, period in enumerate(periods):
        url_catalogo += f'resource_period_str:"{period}"'
        if i != len(periods) - 1:
            url_catalogo += " OR "

    # Fetch the catalog
    if verbose:
        print("Contacting Biblioteca Italiana")
    response = http_get(url_catalogo)

    # Parse response
    if verbose:
        print("Parsing response")
    js = json.loads(response.content)

    # Process documents and yield lines as they are found
    schede = js["response"]["docs"]

    for i, scheda in enumerate(schede):
        if verbose:
            print(f"Retrieving text {i + 1}/{len(schede)}")

        scheda = {k.lower(): v for k, v in scheda.items()}

        # Check if document language is in allowed languages
        if (
            len(set(scheda["resource_language_str"]).intersection(allowed_languages))
            == 0
        ):
            continue

        # Get TEI XML
        tei_url = (
            "http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/"
            + scheda["hdl_txt"][0]
        )
        tei_file = http_get(tei_url)

        if len(tei_file.content) == 0:
            continue

        # Extract text from XML
        try:
            extracted_text = extract_text(tei_file.content)
        except etree.XMLSyntaxError:
            if verbose:
                print("\nXML Syntax Error, skipping one file")
            continue

        # Yield lines as soon as we have them
        for line in extracted_text.split("\n"):
            if len(line) > 0:
                # Apply abbreviations if needed
                if abbreviations:
                    line = abbreviate_text(
                        line, abbreviations, abbreviation_probability
                    )
                yield line


def get_line(
    generator: Optional[Generator[str, None, None]] = None, **kwargs
) -> Optional[str]:
    """
    Get one line of text from the database.

    Args:
        generator: An existing generator from get_line_generator.
                  If None, a new one will be created with the provided kwargs.
        **kwargs: Arguments to pass to get_line_generator if generator is None.

    Returns:
        Optional[str]: A line of text, or None if no more lines are available
    """
    # If no generator provided, create a new one
    if generator is None:
        generator = get_line_generator(**kwargs)

    # Get next line from the generator
    try:
        return next(generator)
    except StopIteration:
        return None


# # Example usage:
# if __name__ == "__main__":
#     # Example abbreviations dictionary
#     abbr = {"quod": ["qd", "q̄"], "est": ["ē", "ẽ"], "et": ["&", "⁊"], "per": ["p̄", "ꝑ"]}
#
#     # Create a generator with abbreviations
#     line_generator = get_line_generator(
#         allowed_languages=["Italian"],
#         periods=["1300-1375", "1376-1500"],
#         abbreviations=abbr,
#         abbreviation_probability=0.7,  # Replace 70% of occurrences
#         verbose=True,
#     )
#
#     # Process lines one by one
#     count = 0
#     for line in line_generator:
#         print(f"Line {count}: {line}")
#         count += 1
#         if count >= 10:  # Just get 10 lines for demo
#             break
#
#     # Alternative: get lines one at a time using get_line
#     generator = get_line_generator(
#         allowed_languages=["Italian"],
#         periods=["1300-1375"],
#         abbreviations=abbr,
#         verbose=True,
#     )
#
#     line = get_line(generator)
#     while line is not None:
#         # Process your line here with your time-expensive function
#         print(f"Processing: {line}")
