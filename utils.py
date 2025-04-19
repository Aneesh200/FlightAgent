# utils.py
import json
import os
import unicodedata
import re
from datetime import datetime, timedelta
import dateparser

# Import config variables needed here
from config import CITIES_IATA_PATH

# --- Global Variable for Loaded IATA Data ---
CITIES_IATA_DATA = {}

def load_cities_iata(file_path=CITIES_IATA_PATH):
    """Loads city-to-IATA data from JSON file."""
    global CITIES_IATA_DATA
    # Check if path is absolute or relative
    if not os.path.isabs(file_path):
        # Assuming the script is run from the directory containing the JSON file
        try:
            # Find the directory where utils.py resides
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, file_path)
            print(f"Relative path detected. Trying absolute path: {file_path}")
        except NameError: # Handle case where __file__ is not defined (e.g., interactive)
             print("Warning: Could not determine script directory. Using CWD for relative path.")
             file_path = os.path.join(os.getcwd(), file_path)


    try:
        with open(file_path, "r", encoding='utf-8') as file:
            CITIES_IATA_DATA = json.load(file)
            print(f"Successfully loaded {len(CITIES_IATA_DATA)} IATA records from {file_path}")
            return True # Indicate success
    except FileNotFoundError:
        print(f"FATAL ERROR: IATA file not found at '{file_path}'. City-to-IATA conversion will fail.")
        CITIES_IATA_DATA = {}
        return False # Indicate failure
    except json.JSONDecodeError:
        print(f"FATAL ERROR: Error decoding JSON from {file_path}.")
        CITIES_IATA_DATA = {}
        return False # Indicate failure
    except Exception as e:
        print(f"FATAL ERROR: Error loading IATA file {file_path}: {e}")
        CITIES_IATA_DATA = {}
        return False # Indicate failure

def calculate_duration(start, end):
    """Calculates duration between two ISO-formatted datetime strings."""
    fmt = "%Y-%m-%dT%H:%M:%S"
    try:
        start_time = datetime.strptime(start, fmt)
        end_time = datetime.strptime(end, fmt)
        duration = end_time - start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes = remainder // 60
        return f"{int(hours)}h {int(minutes)}m"
    except (ValueError, TypeError, AttributeError):
        return "N/A"

def normalize_accents(text):
    """Removes accents from text."""
    if not isinstance(text, str): return text
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def city_to_iata(city_name):
    """Converts city name to IATA code using loaded data."""
    if not CITIES_IATA_DATA:
        print("Warning: IATA data not loaded, cannot convert city name.")
        return "Unknown"
    if not city_name: return "Unknown"
    cleaned_name = city_name.strip().title()

    # Direct match
    iata = CITIES_IATA_DATA.get(cleaned_name)
    if iata: return iata

    # Variations
    name_variations = {"New York": "New York City", "NYC": "New York City", "LA": "Los Angeles", "SF": "San Francisco", "Frisco": "San Francisco", "DC": "Washington", "Wash DC": "Washington"}
    normalized_variation = name_variations.get(cleaned_name, cleaned_name)
    iata = CITIES_IATA_DATA.get(normalized_variation)
    if iata: return iata

    # Normalized match
    normalized_name = normalize_accents(normalized_variation).title()
    iata = CITIES_IATA_DATA.get(normalized_name)
    if iata: return iata

    # Case-insensitive fallback
    cleaned_name_lower = cleaned_name.lower()
    normalized_variation_lower = normalized_variation.lower()
    normalized_name_lower = normalized_name.lower()
    for key, value in CITIES_IATA_DATA.items():
        key_lower = key.lower()
        if key_lower in [cleaned_name_lower, normalized_variation_lower, normalized_name_lower]:
            return value
        normalized_key = normalize_accents(key).lower()
        if normalized_key in [cleaned_name_lower, normalized_variation_lower, normalized_name_lower]:
            return value

    print(f"Warning: Could not find IATA for city '{city_name}'")
    return "Unknown"

# --- Load IATA Data on import ---
if not load_cities_iata():
    print("Exiting due to failure loading mandatory IATA data.")
    # In a real app, you might raise an exception or use sys.exit(1)
    # For this example, we print and continue, but functions will fail.
    pass