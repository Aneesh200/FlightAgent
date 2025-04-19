import os
from dotenv import load_dotenv
from datetime import datetime
import traceback
import socket

# --- Load environment variables ---
try:
    load_dotenv()
    print("Loaded environment variables from .env file.")
except ImportError:
    print("dotenv not installed, skipping .env file loading. Ensure variables are set manually.")

# --- Configuration Flags ---
VOICE_ENABLED = os.getenv("VOICE_ENABLED", "True").lower() == "true"

_BOTO3_FOUND = True
_PLAYSOUND_FOUND = True
_SR_FOUND = True
_GROQ_LIB_FOUND = True

try:
    import boto3
except ImportError: _BOTO3_FOUND = False
try:
    import playsound
except ImportError: _PLAYSOUND_FOUND = False
try:
    import speech_recognition
except ImportError: _SR_FOUND = False
try:
    from groq import Groq
except ImportError: _GROQ_LIB_FOUND = False

USE_TTS_OUTPUT = VOICE_ENABLED and _BOTO3_FOUND and _PLAYSOUND_FOUND
USE_VOICE_INPUT = VOICE_ENABLED and _SR_FOUND and _GROQ_LIB_FOUND

# --- API Keys & Paths ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")
CITIES_IATA_PATH = os.getenv("CITIES_IATA_PATH")

# --- AWS Polly Configuration ---
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# --- Email Configuration ---
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = os.getenv("SMTP_PORT")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# --- LangChain/LLM Configuration ---
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.6))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 3))
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", 90.0))

# --- RAG Configuration ---
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL")

# --- Voice Configuration ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
POLLY_VOICE_ID = os.getenv("POLLY_VOICE_ID")
MIC_TIMEOUT_SECONDS = int(os.getenv("MIC_TIMEOUT_SECONDS", 8))
MIC_PHRASE_LIMIT_SECONDS = int(os.getenv("MIC_PHRASE_LIMIT_SECONDS", 10))
MIC_ADJUST_DURATION = float(os.getenv("MIC_ADJUST_DURATION", 1.5))

# --- Conversation Constants ---
REQUIRED_FIELDS = ["departure_city", "arrival_city", "departure_date", "passengers"]
OPTIONAL_FIELDS = ["class_preference", "meal_preference", "preferred_airline", "return_date"]
INTERNAL_FIELDS = ["suggestion_keywords", "selected_flight_identifier"]
ALL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS + INTERNAL_FIELDS
CONVERSATION_HISTORY_LENGTH = int(os.getenv("CONVERSATION_HISTORY_LENGTH", 10))

# --- Print Configuration ---
def print_configuration():
    print(f"--- Configuration ---")
    print(f"Voice Features Enabled: {VOICE_ENABLED}")
    print(f"  - Initial TTS Capability: {USE_TTS_OUTPUT}")
    print(f"  - Initial STT Capability: {USE_VOICE_INPUT}")
    print(f"GROQ Key Loaded: {'Yes' if GROQ_API_KEY else 'No'}")
    print(f"Amadeus Client ID Loaded: {'Yes' if AMADEUS_CLIENT_ID else 'No'}")
    print(f"Amadeus Client Secret Loaded: {'Yes' if AMADEUS_CLIENT_SECRET else 'No'}")
    print(f"Cities IATA Path: {CITIES_IATA_PATH}")
    if USE_TTS_OUTPUT:
        print(f"AWS Region: {AWS_REGION_NAME}")
        print(f"AWS Access Key ID Loaded: {'Yes' if AWS_ACCESS_KEY_ID else 'No'}")
        print(f"AWS Secret Key Loaded: {'Yes' if AWS_SECRET_ACCESS_KEY else 'No'}")
    print(f"Recipient Email Loaded: {'Yes' if RECIPIENT_EMAIL else 'No'}")
    if SMTP_HOST:
        print(f"SMTP Host: {SMTP_HOST}")
        print(f"SMTP Port: {SMTP_PORT}")
        print(f"SMTP User Loaded: {'Yes' if SMTP_USER else 'No'}")
        print(f"SMTP Password Loaded: {'Yes' if SMTP_PASSWORD else 'No'}")
    else:
        print(f"SMTP Config: Not set")
    print(f"--------------------")

# --- Critical Checks ---
def perform_critical_checks():
    critical_ok = True
    if not GROQ_API_KEY:
        print("FATAL ERROR: GROQ_API_KEY is not set.")
        critical_ok = False
    if not AMADEUS_CLIENT_ID or not AMADEUS_CLIENT_SECRET:
        print("FATAL ERROR: AMADEUS credentials not set.")
        critical_ok = False
    if not RECIPIENT_EMAIL:
        print("WARNING: RECIPIENT_EMAIL is not set.")
    if SMTP_HOST and (not SMTP_PORT or not SMTP_USER or not SMTP_PASSWORD):
        print("WARNING: SMTP settings are incomplete.")
    return critical_ok

# Print config on import
print_configuration()
