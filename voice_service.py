# voice_service.py
import os
import tempfile
import traceback

# Attempt to import optional libraries
try: import boto3
except ImportError: boto3 = None
try: from playsound import playsound
except ImportError: playsound = None
try: import speech_recognition as sr
except ImportError: sr = None
try: from groq import Groq
except ImportError: Groq = None

# Import config variables
from config import (
    USE_TTS_OUTPUT, USE_VOICE_INPUT,
    AWS_REGION_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, POLLY_VOICE_ID,
    GROQ_API_KEY, WHISPER_MODEL,
    MIC_TIMEOUT_SECONDS, MIC_PHRASE_LIMIT_SECONDS, MIC_ADJUST_DURATION
)

# --- Global Client/Recognizer Variables ---
polly_client = None
groq_client = None
stt_recognizer = None
stt_microphone = None
_tts_enabled = False
_stt_enabled = False

# --- Initialization Functions ---
def initialize_tts():
    """Initializes AWS Polly client if configured and libraries exist."""
    global polly_client, _tts_enabled
    if USE_TTS_OUTPUT and boto3 and playsound and AWS_ACCESS_KEY_ID and \
       AWS_ACCESS_KEY_ID != "YOUR_AWS_ACCESS_KEY_ID" and AWS_SECRET_ACCESS_KEY and \
       AWS_SECRET_ACCESS_KEY != "YOUR_AWS_SECRET_ACCESS_KEY" and AWS_REGION_NAME:
        try:
            session = boto3.Session(
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION_NAME
            )
            polly_client = session.client('polly')
            # Test call
            test_tts = polly_client.synthesize_speech(Text="TTS Initialized", OutputFormat='mp3', VoiceId=POLLY_VOICE_ID)
            if 'AudioStream' in test_tts:
                print(f"AWS Polly client initialized successfully for region {AWS_REGION_NAME}.")
                _tts_enabled = True
                return True
            else:
                raise Exception("Polly test call failed.")
        except Exception as e:
            print(f"Error initializing AWS Polly client: {e}")
            polly_client = None
    else:
        print("TTS prerequisites not met (libs missing, config placeholder, or disabled). TTS Disabled.")
    _tts_enabled = False
    return False

def initialize_stt():
    """Initializes Groq Whisper client and SpeechRecognition components."""
    global groq_client, stt_recognizer, stt_microphone, _stt_enabled
    if USE_VOICE_INPUT and sr and Groq and GROQ_API_KEY and GROQ_API_KEY != "YOUR_GROQ_API_KEY":
        # Initialize Groq Client first
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
            print("Groq client for Whisper initialized.")
        except Exception as e:
            print(f"Error initializing Groq client: {e}. STT Disabled.")
            groq_client = None
            _stt_enabled = False
            return False

        # Initialize Speech Recognition
        try:
            stt_recognizer = sr.Recognizer()
            stt_microphone = sr.Microphone()
            with stt_microphone as source:
                print(f"Adjusting microphone for ambient noise ({MIC_ADJUST_DURATION}s)...")
                stt_recognizer.adjust_for_ambient_noise(source, duration=MIC_ADJUST_DURATION)
            print("Speech Recognition components initialized.")
            _stt_enabled = True
            return True
        except sr.RequestError as e:
            print(f"SR API error: {e}. STT Disabled.")
        except AttributeError: # Often PyAudio missing
            print("Microphone init failed (PyAudio installed?). STT Disabled.")
        except Exception as e:
            print(f"Error initializing Speech Recognition: {e}. STT Disabled.")
            traceback.print_exc()

    else:
         print("STT prerequisites not met (libs missing, config placeholder, or disabled). STT Disabled.")

    # Ensure components are None if init failed
    groq_client = None
    stt_recognizer = None
    stt_microphone = None
    _stt_enabled = False
    return False

# --- Core Voice Functions ---
def speak_text(text_to_speak, temp_file_suffix=".mp3"):
    """Synthesizes text using Polly and plays it."""
    if not _tts_enabled or not polly_client or not text_to_speak or not playsound:
        # print("(TTS Skipped)")
        return

    # Basic SSML wrapping for pauses
    if not text_to_speak.strip().startswith('<speak>'):
        processed_text = text_to_speak.replace('. ', '.<break time="300ms"/> ') \
                                      .replace('? ', '?<break time="400ms"/> ') \
                                      .replace('! ', '!<break time="400ms"/> ') \
                                      .replace(', ', ',<break time="150ms"/> ')
        text_to_send = f"<speak>{processed_text}</speak>"
        text_type = 'ssml'
    else:
        text_to_send = text_to_speak
        text_type = 'ssml'

    temp_audio_path = None
    try:
        response = polly_client.synthesize_speech(
            Text=text_to_send, TextType=text_type, OutputFormat='mp3', VoiceId=POLLY_VOICE_ID
        )
        if 'AudioStream' in response:
            with tempfile.NamedTemporaryFile(delete=False, suffix=temp_file_suffix) as fp:
                fp.write(response['AudioStream'].read())
                temp_audio_path = fp.name
            playsound(temp_audio_path)
        else: print("Error: Could not get audio stream from Polly.")
    except playsound.PlaysoundException as e: print(f"Error playing sound: {e}.")
    except boto3.exceptions.Boto3Error as e: print(f"AWS Polly API Error: {e}")
    except Exception as e: print(f"Error during TTS: {e}"); traceback.print_exc()
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try: os.remove(temp_audio_path)
            except Exception as e_rem: print(f"Warn: Failed remove temp audio {temp_audio_path}: {e_rem}")

def listen_for_input():
    """Listens via microphone, transcribes using Groq Whisper."""
    if not _stt_enabled or not stt_recognizer or not stt_microphone or not groq_client:
        print("DEBUG: listen_for_input called but STT is disabled/components missing.")
        return None

    print("\nListening... (Speak clearly)")
    with stt_microphone as source:
        temp_audio_path = None
        try:
            audio = stt_recognizer.listen(
                source, timeout=MIC_TIMEOUT_SECONDS, phrase_time_limit=MIC_PHRASE_LIMIT_SECONDS
            )
            print("-> Audio captured, processing...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio.get_wav_data())
                temp_audio_path = temp_audio.name

            if not temp_audio_path or not os.path.exists(temp_audio_path):
                 print("Error: Failed create temp audio file."); return None

            print("-> Sending audio to Groq Whisper...")
            with open(temp_audio_path, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=(os.path.basename(temp_audio_path), audio_file.read()),
                )
            text = transcription.text if transcription else None
            if text: print(f"-> Heard: \"{text}\""); return text.strip()
            else: print("-> Whisper transcription empty."); return None

        except sr.WaitTimeoutError: print("-> No speech detected."); return None
        except sr.RequestError as e: print(f"-> SR API error: {e}"); return None
        except sr.UnknownValueError: print("-> Whisper could not understand."); return None
        except Exception as e: print(f"-> STT Error: {e}"); traceback.print_exc(); return None
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try: os.remove(temp_audio_path)
                except Exception as e_rem: print(f"Warn: Failed remove temp audio {temp_audio_path}: {e_rem}")
