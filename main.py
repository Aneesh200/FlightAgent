# main.py
import json
import time
import traceback
from collections import deque
import sys # For sys.exit

# --- LangChain Core Imports ---
from langchain_groq import ChatGroq

# --- Project Modules ---
import config # Import config first
from utils import CITIES_IATA_DATA # Access loaded data
from email_service import send_flight_details_email
from voice_service import initialize_tts, initialize_stt, speak_text, listen_for_input
from rag_module import FlightRAG
from langchain_tools import ( # Import Tool classes
    UserInputParsingTool, DataValidationTool, AmadeusFlightSearchTool, FlightAnalysisTool
)
from coordinator import generate_coordinator_response

# --- Global Instances (Initialized after checks) ---
chat_llm = None
flight_rag_instance = None
input_parser_tool = None
validator_tool = None
search_tool = None
analyzer_tool = None

def initialize_systems():
    """Initialize LLM, RAG, Tools, Voice"""
    global chat_llm, flight_rag_instance, input_parser_tool, validator_tool, search_tool, analyzer_tool

    # 1. Perform Critical Config Checks
    if not config.perform_critical_checks():
        print("Critical configuration missing. Exiting.")
        return False

    # 2. Initialize LLM
    try:
        print(f"Initializing LLM: {config.LLM_MODEL}...")
        chat_llm = ChatGroq(
            temperature=config.LLM_TEMPERATURE,
            model_name=config.LLM_MODEL,
            api_key=config.GROQ_API_KEY,
            max_retries=config.LLM_MAX_RETRIES,
            request_timeout=config.LLM_REQUEST_TIMEOUT
        )
        # Simple invoke test (optional, uses tokens)
        # chat_llm.invoke("Hello!")
        print("LLM Initialized successfully.")
    except Exception as e:
        print(f"FATAL: LLM Initialization failed: {e}"); traceback.print_exc()
        return False # Cannot proceed without LLM

    # 3. Initialize RAG
    print("Initializing RAG...")
    flight_rag_instance = FlightRAG() # Uses model from config internally
    if not flight_rag_instance.is_initialized:
        print("Warning: RAG initialization failed. RAG features disabled.")
        # Decide whether to continue without RAG or exit
        # return False # Uncomment to make RAG mandatory

    # 4. Initialize Tools (Pass dependencies)
    print("Initializing LangChain Tools...")
    try:
        input_parser_tool = UserInputParsingTool(llm=chat_llm)
        validator_tool = DataValidationTool() # No dependencies needed in constructor
        search_tool = AmadeusFlightSearchTool() # Manages its own token
        # Pass LLM and RAG instance to Analyzer Tool
        analyzer_tool = FlightAnalysisTool(llm=chat_llm, rag_instance=flight_rag_instance)
        print("Tools Initialized.")
    except Exception as e:
        print(f"FATAL: Tool Initialization failed: {e}"); traceback.print_exc()
        return False

    # 5. Initialize Voice Services (Optional)
    if config.VOICE_ENABLED:
        print("Initializing Voice Services (TTS/STT)...")
        tts_ok = initialize_tts()
        stt_ok = initialize_stt()
        print(f"TTS Active: {tts_ok}, STT Active: {stt_ok}")
        # Update config flags based on actual initialization results
        config.USE_TTS_OUTPUT = tts_ok
        config.USE_VOICE_INPUT = stt_ok
    else:
        print("Voice features disabled by configuration.")

    return True # All essential initializations successful

# --- Main Execution Logic ---
def main():
    """Main function to run the flight assistant."""
    if not initialize_systems():
        sys.exit(1) # Exit if initialization failed

    # Check if IATA data loaded successfully (handled in utils, but double-check)
    if not CITIES_IATA_DATA:
         print("Cannot proceed without IATA city data. Please check cities_iata.json path and format.")
         sys.exit(1)


    # --- Initial State ---
    conversation_state = {
        "requirements": {field: None for field in config.REQUIRED_FIELDS},
        "preferences": {field: None for field in config.OPTIONAL_FIELDS},
        "conversation_mode": "greeting",
        "flights_found_summary": None,
        "last_validation_errors": [],
        "all_flights_raw": [],
        "simplified_options_from_rag": [],
        "selected_flight_for_booking": None,
        "history": deque(maxlen=config.CONVERSATION_HISTORY_LENGTH)
    }

    # --- Initial Greeting ---
    initial_greeting = "👋 Hi there! I'm your flight assistant. How can I help you find flights today?"
    print(f"Assistant: {initial_greeting}")
    if config.USE_TTS_OUTPUT: speak_text(initial_greeting)
    conversation_state["history"].append({"role": "assistant", "content": initial_greeting})
    conversation_state["conversation_mode"] = "gathering"

    # --- Main Loop ---
    while conversation_state["conversation_mode"] != "ended":
        user_input = None
        try:
            # --- Get Input ---
            if config.USE_VOICE_INPUT:
                user_input = listen_for_input()
                if not user_input: # Voice failed/timeout
                    fallback_msg = "I didn't catch that. Could you please type your request?"
                    print(f"Assistant: {fallback_msg}")
                    if config.USE_TTS_OUTPUT: speak_text(fallback_msg)
                    try: user_input = input("You (text fallback): ").strip()
                    except EOFError: user_input = "exit" # Treat EOF as exit request
                    if not user_input: continue # Skip turn if text fallback empty
            else: # Text input only
                 try: user_input = input("You: ").strip()
                 except EOFError: user_input = "exit" # Treat EOF as exit request
                 if not user_input: continue

            if not user_input: continue # Should not happen, but safeguard

            conversation_state["history"].append({"role": "user", "content": user_input})
            print("\n--- Processing Turn ---")
            start_turn_time = time.time()

            # 1. Parse (Tool Call)
            print(f" -> Parsing input...")
            parser_start_time = time.time()
            # Pass history deque converted to list for JSON serialization
            parser_input_dict = {
                 "user_input": user_input,
                 "conversation_state": {**conversation_state, "history": list(conversation_state["history"])}
             }
            parser_result_json = input_parser_tool._run(json.dumps(parser_input_dict))
            print(f"   Parser Tool Duration: {time.time() - parser_start_time:.2f}s")
            # ... (rest of parser result handling - intent, extracted_data, error) ...
            intent = "UNCLEAR"; extracted_data = {}; parser_error = None
            try:
                parser_result = json.loads(parser_result_json)
                intent = parser_result.get("intent", "UNCLEAR")
                extracted_data = parser_result.get("extracted_data", {})
                parser_error = parser_result.get("error")
                if parser_error: print(f"Parser Tool Error: {parser_error}"); intent = "UNCLEAR"
                if not isinstance(extracted_data, dict): extracted_data = {}
                print(f"   Intent: {intent}, Extracted (Raw): {extracted_data}")
            except Exception as e: print(f"Error decoding parser result: {e}"); parser_error = "Decode failure"; intent = "UNCLEAR"


            # 2. Validate (Tool Call)
            validated_data = {}; validation_errors = []
            if extracted_data and not parser_error:
                print(" -> Validating data...")
                validator_start_time = time.time()
                validator_input_json = json.dumps({
                    "extracted_data": extracted_data,
                    "current_requirements": conversation_state.get("requirements", {})})
                validator_result_json = validator_tool._run(validator_input_json)
                print(f"   Validator Tool Duration: {time.time() - validator_start_time:.2f}s")
                # ... (rest of validator result handling - validated_data, errors) ...
                try:
                    validator_result = json.loads(validator_result_json)
                    validated_data = validator_result.get("validated_data", {})
                    validation_errors = validator_result.get("errors", [])
                    if validator_result.get("error"): validation_errors.append(validator_result['error'])
                    print(f"   Validated Subset: {validated_data}, Errors: {validation_errors}")
                except Exception as e: print(f"Error decoding validator result: {e}"); validation_errors.append("Internal validation error.")
            elif parser_error: print(" -> Parser error, skipping validation."); validation_errors.append(f"Input understanding issue ({parser_error}).")
            else: print(" -> No data extracted/relevant, skipping validation.")


            # 3. Coordinate (Function Call - pass dependencies)
            print(" -> Coordinating response...")
            coord_start_time = time.time()
            next_response, next_state, next_action = generate_coordinator_response(
                conversation_state, intent, extracted_data, validated_data, validation_errors, user_input,
                chat_llm=chat_llm, flight_rag_instance=flight_rag_instance # Pass instances
            )
            print(f"   Coordinator Logic & LLM Duration: {time.time() - coord_start_time:.2f}s")

            # --- Output Response & Update State ---
            print(f"Assistant: {next_response}")
            if config.USE_TTS_OUTPUT: speak_text(next_response)
            # Update history using the returned state's history object
            next_state["history"].append({"role": "assistant", "content": next_response})
            conversation_state = next_state
            print(f"   Next Mode: {conversation_state['conversation_mode']}, Action Required: {next_action}")


            # 4. Execute Action (if needed)
            if next_action == 'exit': pass # Loop condition handles exit
            elif next_action == 'initiate_search':
                print("\n--- Initiating Flight Search ---")
                search_start_time = time.time()
                search_criteria = conversation_state.get("requirements", {}).copy()
                search_criteria.update(conversation_state.get("preferences", {}))
                search_criteria.pop("suggestion_keywords", None); search_criteria.pop("selected_flight_identifier", None)
                search_result_json = search_tool._run(json.dumps({"search_criteria": search_criteria}))
                print(f"   Search Tool Duration: {time.time() - search_start_time:.2f}s")
                # ... (Handle search result: success, error, flights) ...
                search_success = False; search_error_msg = None; flights = []
                try:
                    search_data = json.loads(search_result_json)
                    flights = search_data.get("flights"); search_error_msg = search_data.get("error")
                    if isinstance(flights, list) and not search_error_msg: search_success = True
                    elif search_error_msg: print(f"Search API Error: {search_error_msg}")
                    else: search_error_msg = "Invalid flight data format."; print(f"Search Error: {search_error_msg}")
                except Exception as e: print(f"Error parsing search result: {e}"); search_error_msg = "Internal error processing results."

                response_after_search = ""
                if search_success and flights:
                    print(f" -> Found {len(flights)} flight offers.")
                    conversation_state["all_flights_raw"] = flights
                    conversation_state["simplified_options_from_rag"] = [] # Clear old ones
                    # ... (Generate price range summary) ...
                    try:
                        prices = [f['price'] for f in flights if isinstance(f.get('price'), (int, float))]
                        curr = flights[0].get('currency','USD') if flights else 'USD'
                        price_range = f" Prices ~${min(prices):.0f}-${max(prices):.0f} {curr}." if prices else ""
                    except: price_range = ""
                    summary = f"Found {len(flights)} option(s)!{price_range} Ask about details (cheapest, fastest, airlines, non-stop?)"
                    conversation_state["flights_found_summary"] = summary
                    conversation_state["conversation_mode"] = "conversing_flights"
                    response_after_search = summary

                    # Encode flights in RAG (if RAG is working)
                    if flight_rag_instance and flight_rag_instance.is_initialized:
                        print("\n--- Encoding Flights for RAG ---")
                        rag_encode_start = time.time()
                        encode_input = json.dumps({"action": "encode", "flights": flights})
                        encode_result = analyzer_tool._run(encode_input) # Use the tool instance
                        print(f"   RAG Encoding Duration: {time.time() - rag_encode_start:.2f}s")
                        try: # Log RAG status
                            enc_data = json.loads(encode_result)
                            print(f"   RAG Info: {enc_data.get('message', enc_data.get('error', 'Status unknown'))}")
                        except: pass # Ignore logging errors
                    else: print("   Skipping RAG encoding (RAG not available).")

                else: # Search failed
                    fail_msg = search_error_msg or ("No flights found." if not flights else "Unexpected search issue.")
                    response_after_search = f"Sorry, search issue: {fail_msg} Modify request or try again?"
                    conversation_state["conversation_mode"] = "confirming" # Go back
                    conversation_state["all_flights_raw"] = []; conversation_state["simplified_options_from_rag"] = []
                    if flight_rag_instance and flight_rag_instance.is_initialized: flight_rag_instance.encode_flight_data([]) # Clear index

                print(f"Assistant: {response_after_search}")
                if config.USE_TTS_OUTPUT: speak_text(response_after_search)
                conversation_state["history"].append({"role": "assistant", "content": response_after_search})


            elif next_action == 'answer_flight_question':
                print("\n--- Answering Flight Question with RAG ---")
                rag_query_start = time.time()
                query_input = json.dumps({"action": "query", "user_question": user_input})
                rag_result_json = analyzer_tool._run(query_input) # Use the tool instance
                print(f"   RAG Query/Analysis Duration: {time.time() - rag_query_start:.2f}s")
                # ... (Handle RAG result: answer, simplified_options, error) ...
                rag_answer = "Sorry, couldn't find specific details for that."; simplified_opts = []
                try:
                    rag_data = json.loads(rag_result_json)
                    rag_answer = rag_data.get("answer", rag_answer)
                    simplified_opts = rag_data.get("simplified_options", [])
                    if simplified_opts: conversation_state["simplified_options_from_rag"] = simplified_opts; print(f"   RAG: Updated simplified options ({len(simplified_opts)}).")
                    if rag_data.get("error"): print(f"   RAG Warning: {rag_data.get('error')}")
                except Exception as e: print(f"Error parsing RAG answer: {e}")
                final_rag_response = f"{rag_answer} What else about these flights?"
                print(f"Assistant: {final_rag_response}")
                if config.USE_TTS_OUTPUT: speak_text(final_rag_response)
                conversation_state["history"].append({"role": "assistant", "content": final_rag_response})


            elif next_action == 'send_booking_email':
                 print("\n--- Attempting Booking Confirmation Email ---")
                 email_start_time = time.time()
                 selected_flight = conversation_state.get("selected_flight_for_booking")
                 recipient = config.RECIPIENT_EMAIL
                 final_message = ""
                 if not selected_flight: final_message = "Error: No flight selected state."; conversation_state["conversation_mode"] = "conversing_flights"
                 elif not recipient or recipient == "YOUR_RECIPIENT_EMAIL@example.com": final_message = "Error: Recipient email not configured."; conversation_state["conversation_mode"] = "conversing_flights"
                 else:
                     # Add context to flight details for email
                     flight_to_send = selected_flight.copy()
                     flight_to_send["passenger_count"] = conversation_state.get("requirements", {}).get("passengers", "N/A")
                     flight_to_send["class_preference"] = conversation_state.get("preferences", {}).get("class_preference", "Any")
                     success, message = send_flight_details_email(flight_to_send, recipient) # Use imported func
                     print(f"   Email Sending Duration: {time.time() - email_start_time:.2f}s")
                     print(f"   Email Result: Success={success}, Msg='{message}'")
                     if success:
                          final_message = f"Sent details for {selected_flight.get('airline','Flight')} to {recipient}. Check email for booking instructions. Anything else?"
                          conversation_state["conversation_mode"] = "ended_after_booking" # Temp state
                     else:
                          final_message = f"Error sending email: {message}. Try again later? What now?"
                          conversation_state["conversation_mode"] = "conversing_flights"
                 print(f"Assistant: {final_message}")
                 if config.USE_TTS_OUTPUT: speak_text(final_message)
                 conversation_state["history"].append({"role": "assistant", "content": final_message})
                 if conversation_state["conversation_mode"] == "ended_after_booking": conversation_state["conversation_mode"] = "ended"


            # --- End Turn ---
            print(f"--- Turn completed in {time.time() - start_turn_time:.2f} seconds ---")

        except KeyboardInterrupt: print("\nAssistant: Okay, cancelling. Goodbye!"); conversation_state["conversation_mode"] = "ended"
        except EOFError: print("\nAssistant: Input ended. Goodbye!"); conversation_state["conversation_mode"] = "ended"
        except Exception as e:
            print(f"\n*** UNEXPECTED ERROR in Main Loop: {type(e).__name__}: {e} ***"); traceback.print_exc()
            error_msg = "Oh dear, unexpected glitch. Let's try again. Repeat last request or start over?"
            print(f"Assistant: {error_msg}")
            if config.USE_TTS_OUTPUT: speak_text(error_msg)
            conversation_state["history"].append({"role": "assistant", "content": error_msg})
            conversation_state["last_validation_errors"] = ["Internal error."]
            conversation_state["conversation_mode"] = "gathering" # Attempt recovery

    print("\nAssistant: Session ended.")
    if config.USE_TTS_OUTPUT: speak_text("Session ended. Goodbye!")

if __name__ == "__main__":
    main()