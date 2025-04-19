# langchain_tools.py
import json
import re
import dateparser
from datetime import datetime, timedelta
import traceback
import time
import requests

from pydantic import Field, BaseModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_groq import ChatGroq # Keep this import specific for clarity

# Import shared components/config
from config import (
    ALL_FIELDS, REQUIRED_FIELDS, OPTIONAL_FIELDS,
    AMADEUS_CLIENT_ID, AMADEUS_CLIENT_SECRET
)
from utils import city_to_iata, calculate_duration, CITIES_IATA_DATA
# RAG instance will be passed during initialization in main.py

# --- Pydantic Schemas ---
class UserInputParsingToolInput(BaseModel):
    user_input: str = Field(description="The raw text input from the user.")
    conversation_state: dict = Field(description="The current state of the conversation including history.")

class DataValidationToolInput(BaseModel):
    extracted_data: dict = Field(description="Data extracted by the parsing tool.")
    current_requirements: dict = Field(description="Current requirements from conversation state for context.")

class AmadeusFlightSearchToolInput(BaseModel):
    search_criteria: dict = Field(description="Validated search parameters including cities, dates, passengers, etc.")

class FlightAnalysisToolInput(BaseModel):
    action: str = Field(description="Either 'encode' or 'query'.")
    flights: list | None = Field(default=None, description="List of flight dicts for 'encode' action.")
    user_question: str | None = Field(default=None, description="User question for 'query' action.")

# --- Tool Definitions ---

class UserInputParsingTool(BaseTool):
    """Analyzes user message to extract details and intent using an LLM."""
    name: str = "UserInputParser"
    description: str = "Analyzes user message considering conversation history to extract flight details and intent..." # Keep short
    args_schema: type[BaseModel] = UserInputParsingToolInput
    llm: ChatGroq # Expect LLM to be passed during initialization

    def _run(self, input_json: str) -> str:
        # (Keep the complex prompt and logic from the original file here)
        # Make sure it uses self.llm instead of the global chat_llm
        # --- REVISED PROMPT emphasizing history context ---
        system_prompt_template = """
Analyze the user's *latest* message **based on the current conversation state and recent history** to extract flight details and determine the primary intent.

Today's Date: {today_date}

Current State:
- Mode: {mode}
- Missing Required Info: {missing_fields_str}
- Requirements Filled: {requirements_str}
- Preferences Filled: {preferences_str}
- Flights Found: {flights_found_status}

Recent Conversation History (last {history_len} turns):
--- HISTORY START ---
{history_str}
--- HISTORY END ---

Your Tasks:
1.  **Analyze the *latest* user message:**
    - **Use the history for context:** Understand references (e.g., if user says "change it to Tokyo", determine what "it" referred to previously). Understand follow-up questions or corrections.
    - Extract any new or corrected flight details *explicitly mentioned in the current message*.
    - Use keys: {all_fields_str}. Also extract flight identifiers if mentioned (e.g., "option 1", "the Delta flight", "the first one", "the cheapest", "the return flight", "The fastest Flight", "The non-stop one", "The flight in the evening") - use key `selected_flight_identifier`.
    - Format dates as YYYY-MM-DD. Infer relative dates based on Today's Date.
    - Extract passenger count as a number.
    - Capitalize city names properly.
    - Extract suggestion keywords if user asks for ideas (key: `suggestion_keywords`).
    - Extract meal preferences (key: `meal_preference`).
    - Extract preferred airline (key: `preferred_airline`).
    - Extract cabin class (key: `class_preference`).
2.  **Determine the primary Intent** based on the *latest* user message. Choose ONE:
    - PROVIDING_DETAILS: User gives new information like city, date, passengers, preferences.
    - CORRECTING_DETAILS: User explicitly changes previously given info.
    - ASKING_QUESTION_FLIGHTS: User asks about specific flight options *after* a search has been performed.
    - ASKING_QUESTION_GENERAL: User asks general travel questions, about capabilities, policies, etc.
    - ASKING_SUGGESTION: User asks for destination ideas or help deciding.
    - CONFIRMING_SEARCH: User explicitly agrees to proceed with search using gathered details.
    - **CONFIRMING_BOOKING:** User explicitly confirms they want to proceed with booking *a specific flight* that has been discussed or presented (e.g., 'book the first option', 'confirm the United flight', 'yes, book that one', 'let's do it').
    - REQUESTING_EXIT: User wants to stop the conversation.
    - UNCLEAR: If the intent is ambiguous, combines multiple points, doesn't fit, or is unintelligible.

Output Format (Strict JSON):
```json
{{
  "intent": "INTENT_NAME",
  "extracted_data": {{ "field_name": "value", ... , "selected_flight_identifier": "identifier_string_or_null", "suggestion_keywords": "keywords_or_null" }}
}}
```"""
        # --- End of moved code ---
        if not self.llm: return json.dumps({"error": "LLM not available."})
        try:
            tool_input = UserInputParsingToolInput.model_validate_json(input_json)
            # ... rest of the original _run logic using self.llm ...
            prompt = ChatPromptTemplate.from_messages([("system", system_prompt_template), ("human", "Latest User Message: \"{user_input}\"")])
            chain = prompt | self.llm | JsonOutputParser()
            # Prepare input dict...
            conversation_state = tool_input.conversation_state
            requirements = conversation_state.get("requirements", {})
            preferences = conversation_state.get("preferences", {})
            mode = conversation_state.get("conversation_mode", "greeting")
            history_list = conversation_state.get("history", []) # Assume history is passed as list
            history_str = "\n".join([f"{msg.get('role','Unknown').title()}: {msg.get('content','')}" for msg in history_list])
            if not history_str: history_str = "No previous history."
            requirements_str = json.dumps(requirements)
            preferences_str = json.dumps(preferences)
            missing_fields = [f for f in REQUIRED_FIELDS if not requirements.get(f)]
            missing_fields_str = ", ".join(missing_fields) if missing_fields else "None"
            all_fields_str = ", ".join(ALL_FIELDS)
            flights_found_status = "Yes" if conversation_state.get("all_flights_raw") else "No"

            invoke_input = {
                "today_date": datetime.now().strftime('%Y-%m-%d'), "mode": mode,
                "missing_fields_str": missing_fields_str, "requirements_str": requirements_str,
                "preferences_str": preferences_str, "flights_found_status": flights_found_status,
                "history_len": len(history_list), "history_str": history_str,
                "all_fields_str": all_fields_str, "user_input": tool_input.user_input
            }
            response_dict = chain.invoke(invoke_input)
            # ... validation of response_dict ...
            if not isinstance(response_dict, dict): raise ValueError("LLM did not return dict")
            if "intent" not in response_dict: response_dict["intent"] = "UNCLEAR"
            if "extracted_data" not in response_dict or not isinstance(response_dict.get("extracted_data"), dict): response_dict["extracted_data"] = {}
            if "selected_flight_identifier" not in response_dict["extracted_data"]: response_dict["extracted_data"]["selected_flight_identifier"] = None
            if "suggestion_keywords" not in response_dict["extracted_data"]: response_dict["extracted_data"]["suggestion_keywords"] = None

            return json.dumps(response_dict)
        except (json.JSONDecodeError, TypeError) as e:
            return json.dumps({"intent": "UNCLEAR", "extracted_data": {}, "error": f"Invalid JSON input/Pydantic validation error: {e}"})
        except Exception as e:
            print(f"Error in UserInputParsingTool: {e}"); traceback.print_exc()
            return json.dumps({"intent": "UNCLEAR", "extracted_data": {}, "error": f"Internal tool error: {e}"})


class DataValidationTool(BaseTool):
    """Validates extracted flight details against business rules."""
    name: str = "DataValidator"
    description: str = "Validates extracted flight details (cities, dates, passengers, preferences)."
    args_schema: type[BaseModel] = DataValidationToolInput

    def _run(self, input_json: str) -> str:
        # (Keep the logic from the original file here)
        # Ensure it uses city_to_iata from utils
        # --- Start of moved code ---
        try:
            tool_input = DataValidationToolInput.model_validate_json(input_json)
            extracted_data = tool_input.extracted_data
            current_requirements = tool_input.current_requirements
            if not isinstance(extracted_data, dict):
                 return json.dumps({"validated_data": {}, "errors": ["Invalid format: extracted_data must be a dictionary."]})

            errors = []
            validated = {}

            for field, value in extracted_data.items():
                if field not in ALL_FIELDS or value is None: continue
                raw_value = str(value).strip()
                if not raw_value: continue
                if field in ["selected_flight_identifier", "suggestion_keywords"]:
                    if isinstance(value, (str, int, float)): validated[field] = value # Basic pass-through
                    continue

                # Date Validation
                if field in ["departure_date", "return_date"]:
                    try:
                        now = datetime.now()
                        relative_base_date = (now if now.hour < 20 else now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                        parsed_date = dateparser.parse(raw_value, settings={'RELATIVE_BASE': relative_base_date, 'PREFER_DATES_FROM': 'future', 'STRICT_PARSING': False})
                        if parsed_date:
                            date_str = parsed_date.strftime("%Y-%m-%d")
                            if parsed_date.date() < now.date():
                                errors.append(f"Date Error: {field.replace('_', ' ')} '{raw_value}' ({date_str}) is in the past.")
                            elif field == "return_date":
                                departure_date_str = current_requirements.get("departure_date") or validated.get("departure_date")
                                if departure_date_str:
                                    try:
                                        dep_date_obj = datetime.strptime(departure_date_str, "%Y-%m-%d")
                                        if parsed_date.date() <= dep_date_obj.date():
                                            errors.append(f"Date Error: Return date '{raw_value}' ({date_str}) must be after departure ({departure_date_str}).")
                                        else: validated[field] = date_str
                                    except (ValueError, TypeError): validated[field] = date_str # Validate format if dep_date is bad
                                else: validated[field] = date_str # Validate format if dep_date unknown
                            else: validated[field] = date_str # Valid departure date
                        else: errors.append(f"Date Error: Cannot understand date: '{raw_value}'.")
                    except Exception as e: errors.append(f"Date Error: Parsing '{raw_value}': {e}")

                # Passengers Validation
                elif field == "passengers":
                    try:
                        num_passengers = -1
                        word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "a couple": 2, "a few": 3, "just me": 1, "myself": 1, "alone": 1}
                        num_str_cleaned = re.sub(r'[^\w\s-]', '', raw_value.lower()).strip()
                        if num_str_cleaned in word_to_num: num_passengers = word_to_num[num_str_cleaned]
                        else:
                            num_match = re.search(r'\b(\d+)\b', num_str_cleaned)
                            if num_match: num_passengers = int(num_match.group(1))
                            else:
                                # --- FIX START ---
                                # The try...except needs to be on separate lines with proper indentation
                                try:
                                    num_passengers = int(raw_value)
                                except ValueError:
                                    pass # If raw_value cannot be directly converted to int, keep num_passengers as -1
                                # --- FIX END ---
                        if 1 <= num_passengers <= 9: validated[field] = num_passengers
                        elif num_passengers > 9: errors.append("Passenger Error: Max 9 passengers.")
                        else: errors.append(f"Passenger Error: Invalid number: '{raw_value}'. (1-9)")
                    except Exception as e: errors.append(f"Passenger Error: Cannot parse '{raw_value}': {e}")

                # City Validation
                elif field in ["departure_city", "arrival_city"]:
                    city_name = raw_value.strip().title()
                    if len(city_name) > 1 and not city_name.isdigit():
                        iata = city_to_iata(city_name) # Use imported function
                        if iata != "Unknown": validated[field] = city_name # Store user name
                        else:
                             is_likely_iata = len(city_name) == 3 and city_name.isalpha() and city_name.isupper()
                             if is_likely_iata: validated[field] = city_name # Assume user entered IATA
                             else: errors.append(f"City Error: Cannot recognize city '{city_name}'.")
                    else: errors.append(f"City Error: Invalid city name '{raw_value}'.")

                # Class Preference Validation
                elif field == "class_preference":
                    pref = raw_value.lower()
                    class_map = {"economy": "ECONOMY", "coach": "ECONOMY", "cheap": "ECONOMY", "cheapest": "ECONOMY", "standard": "ECONOMY", "basic": "ECONOMY", "lowest": "ECONOMY", "regular": "ECONOMY", "premium economy": "PREMIUM_ECONOMY", "premium": "PREMIUM_ECONOMY", "prem econ": "PREMIUM_ECONOMY", "econ plus": "PREMIUM_ECONOMY", "economy plus": "PREMIUM_ECONOMY", "premium coach": "PREMIUM_ECONOMY", "business": "BUSINESS", "biz": "BUSINESS", "business class": "BUSINESS", "first": "FIRST", "first class": "FIRST"}
                    matched_class = None
                    if pref in class_map: matched_class = class_map[pref]
                    else: # Keyword check
                        if "premium" in pref and "economy" in pref: matched_class = "PREMIUM_ECONOMY"
                        elif "first" in pref: matched_class = "FIRST"
                        elif "business" in pref: matched_class = "BUSINESS"
                        elif "economy" in pref: matched_class = "ECONOMY"
                        elif "coach" in pref: matched_class = "ECONOMY"
                    if matched_class: validated[field] = matched_class
                    else: errors.append(f"Class Error: Unrecognized class '{raw_value}'. Use Economy, Premium Economy, Business, First.")

                # Other Optional Fields
                elif field in ["meal_preference", "preferred_airline"]:
                    if 1 < len(raw_value) < 150:
                         validated[field] = raw_value.title() if field == "preferred_airline" else raw_value
                    else: errors.append(f"Preference Error: {field.replace('_', ' ')} '{raw_value}' seems invalid.")

            return json.dumps({"validated_data": validated, "errors": errors})
        except (json.JSONDecodeError, TypeError) as e:
            return json.dumps({"validated_data": {}, "errors": [f"Invalid JSON input/Pydantic validation error: {e}"]})
        except Exception as e:
            print(f"Error in DataValidationTool: {e}"); traceback.print_exc()
            return json.dumps({"validated_data": {}, "errors": [f"Internal validation error: {e}"]})
        # --- End of moved code ---

class AmadeusFlightSearchTool(BaseTool):
    """Searches Amadeus API for flight offers."""
    name: str = "AmadeusFlightSearch"
    description: str = "Searches Amadeus API for flight offers based on validated criteria."
    args_schema: type[BaseModel] = AmadeusFlightSearchToolInput
    _token: str | None = None
    _token_expiry: datetime | None = None

    def _fetch_token(self):
        # (Keep the token fetching logic from the original file here)
        # --- Start of moved code ---
        if self._token and self._token_expiry and datetime.now() < self._token_expiry:
            return self._token
        print("Fetching new Amadeus token...")
        url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        client_id = AMADEUS_CLIENT_ID
        client_secret = AMADEUS_CLIENT_SECRET
        if not client_id or client_id == "YOUR_AMADEUS_CLIENT_ID" or \
           not client_secret or client_secret == "YOUR_AMADEUS_CLIENT_SECRET":
             print("Error: Amadeus client ID/secret not configured."); return None
        data = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
        try:
            response = requests.post(url, headers=headers, data=data, timeout=30)
            response.raise_for_status()
            token_data = response.json()
            self._token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3599)
            self._token_expiry = datetime.now() + timedelta(seconds=max(expires_in - 120, 300))
            print(f"Amadeus Token fetched. Expires ~: {self._token_expiry.strftime('%Y-%m-%d %H:%M:%S')}")
            return self._token
        except requests.exceptions.Timeout: print("Error fetching Amadeus token: Timeout."); return None
        except requests.exceptions.HTTPError as e:
            error_msg = f"Error fetching Amadeus token: HTTP {e.response.status_code}."
            try: error_msg += f" Response: {json.dumps(e.response.json())}"
            except json.JSONDecodeError: error_msg += f" Raw Response: {e.response.text[:200]}"
            print(error_msg); return None
        except requests.exceptions.RequestException as e: print(f"Error fetching Amadeus token: {e}"); return None
        except json.JSONDecodeError as e:
            status = response.status_code if 'response' in locals() else 'N/A'
            text = response.text[:200] if 'response' in locals() else 'N/A'
            print(f"Error decoding token JSON. Status: {status}. Response: {text}. Error: {e}"); return None
        # --- End of moved code ---


    def _run(self, input_json: str) -> str:
        # (Keep the main search logic from the original file here)
        # Ensure it uses city_to_iata and calculate_duration from utils
        # --- Start of moved code ---
        try:
            tool_input = AmadeusFlightSearchToolInput.model_validate_json(input_json)
            criteria = tool_input.search_criteria
            if not isinstance(criteria, dict): return json.dumps({"flights": [], "error": "Invalid criteria format."})

            dep_city = criteria.get("departure_city"); arr_city = criteria.get("arrival_city")
            dep_date = criteria.get("departure_date"); passengers = criteria.get("passengers")
            return_date = criteria.get("return_date"); cabin_class = criteria.get("class_preference")

            missing = [f for f, v in [("departure city", dep_city), ("arrival city", arr_city), ("departure date", dep_date), ("passengers", passengers)] if not v]
            if missing: return json.dumps({"flights": [], "error": f"Missing required criteria: {', '.join(missing)}."})

            origin_iata = city_to_iata(dep_city); dest_iata = city_to_iata(arr_city)
            if origin_iata == "Unknown" or dest_iata == "Unknown":
                missing_iata = [c for c, iata in [(dep_city, origin_iata), (arr_city, dest_iata)] if iata == "Unknown"]
                return json.dumps({"flights": [], "error": f"Cannot find airport code for {', '.join(missing_iata)}."})

            token = self._fetch_token()
            if not token: return json.dumps({"flights": [], "error": "Failed to get Amadeus token."})

            api_endpoint = "https://test.api.amadeus.com/v2/shopping/flight-offers"
            headers = {"Authorization": f"Bearer {token}"}
            params = {"originLocationCode": origin_iata, "destinationLocationCode": dest_iata, "departureDate": dep_date, "adults": passengers, "currencyCode": "USD", "max": 15}
            if return_date: params["returnDate"] = return_date
            if cabin_class and cabin_class in ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"]: params["travelClass"] = cabin_class

            print(f"Searching Amadeus with params: {params}")
            processed_flights = []; data = None
            start_time = time.time()
            try:
                response = requests.get(api_endpoint, headers=headers, params=params, timeout=60)
                response.raise_for_status(); data = response.json()
                print(f"Amadeus API call successful ({response.status_code}). Duration: {time.time() - start_time:.2f}s")
            except requests.exceptions.Timeout: return json.dumps({"flights": [], "error": "Flight search timed out."})
            except requests.exceptions.HTTPError as e:
                error_msg = f"Amadeus API Error (HTTP {e.response.status_code})."
                user_friendly_error = "Error during flight search."
                try:
                    details = e.response.json().get('errors', [])
                    detail_msgs = [str(d.get('detail', d.get('title', ''))) for d in details if d.get('detail') or d.get('title')]
                    if detail_msgs:
                        error_msg += f" Details: {' | '.join(detail_msgs)}"
                        if any("INVALID DATE" in m.upper() for m in detail_msgs): user_friendly_error = f"Invalid date provided ({dep_date}{' / '+return_date if return_date else ''})."
                        elif any("INVALID LOCATION" in m.upper() for m in detail_msgs): user_friendly_error = f"Unrecognized origin/destination ({dep_city}/{origin_iata} or {arr_city}/{dest_iata})."
                        elif any("NO FLIGHT OFFERS FOUND" in m.upper() for m in detail_msgs): user_friendly_error = "No flights found for your criteria."
                        elif any("MANDATORY DATA MISSING" in m.upper() for m in detail_msgs): user_friendly_error = "Missing required search info."
                        else: user_friendly_error = f"Issue with search request details. (Details: {' | '.join(detail_msgs)})"
                    else: error_msg += f" Raw: {e.response.text[:250]}"
                    if e.response.status_code >= 500: user_friendly_error = "Airline search system temporary issue. Try again later."
                except Exception: error_msg += f" Could not parse error details. Raw: {e.response.text[:250]}"
                print(f"Amadeus Search Error Log: {error_msg}")
                return json.dumps({"flights": [], "error": user_friendly_error})
            except requests.exceptions.RequestException as e: return json.dumps({"flights": [], "error": f"Network error during search: {e}"})
            except json.JSONDecodeError as e: return json.dumps({"flights": [], "error": "Received unreadable response from search system."})

            if data is None: return json.dumps({"flights": [], "error": "Internal error: Failed to process search response."})

            dictionaries = data.get("dictionaries", {}); airlines_dict = dictionaries.get("carriers", {})
            offers = data.get("data", [])
            print(f"Received {len(offers)} flight offers.")

            for i, offer in enumerate(offers):
                try:
                    offer_id = offer.get("id", f"gen_{i}"); price_info = offer.get("price")
                    if not price_info or price_info.get("total") is None: print(f"Warn: Skip offer {offer_id}, no price."); continue
                    itins = offer.get("itineraries", []); outbound_itin = itins[0] if itins else None
                    if not outbound_itin or not outbound_itin.get("segments"): print(f"Warn: Skip offer {offer_id}, no outbound segs."); continue
                    out_segs = outbound_itin["segments"]; first_dep = out_segs[0].get("departure", {}); last_arr = out_segs[-1].get("arrival", {})
                    out_dep_t = first_dep.get("at"); out_arr_t = last_arr.get("at")
                    origin_code = first_dep.get("iataCode"); dest_code = last_arr.get("iataCode")

                    return_itin = itins[1] if len(itins) > 1 else None
                    ret_segs = return_itin.get("segments", []) if return_itin else []; ret_dep_t = None; ret_arr_t = None
                    if ret_segs: ret_dep_t = ret_segs[0].get("departure", {}).get("at"); ret_arr_t = ret_segs[-1].get("arrival", {}).get("at")

                    val_carriers = offer.get("validatingAirlineCodes", [])
                    op_carrier = out_segs[0].get("operating", {}).get("carrierCode")
                    mark_carrier = out_segs[0].get("carrierCode")
                    main_code = val_carriers[0] if val_carriers else op_carrier if op_carrier else mark_carrier if mark_carrier else "N/A"
                    airline_name = airlines_dict.get(main_code, main_code)

                    details = {
                        "offer_id": offer_id, "price": float(price_info.get("total")), "currency": price_info.get("currency", "USD"),
                        "airline_code": main_code, "airline": airline_name, "origin": origin_code, "destination": dest_code,
                        "departure_time": out_dep_t, "arrival_time": out_arr_t, "duration": calculate_duration(out_dep_t, out_arr_t),
                        "stops": len(out_segs) - 1, "type": "Round-trip" if bool(return_itin) else "One-way", "is_round_trip": bool(return_itin),
                        "return_departure_time": ret_dep_t, "return_arrival_time": ret_arr_t,
                        "return_duration": calculate_duration(ret_dep_t, ret_arr_t) if ret_dep_t and ret_arr_t else None,
                        "return_stops": len(ret_segs) - 1 if ret_segs else None,
                    }
                    if not details["is_round_trip"]: # Clean up return fields if one-way
                        for k in list(details.keys()):
                            if k.startswith('return_') and details[k] is None: del details[k]
                    processed_flights.append(details)
                except Exception as e: print(f"Warn: Skipping offer {offer.get('id', f'gen_{i}')} due to processing error: {e}. Snippet: {str(offer)[:200]}")

            processed_flights.sort(key=lambda x: x.get('price', float('inf')))
            if not processed_flights:
                error_msg = "Found offers, but failed processing details." if offers else "No flight offers found matching criteria."
                return json.dumps({"flights": [], "error": error_msg})

            print(f"Successfully processed {len(processed_flights)} flight offers.")
            return json.dumps({"flights": processed_flights, "error": None})

        except (json.JSONDecodeError, TypeError) as e: return json.dumps({"flights": [], "error": f"Invalid JSON input/Pydantic validation error: {e}"})
        except Exception as e: print(f"Critical Error in AmadeusFlightSearchTool: {e}"); traceback.print_exc(); return json.dumps({"flights": [], "error": "Unexpected internal search tool error."})
        # --- End of moved code ---


class FlightAnalysisTool(BaseTool):
    """Analyzes flight offers using RAG (encode/query)."""
    name: str = "FlightDataAnalyzerRAG"
    description: str = "Analyzes flight offers using RAG. Actions: 'encode', 'query'."
    args_schema: type[BaseModel] = FlightAnalysisToolInput
    llm: ChatGroq # Expect LLM passed in
    rag_instance: object # Expect RAG instance passed in (use 'object' for generic type hint)

    def _run(self, input_json: str) -> str:
        # (Keep the logic from the original file here)
        # Make sure it uses self.llm and self.rag_instance
        # --- Start of moved code ---
        if not self.rag_instance or not getattr(self.rag_instance, 'is_initialized', False):
             init_fail_msg = "RAG system not available (failed initialization)."
             return json.dumps({"success": False, "error": init_fail_msg})

        try:
            tool_input = FlightAnalysisToolInput.model_validate_json(input_json)
            action = tool_input.action

            if action == "encode":
                flights = tool_input.flights
                if not isinstance(flights, list): return json.dumps({"success": False, "error": "Invalid 'flights' format for encoding."})
                success = self.rag_instance.encode_flight_data(flights)
                if success:
                     count = getattr(getattr(self.rag_instance, 'index', None), 'ntotal', 0)
                     return json.dumps({"success": True, "message": f"Encoded {count} flights into RAG index."})
                else: return json.dumps({"success": False, "error": "Failed to encode flights into RAG index."})

            elif action == "query":
                user_question = tool_input.user_question
                if not user_question: return json.dumps({"answer": "Question about flights was missing.", "relevant_flights": []})

                rag_index = getattr(self.rag_instance, 'index', None)
                if rag_index is None or rag_index.ntotal == 0:
                    return json.dumps({"answer": "No flight details loaded yet to answer questions.", "relevant_flights": []})

                print(f"RAG: Querying index with: '{user_question}'")
                relevant_flights = self.rag_instance.query(user_question, top_k=3)

                if not relevant_flights:
                    return json.dumps({"answer": f"Couldn't find specific details for '{user_question}'. Ask about price, airlines, duration?", "relevant_flights": []})

                context_parts = []; simplified_options = []
                for i, flight in enumerate(relevant_flights):
                    # Concise summary for LLM and state
                    origin = flight.get('origin', 'N/A'); dest = flight.get('destination', 'N/A')
                    airline = flight.get('airline', 'Unk'); price = flight.get('price'); currency = flight.get('currency', 'USD')
                    price_str = f"${price:.0f} {currency}" if price is not None else "N/A"
                    dur = flight.get('duration', 'N/A'); stops = f"{flight.get('stops')} stop(s)"
                    ftype = flight.get('type', 'Flight')

                    parts = [f"**Option {i+1} ({ftype}):** {airline} ({flight.get('airline_code', 'N/A')}) {origin}->{dest}."]
                    parts.append(f"Price ~{price_str}. Dur: {dur}, Stops: {stops}.")
                    if flight.get('is_round_trip'): parts.append(f"Return dur: {flight.get('return_duration', 'N/A')}, Stops: {flight.get('return_stops', 'N/A')} stop(s).")
                    context_parts.append(" ".join(parts))
                    simplified_options.append({"option_number": i + 1, "airline": airline, "origin": origin, "destination": dest, "price_summary": price_str, "offer_id": flight.get("offer_id")})

                context_str = "\n\n".join(context_parts)
                print(f"RAG: Context for LLM:\n{context_str}")

                synthesis_prompt = ChatPromptTemplate.from_messages([
                    ("system", """Answer user questions based *only* on the provided flight snippets. Be concise.
                    - Present relevant options clearly numbered (e.g., **Option 1:**) IF needed for comparison.
                    - If info isn't in snippets, state that clearly (e.g., "Based on Option 1, I don't see info about X.").
                    - Do NOT invent details. Be direct.
                    Relevant Flight Snippets:\n--- START DATA ---\n{context}\n--- END DATA ---"""),
                    ("human", "User Question: \"{question}\"")])
                synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()

                try:
                     start_time = time.time()
                     final_answer = synthesis_chain.invoke({"context": context_str, "question": user_question})
                     print(f"RAG: LLM Synthesis took {time.time() - start_time:.2f}s.")

                     refusal_phrases = ["cannot find", "don't have information", "not mentioned", "details provided do not include"]
                     if any(p in final_answer.lower() for p in refusal_phrases) and "option" not in final_answer.lower():
                         final_answer = f"Based on relevant options for '{user_question}':\n" + final_answer
                         print("RAG: LLM refused generically; prepended context.")

                     return json.dumps({"answer": final_answer.strip(), "relevant_flights_summary": f"Found {len(relevant_flights)} relevant options.", "simplified_options": simplified_options})
                except Exception as llm_e:
                     print(f"Error calling LLM for RAG synthesis: {llm_e}"); traceback.print_exc()
                     return json.dumps({"answer": f"Found {len(relevant_flights)} relevant options, but had trouble summarizing.", "relevant_flights_summary": f"Found {len(relevant_flights)} options.", "simplified_options": simplified_options})

            else: return json.dumps({"success": False, "error": f"Invalid action '{action}'. Use 'encode' or 'query'."})
        except (json.JSONDecodeError, TypeError) as e: return json.dumps({"error": f"Invalid JSON input/Pydantic validation error: {e}"})
        except Exception as e: print(f"Error in FlightAnalysisTool: {e}"); traceback.print_exc(); return json.dumps({"error": "Internal analysis tool error."})
        # --- End of moved code ---