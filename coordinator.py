# coordinator.py
import json
import re
from datetime import datetime
from collections import deque
import traceback
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq # Keep specific import

# Import shared components/config
from config import REQUIRED_FIELDS, OPTIONAL_FIELDS, RECIPIENT_EMAIL
# RAG instance and LLM will be passed into the function

def generate_coordinator_response(
    conversation_state,
    intent,
    extracted_data, # Raw from parser
    validated_data, # Validated subset
    validation_errors,
    user_input,
    chat_llm: ChatGroq, # Pass LLM instance
    flight_rag_instance: object # Pass RAG instance (use 'object' type hint)
):
    """
    Determines the next state, action, and crafts the user response instruction.
    Uses LLM for crafting the final text based on determined logic.
    """
    # (Keep the complex logic from the original file here)
    # ... [Approx 300+ lines of the original generate_coordinator_response function] ...
    # Ensure it uses chat_llm and flight_rag_instance where needed
    # --- Start of moved code ---
    potential_next_state = conversation_state.copy()
    potential_next_state["selected_flight_for_booking"] = None # Initialize

    current_reqs = potential_next_state.get("requirements", {})
    current_prefs = potential_next_state.get("preferences", {})
    current_mode = potential_next_state.get("conversation_mode", "greeting")
    flights_available = bool(potential_next_state.get("all_flights_raw"))
    history = potential_next_state.get("history", deque()) # Get deque
    last_error = potential_next_state.get("last_validation_errors", [])
    selected_flight_identifier = extracted_data.get("selected_flight_identifier")

    updated_reqs = current_reqs.copy(); updated_prefs = current_prefs.copy()
    next_action_required = 'none'; next_mode = current_mode
    response_prompt_context = ""
    update_performed = False

    is_providing_data_intent = intent in ['PROVIDING_DETAILS', 'CORRECTING_DETAILS'] or \
                               (intent == 'UNCLEAR' and validated_data and not validation_errors)

    if validated_data and not validation_errors:
        print(f"Coordinator: Updating state with validated data: {validated_data}")
        for field, value in validated_data.items():
            if field in ["selected_flight_identifier", "suggestion_keywords"]: continue
            if field in REQUIRED_FIELDS:
                if updated_reqs.get(field) != value: updated_reqs[field] = value; update_performed = True
            elif field in OPTIONAL_FIELDS:
                 if updated_prefs.get(field) != value: updated_prefs[field] = value; update_performed = True
        extracted_keywords = extracted_data.get("suggestion_keywords")
        if isinstance(extracted_keywords, str) and extracted_keywords:
            current_keywords_list = updated_prefs.get("suggestion_keywords", [])
            if isinstance(current_keywords_list, str): current_keywords_list = [current_keywords_list]
            if extracted_keywords not in current_keywords_list:
                updated_prefs["suggestion_keywords"] = current_keywords_list + [extracted_keywords]
                update_performed = True
        if not update_performed and is_providing_data_intent: print("Coordinator: Valid data, but no state change.")

    potential_next_state["requirements"] = updated_reqs
    potential_next_state["preferences"] = updated_prefs
    potential_next_state["last_validation_errors"] = validation_errors if validation_errors else []

    # Details for LLM prompt context
    prompt_details_for_llm = {
            "dep_city": updated_reqs.get("departure_city", "Not set"), "arr_city": updated_reqs.get("arrival_city", "Not set"),
            "dep_date": updated_reqs.get("departure_date", "Not set"), "ret_date": updated_reqs.get("return_date") or "Not set",
            "pax": updated_reqs.get("passengers", "Not set"), "cabin": updated_prefs.get("class_preference") or "Any",
            "suggestion_keywords": ", ".join(updated_prefs.get("suggestion_keywords", [])) }

    # --- Intent Handling Logic ---
    if intent == 'REQUESTING_EXIT':
        next_action_required = 'exit'; next_mode = 'ended'
        response_prompt_context = "User wants to end. Provide polite closing."
    elif intent == 'CONFIRMING_BOOKING':
        recipient_email_check = RECIPIENT_EMAIL and RECIPIENT_EMAIL != "YOUR_RECIPIENT_EMAIL@example.com"
        if not recipient_email_check:
             next_mode = 'conversing_flights'
             response_prompt_context = "Cannot book, recipient email not configured."
             potential_next_state["requirements"] = current_reqs # Revert potential changes
             potential_next_state["preferences"] = current_prefs
        elif not flights_available or not conversation_state.get("all_flights_raw"):
            next_mode = 'gathering'
            response_prompt_context = "No flights found/searched yet. Need to search first."
            # Add check for missing required fields and prompt if needed
            all_req_filled = all(potential_next_state["requirements"].get(f) for f in REQUIRED_FIELDS)
            if not all_req_filled:
                missing = [f for f in REQUIRED_FIELDS if not potential_next_state["requirements"].get(f)]
                field_map = { "departure_city": "origin city", "arrival_city": "destination city", "departure_date": "departure date", "passengers": "number of passengers" }
                missing_name = field_map.get(missing[0], missing[0].replace('_', ' '))
                response_prompt_context = f"Need flights first. Missing details: {missing_name}. Please provide."
                next_mode = "gathering"

        else: # Flights available, identify target
            target_flight = None; all_raw_flights = conversation_state.get("all_flights_raw", [])
            simplified_options = conversation_state.get("simplified_options_from_rag", [])

            if selected_flight_identifier:
                print(f"Coordinator: Identifier: '{selected_flight_identifier}'")
                identifier_lower = str(selected_flight_identifier).lower(); matched_by_id = []
                # 1. Match Option Number (using simplified RAG options)
                if simplified_options:
                    num_match = re.search(r'\b(\d+)\b', identifier_lower)
                    if num_match:
                        try:
                            option_num = int(num_match.group(1))
                            target_offer_id = next((s_opt.get("offer_id") for s_opt in simplified_options if s_opt.get("option_number") == option_num), None)
                            if target_offer_id:
                                target_flight_match = next((f for f in all_raw_flights if f.get("offer_id") == target_offer_id), None)
                                if target_flight_match: matched_by_id = [target_flight_match]; print(f"   Matched Option {option_num} via RAG options.")
                        except Exception: pass
                # 2. Match Airline Name
                if not matched_by_id:
                    matched_airlines = [f for f in all_raw_flights if f.get('airline') and f.get('airline').lower() in identifier_lower]
                    if matched_airlines: matched_by_id = matched_airlines; print(f"   Matched by airline name: {[f['airline'] for f in matched_by_id]}")
                # 3. Match Common Terms ("first", "cheapest", "non-stop")
                if not matched_by_id:
                     if "first" in identifier_lower or "cheapest" in identifier_lower:
                         if all_raw_flights: matched_by_id = [all_raw_flights[0]]; print("   Matched by 'first'/'cheapest'.")
                     elif "non-stop" in identifier_lower or "direct" in identifier_lower:
                         non_stops = [f for f in all_raw_flights if f.get('stops') == 0]
                         if non_stops: matched_by_id = non_stops; print(f"   Matched by 'non-stop' ({len(matched_by_id)} found).")

                # Check match results
                if len(matched_by_id) == 1: target_flight = matched_by_id[0]; print(f"   Single target identified: Offer ID {target_flight.get('offer_id')}")
                elif len(matched_by_id) > 1:
                     print(f"   Ambiguous identifier '{selected_flight_identifier}', matched {len(matched_by_id)}.")
                     next_mode = 'conversing_flights'
                     ambiguous_summary = ", ".join([f"{fl.get('airline','N/A')} (~${fl.get('price',0):.0f})" for fl in matched_by_id[:3]])
                     response_prompt_context = f"Found a few options for '{selected_flight_identifier}' (like {ambiguous_summary}). Be more specific? (e.g., Option number, airline)."
                     potential_next_state["requirements"] = current_reqs; potential_next_state["preferences"] = current_prefs
                else:
                     print(f"   Identifier '{selected_flight_identifier}' not matched.")
                     next_mode = 'conversing_flights'
                     response_prompt_context = f"Sorry, couldn't identify '{selected_flight_identifier}'. Clarify which flight? (Option number, airline)."
                     potential_next_state["requirements"] = current_reqs; potential_next_state["preferences"] = current_prefs

            elif len(all_raw_flights) == 1: target_flight = all_raw_flights[0]; print(f"   No identifier, assuming the only flight.")
            elif len(all_raw_flights) > 1:
                 print("   No identifier, multiple flights. Asking clarification."); next_mode = 'conversing_flights'
                 response_prompt_context = "Several options here. Which one to book? (Option number, airline)."
                 potential_next_state["requirements"] = current_reqs; potential_next_state["preferences"] = current_prefs
            else: # Should be caught earlier
                 next_mode = 'gathering'; response_prompt_context = "No flight details to book. Let's search again."
                 potential_next_state["all_flights_raw"] = []; potential_next_state["simplified_options_from_rag"] = []

            if target_flight: # Target flight identified
                if not recipient_email_check: # Final check
                    next_mode = 'conversing_flights'; response_prompt_context = "Identified flight, but cannot send email (recipient config issue)."
                else:
                    potential_next_state["selected_flight_for_booking"] = target_flight
                    next_action_required = 'send_booking_email'
                    next_mode = 'booking_confirmation'
                    summary = f"{target_flight.get('airline','Flight')} from {target_flight.get('origin','?')} to {target_flight.get('destination','?')}"
                    response_prompt_context = f"Okay! Sending details for the {summary} to {RECIPIENT_EMAIL}. One moment..."

    elif validation_errors:
        next_mode = "gathering"; first_error = validation_errors[0]
        response_prompt_context = f"Issue found: '{first_error}'. Could you please provide correct details?"
        potential_next_state["requirements"] = current_reqs # Revert changes from invalid input
        potential_next_state["preferences"] = current_prefs
        potential_next_state["last_validation_errors"] = validation_errors
        print("Coordinator: Validation failed, reverting state for this turn.")

    elif intent == 'ASKING_QUESTION_FLIGHTS':
        rag_ready = flight_rag_instance and getattr(flight_rag_instance, 'is_initialized', False) and \
                    getattr(getattr(flight_rag_instance, 'index', None), 'ntotal', 0) > 0
        if flights_available and rag_ready:
            next_action_required = 'answer_flight_question'; next_mode = 'conversing_flights'
            response_prompt_context = f"User asking about flights ('{user_input}'). Acknowledge checking details (e.g., 'Let me check that...')."
        else: # No flights or RAG not ready
            all_req_filled = all(potential_next_state["requirements"].get(f) for f in REQUIRED_FIELDS)
            if all_req_filled:
                 next_mode = "confirming"; response_prompt_context = "Haven't searched yet. Have main details. Shall I search? Summarize details (Origin, Dest, Date(s), Pax) and ask."
            else:
                 next_mode = "gathering"; missing = [f for f in REQUIRED_FIELDS if not potential_next_state["requirements"].get(f)]
                 field_map = { "departure_city": "origin city", "arrival_city": "destination city", "departure_date": "departure date", "passengers": "number of passengers" }
                 missing_name = field_map.get(missing[0], missing[0].replace('_', ' '))
                 response_prompt_context = f"Haven't searched yet and missing info (like {missing_name}). Please provide the {missing_name}?"

    elif intent == 'ASKING_SUGGESTION':
        next_mode = 'gathering'; current_keywords_list = potential_next_state["preferences"].get("suggestion_keywords", [])
        prompt_details_for_llm["suggestion_keywords"] = ", ".join(current_keywords_list) # Update for prompt
        context_parts = [] # Build context summary... (as before)
        if prompt_details_for_llm.get("dep_city") not in [None, 'Not set']: context_parts.append(f"from {prompt_details_for_llm['dep_city']}")
        # ... add other parts: dep_date, ret_date, pax, cabin ...
        current_keywords_str = prompt_details_for_llm.get("suggestion_keywords", "")
        if current_keywords_str: context_parts.append(f"interested in: {current_keywords_str}")
        llm_context_summary = ("Based on our chat: " + ", ".join(context_parts) + ".") if context_parts else "Looking for travel suggestions."

        missing_info_q = [] # Check missing info... (as before)
        if not potential_next_state["requirements"].get("departure_city"): missing_info_q.append("departure city")
        # ... add dep_date, passengers ...
        missing_info_note = ""
        if missing_info_q: missing_info_note = f"Knowing {', '.join(missing_info_q)} would help tailor ideas."

        ack = f"Okay, let's brainstorm ideas!"
        if current_keywords_str: ack = f"Okay, ideas for a {current_keywords_str} trip!"
        # Construct CoT prompt for LLM (as before)
        response_prompt_context = f"""
SYSTEM INSTRUCTION: User wants travel suggestions. Provide 2-3 diverse, reasoned flight destinations. Use Chain-of-Thought.
1. Acknowledge & Summarize: Start with '{ack}'. Summarize context: {llm_context_summary}.
2. Reason (Internal): Consider theme ('{current_keywords_str or 'general'}'), context -> destinations.
3. Suggest: Offer 2-3 concrete places reachable by flight (e.g., **City Name**), briefly state why.
4. Clarify & Nudge: Ask open-ended Q (e.g., "Do these sound interesting?", "What vibe?").
5. Mention Missing Info: Gently add '{missing_info_note}'.
Generate the response now.
""".strip()

    elif intent == 'ASKING_QUESTION_GENERAL':
        all_req_filled = all(potential_next_state["requirements"].get(f) for f in REQUIRED_FIELDS)
        next_mode = 'confirming' if all_req_filled else 'gathering'
        llm_instruction = f"User asked general question: '{user_input}'. " \
                          f"Answer briefly IF related to general travel/flights/my capabilities. If unrelated/too specific, politely state limitation. "
        if next_mode == 'gathering':
            missing = [f for f in REQUIRED_FIELDS if not potential_next_state["requirements"].get(f)]
            field_map = { "departure_city": "origin city", "arrival_city": "destination city", "departure_date": "departure date", "passengers": "number of passengers" }
            missing_name = field_map.get(missing[0], missing[0].replace('_', ' '))
            llm_instruction += f"Then, gently prompt for next detail: 'To continue, provide the {missing_name}?'"
        else: # Confirming mode
             dep_c = potential_next_state["requirements"].get("departure_city", "N/A") # Use potential state
             arr_c = potential_next_state["requirements"].get("arrival_city", "N/A")
             dep_d = potential_next_state["requirements"].get("departure_date", "N/A")
             summary = f"{dep_c} to {arr_c} on {dep_d}"
             llm_instruction += f"Then, ask to search: 'Shall I search for flights for {summary}?'"
        response_prompt_context = llm_instruction

    elif intent == 'UNCLEAR':
        all_req_filled = all(potential_next_state["requirements"].get(f) for f in REQUIRED_FIELDS)
        next_mode = 'confirming' if all_req_filled else 'gathering'
        prompt_question = ""
        if next_mode == 'gathering':
            missing = [f for f in REQUIRED_FIELDS if not potential_next_state["requirements"].get(f)]
            field_map = { "departure_city": "Where from?", "arrival_city": "Where to?", "departure_date": "What date?", "passengers": "How many people?" }
            prompt_question = field_map.get(missing[0], f"Provide the {missing[0].replace('_',' ')}?")
        else: prompt_question = "Shall I search based on details I have?"
        history_list_short = list(history)[-4:]; history_str_short = "\n".join([f"{m['role']}: {m['content']}" for m in history_list_short])
        response_prompt_context = f"User msg ('{user_input}') unclear. History:\n{history_str_short}\n\nAsk politely for clarification. Then guide: '{prompt_question}'"

    elif intent == 'CONFIRMING_SEARCH':
        all_req_filled = all(potential_next_state["requirements"].get(f) for f in REQUIRED_FIELDS)
        if all_req_filled:
             next_action_required = 'initiate_search'; next_mode = 'searching'
             response_prompt_context = "User confirmed search. Acknowledge enthusiastically and state search starting."
        else: # Confirmed but info missing
             missing = [f for f in REQUIRED_FIELDS if not potential_next_state["requirements"].get(f)]
             next_mode = 'gathering'
             field_map = { "departure_city": "origin city", "arrival_city": "destination city", "departure_date": "departure date", "passengers": "number of passengers" }
             missing_name = field_map.get(missing[0], missing[0].replace('_', ' ')) if missing else "details"
             response_prompt_context = f"User confirmed search, but missing info ({missing_name}). Politely point out and ask for first missing item again."

    else: # PROVIDING_DETAILS, CORRECTING_DETAILS, or UNCLEAR with data
        all_req_filled = all(potential_next_state["requirements"].get(f) for f in REQUIRED_FIELDS)
        ack = ""
        if update_performed and is_providing_data_intent:
            updated_fields = [] # Find updated fields... (as before)
            for field, value in validated_data.items():
                 if field in REQUIRED_FIELDS and current_reqs.get(field) != value: updated_fields.append(field.replace('_',' '))
                 elif field in OPTIONAL_FIELDS and current_prefs.get(field) != value: updated_fields.append(field.replace('_',' '))
            if "suggestion_keywords" in updated_prefs and set(current_prefs.get("suggestion_keywords",[])) != set(updated_prefs["suggestion_keywords"]): updated_fields.append("interests")
            ack = f"Okay, noted the {', '.join(updated_fields)}. " if updated_fields else "Okay, got that. "

        if all_req_filled:
            next_mode = 'confirming'
            # Build confirmation summary... (as before)
            reqs = potential_next_state["requirements"]; prefs = potential_next_state["preferences"]
            summary = f"Flying from {reqs.get('departure_city', 'N/A')} to {reqs.get('arrival_city', 'N/A')} " \
                      f"on {reqs.get('departure_date', 'N/A')} for {reqs.get('passengers', 'N/A')} pax"
            if reqs.get('return_date'): summary += f", returning {reqs.get('return_date')}"
            if prefs.get('class_preference') and prefs['class_preference'] != "Any": summary += f", in {prefs['class_preference'].lower().replace('_',' ')}"
            summary += "."
            response_prompt_context = f"{ack}Great, I have the main details. Double-check: {summary} Shall I search?"
        else:
            next_mode = 'gathering'; missing = [f for f in REQUIRED_FIELDS if not potential_next_state["requirements"].get(f)]
            field_map = { "departure_city": "Where from?", "arrival_city": "Where to?", "departure_date": "What date?", "passengers": "How many people?" }
            next_q = field_map.get(missing[0], f"What's the {missing[0].replace('_',' ')}?")
            response_prompt_context = f"{ack}{next_q}".strip()

    # Update final state
    final_next_state = potential_next_state
    final_next_state["conversation_mode"] = next_mode
    if current_mode == "conversing_flights" and next_mode != "conversing_flights":
         final_next_state["simplified_options_from_rag"] = []

    # Use LLM for final response text
    final_response_text = "Okay." # Fallback
    if not chat_llm: final_response_text = "LLM unavailable."; print("Coordinator Warn: LLM unavailable.")
    elif not response_prompt_context: print(f"Coordinator Warn: Empty prompt context for intent '{intent}', mode '{current_mode}'. Using basic fallback."); # Add basic fallback based on next_mode
    else:
        response_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly, helpful flight booking assistant. Generate a concise, conversational response based ONLY on the instruction below. Be natural.
            - Follow the instruction precisely.
            - Use summary info from instruction if needed.
            - Avoid robotic phrases. Use "Okay", "Got it", etc.
            - Do NOT reveal internal state names ('mode', 'intent').
            - Keep responses brief unless asked to summarize/suggest.
            Instruction:
            ---
            {instruction}
            ---
            Current Summary (for context if needed): From: {dep_city} To: {arr_city}, Depart: {dep_date} {ret_date_part}, Pax: {pax} {cabin_part} {keywords_part}"""),
        ])
        response_chain = response_gen_prompt | chat_llm | StrOutputParser()
        final_reqs = final_next_state.get("requirements", {}); final_prefs = final_next_state.get("preferences", {})
        ret_part = f"Return: {final_reqs.get('return_date')}" if final_reqs.get('return_date') else ""
        cab_part = f"Class: {final_prefs.get('class_preference', 'Any')}" if final_prefs.get('class_preference') and final_prefs['class_preference'] != "Any" else ""
        key_part = f"Interests: {', '.join(final_prefs.get('suggestion_keywords',[]))}" if final_prefs.get('suggestion_keywords') else ""
        try:
            start_time = time.time()
            final_response_text = response_chain.invoke({
                "instruction": response_prompt_context,
                "dep_city": final_reqs.get("departure_city", "Not set"), "arr_city": final_reqs.get("arrival_city", "Not set"),
                "dep_date": final_reqs.get("departure_date", "Not set"), "ret_date_part": ret_part,
                "pax": final_reqs.get("passengers", "Not set"), "cabin_part": cab_part, "keywords_part": key_part,
            })
            print(f"Coordinator: LLM Response Gen took {time.time() - start_time:.2f}s.")
            if not final_response_text or len(final_response_text) < 3: raise ValueError("LLM response too short")
        except Exception as e:
            print(f"Error generating response text: {e}. Using fallback logic."); traceback.print_exc()
            # Add enhanced fallback logic here based on next_mode/action (as in original)
            if next_action_required == 'exit': final_response_text = "Okay, goodbye!"
            elif validation_errors: final_response_text = f"Sorry, issue with info: {validation_errors[0]}. Please clarify?"
            elif next_mode == 'confirming': final_response_text = "Got it. Shall I search?" # Simplified fallback
            elif next_mode == 'gathering': final_response_text = "Okay. What's next?" # Simplified fallback
            # ... etc. for other modes/intents ...
            else: final_response_text = "Okay."

    return final_response_text.strip(), final_next_state, next_action_required
    # --- End of moved code ---