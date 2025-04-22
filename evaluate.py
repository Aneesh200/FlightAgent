# evaluate.py
import json
import time
import traceback
from collections import defaultdict
import re # For parsing LLM judge output

# --- Project Modules ---
import config
from utils import load_cities_iata, CITIES_IATA_DATA # Make sure CITIES_IATA_DATA is accessible
from langchain_groq import ChatGroq
from rag_module import FlightRAG
from langchain_tools import UserInputParsingTool, DataValidationTool, FlightAnalysisTool

# --- Evaluation Specific Imports ---
from sklearn.metrics import accuracy_score # For parser intent

# --- Global Instances ---
chat_llm = None # Used by tools AND as judge
flight_rag_instance = None
input_parser_tool = None
validator_tool = None # Keep if needed for parser/other tests
analyzer_tool = None # The tool being tested for RAG

# --- Constants for LLM Judge ---
JUDGE_MODEL = config.LLM_MODEL # Use the same model or specify another like "mixtral-8x7b-32768"
JUDGE_TEMPERATURE = 0.0 # Low temp for consistent judging
JUDGE_MAX_RETRIES = 1
JUDGE_TIMEOUT = 45 # Allow time for judgment call

FAITHFULNESS_PROMPT_TEMPLATE = """
System: You are an impartial judge evaluating an AI assistant's answer based *only* on the provided context. Do not use any prior knowledge.

Context (Flight Information):
{context}

User Question:
{question}

Assistant's Answer:
{answer}

Task: Evaluate if the Assistant's Answer is faithfully supported by the Context.
- The answer must be derivable *exclusively* from the text in the Context.
- Do not penalize the answer for being incomplete if the asked-for information is missing from the context.
- Penalize the answer if it includes information NOT present in the context (hallucination).
- Penalize the answer if it contradicts information in the context.

Provide your evaluation in the following format:
Faithfulness Score: [1-5] (1=Completely Unfaithful/Hallucinated, 5=Fully Faithful)
Reasoning: [Brief explanation for the score, citing context details or lack thereof]
"""

RELEVANCE_PROMPT_TEMPLATE = """
System: You are an impartial judge evaluating an AI assistant's answer based on its relevance to the User Question.

User Question:
{question}

Assistant's Answer:
{answer}

Task: Evaluate if the Assistant's Answer directly addresses the User Question.
- Does the answer attempt to answer the specific question asked?
- Ignore whether the answer is factually correct or faithful (that's evaluated separately). Focus only on relevance.
- An answer like "I cannot answer that based on the provided flight data" IS relevant if the question asked for something outside the scope (e.g., weather).
- An answer discussing unrelated topics is irrelevant.

Provide your evaluation in the following format:
Relevance Score: [1-5] (1=Completely Irrelevant, 5=Directly Relevant)
Reasoning: [Brief explanation for the score]
"""


def initialize_evaluation_systems():
    """Initialize systems needed for evaluation."""
    global chat_llm, flight_rag_instance, input_parser_tool, validator_tool, analyzer_tool

    print("Initializing systems for evaluation...")
    # *** MODIFIED LINE HERE ***
    if not config.perform_critical_checks(): # Removed ignore_voice=True
        print("Critical configuration missing. Exiting evaluation.")
        return False

    # Initialize LLM (Used for tools AND Judge)
    try:
        chat_llm = ChatGroq(
            temperature=JUDGE_TEMPERATURE, # Use judge temp default here
            model_name=JUDGE_MODEL,
            api_key=config.GROQ_API_KEY,
            max_retries=JUDGE_MAX_RETRIES,
            request_timeout=JUDGE_TIMEOUT
        )
        print("LLM Initialized (for tools and judging).")
    except Exception as e:
        print(f"FATAL: LLM Init failed: {e}"); return False

    # Initialize RAG
    flight_rag_instance = FlightRAG() # Uses config model internally
    if not flight_rag_instance.is_initialized:
        print("Warning: RAG init failed. RAG tests will be skipped.")
        # Allow other tests to run

    # Initialize Tools
    try:
        # Make sure tools use the primary LLM instance if they need one
        # Consider if parser needs different temp settings
        parser_llm = ChatGroq(
            temperature=config.LLM_TEMPERATURE, # Default temp from config
            model_name=config.LLM_MODEL,
            api_key=config.GROQ_API_KEY,
            max_retries=config.LLM_MAX_RETRIES,
            request_timeout=config.LLM_REQUEST_TIMEOUT
        )
        input_parser_tool = UserInputParsingTool(llm=parser_llm) # Give parser its own LLM instance
        validator_tool = DataValidationTool()
        analyzer_tool = FlightAnalysisTool(llm=chat_llm, rag_instance=flight_rag_instance) # Analyzer uses main (judge) LLM instance
        print("Tools Initialized.")
    except Exception as e:
        print(f"FATAL: Tool Init failed: {e}"); return False

    if not load_cities_iata():
         print("Warning: Failed to load IATA city data. Validation/parsing steps might be limited.")

    print("Evaluation systems initialized.")
    return True

def parse_judge_response(response_text, score_name):
    """Parses the score and reasoning from the LLM judge's response."""
    score = 0
    reasoning = "Parsing failed."
    try:
        # Improved regex to handle optional brackets and varying whitespace
        score_match = re.search(rf"{score_name}\s*Score:\s*\[?(\d)\]?", response_text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
            # Validate score is within expected range (1-5)
            if not 1 <= score <= 5:
                score = 0 # Invalid score
                reasoning = f"Parsed score '{score_match.group(1)}' out of range (1-5)."

        else: # If score pattern not found at all
            reasoning = f"'{score_name} Score: [1-5]' pattern not found."

        # Make reasoning parsing more robust
        reasoning_match = re.search(r"Reasoning:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            # Only update reasoning if score parsing didn't already set an error message
            if score != 0 or "' pattern not found" in reasoning:
                 reasoning = reasoning_match.group(1).strip()
        elif score != 0: # If score was found but no reasoning line
             reasoning = "Reasoning line not found after score."

    except Exception as e:
        reasoning = f"Error parsing judge response: {e}"
        score = 0 # Ensure score is 0 on parsing exception
    return score, reasoning

# --- Evaluation Functions ---

def evaluate_parser(test_case):
    """Evaluates the UserInputParsingTool."""
    if not input_parser_tool: return {"accuracy": 0, "f1": 0, "error": "Parser tool not initialized"}

    print(f"\n--- Testing Parser: {test_case['id']} ---")
    print(f"Input: {test_case['user_input']}")
    start_time = time.time()

    parser_input_dict = {
        "user_input": test_case['user_input'],
        "conversation_state": {"history": []} # Minimal state
    }
    parser_result_json = input_parser_tool._run(json.dumps(parser_input_dict))
    duration = time.time() - start_time
    print(f"Raw Output: {parser_result_json} (Duration: {duration:.2f}s)")

    results = {"id": test_case['id'], "duration": duration, "pass": False}
    try:
        parser_result = json.loads(parser_result_json)
        actual_intent = parser_result.get("intent", "UNCLEAR")
        actual_entities = parser_result.get("extracted_data", {})
        parser_error = parser_result.get("error")

        if parser_error:
            print(f"Parser tool reported error: {parser_error}")
            results["error"] = parser_error
            return results

        # Intent Matching (Accuracy)
        expected_intent = test_case.get("expected_intent")
        intent_correct = (actual_intent == expected_intent)
        print(f"Intent: Expected='{expected_intent}', Actual='{actual_intent}' -> {'Correct' if intent_correct else 'Incorrect'}")
        results["intent_correct"] = intent_correct

        # Entity Matching (Slot Accuracy - simplified)
        expected_entities = test_case.get("expected_entities", {})
        correct_slots = 0
        present_actual_relevant = {}
        for key, expected_value in expected_entities.items():
             if key in actual_entities:
                 present_actual_relevant[key] = actual_entities[key]
                 # Simple case-insensitive comparison for robustness
                 if str(actual_entities[key]).strip().lower() == str(expected_value).strip().lower():
                     correct_slots += 1

        num_expected = len(expected_entities)
        slot_accuracy = correct_slots / num_expected if num_expected > 0 else 1.0
        print(f"Entities: Expected={expected_entities}, Actual Relevant={present_actual_relevant}")
        print(f"Slot Accuracy (Exact match on expected slots): {slot_accuracy:.2f} ({correct_slots}/{num_expected})")

        results["slot_accuracy"] = slot_accuracy
        results["actual_intent"] = actual_intent
        results["actual_entities"] = actual_entities
        # Adjust pass criteria as needed
        results["pass"] = intent_correct and (slot_accuracy >= 0.8 or num_expected == 0)

    except Exception as e:
        print(f"Error processing parser result: {e}")
        results["error"] = str(e)
        traceback.print_exc()

    return results


def evaluate_rag_llm_judge(test_case):
    """Evaluates the FlightAnalysisTool's RAG capability using an LLM judge."""
    if not analyzer_tool or not flight_rag_instance or not chat_llm:
        return {"error": "Analyzer, RAG, or Judge LLM not initialized"}
    if not flight_rag_instance.is_initialized:
         return {"error": "RAG module not initialized, cannot test"}

    print(f"\n--- Testing RAG (LLM Judge): {test_case['id']} ---")
    start_time = time.time()

    # 1. Prepare Context for RAG and Judge
    context_flights = test_case.get("context_flights", [])
    context_string = "\n".join([f"- {json.dumps(f)}" for f in context_flights])
    if not context_string: context_string = "No flight data provided in context."

    # 2. Encode context into RAG instance for the test
    try:
        print("   Encoding context flights into RAG...")
        encode_input = json.dumps({"action": "encode", "flights": context_flights})
        encode_result = analyzer_tool._run(encode_input)
        print(f"   Encoding result: {encode_result}")
    except Exception as e:
         print(f"   Error encoding flights for test: {e}")
         return {"id": test_case['id'], "error": f"Failed to encode flights: {e}"}

    # 3. Run the actual RAG query using the Analyzer Tool
    user_question = test_case["user_question"]
    print(f"   User Question: {user_question}")
    actual_answer = "Error during RAG query execution."
    rag_duration = 0
    try:
        query_input_json = json.dumps({
            "action": "query",
            "user_question": user_question
        })
        query_start_time = time.time()
        rag_result_json = analyzer_tool._run(query_input_json)
        rag_duration = time.time() - query_start_time
        print(f"   Raw RAG Output: {rag_result_json} (Duration: {rag_duration:.2f}s)")
        rag_result = json.loads(rag_result_json)
        actual_answer = rag_result.get("answer", "No answer provided by tool.")
        if rag_result.get("error"):
            actual_answer = f"Tool Error: {rag_result.get('error')}"
    except Exception as e:
        print(f"   Error running RAG query: {e}")
        actual_answer = f"Failed to run/parse RAG query: {e}"
        rag_duration = time.time() - query_start_time # Record duration up to failure

    print(f"   Actual Answer: {actual_answer}")

    # 4. Call LLM Judge for Faithfulness
    print("   Judging Faithfulness...")
    faithfulness_score = 0
    faithfulness_reasoning = "Judging failed."
    judge_faith_duration = 0
    try:
        judge_start_time = time.time()
        faithfulness_prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
            context=context_string,
            question=user_question,
            answer=actual_answer
        )
        # Use the specific judge LLM instance
        judge_response = chat_llm.invoke(faithfulness_prompt)
        judge_faith_duration = time.time() - judge_start_time
        print(f"   Judge Faithfulness Response: {judge_response.content}")
        faithfulness_score, faithfulness_reasoning = parse_judge_response(judge_response.content, "Faithfulness")
    except Exception as e:
        print(f"   Error calling/parsing faithfulness judge: {e}")
        faithfulness_reasoning = f"Faithfulness judging error: {e}"
        judge_faith_duration = time.time() - judge_start_time
        faithfulness_score = 0 # Ensure score is 0 on error

    print(f"   Faithfulness Score: {faithfulness_score}/5 (Judge Duration: {judge_faith_duration:.2f}s)")

    # 5. Call LLM Judge for Relevance
    print("   Judging Relevance...")
    relevance_score = 0
    relevance_reasoning = "Judging failed."
    judge_rel_duration = 0
    try:
        judge_start_time = time.time()
        relevance_prompt = RELEVANCE_PROMPT_TEMPLATE.format(
            question=user_question,
            answer=actual_answer
        )
        # Use the specific judge LLM instance
        judge_response = chat_llm.invoke(relevance_prompt)
        judge_rel_duration = time.time() - judge_start_time
        print(f"   Judge Relevance Response: {judge_response.content}")
        relevance_score, relevance_reasoning = parse_judge_response(judge_response.content, "Relevance")
    except Exception as e:
        print(f"   Error calling/parsing relevance judge: {e}")
        relevance_reasoning = f"Relevance judging error: {e}"
        judge_rel_duration = time.time() - judge_start_time
        relevance_score = 0 # Ensure score is 0 on error

    print(f"   Relevance Score: {relevance_score}/5 (Judge Duration: {judge_rel_duration:.2f}s)")

    total_duration = time.time() - start_time

    # Optional: Check against expected keywords
    keywords_match_details = {"match": "N/A", "missing": [], "present": []}
    if "expected_keywords" in test_case and test_case["expected_keywords"]:
        present_keywords = []
        missing_keywords = []
        all_match = True
        for keyword in test_case["expected_keywords"]:
            if keyword.lower() in actual_answer.lower():
                present_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
                all_match = False
        keywords_match_details["match"] = all_match
        keywords_match_details["missing"] = missing_keywords
        keywords_match_details["present"] = present_keywords
        print(f"   Expected Keywords Match: {all_match} (Present: {present_keywords}, Missing: {missing_keywords})")
    else:
         print("   Expected Keywords: Not defined for this test case.")


    return {
        "id": test_case['id'],
        "question": user_question,
        "context_flights": context_flights,
        "actual_answer": actual_answer,
        "rag_duration": rag_duration,
        "faithfulness_score": faithfulness_score,
        "faithfulness_reasoning": faithfulness_reasoning,
        "judge_faith_duration": judge_faith_duration,
        "relevance_score": relevance_score,
        "relevance_reasoning": relevance_reasoning,
        "judge_rel_duration": judge_rel_duration,
        "keywords_match_details": keywords_match_details,
        "total_duration": total_duration,
        # Adjust pass criteria as needed (e.g., allow lower scores for certain cases)
        "pass": faithfulness_score >= 3 and relevance_score >= 3
    }


# --- Main Evaluation Runner ---

def run_evaluation():
    """Loads test suite and runs different evaluation types."""
    if not initialize_evaluation_systems():
        print("Exiting due to initialization failure.")
        return

    try:
        with open("test_suite.json", 'r') as f:
            test_suite = json.load(f)
    except FileNotFoundError:
        print("Error: test_suite.json not found.")
        return
    except json.JSONDecodeError:
        print("Error: test_suite.json is not valid JSON.")
        return
    except Exception as e:
        print(f"Error loading test_suite.json: {e}")
        return

    parser_results = []
    rag_results = []

    print(f"\nFound {len(test_suite)} test cases in test_suite.json")
    for i, test_case in enumerate(test_suite):
        test_id = test_case.get('id', f'unnamed_test_{i+1}')
        test_type = test_case.get("type")
        print(f"\n[{i+1}/{len(test_suite)}] Running test: {test_id} (Type: {test_type})")

        if test_type == "parser":
            result = evaluate_parser(test_case)
            parser_results.append(result)
        elif test_type == "rag_llm_judge":
            result = evaluate_rag_llm_judge(test_case)
            rag_results.append(result)
        else:
            print(f"Skipping test case {test_id} with unknown type: {test_type}")

        # Optional: Add a small delay between tests to avoid rate limits
        time.sleep(config.LLM_REQUEST_TIMEOUT / 10 if config.LLM_REQUEST_TIMEOUT else 1.0) # Sleep based on timeout

    # --- Summarize Parser Results ---
    if parser_results:
        print("\n\n" + "="*20 + " Parser Evaluation Summary " + "="*20)
        total_tests = len(parser_results)
        passed_tests = sum(1 for r in parser_results if r.get("pass"))
        avg_slot_accuracy = sum(r.get("slot_accuracy", 0) for r in parser_results) / total_tests if total_tests > 0 else 0
        avg_duration = sum(r.get("duration", 0) for r in parser_results) / total_tests if total_tests > 0 else 0

        print(f"Total Parser Tests: {total_tests}")
        print(f"Passed Tests: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")

        # Calculate overall intent accuracy
        intents_true = []
        intents_pred = []
        for tc in test_suite:
            if tc.get("type") == "parser":
                matching_result = next((r for r in parser_results if r.get("id") == tc.get("id")), None)
                if matching_result and "error" not in matching_result: # Only include if test ran without error
                    intents_true.append(tc.get("expected_intent"))
                    intents_pred.append(matching_result.get("actual_intent"))

        if intents_true: # Ensure we have results to compare
            intent_accuracy = accuracy_score(intents_true, intents_pred)
            print(f"Overall Intent Accuracy (on successful runs): {intent_accuracy*100:.1f}%")
        else:
             print("Overall Intent Accuracy: N/A (No successful parser runs)")

        print(f"Average Slot Accuracy (on expected slots): {avg_slot_accuracy:.2f}")
        print(f"Average Duration: {avg_duration:.2f}s")
        errors = [f"  - {r['id']}: {r.get('error', 'Unknown error')}" for r in parser_results if not r.get("pass") or r.get("error")]
        if errors: print(f"\nFailed/Error Cases ({len(errors)}):")
        for err in errors: print(err)

    # --- Summarize RAG LLM-Judge Results ---
    if rag_results:
        print("\n\n" + "="*20 + " RAG Evaluation (LLM-as-Judge) Summary " + "="*20)
        total_tests = len(rag_results)
        passed_tests = sum(1 for r in rag_results if r.get("pass"))
        # Filter out potential errors before calculating averages
        valid_faith_scores = [r.get("faithfulness_score", 0) for r in rag_results if "error" not in r]
        valid_rel_scores = [r.get("relevance_score", 0) for r in rag_results if "error" not in r]
        valid_rag_durations = [r.get("rag_duration", 0) for r in rag_results if "error" not in r]
        valid_judge_durations = [r.get("judge_faith_duration", 0) + r.get("judge_rel_duration", 0) for r in rag_results if "error" not in r]

        avg_faithfulness = sum(valid_faith_scores) / len(valid_faith_scores) if valid_faith_scores else 0
        avg_relevance = sum(valid_rel_scores) / len(valid_rel_scores) if valid_rel_scores else 0
        avg_rag_duration = sum(valid_rag_durations) / len(valid_rag_durations) if valid_rag_durations else 0
        avg_judge_duration = sum(valid_judge_durations) / len(valid_judge_durations) if valid_judge_durations else 0

        print(f"Total RAG Tests: {total_tests}")
        print(f"Passed Tests (Faithfulness >= 3 AND Relevance >= 3): {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"Average Faithfulness Score (on successful runs): {avg_faithfulness:.2f}/5")
        print(f"Average Relevance Score (on successful runs): {avg_relevance:.2f}/5")
        print(f"Average RAG Query Duration: {avg_rag_duration:.2f}s")
        print(f"Average Judge Duration (Faith+Rel): {avg_judge_duration:.2f}s")

        failed_cases = [r for r in rag_results if not r.get("pass") or r.get("error")]
        print(f"\nFailed/Low-Score/Error Cases ({len(failed_cases)}):")
        for r in failed_cases:
            print(f"  - ID: {r['id']}")
            if r.get("error"):
                print(f"    ERROR: {r['error']}")
                continue # Skip detailed scores if there was a fundamental error
            print(f"    Q: {r['question'][:80]}...")
            print(f"    A: {r['actual_answer'][:100]}...")
            print(f"    Faithfulness: {r.get('faithfulness_score', 'N/A')}/5 | Reason: {r.get('faithfulness_reasoning', 'N/A')[:100]}...")
            print(f"    Relevance:    {r.get('relevance_score', 'N/A')}/5 | Reason: {r.get('relevance_reasoning', 'N/A')[:100]}...")
            kwd_details = r.get('keywords_match_details', {})
            kwd_match = kwd_details.get('match', 'N/A')
            kwd_missing = kwd_details.get('missing', [])
            if kwd_match != 'N/A':
                print(f"    Keywords Match: {kwd_match} {'(Missing: ' + ', '.join(kwd_missing) + ')' if kwd_missing else ''}")

    print("\n" + "="*25 + " Evaluation Complete " + "="*25)

    # Optionally save detailed results to a JSON file
    results_data = {"parser": parser_results, "rag": rag_results}
    results_filename = "evaluation_results.json"
    try:
        with open(results_filename, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\nDetailed results saved to: {results_filename}")
    except Exception as e:
        print(f"\nError saving results to {results_filename}: {e}")


if __name__ == "__main__":
    run_evaluation()