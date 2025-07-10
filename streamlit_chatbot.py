import streamlit as st
from openai import OpenAI
import time
import json
import pandas as pd
import numpy as np
from urllib.parse import unquote
import re
import uuid
import textwrap
from typing import List, Dict, Optional, Tuple
from nltk.util import ngrams
from num2words import num2words
import string
from zoneinfo import ZoneInfo
import sqlite3
from datetime import datetime
from pathlib import Path
from fuzzywuzzy import fuzz


# --- NEW: Firebase Firestore Imports ---
from google.cloud import firestore
from google.oauth2 import service_account

# --- NEW: Firebase Firestore Client Initialization ---
@st.cache_resource
def get_firestore_client():
    """Initializes and returns a Cloud Firestore client."""
    try:
        key_dict = json.loads(st.secrets["firestore"]["textkey"])
        creds = service_account.Credentials.from_service_account_info(key_dict)
        db = firestore.Client(credentials=creds, project=key_dict["project_id"])
        return db
    except Exception as e:
        st.error(f"Error initializing Firestore: {e}. "
                 f"Please ensure your .streamlit/secrets.toml is correctly configured.")
        st.stop()

db = get_firestore_client()
CHATBOT_LOGS_COLLECTION = "chatbot_interaction_logs"
logs_ref = db.collection(CHATBOT_LOGS_COLLECTION)

# --- Existing App Setup ---
client = OpenAI(api_key=st.secrets["openai"]["api_key"])
query_params = st.query_params
question_id = query_params.get("qid", "Q1")
question_text = query_params.get("qtext", "What is your decision?")
options_raw = query_params.get("opts", "Option 1|Option 2|Option 3|Option 4")
options = options_raw.split("|")

while len(options) < 4:
    options.append("")

option_mapping = {f"option {i+1}": options[i] for i in range(4)}
option_mapping.update({f"option{i+1}": options[i] for i in range(4)})
participant_id = query_params.get("pid", str(uuid.uuid4()))

if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'last_recommendation' not in st.session_state:
    st.session_state.last_recommendation = None
if 'last_question_id' not in st.session_state:
    st.session_state.last_question_id = None
if 'first_load' not in st.session_state:
    st.session_state.first_load = True
if 'already_saved' not in st.session_state:
    st.session_state.already_saved = False
if 'original_recommendation' not in st.session_state:
    st.session_state.original_recommendation = None
if 'followup_questions' not in st.session_state:
    st.session_state.followup_questions = []
if 'Youtubes' not in st.session_state:
    st.session_state.Youtubes = []
if 'last_followup_time' not in st.session_state:
    st.session_state.last_followup_time = 0

if "original_options" not in st.session_state:
    st.session_state.original_options = options
    st.session_state.option_mapping = {
        f"option{i+1}": options[i] for i in range(len(options))
    }

if 'usage_data' not in st.session_state:
    st.session_state.usage_data = {
        'participant_id': participant_id,
        'question_id': question_id,
        'chatbot_used': False,
        'total_questions_asked': 0,
        'get_recommendation': False,
        'followup_used': False,
        'start_time': None,
        'total_time': 0
    }

if 'interaction_active' not in st.session_state:
    st.session_state.interaction_active = False
if 'total_interaction_time' not in st.session_state:
    st.session_state.total_interaction_time = 0
if 'last_interaction_time' not in st.session_state:
    st.session_state.last_interaction_time = None
if 'get_recommendation_used' not in st.session_state:
    st.session_state.get_recommendation_used = False
if 'followup_used' not in st.session_state:
    st.session_state.followup_used = False

# --- Utility Functions (unchanged) ---
def normalize_numbers(text: str) -> str:
   return re.sub(r'\b\d+\b', lambda m: num2words(int(m.group())), text)

def has_continuous_match(option_text: str, user_input: str, min_len=2, max_len=5) -> bool:
    option_tokens = option_text.split()
    user_tokens = user_input.split()
    for n in range(max_len, min_len - 1, -1):
        option_ngrams = list(ngrams(option_tokens, n))
        user_ngrams = list(ngrams(user_tokens, n))
        for opt_ng in option_ngrams:
            if opt_ng in user_ngrams:
                return True
    return False

@st.cache_resource
# --- Utility Functions ---
def normalize_numbers(text: str) -> str:
   """Converts numerical digits in text to their word form."""
   return re.sub(r'\b\d+\b', lambda m: num2words(int(m.group())), text)


def has_continuous_match(option_text: str, user_input: str, min_len=2, max_len=5) -> bool:
   """
   Checks for continuous n-gram matches between option text and user input.
   Used for more robust matching of user's referenced options.
   """
   option_tokens = option_text.split()
   user_tokens = user_input.split()


   for n in range(max_len, min_len - 1, -1):
       option_ngrams = list(ngrams(option_tokens, n))
       user_ngrams = list(ngrams(user_tokens, n))


       for opt_ng in option_ngrams:
           if opt_ng in user_ngrams:
               return True
   return False


def extract_referenced_option(user_input: str, options: List[str]) -> Optional[str]:
   """
   Identifies if the user's input references one of the available options
   using exact, n-gram, and fuzzy matching.
   """
   if not user_input or not options:
       return None


   user_input_lower = user_input.lower()


   # 1. Check for exact presence (case-insensitive) of the option text
   for opt in options:
       opt_lower = opt.lower()
       if opt_lower in user_input_lower:
           return opt


   # Normalize user input and options for partial/fuzzy matching
   user_input_clean = re.sub(r'[.,;!?]', '', user_input_lower)
   user_input_norm = normalize_numbers(user_input_clean)


   for opt in options:
       opt_lower = opt.lower()
       opt_norm = normalize_numbers(opt_lower)


       # 2. Check for continuous n-gram matches
       if has_continuous_match(opt_norm, user_input_norm):
           return opt


       # 3. Use fuzzy partial ratio match as fallback
       if fuzz.partial_ratio(opt_norm, user_input_norm) > 90:
           return opt


   # Optional: Check explicit "option N" patterns (e.g., "option 1")
   explicit_option_patterns = [
       r'\b(?:option|opt|choice|selection)\s*(\d+)',
   ]
   user_input_no_punct = re.sub(r'[.,;!?]', '', user_input_lower)
   for pattern in explicit_option_patterns:
       match = re.search(pattern, user_input_no_punct)
       if match:
           try:
               option_num = int(match.group(1))
               if 1 <= option_num <= len(options):
                   return options[option_num - 1]
           except ValueError:
               pass


   return None

def update_interaction_time():
    now = time.time()
    if not st.session_state.interaction_active:
        st.session_state.interaction_start_time = now
        st.session_state.interaction_active = True
    st.session_state.last_interaction_time = now

def end_interaction_and_accumulate_time():
    if st.session_state.interaction_active and st.session_state.interaction_start_time:
        now = time.time()
        duration = now - st.session_state.interaction_start_time
        st.session_state.total_interaction_time += duration
        st.session_state.interaction_active = False
        st.session_state.interaction_start_time = None

def flatten_json(y):
  out = {}
  def flatten(x, name=''):
      if type(x) is dict:
          for a in x:
              flatten(x[a], name + a + '_')
      elif type(x) is list:
          for i,a in enumerate(x):
              flatten(a, name + str(i) + '_')
      else:
          out[name[:-1]] = x
  flatten(y)
  return out


# --- MODIFIED: save_session_data (for main summary document) ---
def save_session_data() -> bool:
    """
    Saves/updates the main summary document for the participant_id and question_id.
    This logs aggregated metrics for the current survey question.
    """
    try:
        doc_id = f"{participant_id}_{question_id}"
        
        data_to_save = {
            "participant_id": participant_id,
            "question_id": question_id,
            "chatbot_used": "yes" if (st.session_state.usage_data['chatbot_used'] or
                                     st.session_state.usage_data['followup_used']) else "no",
            "total_questions_asked_to_chatbot": st.session_state.usage_data['total_questions_asked'],
            "total_chatbot_time_seconds": round(st.session_state.get('total_interaction_time', 0), 2),
            "got_recommendation": "yes" if st.session_state.usage_data['get_recommendation'] else "no",
            "asked_followup": "yes" if st.session_state.usage_data['followup_used'] else "no",
            "last_updated_timestamp": firestore.SERVER_TIMESTAMP,
        }
        
        logs_ref.document(doc_id).set(data_to_save, merge=True)
        st.session_state.already_saved = True
        return True
    except Exception as e:
        st.error(f"Failed to save summary data to Firestore: {str(e)}")
        return False

# --- MODIFIED FUNCTION: Log individual chatbot interactions (to subcollection) ---
def log_individual_chatbot_interaction(
    current_survey_question_text: str, # Changed parameter name for clarity
    user_input_text: str,
    interaction_type: str, # "initial_recommendation" or "follow_up"
    was_answered: bool, # True if chatbot provided a relevant answer
    total_questions_asked_so_far: int
):
    """
    Logs each individual user question and its answered status to a subcollection.
    The chatbot's full response is NOT saved here.
    """
    try:
        main_doc_id = f"{participant_id}_{question_id}"
        main_doc_ref = logs_ref.document(main_doc_id)
        subcollection_ref = main_doc_ref.collection("individual_interactions")

        interaction_data = {
            "interaction_index": total_questions_asked_so_far,
            "survey_question_text": current_survey_question_text,
            "user_question": user_input_text,
            "interaction_type": interaction_type,
            "answered_relevantly": "yes" if was_answered else "no",
            "timestamp": firestore.SERVER_TIMESTAMP,
        }

        subcollection_ref.add(interaction_data)
    except Exception as e:
        st.error(f"Failed to log individual interaction to Firestore: {str(e)}")


def get_gpt_recommendation(
   question: str,
   options: List[str] = None,
   is_followup: bool = False,
   follow_up_question: Optional[str] = None,
   referenced_option: Optional[int] = None,
   user_input_for_logging: Optional[str] = None,
) -> Tuple[str, bool]:
 # Return type now includes bool for answered_relevantly


    try:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        messages = st.session_state.chat_history.copy()
        if options is None:
            options = st.session_state.get('original_options', [])

        
        # Load and flatten JSON data only if it's a follow-up
        if is_followup:
            json_data_path = "data/dashboard_data.json"
            try:
                with open(json_data_path, 'r') as file:
                    json_data = json.load(file)
                flat_data = flatten_json(json_data)
                # Decide on the best format for prompt_context_data
                # Option 1: Compact, space-separated (your original attempt)
                # prompt_context_data = " ".join([f"{k}:{v}" for k,v in flat_data.items()])
                # Option 2: More readable JSON block (recommended for clarity)
                prompt_context_data = f"\nSupermarket Data (flattened JSON):\n```json\n{json.dumps(flat_data, indent=2)}\n```"
                # Option 3: Bulleted list for readability (if not too large)
                # prompt_context_data = "\nAvailable Data (key: value):\n" + "\n".join([f"- {k}: {v}" for k,v in flat_data.items()])

            except FileNotFoundError:
                st.warning(f"Warning: Data file not found at {json_data_path}. Chatbot will operate without specific data context.")
                prompt_context_data = "" # No data available
            except json.JSONDecodeError:
                st.warning(f"Warning: Error decoding JSON from {json_data_path}. Chatbot will operate without specific data context.")
                prompt_context_data = "" # No data available


            original_rec = st.session_state.get("original_recommendation")
            context_parts = []
            if original_rec:
                context_parts.append(
                    f"Earlier Recommendation:\n"
                    f"Recommended option: {original_rec['text']}\n"
                    f"Reason: {original_rec['reasoning']}\n"
                )
            # Always include the data context if it was loaded for follow-ups
            if prompt_context_data:
                context_parts.append(prompt_context_data)


            prompt_content = f"""You are a chatbot that answers questions strictly related to supermarket scenarios, including sales, marketing, and data insights provided in a specific JSON file. Your responses must always adhere to the following rules and context.

Context:
- Original question: {question}
- Options:
{chr(10).join(options)}

{''.join(context_parts) if context_parts else ''}
{f"- Referenced option: {referenced_option}" if referenced_option else ''}

Rules:
1. First, check if the question pertains to the JSON data. If it does, use the data to provide your answer.
2. If not, check if the topic is relevant to supermarket scenarios, retail operations, marketing, or sales.
3. If the question is about the original question, the four options, or challenges to the recommended solution:
   - Respond using all the available context and justify the recommended option.
4. If a user suggests an alternative option, acknowledge their reasoning but explain clearly and concisely why the recommended option is better, based on logic, operational research, or available data.
5. If the question does not match any of the above categories, strictly respond with :
   "Please ask a question related to the survey." donot change this sentence
6. If the user greets (e.g., says "hello", "hi", etc.), respond with a short friendly greeting and prompt them to ask a question related to the survey.
7. If the user asks "why" regarding the recommendation, explain the reason provided in the context, clearly and concisely.

Limit every response to 50 words or fewer.

Respond in this format:
"<your answer here>"
"""
        else: # Initial question
            options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]) if options else ""
            prompt_content = f"""Survey Question: {question}

Available Options:
{options_text}

Please recommend the best option with reasoning (limit to 50 words).

Format:
Recommended option: <option number or text>
Reason: <short explanation>
"""

        messages.append({"role": "user", "content": prompt_content})

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=100,
            temperature=0,
            timeout=3,
            stream=True
        )

        placeholder = st.empty()
        result = ""

        for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta'):
                    content_part = getattr(choice.delta, 'content', None)
                    if content_part:
                        result += content_part
                        placeholder.markdown(f"**Chatbot:** {result}")

        if not is_followup and options: # Logic for processing initial recommendation
            if "Recommended option:" in result and "Reason:" in result:
                # Be careful with multiple "Recommended option:" or "Reason:"
                try:
                    rec_text_part = result.split("Recommended option:")[1]
                    reasoning_part = rec_text_part.split("Reason:")
                    rec_text = reasoning_part[0].strip()
                    reasoning = reasoning_part[1].strip()
                except IndexError: # Fallback if expected format isn't strictly followed by LLM
                    rec_text = result
                    reasoning = "Based on overall analysis."
            else:
                rec_text = result
                reasoning = "Based on overall analysis of options and dashboard trends."

            st.session_state.original_recommendation = {
                'text': rec_text,
                'reasoning': reasoning,
                'options': options.copy(),
                'timestamp': time.time()
            }
       
       # Determine if the chatbot answered relevantly based on its response
        answered_relevantly = "Please ask a question related to the survey" not in result

        messages.append({"role": "assistant", "content": result})
        st.session_state.chat_history = messages[-30:]

        return result, answered_relevantly

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        # Log the full traceback for debugging in a real application
        import traceback
        st.error(traceback.format_exc())
        return "An internal error occurred. Please try again later.", False


def display_conversation():
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []


# Track question changes
if question_id != st.session_state.get('last_question_id'):
    st.session_state.followup_questions = []
    st.session_state.Youtubes = []
    st.session_state.conversation = []
    st.session_state.last_question_id = question_id
    st.session_state.already_saved = False

# --- Streamlit UI and Interaction Logic ---



if st.button("Get Recommendation"):
    update_interaction_time()
    
    st.session_state.usage_data['total_questions_asked'] += 1
    user_input_for_logging = question_text # Initial prompt is the "user question" here
    
    recommendation, answered_relevantly = get_gpt_recommendation(
        question_text, 
        options=options, 
        is_followup=False,
        user_input_for_logging=user_input_for_logging
    )
    st.session_state.conversation.append(("assistant", recommendation))
    end_interaction_and_accumulate_time()

    # Log this individual interaction, WITHOUT chatbot_response_text
    log_individual_chatbot_interaction(
        current_survey_question_text=question_text,
        user_input_text=user_input_for_logging,
        interaction_type="initial_recommendation",
        was_answered=answered_relevantly,
        total_questions_asked_so_far=st.session_state.usage_data['total_questions_asked']
    )

    st.session_state.usage_data.update({
        'chatbot_used': True,
        'get_recommendation': True,
        'total_time': st.session_state.get('total_interaction_time', 0)
    })
    save_session_data()


# Display chatbot conversation
# if st.session_state.conversation:
#     st.markdown("---")
#     st.subheader("Chat History")
#     for role, message in st.session_state.conversation:
#         if role == "user":
#             st.markdown(f"**You:** {message}")
#         else:
#             st.markdown(f"**Chatbot:** {message}")
#     st.markdown("---")


user_input = st.text_input("Ask a follow-up question:")
if st.button("Send") and user_input.strip():
    update_interaction_time()
    st.session_state.conversation.append(("user", user_input))

    st.session_state.usage_data['total_questions_asked'] += 1


    referenced_option = extract_referenced_option(user_input, options) # Call it once here

    option_num = options.index(referenced_option) + 1 if referenced_option else None


    if user_input.lower().strip() in ['help', '?']:
        response = (
            "I can help with:\n"
            "- Explaining dashboard terms\n"
            "- Analyzing trends\n"
            "- Making recommendations\n"
            "Ask me anything about the supermarket data!"
        )
        answered_relevantly = True
    elif referenced_option: # Check if a referenced_option was found (it will be truthy if not None)
        response, answered_relevantly = get_gpt_recommendation(
            question_text,
            options=options,
            is_followup=True,
            referenced_option=option_num, # Pass the found option
            user_input_for_logging=user_input
        )
    else: # If no help requested and no option was referenced
        response, answered_relevantly = get_gpt_recommendation(
            question_text,
            options=options,
            is_followup=True,
            user_input_for_logging=user_input
        )


    st.session_state.conversation.append(("assistant", response))
    end_interaction_and_accumulate_time()

    # Log this individual interaction, WITHOUT chatbot_response_text
    log_individual_chatbot_interaction(
        current_survey_question_text=question_text,
        user_input_text=user_input,
        interaction_type="follow_up",
        was_answered=answered_relevantly,
        total_questions_asked_so_far=st.session_state.usage_data['total_questions_asked']
    )

    st.session_state.usage_data.update({
        'chatbot_used': True,
        'followup_used': True,
        'total_time': st.session_state.total_interaction_time
    })
    save_session_data()


# Debug information
if query_params.get("debug", "false") == "true":
    st.write("### Debug Information")
    st.write("Query Parameters:", query_params)
    st.write("Current Question ID:", question_id)
    st.write("Participant ID:", participant_id)
    st.write("Session State:", {
        k: v for k, v in st.session_state.items()
        if k not in ['conversation', '_secrets', 'chat_history']
    })
    
    st.subheader("Recent Individual Interactions (Debug - from Firestore)")
    try:
        main_doc_id = f"{participant_id}_{question_id}"
        # Ensure the main document exists before trying to access its subcollection
        if logs_ref.document(main_doc_id).get().exists:
            recent_individual_interactions = logs_ref.document(main_doc_id).collection("individual_interactions") \
                                                     .order_by("timestamp", direction=firestore.Query.DESCENDING) \
                                                     .limit(5).stream()
            found_interactions = False
            for doc in recent_individual_interactions:
                st.json(doc.to_dict())
                found_interactions = True
            if not found_interactions:
                st.info("No individual interactions found for this session yet.")
        else:
            st.info("Main session document does not exist yet. No individual interactions to display.")
    except Exception as e:
        st.error(f"Error fetching individual interactions for debug: {e}")