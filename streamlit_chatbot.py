

import streamlit as st
from openai import OpenAI
import time
import json
import pandas as pd
import numpy as np
from urllib.parse import unquote
import gspread
from gspread_dataframe import set_with_dataframe
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
import sqlite3
from pathlib import Path


# Database setup
DB_PATH = "data_chat.db"


from fuzzywuzzy import fuzz  # For fuzzy string matchin
client = OpenAI(api_key=st.secrets["openai"]["api_key"])
query_params = st.query_params
question_id = query_params.get("qid", "Q1")
question_text = query_params.get("qtext", "What is your decision?")
#&options_raw = query_params.get("opts", "Option A|Option B|Option C")

#&options = options_raw.split("|")
options_raw = query_params.get("opts", "Option 1|Option 2|Option 3|Option 4")  # Default now has 4 options
options = options_raw.split("|")
##&



while len(options) < 4:
 options.append("") # Ensure at least 4 options, padding with empty strings if needed


option_mapping = {f"option {i+1}": options[i] for i in range(4)}
option_mapping.update({f"option{i+1}": options[i] for i in range(4)})
# for key, value in option_mapping.items():
#     st.write(f"{key}: {value}")
participant_id = query_params.get("pid", str(uuid.uuid4()))




if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'last_recommendation' not in st.session_state:
    st.session_state.last_recommendation = None
if 'last_question_id' not in st.session_state:
    st.session_state.last_question_id = None
if 'first_load' not in st.session_state:
    st.session_state.first_load = True
if 'sheet_initialized' not in st.session_state:
    st.session_state.sheet_initialized = False
if 'already_saved' not in st.session_state:  # New flag to track saves
    st.session_state.already_saved = False
if 'original_recommendation' not in st.session_state:
    st.session_state.original_recommendation = None
if 'followup_questions' not in st.session_state:
    st.session_state.followup_questions = []
if 'question_answers' not in st.session_state:
    st.session_state.question_answers = []


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
        'total_time': 0  # This will accumulate all interaction time
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

def initialize_database():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                participant_id TEXT,
                question_id TEXT,
                chatbot_used TEXT,
                total_questions_asked INTEGER,
                total_time_seconds REAL,
                got_recommendation TEXT,
                asked_followup TEXT,
                record_timestamp TEXT,
                user_question TEXT,
                question_answered TEXT,
                PRIMARY KEY (participant_id, question_id)
            )
        """)
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_id TEXT,
                question_id TEXT,
                message_type TEXT,
                content TEXT
            )
        """)
        conn.commit()
e

def save_session_data():
    try:
        # Build data dictionary
        data = {
            "participant_id": participant_id,
            "question_id": question_id,
            "chatbot_used": "yes" if (st.session_state.usage_data['chatbot_used'] or
                                      st.session_state.usage_data['followup_used']) else "no",
            "total_questions_asked": st.session_state.usage_data['total_questions_asked'],
            "total_time_seconds": round(st.session_state.get('total_interaction_time', 0), 2),
            "got_recommendation": "yes" if st.session_state.usage_data['get_recommendation'] else "no",
            "asked_followup": "yes" if st.session_state.usage_data['followup_used'] else "no",
            "record_timestamp": pd.Timestamp.now(tz=ZoneInfo("Europe/Berlin")).isoformat(),
            "user_question": st.session_state.usage_data.get("user_question", ""),
            "question_answered": st.session_state.usage_data.get("question_answered", "")
        }

        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()

            # Upsert into usage_logs
            c.execute("""
                INSERT INTO usage_logs (
                    participant_id, question_id, chatbot_used,
                    total_questions_asked, total_time_seconds,
                    got_recommendation, asked_followup,
                    record_timestamp, user_question, question_answered
                ) VALUES (
                    :participant_id, :question_id, :chatbot_used,
                    :total_questions_asked, :total_time_seconds,
                    :got_recommendation, :asked_followup,
                    :record_timestamp, :user_question, :question_answered
                )
                ON CONFLICT(participant_id, question_id)
                DO UPDATE SET
                    chatbot_used = excluded.chatbot_used,
                    total_questions_asked = excluded.total_questions_asked,
                    total_time_seconds = excluded.total_time_seconds,
                    got_recommendation = excluded.got_recommendation,
                    asked_followup = excluded.asked_followup,
                    record_timestamp = excluded.record_timestamp,
                    user_question = excluded.user_question,
                    question_answered = excluded.question_answered
            """, data)

            # Save conversation history
            if 'conversation' in st.session_state:
                for role, message in st.session_state.conversation:
                    c.execute("""
                        INSERT INTO conversation_history (
                            participant_id, question_id, message_type, content
                        ) VALUES (?, ?, ?, ?)
                    """, (data['participant_id'], data['question_id'], role, message))

            conn.commit()
        return True

    except Exception as e:
        st.error(f"Database save failed: {e}")
        return False




# --- Data Loading (for embeddings/followup questions) ---
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
    """Starts or updates the timer for user interaction."""
    now = time.time()
    if not st.session_state.interaction_active:
        st.session_state.interaction_start_time = now
        st.session_state.interaction_active = True
    st.session_state.last_interaction_time = now


def end_interaction_and_accumulate_time():
    """Ends the current interaction segment and adds its duration to total time."""
    if st.session_state.interaction_active and st.session_state.interaction_start_time:
        now = time.time()
        duration = now - st.session_state.interaction_start_time
        st.session_state.total_interaction_time += duration
        st.session_state.interaction_active = False
        st.session_state.interaction_start_time = None






# def initialize_gsheet():
# """Initialize the Google Sheet with proper unique headers"""
#     try:
#         gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
#         sheet = gc.open("Chatbot Usage Log")
#     try:
#         worksheet = sheet.worksheet("Logs_with_explanation")
#     except:
#         worksheet = sheet.add_worksheet(title="Logs_with_explanation", rows=5000, cols=20)
#      # Define and verify headers - ensure all are unique
#     expected_headers = [
#          "participant_id", "question_id", "chatbot_used",
#          "total_questions_asked", "total_time_seconds",
#          "got_recommendation", "asked_followup", "record_timestamp",
#          "user_question", "question_answered"

#     ]
#      current_headers = worksheet.row_values(1)
#      # Only update headers if they don't match exactly
#     if not current_headers or set(current_headers) != set(expected_headers):
#         worksheet.clear()
#         worksheet.append_row(expected_headers)
#      return worksheet
#  except Exception as e:
#     st.error(f"Google Sheets initialization failed: {str(e)}")
#     return None

def save_session_data():
    try:
        # Build data dictionary
        data = {
            "participant_id": participant_id,
            "question_id": question_id,
            "chatbot_used": "yes" if (st.session_state.usage_data['chatbot_used'] or
                                      st.session_state.usage_data['followup_used']) else "no",
            "total_questions_asked": st.session_state.usage_data['total_questions_asked'],
            "total_time_seconds": round(st.session_state.get('total_interaction_time', 0), 2),
            "got_recommendation": "yes" if st.session_state.usage_data['get_recommendation'] else "no",
            "asked_followup": "yes" if st.session_state.usage_data['followup_used'] else "no",
            "record_timestamp": pd.Timestamp.now(tz=ZoneInfo("Europe/Berlin")).isoformat(),
            "user_question": st.session_state.usage_data.get("user_question", ""),
            "question_answered": st.session_state.usage_data.get("question_answered", "")
        }

        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()

            # Upsert into usage_logs
            c.execute("""
                INSERT INTO usage_logs (
                    participant_id, question_id, chatbot_used,
                    total_questions_asked, total_time_seconds,
                    got_recommendation, asked_followup,
                    record_timestamp, user_question, question_answered
                ) VALUES (
                    :participant_id, :question_id, :chatbot_used,
                    :total_questions_asked, :total_time_seconds,
                    :got_recommendation, :asked_followup,
                    :record_timestamp, :user_question, :question_answered
                )
                ON CONFLICT(participant_id, question_id)
                DO UPDATE SET
                    chatbot_used = excluded.chatbot_used,
                    total_questions_asked = excluded.total_questions_asked,
                    total_time_seconds = excluded.total_time_seconds,
                    got_recommendation = excluded.got_recommendation,
                    asked_followup = excluded.asked_followup,
                    record_timestamp = excluded.record_timestamp,
                    user_question = excluded.user_question,
                    question_answered = excluded.question_answered
            """, data)

            # Save conversation history
            if 'conversation' in st.session_state:
                for role, message in st.session_state.conversation:
                    c.execute("""
                        INSERT INTO conversation_history (
                            participant_id, question_id, message_type, content
                        ) VALUES (?, ?, ?, ?)
                    """, (data['participant_id'], data['question_id'], role, message))

            conn.commit()
        return True

    except Exception as e:
        st.error(f"Database save failed: {e}")
        return False



# def save_to_gsheet(data_dict: Dict) -> bool:
#     try:
#         worksheet = initialize_gsheet()
#         if not worksheet:
#             return False

#     # Get all records with expected headers to avoid duplicates
#         records = worksheet.get_all_records(expected_headers=[
#          "participant_id", "question_id", "chatbot_used",
#          "total_questions_asked", "total_time_seconds",
#          "got_recommendation", "asked_followup", "record_timestamp", "user_question",
#         "question_answered"
#         ])
#      # Find existing record
#         row_index = None
#         for i, record in enumerate(records):
#             pid_match = str(record.get("participant_id", "")).strip() == str(data_dict.get("participant_id", "")).strip()
#             qid_match = str(record.get("question_id", "")).strip() == str(data_dict.get("question_id", "")).strip()
#             if pid_match and qid_match:
#                 row_index = i + 2  # +2 to account for header row and 1-based indexing
#                 break

#     # Prepare complete data row
#         headers = worksheet.row_values(1)
#         row_data = {k: data_dict.get(k, "") for k in headers}
#         if row_index:
#         # Update existing row
#             worksheet.update(
#                 f"A{row_index}:{chr(65 + len(headers) - 1)}{row_index}",
#                 [[row_data.get(h, "") for h in headers]]
#             )
#         else:
#         # Add new row
#             worksheet.append_row([row_data.get(h, "") for h in headers])
#             return True

#     except Exception as e:
#         st.error(f"Failed to save to Google Sheets: {str(e)}")
#         return False

#     return "Sorry, I encountered an error processing your question."






## --- AI and Chatbot Logic ---
def validate_followup(user_input: str, question_id: str, options: List[str], question_text: str = "") -> str:
    """Validates user's follow-up questions and initiates GPT recommendation."""
    try:
        user_input = user_input.strip()

        if not options:
            options = st.session_state.get('original_options', [])
        if not user_input:
            return "Please enter a valid question."

        placeholder = st.empty()

        # Extract referenced option if any
        referenced_option = extract_referenced_option(user_input, options)
        option_num = options.index(referenced_option) + 1 if referenced_option else None

        if option_num is not None:
            return get_gpt_recommendation(
                question=question_text,
                options=options,
                referenced_option=option_num,
                is_followup=True,
                follow_up_question=user_input,
            )

        return get_gpt_recommendation(
            question=question_text,
            is_followup=True,
            follow_up_question=user_input
        )

    except Exception as e:
        st.error(f"Error in followup validation: {str(e)}")
        return "Sorry, I encountered an error processing your question."

def get_gpt_recommendation(
    question: str,
    options: List[str] = None,
    is_followup: bool = False,
    follow_up_question: Optional[str] = None,
    referenced_option: Optional[str] = None,
    non_dashboard: bool = False,
) -> str:
    """
    Generates a recommendation or response from the GPT model based on the context.
    Supports initial recommendations and follow-up questions.
    """
    try:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        messages = st.session_state.chat_history.copy()
        if options is None:
            options = st.session_state.get('original_options', [])

        if is_followup:
            original_rec = st.session_state.get("original_recommendation")
            context_parts = []
            if original_rec:
                context_parts.append(
                    f"Earlier Recommendation:\n"
                    f"Recommended option: {original_rec['text']}\n"
                    f"Reason: {original_rec['reasoning']}\n"
                )

            prompt = f"""You are a chatbot that answers questions strictly related to supermarket scenarios, including sales, marketing, and data insights provided in a specific JSON file. Your responses must always adhere to the following rules and context.

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
5. If the question does not match any of the above categories, respond with:
   "Please ask a question related to the survey."
6. If the user greets (e.g., says "hello", "hi", etc.), respond with a short friendly greeting and prompt them to ask a question related to the survey.
7. If the user asks "why" regarding the recommendation, explain the reason provided in the context, clearly and concisely.

Limit every response to 50 words or fewer.

Respond in this format:
Chatbot answer: "<your answer here>"
"""
            st.session_state.followup_questions.append(follow_up_question)

            # Determine if the response was a valid answer or a rejection
            if "Please ask a question related to the survey" in result:
                answered = "No"
            else:
                answered = "Yes"
            index = len(st.session_state.followup_questions)
            st.session_state.Youtubes.append(f"{index}. {answered}")

            # Store formatted questions and answers in usage_data
            st.session_state.usage_data.update({
                'user_question': "\n".join([f"{i+1}. {q}" for i, q in enumerate(st.session_state.followup_questions)]),
                'Youtubeed': "\n".join(st.session_state.Youtubes),
            })

        else:  # Initial recommendation logic
            options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]) if options else ""
            prompt = f"""Survey Question: {question}

Available Options:
{options_text}

Please recommend the best option with reasoning (limit to 50 words).

Format:
Recommended option: <option number or text>

Reason: <short explanation>
"""

        # Add user message to chat history
        messages.append({"role": "user", "content": prompt})

        # Start streaming response from OpenAI
        stream = client.chat.completions.create(
            model="gpt-4.1-mini", # Assuming this model ID is correct/available
            messages=messages,
            max_tokens=100,
            temperature=0,
            timeout=3,
            stream=True
        )

        # Live output with placeholder
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

        # Store original recommendation if this was an initial recommendation
        if not is_followup and options:
            if "Recommended option:" in result and "Reason:" in result:
                rec_text = result.split("Recommended option:")[1].split("Reason:")[0].strip()
                reasoning = result.split("Reason:")[1].strip()
            else:
                rec_text = result
                reasoning = "Based on overall analysis of options and dashboard trends."

            st.session_state.original_recommendation = {
                'text': rec_text,
                'reasoning': reasoning,
                'options': options.copy(),
                'timestamp': time.time()
            }

        # Update chat history with the assistant's response
        messages.append({"role": "assistant", "content": result})
        st.session_state.chat_history = messages[-30:] # Keep last 30 messages for context

        if is_followup:
            st.session_state.followup_questions.append(follow_up_question)

            # Determine if the response was a valid answer or a rejection
            if "Please ask a question related to the survey" in result:
                answered = "No"
            else:
                answered = "Yes"
            index = len(st.session_state.followup_questions)
            st.session_state.Youtubes.append(f"{index}. {answered}")

            # Store formatted questions and answers in usage_data
            st.session_state.usage_data.update({
                'user_question': "\n".join([f"{i+1}. {q}" for i, q in enumerate(st.session_state.followup_questions)]),
                'Youtubeed': "\n".join(st.session_state.Youtubes),
            })
            # Save updated usage data after a follow-up

        return result

    except Exception as e:
        st.error(f"Recommendation generation failed: {str(e)}")
        return "Sorry, I couldn't generate a recommendation."

def display_conversation():
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if len(st.session_state.conversation) > 0:
        role, message = st.session_state.conversation[-1]
        if role != "user":
            st.markdown(f"**Chatbot:** {message}")

def save_progress():
    """Save or update progress in Google Sheets"""
    if st.session_state.already_saved:
        return True

    if not st.session_state.usage_data.get('start_time'):
        return False

    try:
        # Ensure timing variables are defined
        start_time = st.session_state.get("interaction_start_time")
        end_time = st.session_state.get("interaction_end_time")
        total_time = round(end_time - start_time, 2) if start_time and end_time else 0

        usage_data = {
            "participant_id": participant_id,
            "question_id": question_id,
            "chatbot_used": "yes" if (st.session_state.get("get_recommendation_used") or st.session_state.get("followup_used")) else "no",
            "total_total_questions_asked_to_chatbot": st.session_state.usage_data.get('followups_asked', 0),
            "total_chatbot_time_seconds": total_time,
            "get_recommendation": "yes" if st.session_state.get("get_recommendation_used") else "no",
            "further_question_asked": "yes" if st.session_state.get("followup_used") else "no",
            "timestamp": pd.Timestamp.now().isoformat()
        }

        if save_to_gsheet(usage_data):
            st.session_state.usage_data['start_time'] = time.time()
            st.session_state.already_saved = True
            return True

        return False  # Only one save attempt

    except Exception as e:
        st.error(f"Progress save failed: {str(e)}")
        return False

# Initialize Google Sheet on first load
# if st.session_state.first_load and not st.session_state.sheet_initialized:
#     initialize_gsheet()
#     st.session_state.sheet_initialized = True

# Track question changes
if question_id != st.session_state.get('last_question_id'):
    st.session_state.followup_questions = []
    st.session_state.question_answers = []
    st.session_state.conversation = []
    st.session_state.last_question_id = question_id
    st.session_state.already_saved = False  # Reset saved flag for new question

if st.button("Get Recommendation"):
    update_interaction_time()
    recommendation = get_gpt_recommendation(question_text, options=options, is_followup=False)
    st.session_state.conversation.append(("assistant", recommendation))
    end_interaction_and_accumulate_time()
    # Update usage data
    st.session_state.usage_data.update({
        'chatbot_used': True,
        'total_questions_asked': st.session_state.usage_data.get('total_questions_asked', 0) + 1,
        'get_recommendation': True,
        'total_time': st.session_state.get('total_interaction_time', 0)
    })
    save_session_data()

user_input = st.text_input("Ask a follow-up question:")
if st.button("Send") and user_input.strip():
    update_interaction_time()
    st.session_state.conversation.append(("user", user_input))
    if user_input.lower().strip() in ['help', '?']:
        response = "I can help with:\n- Explaining dashboard terms\n- Analyzing trends\n- Making recommendations\nAsk me anything about the supermarket data!"
    else:
        response = validate_followup(user_input, question_id, options=options)
    st.session_state.conversation.append(("assistant", response))
    end_interaction_and_accumulate_time()

    # Update usage data
    st.session_state.usage_data.update({
        'chatbot_used': True,
        'followup_used': True,
        'total_questions_asked': st.session_state.usage_data.get('total_questions_asked', 0) + 1,
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
        if k not in ['conversation', '_secrets']
    })