import streamlit as st
from openai import OpenAI
import time
import json
import pandas as pd
from urllib.parse import unquote
import gspread
import re
import uuid
from typing import List, Dict, Optional, Tuple
from nltk.util import ngrams
from num2words import num2words
from zoneinfo import ZoneInfo
from fuzzywuzzy import fuzz

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Get query parameters
query_params = st.query_params
question_id = query_params.get("qid", "Q1")
question_text = query_params.get("qtext", "What is your decision?")

options_raw = query_params.get("opts", "Option 1|Option 2|Option 3|Option 4")
options = options_raw.split("|")

# Ensure at least 4 options, padding with empty strings if needed
while len(options) < 4:
    options.append("")

option_mapping = {f"option {i+1}": options[i] for i in range(4)}
option_mapping.update({f"option{i+1}": options[i] for i in range(4)})

participant_id = query_params.get("pid", str(uuid.uuid4()))

# --- Session State Initializations ---
# These flags and variables manage the state of the application and interaction
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'gsheet_row_index' not in st.session_state:
    st.session_state.gsheet_row_index = None  # Stores the row index for the current question in Google Sheet

if 'gsheet_worksheet' not in st.session_state:
    st.session_state.gsheet_worksheet = None  # Caches the Google Sheet worksheet object

if 'gsheet_headers' not in st.session_state:
    st.session_state.gsheet_headers = None  # Caches the Google Sheet headers

if 'last_recommendation' not in st.session_state:
    st.session_state.last_recommendation = None
if 'last_question_id' not in st.session_state:
    st.session_state.last_question_id = None
if 'sheet_initialized' not in st.session_state:
    st.session_state.sheet_initialized = False  # Flag to confirm sheet is ready
if 'already_saved' not in st.session_state:
    st.session_state.already_saved = False  # Flag to track saves for the current question
if 'original_recommendation' not in st.session_state:
    st.session_state.original_recommendation = None
if 'followup_questions' not in st.session_state:
    st.session_state.followup_questions = []
if 'Youtubes' not in st.session_state:
    st.session_state.Youtubes = []

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
        'total_time': 0  # Accumulates total interaction time
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

# --- Google Sheets Initialization (Cached) ---
@st.cache_resource
def get_gsheet_worksheet_and_headers() -> Tuple[Optional[gspread.Worksheet], Optional[List[str]]]:
    """
    Connects to Google Sheets, gets/creates the worksheet, and fetches headers.
    This function is cached to run only once per app deployment/session.
    """
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        sheet = gc.open("Chatbot Usage Log")

        try:
            worksheet = sheet.worksheet("Logs_with_explanation") # Consistent worksheet name
        except gspread.exceptions.WorksheetNotFound:
            # If worksheet doesn't exist, create it
            worksheet = sheet.add_worksheet(title="Logs_with_explanation", rows=5000, cols=20) # Consistent worksheet name
            st.warning("Created new worksheet: 'Logs_with_explanation'.")

        # Define expected headers for the sheet
        expected_headers = [
            "participant_id", "question_id", "chatbot_used",
            "total_questions_asked", "total_time_seconds",
            "got_recommendation", "asked_followup", "record_timestamp",
            "user_question", "Youtubeed"
        ]

        # Get current headers from the sheet's first row
        current_headers = worksheet.row_values(1)

        # Clear and append headers only if they are missing or don't match
        if not current_headers or set(current_headers) != set(expected_headers):
            st.info("Updating Google Sheet headers...")
            worksheet.clear()  # Clears entire sheet if headers need correction
            worksheet.append_row(expected_headers)
            headers = expected_headers  # Use the expected headers after setting them
        else:
            headers = current_headers  # Use the existing headers if they match

        return worksheet, headers

    except Exception as e:
        st.error(f"Google Sheets connection failed: {str(e)}")
        return None, None

# Initialize Google Sheet and cache worksheet/headers on first load of the app
# This block runs once at the start of a session or if the cached resources are not available.
if st.session_state.gsheet_worksheet is None or st.session_state.gsheet_headers is None:
    st.session_state.gsheet_worksheet, st.session_state.gsheet_headers = get_gsheet_worksheet_and_headers()
    if st.session_state.gsheet_worksheet and st.session_state.gsheet_headers:
        st.session_state.sheet_initialized = True
    else:
        st.error("Failed to initialize Google Sheet. Data saving will not work.")
        st.session_state.sheet_initialized = False

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

# --- Google Sheets Saving Functions ---
def save_session_data() -> bool:
    """
    Prepares session data and calls the optimized save_to_gsheet function.
    Returns True if data is saved successfully, False otherwise.
    """
    try:
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
            "Youtubeed": st.session_state.usage_data.get("Youtubeed", "")
        }

        if save_to_gsheet(data):
            st.session_state.already_saved = True
            return True
        return False
    except Exception as e:
        st.error(f"Session save failed: {str(e)}")
        return False

def save_to_gsheet(data_dict: Dict) -> bool:
    """
    Saves or updates a row in the Google Sheet using cached worksheet and efficient methods.
    It prioritizes updating an existing row if found, otherwise appends a new one.
    """
    try:
        worksheet = st.session_state.gsheet_worksheet
        headers = st.session_state.gsheet_headers

        if not worksheet or not headers:
            st.error("Google Sheet is not initialized. Cannot save data.")
            return False

        # Prepare the row data based on the cached headers
        row_data = [data_dict.get(h, "") for h in headers]

        # Check if we already know the row number for the current question/participant
        row_index = st.session_state.get('gsheet_row_index')

        if row_index:
            # If we have an index, just update the row directly (most efficient)
            worksheet.update(
                f"A{row_index}:{chr(65 + len(headers) - 1)}{row_index}",
                [row_data]
            )
        else:
            # If no index, try to find the participant's row, or append new
            try:
                # Find the participant_id's cell to get the row number
                cell = worksheet.find(data_dict["participant_id"])
                row_index = cell.row  # Get the row number

                # Store the row number in session state for future updates to this question
                st.session_state.gsheet_row_index = row_index

                # Update the found row directly
                worksheet.update(
                    f"A{row_index}:{chr(65 + len(headers) - 1)}{row_index}",
                    [row_data]
                )
            except gspread.exceptions.CellNotFound:
                # If participant_id not found, append a new row
                worksheet.append_row(row_data)
                # After appending, get the new row number (least efficient, but only for new participants)
                st.session_state.gsheet_row_index = len(worksheet.get_all_values())

        return True

    except Exception as e:
        st.error(f"Failed to save to Google Sheets: {str(e)}")
        return False

# --- Question Change Detection ---
# This block resets relevant session states when the question ID changes,
# ensuring a fresh start for a new question's data logging.
if question_id != st.session_state.get('last_question_id'):
    st.session_state.followup_questions = []
    st.session_state.Youtubes = []
    st.session_state.conversation = []
    st.session_state.gsheet_row_index = None  # CRITICAL: Reset the row index for a new question
    st.session_state.last_question_id = question_id
    st.session_state.already_saved = False  # Reset saved flag for new question

# --- AI and Chatbot Logic ---
def validate_followup(user_input: str, question_id: str, options: List[str], question_text: str = "") -> str:
    """
    Validates user's follow-up questions and initiates GPT recommendation.
    Checks for referenced options and handles invalid inputs.
    """
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
            save_session_data() # Save updated usage data after a follow-up

        return result

    except Exception as e:
        st.error(f"Recommendation generation failed: {str(e)}")
        return "Sorry, I couldn't generate a recommendation."

def display_conversation():
    """Displays the latest chatbot message in the conversation."""
    if 'conversation' not in st.session_state and len(st.session_state.conversation) == 0:
        return # No conversation to display

    if len(st.session_state.conversation) > 0:
        role, message = st.session_state.conversation[-1]
        # Only display chatbot's last message, as per original logic (can be expanded)
        if role != "user":
            st.markdown(f"**Chatbot:** {message}")

# --- Streamlit UI Elements ---

# Button for getting initial recommendation
if st.button("Get Recommendation"):
    update_interaction_time()
    recommendation = get_gpt_recommendation(question_text, options=options, is_followup=False)
    st.session_state.conversation.append(("assistant", recommendation))
    end_interaction_and_accumulate_time()

    # Update usage data for initial recommendation
    st.session_state.usage_data.update({
        'chatbot_used': True,
        'total_questions_asked': st.session_state.usage_data.get('total_questions_asked', 0) + 1,
        'get_recommendation': True,
        'total_time': st.session_state.get('total_interaction_time', 0)
    })
    save_session_data() # Save data to Google Sheet

# Text input for follow-up questions
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

    # Update usage data for follow-up question
    st.session_state.usage_data.update({
        'chatbot_used': True,
        'followup_used': True,
        'total_questions_asked': st.session_state.usage_data.get('total_questions_asked', 0) + 1,
        'total_time': st.session_state.total_interaction_time
    })
    save_session_data() # Save data to Google Sheet

# Debug information (optional, controlled by 'debug' query param)
if query_params.get("debug", "false") == "true":
    st.write("### Debug Information")
    st.write("Query Parameters:", query_params)
    st.write("Current Question ID:", question_id)
    st.write("Participant ID:", participant_id)
    st.write("Session State:", {
        k: v for k, v in st.session_state.items()
        if k not in ['conversation', '_secrets', 'chat_history'] # Exclude large/sensitive objects from general debug vie
    })