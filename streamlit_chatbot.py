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
from typing import List, Dict, Optional, Tuple

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai"]["api_key"])
st.set_page_config(page_title="Survey Chatbot", layout="wide")

# Session State Initialization
for key, default in {
    'conversation': [],
    'last_recommendation': None,
    'last_question_id': None,
    'first_load': True,
    'sheet_initialized': False,
    'already_saved': False,
    'interaction_start_time': None,
    'interaction_end_time': None,
    'get_recommendation_used': False,
    'followup_used': False,
    'usage_data': {
        'start_time': time.time(),
        'questions_asked': 0,
        'followups_asked': 0,
        'last_saved_followups': 0
    }
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Load Embedding Data
@st.cache_resource
def load_embedding_data():
    try:
        with open("data/followup_embeddings_list.json", "r") as f:
            return json.load(f)
    except:
        return {"general_followups": [], "questions": []}

data = load_embedding_data()

# Utility Functions
def get_embedding(text: str) -> List[float]:
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except:
        return []

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    try:
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) 
    except:
        return 0.0

def extract_referenced_option(user_input: str, options: List[str]) -> Optional[str]:
    match = re.search(r"option\s*(\d+)", user_input.lower())
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(options):
            return options[idx]
    return None

# Google Sheets Integration
def initialize_gsheet():
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        sheet = gc.open("Chatbot Usage Log")
        try:
            worksheet = sheet.worksheet("Logs")
        except:
            worksheet = sheet.add_worksheet(title="Logs", rows=1000, cols=20)
        expected_headers = [
            "participant_id", "question_id", "chatbot_used",
            "questions_asked_to_chatbot", "total_chatbot_time_seconds",
            "get_recommendation", "further_question_asked"
        ]
        current_headers = worksheet.row_values(1)
        if not current_headers or current_headers != expected_headers:
            worksheet.clear()
            worksheet.append_row(expected_headers)
        return worksheet
    except:
        return None

def save_to_gsheet(data_dict: Dict) -> bool:
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        sheet = gc.open("Chatbot Usage Log")
        worksheet = sheet.worksheet("Logs")
        records = worksheet.get_all_records()
        row_num = None
        for i, record in enumerate(records, start=2):
            if (str(record.get("participant_id", "")) == str(data_dict["participant_id"]) and 
                str(record.get("question_id", "")) == str(data_dict["question_id"])):
                row_num = i
                break
        headers = worksheet.row_values(1)
        new_values = [str(data_dict.get(h, "")).strip() or None for h in headers]
        if row_num:
            cells = [gspread.Cell(row=row_num, col=col_num, value=value) for col_num, value in enumerate(new_values, start=1)]
            worksheet.update_cells(cells)
        else:
            worksheet.append_row(new_values)
        return True
    except:
        return False

# Chatbot Core
def validate_followup(user_question: str, question_id: str, options: List[str]) -> str:
    user_embedding = get_embedding(user_question)
    referenced_option = extract_referenced_option(user_question, options)
    history = []
    if st.session_state.last_recommendation:
        history.append((f"Original survey question: {question_text}", st.session_state.last_recommendation))
    history.append((f"Follow-up: {user_question}", ""))
    if referenced_option:
        history.append((f"The user mentioned: {referenced_option}", "Acknowledged."))
    for source in data["general_followups"] + data["questions"]:
        if source.get("embedding") and (source.get("question_id") == question_id or "question_id" not in source):
            score = cosine_similarity(user_embedding, source["embedding"])
            if score >= 0.70:
                return get_gpt_recommendation(user_question, options=options, history=history)
    return "Please ask a question related to the current survey topic."

def get_gpt_recommendation(question: str, options: List[str] = None, history: List[Tuple[str, str]] = None) -> str:
    messages = []
    followup_mode = False
    if history:
        for q, a in history:
            if "Follow-up:" in q:
                followup_mode = True
            if q.strip():
                messages.append({"role": "user", "content": q})
            if a.strip():
                messages.append({"role": "assistant", "content": a})
    if followup_mode:
        prompt = (
            "The user has asked a follow-up question about a survey recommendation.\n"
            "You must use prior context and reasoning to answer concisely in under 50 words.\n\n"
            'Respond in this format:\n"Answer: <your answer to the follow-up question>"'
        )
    else:
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]) if options else ""
        prompt = (
            f"Survey Question: {question}\nAvailable Options:\n{options_text}\n\n"
            "Please recommend the best option with reasoning. Limit your response to 50 words.\n\n"
            'Respond in this format:\n"Recommended option: <text>"\n"Reason: <short explanation>"'
        )
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )
    result = response.choices[0].message.content
    st.session_state.last_recommendation = result
    return result

def display_conversation():
    if st.session_state.conversation:
        role, message = st.session_state.conversation[-1]
        if role != "user":
            st.markdown(f"**Chatbot:** {message}")

def save_progress():
    new_followups = st.session_state.usage_data['followups_asked'] - st.session_state.usage_data['last_saved_followups']
    if new_followups <= 0 and not st.session_state.get_recommendation_used:
        return True
    total_time = round(time.time() - st.session_state.usage_data['start_time'], 2)
    usage_data = {
        "participant_id": participant_id,
        "question_id": question_id,
        "chatbot_used": "yes",
        "questions_asked_to_chatbot": st.session_state.usage_data['followups_asked'],
        "total_chatbot_time_seconds": total_time,
        "get_recommendation": "yes" if st.session_state.get_recommendation_used else "no",
        "further_question_asked": "yes" if st.session_state.usage_data['followups_asked'] > 0 else "no"
    }
    if save_to_gsheet(usage_data):
        st.session_state.usage_data['last_saved_followups'] = st.session_state.usage_data['followups_asked']
        st.session_state.get_recommendation_used = False
        return True
    return False

# Get Query Parameters
query_params = st.query_params
question_id = query_params.get("qid", "Q1")
question_text = query_params.get("qtext", "What is your decision?")
options_raw = query_params.get("opts", "Option A|Option B|Option C")
participant_id = query_params.get("pid", str(uuid.uuid4()))
options = [opt.strip() for opt in unquote(options_raw).split(";") if opt.strip()]

# Initialize Sheet on First Load
if st.session_state.first_load and not st.session_state.sheet_initialized:
    initialize_gsheet()
    st.session_state.sheet_initialized = True

# Reset conversation if question changed
if question_id != st.session_state.get('last_question_id'):
    st.session_state.conversation = []
    st.session_state.last_question_id = question_id
    st.session_state.already_saved = False

# Track Interaction Time
st.session_state.interaction_start_time = time.time()
st.session_state.interaction_end_time = time.time()

# Recommendation Button
if st.button("Get Recommendation"):
    st.session_state.usage_data['start_time'] = time.time()
    recommendation = get_gpt_recommendation(question_text, options)
    st.session_state.conversation.append(("assistant", recommendation))
    st.session_state.get_recommendation_used = True
    st.session_state.usage_data['followups_asked'] = 0
    st.session_state.usage_data['last_saved_followups'] = 0
    st.session_state.first_load = False
    save_progress()
    st.rerun()

# Follow-up Input
user_input = st.text_input("Ask a follow-up question:")
if user_input and (not st.session_state.conversation or st.session_state.conversation[-1][1] != user_input):
    # Add user message
    st.session_state.conversation.append(("user", user_input))

    # Call validation logic
    response = validate_followup(user_input, question_id, options)
    st.session_state.conversation.append(("assistant", response))

    # Track follow-up question count manually
    st.session_state.usage_data['followups_asked'] += 1

    if save_progress():
        st.success("Response saved!")
    time.sleep(0.3)
    st.rerun()


# Display conversation
display_conversation()



# Display conversation
display_conversation()

# Final save when leaving the page
# if not st.session_state.first_load and not st.session_state.already_saved:
#     save_progress()

# Debug information
# if query_params.get("debug", "false") == "true":
#     st.write("### Debug Information")
#     st.write("Query Parameters:", query_params)
#     st.write("Current Question ID:", question_id)
#     st.write("Participant ID:", participant_id)
#     st.write("Session State:", {
#         k: v for k, v in st.session_state.items() 
#         if k not in ['conversation', '_secrets']
#     })

