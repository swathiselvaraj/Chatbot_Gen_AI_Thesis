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

# --- Session State Initialization ---
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

if 'usage_data' not in st.session_state:
    st.session_state.usage_data = {
        'start_time': time.time(),
        'questions_asked': 0,
        'followups_asked': 0
    }

# --- Data Loading ---
@st.cache_resource
def load_embedding_data():
    try:
        with open("data/followup_embeddings_list.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load embeddings: {str(e)}")
        return {"general_followups": [], "questions": []}

data = load_embedding_data()

# --- Utility Functions ---
def get_embedding(text: str) -> List[float]:
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding generation failed: {str(e)}")
        return []

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    try:
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) 
    except Exception as e:
        st.error(f"Similarity calculation failed: {str(e)}")
        return 0.0

def extract_referenced_option(user_input: str, options: List[str]) -> Optional[str]:
    try:
        match = re.search(r"option\s*(\d+)", user_input.lower())
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(options):
                return options[idx]
        return None
    except:
        return None

# --- Google Sheets Integration ---
def initialize_gsheet():
    """Initialize the Google Sheet with proper headers"""
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        sheet = gc.open("Chatbot Usage Log")
        
        try:
            worksheet = sheet.worksheet("Logs")
        except:
            worksheet = sheet.add_worksheet(title="Logs", rows=1000, cols=20)
        
        # Define and verify headers
        expected_headers = [
            "participant_id", "question_id", "timestamp",
            "duration_seconds", "questions_asked",
            "followups_asked", "last_recommendation",
            "conversation_snapshot"
        ]
        
        current_headers = worksheet.row_values(1)
        
        if not current_headers or current_headers != expected_headers:
            worksheet.clear()
            worksheet.append_row(expected_headers)
        
        return worksheet
        
    except Exception as e:
        st.error(f"Google Sheets initialization failed: {str(e)}")
        return None

def save_to_gsheet(data_dict: Dict) -> bool:
    """Save data to Google Sheets with duplicate prevention"""
    try:
        worksheet = initialize_gsheet()
        if not worksheet:
            return False
            
        # Get existing records
        records = worksheet.get_all_records()
        
        # Check if this question has already been saved for this participant
        for record in records:
            if (record["participant_id"] == data_dict["participant_id"] and
                record["question_id"] == data_dict["question_id"]):
                return True  # Consider it a success but don't save again
                
        # If not found, append new row
        worksheet.append_row(list(data_dict.values()))
        return True
        
    except Exception as e:
        st.error(f"Failed to save to Google Sheets: {str(e)}")
        return False

# --- Core Chatbot Functions ---
def validate_followup(user_question: str, question_id: str, options: List[str]) -> str:
    try:
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
    except Exception as e:
        st.error(f"Follow-up validation failed: {str(e)}")
        return "Sorry, I encountered an error processing your question."

def get_gpt_recommendation(question: str, options: List[str] = None, history: List[Tuple[str, str]] = None) -> str:
    try:
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
            prompt = f"""The user has asked a follow-up question about a survey recommendation.
You must use prior context and reasoning to answer concisely in under 50 words.

Respond in this format:
"Answer: <your answer to the follow-up question>"
"""
        else:
            options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]) if options else ""
            prompt = f"""Survey Question: {question}
Available Options:
{options_text}

Please recommend the best option with reasoning. Limit your response to 50 words.

Respond in this format:
"Recommended option: <text>"
"Reason: <short explanation>"
"""

        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )

        result = response.choices[0].message.content
        st.session_state.last_recommendation = result
        return result
    except Exception as e:
        st.error(f"Recommendation generation failed: {str(e)}")
        return "Sorry, I couldn't generate a recommendation due to an error."

def display_conversation():
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if len(st.session_state.conversation) > 0:
        role, message = st.session_state.conversation[-1]
        if role != "user":
            st.markdown(f"**Chatbot:** {message}")

def save_progress():
    """Save progress to Google Sheets only if not already saved"""
    if st.session_state.already_saved:
        return True
        
    if not st.session_state.usage_data['start_time']:
        return False
        
    try:
        duration = time.time() - st.session_state.usage_data['start_time']
        
        usage_data = {
            "participant_id": participant_id,
            "question_id": question_id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "duration_seconds": round(duration, 2),
            "questions_asked": st.session_state.usage_data['questions_asked'],
            "followups_asked": st.session_state.usage_data['followups_asked'],
            "last_recommendation": (
                str(st.session_state.last_recommendation)[:500] 
                if st.session_state.last_recommendation is not None 
                else ""
            ),
            "conversation_snapshot": json.dumps(st.session_state.conversation[-3:], ensure_ascii=False)
        }
        
        if save_to_gsheet(usage_data):
            st.session_state.usage_data['start_time'] = time.time()
            st.session_state.already_saved = True
            return True
        return False
        
    except Exception as e:
        st.error(f"Progress save failed: {str(e)}")
        return False

# --- Main App Logic ---
# Get query parameters
query_params = st.query_params
question_id = query_params.get("qid", "Q1")
question_text = query_params.get("qtext", "What is your decision?")
options_raw = query_params.get("opts", "Option A|Option B|Option C")
options = options_raw.split("|")
participant_id = query_params.get("pid", str(uuid.uuid4()))

# Initialize Google Sheet on first load
if st.session_state.first_load and not st.session_state.sheet_initialized:
    initialize_gsheet()
    st.session_state.sheet_initialized = True

# Track question changes
if question_id != st.session_state.get('last_question_id'):
    st.session_state.conversation = []
    st.session_state.last_question_id = question_id
    st.session_state.already_saved = False  # Reset saved flag for new question
    if not st.session_state.first_load:
        save_progress()

# Recommendation button
if st.button("Get Recommendation"):
    recommendation = get_gpt_recommendation(question_text, options)
    st.session_state.conversation.append(("assistant", recommendation))
    st.session_state.usage_data['questions_asked'] += 1
    st.session_state.first_load = False
    save_progress()

# Follow-up input
user_input = st.text_input("Ask a follow-up question:")
if user_input:
    st.session_state.conversation.append(("user", user_input))
    response = validate_followup(user_input, question_id, options)
    st.session_state.conversation.append(("assistant", response))
    st.session_state.usage_data['followups_asked'] += 1
    st.session_state.first_load = False
    save_progress()

# Display conversation
display_conversation()

# Final save when leaving the page
if not st.session_state.first_load and not st.session_state.already_saved:
    save_progress()

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