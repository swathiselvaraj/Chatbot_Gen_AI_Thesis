import streamlit as st
from openai import OpenAI
import time
import json
import pandas as pd
import numpy as np
import gspread
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
if 'participant_id' not in st.session_state:
    st.session_state.participant_id = str(uuid.uuid4())
if 'chatbot_question_count' not in st.session_state:
    st.session_state.chatbot_question_count = 0
if 'chatbot_use_total_time' not in st.session_state:
    st.session_state.chatbot_use_total_time = 0.0
if 'chatbot_start_time' not in st.session_state:
    st.session_state.chatbot_start_time = time.time()
if 'question_start_time' not in st.session_state:
    st.session_state.question_start_time = time.time()

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
            "participant_id", 
            "question_id",
            "chatbot_used",
            "questions_asked_to_chatbot",
            "total_chatbot_time_seconds",
            "timestamp"
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
                return True  # Already saved, skip
                
        # If not found, append new row
        worksheet.append_row([
            data_dict["participant_id"],
            data_dict["question_id"],
            data_dict["chatbot_used"],
            data_dict["questions_asked_to_chatbot"],
            data_dict["total_chatbot_time_seconds"],
            data_dict["timestamp"]
        ])
        return True
        
    except Exception as e:
        st.error(f"Failed to save to Google Sheets: {str(e)}")
        return False

def save_progress():
    """Save progress to Google Sheets when moving to next question"""
    if st.session_state.first_load:
        return False
        
    try:
        # Calculate time spent on this question
        question_time = time.time() - st.session_state.question_start_time
        st.session_state.chatbot_use_total_time += question_time
        
        data_dict = {
            "participant_id": st.session_state.participant_id,
            "question_id": question_id,
            "chatbot_used": st.session_state.chatbot_question_count > 0,
            "questions_asked_to_chatbot": st.session_state.chatbot_question_count,
            "total_chatbot_time_seconds": round(st.session_state.chatbot_use_total_time, 2),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return save_to_gsheet(data_dict)
        
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

# Override participant ID if provided in URL
if query_params.get("pid"):
    st.session_state.participant_id = query_params.get("pid")

# Initialize Google Sheet on first load
if st.session_state.first_load and not st.session_state.sheet_initialized:
    initialize_gsheet()
    st.session_state.sheet_initialized = True

# Track question changes
if question_id != st.session_state.get('last_question_id'):
    # Save previous question data before switching
    if not st.session_state.first_load:
        save_progress()
    
    # Reset for new question
    st.session_state.conversation = []
    st.session_state.last_question_id = question_id
    st.session_state.chatbot_question_count = 0
    st.session_state.question_start_time = time.time()
    st.session_state.first_load = False

# Recommendation button
if st.button("Get Recommendation"):
    recommendation = get_gpt_recommendation(question_text, options)
    st.session_state.conversation.append(("assistant", recommendation))
    st.session_state.chatbot_question_count += 1
    st.session_state.last_recommendation = recommendation

# Follow-up input
user_input = st.text_input("Ask a follow-up question:")
if user_input:
    st.session_state.conversation.append(("user", user_input))
    response = validate_followup(user_input, question_id, options)
    st.session_state.conversation.append(("assistant", response))
    st.session_state.chatbot_question_count += 1

# Final save when leaving the page (handled by question change tracking)

# Debug information
if query_params.get("debug", "false") == "true":
    st.write("### Debug Information")
    st.write("Participant ID:", st.session_state.participant_id)
    st.write("Question ID:", question_id)
    st.write("Chatbot Questions Count:", st.session_state.chatbot_question_count)
    st.write("Total Chatbot Time:", round(st.session_state.chatbot_use_total_time, 2))