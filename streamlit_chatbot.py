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
if 'usage_data' not in st.session_state:
    st.session_state.usage_data = {
        'start_time': None,
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
def connect_to_gsheet():
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        return gc.open("Chatbot Usage Log").sheet1
    except Exception as e:
        st.error(f"Google Sheets connection failed: {str(e)}")
        return None

def save_to_gsheet(data_dict: Dict):
    try:
        worksheet = connect_to_gsheet()
        if worksheet:
            records = worksheet.get_all_records()
            df = pd.DataFrame(records)
            new_df = pd.DataFrame([data_dict])
            updated_df = pd.concat([df, new_df], ignore_index=True)
            worksheet.clear()
            set_with_dataframe(worksheet, updated_df)
    except Exception as e:
        st.error(f"Failed to save to Google Sheets: {str(e)}")

# --- Core Chatbot Functions ---
def validate_followup(user_question: str, question_id: str, options: List[str]) -> str:
    try:
        user_embedding = get_embedding(user_question)
        referenced_option = extract_referenced_option(user_question, options)
        
        # Build conversation history
        history = []
        if st.session_state.last_recommendation:
            history.append((question_text, st.session_state.last_recommendation))
            if referenced_option:
                history.append((f"User referenced option: {referenced_option}", "Noted."))
        
        # Check general followups
        for general in data["general_followups"]:
            if general.get("embedding"):
                score = cosine_similarity(user_embedding, general["embedding"])
                if score >= 0.70:
                    return get_gpt_recommendation(
                        user_question, 
                        options=options,
                        history=history
                    )
        
        # Check question-specific followups
        for question in data["questions"]:
            if question["question_id"] == question_id and question.get("embedding"):
                score = cosine_similarity(user_embedding, question["embedding"])
                if score >= 0.70:
                    return get_gpt_recommendation(
                        user_question,
                        options=options,
                        history=history
                    )
        
        return "Please ask a question related to the current survey topic."
    except Exception as e:
        st.error(f"Follow-up validation failed: {str(e)}")
        return "Sorry, I encountered an error processing your question."

def get_gpt_recommendation(question: str, options: List[str] = None, history: List[Tuple[str, str]] = None) -> str:
    try:
        messages = []
        
        # Add conversation history if exists
        if history:
            for q, a in history:
                messages.extend([
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ])
        
        # Build the prompt
        if options:
            options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
            prompt = f"""Survey Question: {question}
Available Options:
{options_text}

Please recommend the best option with reasoning in this format:
"Recommended option: <text>"
"Reason: <detailed explanation>"
"""
        else:
            prompt = f"""Survey Question: {question}
Please provide your recommendation with reasoning in this format:
"Recommendation: <text>"
"Reason: <detailed explanation>"
"""
        
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )
        
        result = response.choices[0].message.content
        st.session_state.last_recommendation = result  # Store the last recommendation
        return result
    except Exception as e:
        st.error(f"Recommendation generation failed: {str(e)}")
        return "Sorry, I couldn't generate a recommendation due to an error."

# #--- UI Components ---
# def display_conversation():
#     #st.write("### Conversation History")
#     for role, message in st.session_state.conversation:
#         if role == "user":
#             st.markdown(f"**You:** {message}")
#         else:
#             st.markdown(f"**Chatbot:** {message}")


# def display_conversation():
#     # Clear conversation at the start of each function call
#   # This will clear the chat history

#     # Display each message from the conversation
#     for role, message in st.session_state.conversation:
#         if role == "assistant":
#             st.markdown(message)

import streamlit as st

def display_conversation():
    # Check if the conversation list exists and has messages
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Only display the latest assistant message
    if len(st.session_state.conversation) > 0:
        role, message = st.session_state.conversation[-1]  # Get the latest message
        if role != "user":  # Display only if the message is from the chatbot (not user)
            st.markdown(f"**Chatbot:** {message}")



# --- Main App Logic ---
# Get query parameters
query_params = st.query_params
question_id = query_params.get("qid", "Q1")
question_text = query_params.get("qtext", "What is your decision?")
options_raw = query_params.get("opts", "Option A|Option B|Option C")
options = options_raw.split("|")

# Display survey question
# st.write(f"### Survey Question ({question_id})")
# st.write(question_text)
# if options:
#     st.write("**Options:**")
#     for i, opt in enumerate(options):
#         st.write(f"{i+1}. {opt}")

# Recommendation button
if st.button("Get Recommendation"):
    if st.session_state.usage_data['start_time'] is None:
        st.session_state.usage_data['start_time'] = time.time()
    
    recommendation = get_gpt_recommendation(question_text, options)
    st.session_state.conversation.append(("assistant", recommendation))
    st.session_state.usage_data['questions_asked'] += 1

# Follow-up input
user_input = st.text_input("Ask a follow-up question:")
if user_input:
    if st.session_state.usage_data['start_time'] is None:
        st.session_state.usage_data['start_time'] = time.time()
    
    st.session_state.conversation.append(("user", user_input))
    response = validate_followup(user_input, question_id, options)
    st.session_state.conversation.append(("assistant", response))
    st.session_state.usage_data['followups_asked'] += 1

# Display conversation
display_conversation()

# Save data when done
auto_finish = query_params.get("auto_finish") == "true"
participant_id = query_params.get("pid", str(uuid.uuid4()))  # Will default only if pid is missing

if auto_finish:
    if st.session_state.usage_data['start_time']:
        duration = time.time() - st.session_state.usage_data['start_time']
        
        usage_data = {
            "participant_id": participant_id,  # ✅ Use the real Prolific ID
            "question_id": question_id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "duration_seconds": round(duration, 2),
            "questions_asked": st.session_state.usage_data['questions_asked'],
            "followups_asked": st.session_state.usage_data['followups_asked'],
            "last_recommendation": st.session_state.last_recommendation[:500] if st.session_state.last_recommendation else None
        }

        
        save_to_gsheet(usage_data)
        st.success("Survey completed! Data saved.")
        
        # Reset session state
        st.session_state.conversation = []
        st.session_state.last_recommendation = None
        st.session_state.usage_data = {
            'start_time': None,
            'questions_asked': 0,
            'followups_asked': 0
        }