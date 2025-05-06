import streamlit as st
from openai import OpenAI
import time
import json
import pandas as pd
from typing import List
import uuid
import numpy as np
from urllib.parse import unquote
from typing import Tuple, List
import gspread
from gspread_dataframe import set_with_dataframe

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

st.set_page_config(page_title="Survey Chatbot")

@st.cache_resource
def load_embedding_data():
    file_path = "data/followup_embeddings_list.json"
    with open(file_path, "r") as f:
        return json.load(f)

data = load_embedding_data()

# Get embedding using OpenAI SDK v1
def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Similarity Calculation
def cosine_similarity(vec1, vec2):
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def connect_to_gsheet():
    gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
    sh = gc.open("Chatbot Usage Log")
    return sh.sheet1

def save_to_gsheet(data_dict):
    worksheet = connect_to_gsheet()
    records = worksheet.get_all_records()
    df = pd.DataFrame(records)
    new_df = pd.DataFrame([data_dict])
    updated_df = pd.concat([df, new_df], ignore_index=True)
    worksheet.clear()
    set_with_dataframe(worksheet, updated_df)

# Follow-Up questions Validation

import re

# Utility: Extract a referenced option by number (e.g., "option 1")
def extract_referenced_option(user_input: str, options: List[str]) -> str:
    match = re.search(r"option\s*(\d+)", user_input.lower())
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(options):
            return options[idx]
    return None

# Main follow-up validation function
def validate_followup(user_question: str, question_id: str) -> str:
    user_embedding = get_embedding(user_question)

    # Try to detect if user is referring to a specific option
    referenced_option = extract_referenced_option(user_question, options)

    # If general follow-up matches
    for general in data["general_followups"]:
        general_embedding = general.get("embedding")
        if general_embedding:
            score = cosine_similarity(user_embedding, general_embedding)
            if score >= 0.70:
                original_question = question_text
                original_recommendation = get_gpt_recommendation(original_question, options)

                # Add referenced option context if found
                history = [(original_question, original_recommendation)]
                if referenced_option:
                    clarification = f'The user is asking why this option was not recommended: "{referenced_option}"'
                    history.append((clarification, "Okay."))

                return get_gpt_recommendation(user_question, history=history)

    # If question-specific follow-up matches
    for question in data["questions"]:
        if question["question_id"] == question_id:
            followup_embedding = question.get("embedding")
            if followup_embedding:
                score = cosine_similarity(user_embedding, followup_embedding)
                if score >= 0.70:
                    original_question = question_text
                    original_recommendation = get_gpt_recommendation(original_question, options)

                    history = [(original_question, original_recommendation)]
                    if referenced_option:
                        clarification = f'The user is asking why this option was not recommended: "{referenced_option}"'
                        history.append((clarification, "Okay."))

                    return get_gpt_recommendation(user_question, history=history)

    return "Sorry, can you ask a question related to the survey?"




def get_gpt_recommendation(question, options=None, history=None):
    messages = []

    # If there is previous conversation, include it
    if history:
        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    if options:
        options_text = f"The available options are:\n{chr(10).join([f'{i+1}. {opt}' for i, opt in enumerate(options)])}"
        messages.append({
            "role": "user",
            "content": f"""
You are helping a user complete a survey.
The question is: "{question}"
{options_text}

Based on general best practices or knowledge, recommend the best option.
Reply in this format:
"Recommended option: <text>"
"Reason: <brief explanation>"
""".strip()
        })
    else:
        messages.append({
            "role": "user",
            "content": f"""
You are helping a user complete a survey.
The user may ask follow-up questions after your first recommendation.

The current question is: "{question}"

Provide an answer.
Reply in this format:
"Explanation: <text>"
""".strip()
        })

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content



# Add this at the top of your file



# Get query parameters
# Modern Streamlit: no need for experimental version or unquote
query_params = st.query_params

# Safely access values with proper defaults
#question_id = query_params.get("qid", ["Q1"])[0] if isinstance(query_params.get("qid"), list) else query_params.get("qid", "Q1")
#question_text = query_params.get("qtext", ["What is your decision?"])[0] if isinstance(query_params.get("qtext"), list) else query_params.get("qtext", "What is your decision?")
#options_raw = query_params.get("opts", ["Option A|Option B|Option C"])[0] if isinstance(query_params.get("opts"), list) else query_params.get("opts", "Option A|Option B|Option C")

# Split the options
#options = options_raw.split("|")

# Decode the options and question parameters
# Display Question and Options
#st.markdown(f"Survey Help Chatbot")
#st.markdown(f"**Survey Question ({question_id}):** {question_text}")
#st.markdown("**Options:**")
#for i, opt in enumerate(options):
    #st.markdown(f"{i+1}. {opt}")

# Modern Streamlit: get query parameters (already decoded)
query_params = st.query_params

# Safely access values with defaults
question_id = query_params.get("qid", "Q1")
question_text = query_params.get("qtext", "What is your decision?")
options_raw = query_params.get("opts", "Option A|Option B|Option C")

# Split the options using pipe delimiter
options = options_raw.split("|")

# Display Question and Options
st.write("### Survey Help Chatbot")
st.write(f"**Survey Question ({question_id}):** {question_text}")
st.write("**Options:**")
for i, opt in enumerate(options):
    st.write(f"{i+1}. {opt}")


# Handle Recommendation Button
if st.button(" Get Recommendation"):
    recommendation = get_gpt_recommendation(question_text, options)
    st.write(f"### Chatbot Recommendation:")
    st.write(recommendation)

    # ✅ Save to Google Sheets
    save_to_gsheet({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": "recommendation",
        "question_id": question_id,
        "question_text": question_text,
        "user_input": "",
        "response": recommendation
    })


# User Input for Follow-Up Questions
user_input = st.text_input("Ask a follow-up question:")

if user_input:
    validation_feedback = validate_followup(user_input, question_id=question_id)
    st.write(f"Chatbot Follow-up:")
    st.write(validation_feedback)

    # ✅ Save to Google Sheets
    save_to_gsheet({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": "followup",
        "question_id": question_id,
        "question_text": question_text,
        "user_input": user_input,
        "response": validation_feedback
    })
