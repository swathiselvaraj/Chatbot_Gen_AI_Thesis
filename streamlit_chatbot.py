import streamlit as st
from openai import OpenAI
import time
import json
import pandas as pd
from typing import List
import uuid
import numpy as np
from urllib.parse import unquote

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

# Follow-Up questions Validation

def validate_followup(user_question: str, question_id: str) -> str:
    user_embedding = get_embedding(user_question)

    # Find a match among general followups
    for general in data["general_followups"]:
        general_embedding = general.get("embedding")
        if general_embedding:
            score = cosine_similarity(user_embedding, general_embedding)
            if score >= 0.70:
                # Get original question and recommendation to build context
                original_question = question_text  # pulled from Streamlit earlier
                original_recommendation = get_gpt_recommendation(original_question, options)

                # Provide both as history to follow-up question
                history = [(original_question, original_recommendation)]
                return get_gpt_recommendation(user_question, history=history)

    # Check question-specific followups
    for question in data["questions"]:
        if question["question_id"] == question_id:
            followup_embedding = question.get("embedding")
            if followup_embedding:
                score = cosine_similarity(user_embedding, followup_embedding)
                if score >= 0.70:
                    original_question = question_text
                    original_recommendation = get_gpt_recommendation(original_question, options)
                    history = [(original_question, original_recommendation)]
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

# User Input for Follow-Up Questions
user_input = st.text_input("Ask a follow-up question:")

if user_input:
    validation_feedback = validate_followup(user_input, question_id=question_id)
    st.write(f"Chatbot Follow-up:")
    st.write(validation_feedback)
