
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


query_params = st.query_params
question_id = query_params.get("qid", "Q1")
question_text = query_params.get("qtext", "What is your decision?")
#&options_raw = query_params.get("opts", "Option A|Option B|Option C")


#&options = options_raw.split("|")
options_raw = query_params.get("opts", "Option 1|Option 2|Option 3|Option 4")  # Default now has 4 options
options = options_raw.split("|")
##&
while len(options) < 4:
   options.append("")

option_mapping = {f"option {i+1}": options[i] for i in range(4)}
option_mapping.update({f"option{i+1}": options[i] for i in range(4)})  # Also handle "option1" format




# Ensure we have exactly 4 options, pad with empty strings if needed


participant_id = query_params.get("pid", str(uuid.uuid4()))





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
       'questions_asked': 0,
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
    """
    Extracts referenced survey option from user input (1-4 only)
    Handles formats: 'option X', 'optionX', 'why not option X', 'X', 'option [word]'
    """
    if not user_input or len(options) != 4:
        return None

    try:
        user_input_lower = user_input.lower().strip()
        number_map = {'one': 1, 'two': 2, 'three': 3, 'four': 4}
        
        # Check all possible patterns
        for i in range(1, 5):
            # Match: option 1, option1, option one
            if (re.search(rf'option[\s\-_]*(0?{i}|{list(number_map.keys())[i-1]})\b', user_input_lower) or
                re.search(rf'(^|\b)why\s+not\s+option[\s\-_]*(0?{i}|{list(number_map.keys())[i-1]})\b', user_input_lower) or
                re.search(rf'(^|\b)0?{i}\b', user_input_lower)):
                
                idx = i - 1
                if idx < len(options) and options[idx]:
                    return options[idx]
        
        return None

    except Exception as e:
        print(f"Option extraction error: {str(e)}")
        return None


# Add these near your other utility functions
def get_contextual_prompt(question_type: str, user_input: str, referenced_option: str = None) -> str:
    options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options) if opt.strip()])
    
    base_context = f"""
    ## Supermarket Dashboard Context ##
    Role: Analyze LimeSurvey options with dashboard insights.

    Current Options:
    {options_text}

    Key Metrics:
    - Sales trends (last 3 months)
    - Customer preference scores
    - Inventory turnover rates
    """

    if referenced_option:
        # Handle ALL option-related questions
        return f"""{base_context}
        
        User Question: "{user_input}"
        Referenced Option: {referenced_option}

        Task:
        1. For "why not" questions:
           - List 1-2 drawbacks of '{referenced_option}'
           - Suggest better alternatives
        2. For "explain" questions:
           - Describe pros/cons of '{referenced_option}'
           - Reference dashboard data
        3. For "compare" questions:
           - Contrast with other options
           - Use metrics (e.g., "Option 3 has 15% higher...")
        4. Always:
           - Be specific to THESE options
           - Avoid generic phrases
        """
    else:
        # Non-option questions (keep your existing logic)
        return f"{base_context}\n\nUser Question: \"{user_input}\"\n\nTask: Answer concisely."

def classify_question(user_input: str) -> str:
    """Determine question type for routing"""
    user_input = user_input.lower()
    
    if any(word in user_input for word in ["what is", "define", "explain", "meaning of"]):
        return "definition"
    elif any(word in user_input for word in ["recommend", "suggest", "should i", "which is better"]):
        return "recommendation"
    elif any(word in user_input for word in ["how to", "improve", "increase", "reduce"]):
        return "actionable"
    else:
        return "general"



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


def initialize_gsheet():
   """Initialize the Google Sheet with proper unique headers"""
   try:
       gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
       sheet = gc.open("Chatbot Usage Log")
      
       try:
           worksheet = sheet.worksheet("Logs")
       except:
           worksheet = sheet.add_worksheet(title="Logs", rows=1000, cols=20)
      
       # Define and verify headers - ensure all are unique
       expected_headers = [
           "participant_id", "question_id", "chatbot_used",
           "questions_asked", "total_time_seconds",
           "got_recommendation", "asked_followup", "record_timestamp"
       ]
      
       current_headers = worksheet.row_values(1)
      
       # Only update headers if they don't match exactly
       if not current_headers or set(current_headers) != set(expected_headers):
           worksheet.clear()
           worksheet.append_row(expected_headers)
      
       return worksheet
      
   except Exception as e:
       st.error(f"Google Sheets initialization failed: {str(e)}")
       return None



def save_session_data():
   try:
       # Use total_interaction_time instead of calculating fresh
       data = {
           "participant_id": participant_id,
           "question_id": question_id,
           "chatbot_used": "yes" if (st.session_state.usage_data['chatbot_used'] or
                                    st.session_state.usage_data['followup_used']) else "no",
           "questions_asked": st.session_state.usage_data['questions_asked'],
           "total_time_seconds": round(st.session_state.total_interaction_time, 2),
           "got_recommendation": "yes" if st.session_state.usage_data['get_recommendation'] else "no",
           "asked_followup": "yes" if st.session_state.usage_data['followup_used'] else "no",
           "record_timestamp": pd.Timestamp.now().isoformat()
       }


       if save_to_gsheet(data):
           st.session_state.already_saved = True
           return True
       return False
   except Exception as e:
       st.error(f"Session save failed: {str(e)}")
       return False


def save_to_gsheet(data_dict: Dict) -> bool:
   try:
       worksheet = initialize_gsheet()
       if not worksheet:
           return False


       # Get all records with expected headers to avoid duplicates
       records = worksheet.get_all_records(expected_headers=[
           "participant_id", "question_id", "chatbot_used",
           "questions_asked", "total_time_seconds",
           "got_recommendation", "asked_followup", "record_timestamp"
       ])
      
       # Find existing record
       row_index = None
       # for i, record in enumerate(records):
       #     if (str(record.get("participant_id")) == str(data_dict.get("participant_id")) and
       #         str(record.get("question_id")) == str(data_dict.get("question_id"))):
       #         row_index = i + 2  # +2 for header and 1-based index
       #         break
       for i, record in enumerate(records):
           pid_match = str(record.get("participant_id", "")).strip() == str(data_dict.get("participant_id", "")).strip()
           qid_match = str(record.get("question_id", "")).strip() == str(data_dict.get("question_id", "")).strip()
           if pid_match and qid_match:
               row_index = i + 2  # +2 to account for header row and 1-based indexing
               break


       # Prepare complete data row
       headers = worksheet.row_values(1)
       row_data = {k: data_dict.get(k, "") for k in headers}
      
       if row_index:
           # Update existing row
           worksheet.update(
               f"A{row_index}:{chr(65 + len(headers) - 1)}{row_index}",
               [[row_data.get(h, "") for h in headers]]
           )
       else:
           # Add new row
           worksheet.append_row([row_data.get(h, "") for h in headers])
      
       return True


   except Exception as e:
       st.error(f"Failed to save to Google Sheets: {str(e)}")
       return False

def validate_followup(user_question: str, question_id: str, options: List[str]) -> str:
    try:
        user_question = user_question.strip()
        if not user_question:
            return "Please enter a valid question."

        # Handle greetings
        greetings = {"hi", "hello", "hey", "greetings"}
        if user_question.lower().rstrip('!?.,') in greetings:
            st.session_state.last_recommendation = None
            return "Hello! I can help with supermarket dashboard questions. What would you like to know?"

        # Extract referenced option if any
        referenced_option = extract_referenced_option(user_question, options)
        
        # Classify question type
        question_type = classify_question(user_question)
        
        # Get question embedding
        user_embedding = get_embedding(user_question)
        if not user_embedding:
            return "Sorry, I couldn't process your question. Please try again."

        # Check against general followups
        general_threshold = 0.50
        general_scores = []
        for source in data.get("general_followups", []):
            if source.get("embedding"):
                score = cosine_similarity(user_embedding, source["embedding"])
                if score >= general_threshold:
                    general_scores.append((score, source))
        
        # Check against question-specific followups
        question_threshold = 0.70
        question_scores = []
        for source in data.get("questions", []):
            if (source.get("embedding") and 
                source.get("question_id", "") == question_id):
                score = cosine_similarity(user_embedding, source["embedding"])
                if score >= question_threshold:
                    question_scores.append((score, source))
        
        # Three-tier response logic
        if question_scores and max([s[0] for s in question_scores]) > 0.7:
            # High confidence match - use predefined answer
            best_score, best_match = max(question_scores, key=lambda x: x[0])
            return best_match["answer"]
        elif general_scores or question_scores:
            # Medium confidence - use GPT with context
            context_prompt = get_contextual_prompt(
                question_type,
                user_question,
                referenced_option
            )
            return get_gpt_response_with_context(context_prompt)
        else:
            # Low confidence - still try with general context
            return get_gpt_response_with_context(
                get_contextual_prompt("general", user_question)
            )

    except Exception as e:
        st.error(f"Error in followup validation: {str(e)}")
        return "Sorry, I encountered an error processing your question."


def get_gpt_response_with_context(prompt: str) -> str:
    """Get GPT response with enhanced context handling"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"GPT response failed: {str(e)}")
        return "I couldn't generate a response. Please try again."

def get_gpt_recommendation(
    question: str,
    options: List[str] = None,
    history: List[Tuple[str, str]] = None,
    is_followup: bool = False,
    referenced_option: Optional[str] = None
) -> str:
    try:
        # Build conversation history
        messages = []
        if history:
            for q, a in history:
                if q.strip():
                    messages.append({"role": "user", "content": q})
                if a.strip():
                    messages.append({"role": "assistant", "content": a})

        # Generate context-aware prompt
        question_type = classify_question(question)
        prompt = get_contextual_prompt(
            question_type,
            question,
            referenced_option
        )

        # Add options if available
        if options:
            options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
            prompt += f"\n\nAvailable Options:\n{options_text}"

        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )

        result = response.choices[0].message.content
        st.session_state.last_recommendation = result
        
        # Log the interaction for improvement
        log_interaction(question, result, question_type)
        
        return result
    
    except Exception as e:
        st.error(f"Recommendation generation failed: {str(e)}")
        return "Sorry, I couldn't generate a recommendation."


def log_interaction(question: str, response: str, question_type: str):
    """Log interactions for continuous improvement"""
    try:
        log_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "participant_id": participant_id,
            "question_id": question_id,
            "question": question,
            "response": response,
            "question_type": question_type,
            "similarity_score": None,  # Can be calculated later
            "feedback": None  # Can be added if you implement feedback
        }
        
        # Initialize logs if not exists
        if 'interaction_logs' not in st.session_state:
            st.session_state.interaction_logs = []
        
        st.session_state.interaction_logs.append(log_entry)
        
    except Exception as e:
        print(f"Failed to log interaction: {str(e)}")
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
           "questions_asked_to_chatbot": st.session_state.usage_data.get('followups_asked', 0),
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




# --- Main App Logic ---
# Get query parameters


# Initialize Google Sheet on first load
if st.session_state.first_load and not st.session_state.sheet_initialized:
  initialize_gsheet()
  st.session_state.sheet_initialized = True




# Track question changes
if question_id != st.session_state.get('last_question_id'):
  st.session_state.conversation = []
  st.session_state.last_question_id = question_id
  st.session_state.already_saved = False  # Reset saved flag for new question


if st.button("Get Recommendation"):
   update_interaction_time()
   recommendation = get_gpt_recommendation(question_text, options)
   st.session_state.conversation.append(("assistant", recommendation))
   end_interaction_and_accumulate_time()
  
   # Update usage data
   st.session_state.usage_data.update({
       'chatbot_used': True,
       'questions_asked': st.session_state.usage_data.get('questions_asked', 0) + 1,
       'get_recommendation': True,
       'total_time': st.session_state.total_interaction_time
   })
   save_session_data()




user_input = st.text_input("Ask a follow-up question:")
# In your main app section, modify the user input handling:
if user_input:
    update_interaction_time()
    st.session_state.conversation.append(("user", user_input))
    
    # First check if it's a simple command
    if user_input.lower().strip() in ['help', '?']:
        response = "I can help with:\n- Explaining dashboard terms\n- Analyzing trends\n- Making recommendations\nAsk me anything about the supermarket data!"
    else:
        response = validate_followup(user_input, question_id, options)
    
    st.session_state.conversation.append(("assistant", response))
    end_interaction_and_accumulate_time()

    # Update usage data
    st.session_state.usage_data.update({
        'chatbot_used': True,
        'followup_used': True,
        'questions_asked': st.session_state.usage_data.get('questions_asked', 0) + 1,
        'total_time': st.session_state.total_interaction_time
    })

    save_session_data()




# Display conversation
display_conversation()




# Final save when leaving the page
# if not st.session_state.first_load and not st.session_state.already_saved:
#     save_progress()




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
