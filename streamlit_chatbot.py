
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


client = OpenAI(api_key=st.secrets["openai"]["api_key"])
query_params = st.query_params
question_id = query_params.get("qid", "Q1")
question_text = query_params.get("qtext", "What is your decision?")
#&options_raw = query_params.get("opts", "Option A|Option B|Option C")


#&options = options_raw.split("|")
options_raw = query_params.get("opts", "Option 1|Option 2|Option 3|Option 4")  # Default now has 4 options
options = options_raw.split("|")
##&
options = sorted(options_raw.split("|"))
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
if 'original_recommendation' not in st.session_state:
    st.session_state.original_recommendation = None

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
    Extracts referenced survey option from user input with improved matching
    Handles formats: 
    - 'option X', 'optionX', 'option [text]'
    - 'X' (just the number)
    - 'why not option X', 'why option X'
    - Direct mentions of option text
    """
    if not user_input or not options:
        return None

    user_input_lower = user_input.lower()
    
    # Check for direct number references (1, 2, 3, 4)
    for i in range(len(options)):
        # Match: "1", "option 1", "option1", "why not option 1"
        if (re.search(rf'(^|\b)(option\s*)?({i+1})\b', user_input_lower) or
            re.search(rf'(why\s+(not\s+)?option\s*)?({i+1})\b', user_input_lower)):
            return options[i]

    # Check for direct text matches (if user quotes part of the option)
    for option in options:
        # Simple text matching (case insensitive)
        option_lower = option.lower()
        if len(option_lower) > 5 and option_lower in user_input_lower:
            return option

    return None

# Add these near your other utility functions

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
           "total_time_seconds": round(st.session_state.get('total_interaction_time', 0), 2),
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
      return "Sorry, I encountered an error processing your question."

def validate_followup(user_input: str, question_id: str, options: List[str], question_text: str = "") -> str:
    try:
        user_input = user_input.strip()
        if not user_input:
            return "Please enter a valid question."

        # Handle greetings
        greetings = {"hi", "hello", "hey", "greetings"}
        if user_input.lower().rstrip('!?.,') in greetings:
            st.session_state.last_recommendation = None
            return "Hello! I can help with survey questions. What would you like to know?"

        # Extract referenced option if any
        referenced_option = extract_referenced_option(user_input, options)
        option_num = options.index(referenced_option) + 1 if referenced_option else None

        if option_num is not None:
           return get_gpt_recommendation(question =  question_text,referenced_option = option_num )

        

        # Get question embedding for similarity checks
        user_embedding = get_embedding(user_input)
        # if not user_embedding:
        #     return "Sorry, I couldn't process your question. Please try again."

        # Check against general followups
        dashboard_threshold = 0.50
        dashboard_scores = []
        for source in data.get("dashboard_followups", []):
            if source.get("embedding"):
                score = cosine_similarity(user_embedding, source["embedding"])
                if score >= dashboard_threshold:
                    dashboard_scores.append((score, source))

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



        # If we have medium confidence matches (either general or question-specific)
        if general_scores or question_scores:
            # Classify question type for contextual prompt
            return get_gpt_recommendation(question =  question_text)

        elif dashboard_scores:
            return get_gpt_recommendation(question =  question_text, dashboard = 'yes')
        
        else:
            return "Please ask a question related to the Survey"
    except Exception as e:
        st.error(f"Error in followup validation: {str(e)}")
        return "Sorry, I encountered an error processing your question."




def get_gpt_recommendation(
    question: str,
    options: List[str] = None,
    is_followup: bool = False,
    follow_up_question: Optional[str] = None,
    referenced_option: Optional[str] = None,
    dashboard: bool = False
) -> str:
    try:
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        messages = st.session_state.chat_history.copy()

        # Include dashboard context if needed
        if dashboard:
            json_data_path = "data/dashboard_data.json"
            try:
                with open(json_data_path, 'r') as file:
                    json_data = json.load(file)
                    json_context = json.dumps(json_data, indent=2)

                    system_prompt = textwrap.dedent(f"""\
                    Current dashboard data (in JSON format):
                    {json_context}

                    Important: When making recommendations:
                    1. Your knowledge is limited only to the data from this dashboard
                    2. Reference specific metrics when available
                    3. If data contradicts standard recommendations, explain why
                    """)
                    messages.insert(0, {"role": "system", "content": system_prompt})
            except Exception as e:
                print(f"Warning: Could not load JSON data - {str(e)}")

        # =========================
        # Construct the user prompt
        # =========================
        if is_followup:
            original_rec = st.session_state.get("original_recommendation")

            if original_rec:
                recommendation_summary = f"""
                Earlier Recommendation:
                Recommended option: {original_rec['text']}
                Reason: {original_rec['reasoning']}
                """
            else:
                recommendation_summary = "No previous recommendation found."

            prompt = f"""{recommendation_summary}

            Follow-up Question: {follow_up_question or question}
            
            Instructions:
            - If the follow-up refers to the options or prior recommendation, compare and clarify.
            - If not, just answer directly using the original context.
            - Keep the response under 50 words.

            Format:
            Answer: <your response>
            """

        else:
            options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]) if options else ""
            prompt = f"""Survey Question: {question}

            Available Options:
            {options_text}

            Please recommend the best option with reasoning (limit to 50 words).

            Format:
            Recommended option: <option>
            Reason: <short explanation>
            """

        messages.append({"role": "user", "content": prompt})

        # Call GPT
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )

        result = response.choices[0].message.content

        # Store original recommendation if not a follow-up
        if not is_followup:
            if "Recommended option:" in result and "Reason:" in result:
                rec_text = result.split("Recommended option:")[1].split("Reason:")[0].strip()
                reasoning = result.split("Reason:")[1].strip()
            else:
                rec_text = result
                reasoning = "Based on overall analysis of options and dashboard trends."
            
            st.session_state.original_recommendation = {
                'text': rec_text,
                'reasoning': reasoning,
                'options': options.copy() if options else [],
                'timestamp': time.time()
            }

        st.session_state.last_recommendation = result

        # Add response to chat history and trim
        messages.append({"role": "assistant", "content": result})
        st.session_state.chat_history = messages[-30:]  # Keep last 30 exchanges

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
       'total_time': st.session_state.get('total_interaction_time', 0)
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

