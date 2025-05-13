
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


# Ensure we have exactly 4 options, pad with empty strings if needed


participant_id = query_params.get("pid", str(uuid.uuid4()))


##&
while len(options) < 4:
   options.append("")


option_mapping = {f"option {i+1}": options[i] for i in range(4)}
option_mapping.update({f"option{i+1}": options[i] for i in range(4)})  # Also handle "option1" format


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




# if 'usage_data' not in st.session_state:
#    st.session_state.usage_data = {
#        'start_time': time.time(),
#        'questions_asked': 0,
#        'followups_asked': 0
#    }


# if 'usage_data' not in st.session_state:
#     st.session_state.usage_data = {
#         'participant_id': participant_id,
#         'question_id': question_id,
#         'chatbot_used': False,
#         'questions_asked': 0,
#         'get_recommendation': False,
#         'followup_used': False,
#         'start_time': None,
#         'total_time': 0
#     }
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


# --- New Session State Initialization for Time Tracking ---
# if 'interaction_start_time' not in st.session_state:
#    st.session_state.interaction_start_time = None
# if 'interaction_end_time' not in st.session_state:
#    st.session_state.interaction_end_time = None
# At the top with other session state initializations
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




# def extract_referenced_option(user_input: str, options: List[str]) -> Optional[str]:
#    try:
#        match = re.search(r"option\s*(\d+)", user_input.lower())
#        if match:
#            idx = int(match.group(1)) - 1
#            if 0 <= idx < len(options):
#                return options[idx]
#        return None
#    except:
#        return None






# def extract_referenced_option(user_input: str, options: List[str]) -> Optional[str]:
#     try:
#         # First try to find exact matches with the mapping
#         user_input_lower = user_input.lower()
#         for option_ref, option_text in option_mapping.items():
#             if option_ref in user_input_lower and option_text:
#                 return option_text
      
#         # Fallback to regex if no direct match found
#         match = re.search(r"option\s*(\d+)", user_input_lower)
#         if match:
#             idx = int(match.group(1)) - 1
#             if 0 <= idx < len(options):
#                 return options[idx]
#         return None
#     except:
#         return None
#-----------
# def extract_referenced_option(user_input: str, options: List[str]) -> Optional[Tuple[str, str]]:
#     """
#     Returns a tuple of (option_number, option_text) if found in user input
#     Example: ("4", "Launch a promotion for cheese...") for input "why not option 4"
#     """
#     try:
#         # Match patterns like "option 4" or "option4" or just "4"
#         match = re.search(r"(?:option\s*)?(\d+)", user_input.lower())
#         if match:
#             option_num = match.group(1)
#             # Find option that starts with this number
#             for option in options:
#                 if option.startswith(f"{option_num}."):
#                     return (option_num, option[len(option_num)+1:].strip())  # Return number and text
#             # Alternative pattern if number isn't at start
#             for option in options:
#                 if re.search(rf"\b{option_num}\.", option):
#                     return (option_num, option)
#         return None
#     except Exception as e:
#         st.error(f"Option extraction failed: {str(e)}")
#         return None


##&
# Modify the extract_referenced_option function to use the mapping:
# def extract_referenced_option(user_input: str, options: List[str]) -> Optional[str]:
#    try:
#        # First try to find exact matches with the mapping
#        user_input_lower = user_input.lower()
#        for option_ref, option_text in option_mapping.items():
#            if option_ref in user_input_lower and option_text:
#                return option_text
      
#        # Fallback to regex if no direct match found
#        match = re.search(r"option\s*(\d+)", user_input_lower)
#        if match:
#            idx = int(match.group(1)) - 1
#            if 0 <= idx < len(options):
#                return options[idx]
#        return None
#    except:
#        return None

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


# def update_interaction_time():
#     now = time.time()
  
#     # If this is the first interaction, start the timer
#     if not st.session_state.interaction_active:
#         st.session_state.interaction_start_time = now
#         st.session_state.interaction_active = True
  
#     # Update the last interaction time
#     st.session_state.last_interaction_time = now
  
#     # Calculate total time so far
#     if st.session_state.interaction_start_time:
#         st.session_state.total_interaction_time = now - st.session_state.interaction_start_time


# def update_interaction_time():
#     now = time.time()
#     if not st.session_state.interaction_active:
#         st.session_state.interaction_start_time = now
#         st.session_state.interaction_active = True
#     st.session_state.last_interaction_time = now
#     st.session_state.total_interaction_time = now - st.session_state.interaction_start_time


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
# --- Google Sheets Integration ---
# def initialize_gsheet():
#    """Initialize the Google Sheet with proper headers"""
#    try:
#        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
#        sheet = gc.open("Chatbot Usage Log")
    
#        try:
#            worksheet = sheet.worksheet("Logs")
#        except:
#            worksheet = sheet.add_worksheet(title="Logs", rows=1000, cols=20)
    
#        # Define and verify headers
#        expected_headers = [
#            "participant_id", "question_id", "chatbot_used",
#            "questions_asked_to_chatbot", "total_chatbot_time_seconds",
#            "get_recommendation", "further_question_asked", "timestamp"
#        ]
    
#        current_headers = worksheet.row_values(1)
    
#        if not current_headers or current_headers != expected_headers:
#            worksheet.clear()
#            worksheet.append_row(expected_headers)
    
#        return worksheet
    
#    except Exception as e:
#        st.error(f"Google Sheets initialization failed: {str(e)}")
#        return None


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




# def save_session_data():
#     try:
#         if not st.session_state.usage_data['start_time']:
#             st.session_state.usage_data['start_time'] = time.time()  # ensure start_time is set


#         total_time = time.time() - st.session_state.usage_data['start_time']


#         data = {
#             "participant_id": st.session_state.usage_data['participant_id'],
#             "question_id": st.session_state.usage_data['question_id'],
#             "chatbot_used": "yes" if st.session_state.usage_data['chatbot_used'] else "no",
#             "questions_asked": st.session_state.usage_data['questions_asked'],
#             "total_time_seconds": round(total_time, 2),
#             "got_recommendation": "yes" if st.session_state.usage_data['get_recommendation'] else "no",
#             "asked_followup": "yes" if st.session_state.usage_data['followup_used'] else "no",
#             "record_timestamp": pd.Timestamp.now().isoformat()
#         }


#         # ✅ Move this AFTER total_time is calculated
#         if save_to_gsheet(data):
#             st.session_state.already_saved = True
#             return True


#         return False


#     except Exception as e:
#         st.error(f"Session save failed: {str(e)}")
#         return False


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
# def save_to_gsheet(data_dict: Dict) -> bool:
#    try:
#        worksheet = initialize_gsheet()
#        if not worksheet:
#            return False




#        headers = worksheet.row_values(1)
#        records = worksheet.get_all_records()




#        target_row = None
#        for i, record in enumerate(records):
#            if (record["participant_id"] == data_dict["participant_id"] and
#                record["question_id"] == data_dict["question_id"]):
#                target_row = i + 2  # +2 because gspread is 1-indexed and skips headers
#                break




#        if target_row:
#            # Row exists – update each column in data_dict
#            for key, value in data_dict.items():
#                if key in headers:
#                    col_index = headers.index(key) + 1
#                    worksheet.update_cell(target_row, col_index, value)
#        else:
#            # Row doesn't exist – fill in row with defaults
#            new_row = ["" for _ in headers]
#            for key, value in data_dict.items():
#                if key in headers:
#                    idx = headers.index(key)
#                    new_row[idx] = value
#            worksheet.append_row(new_row)




#        return True




#    except Exception as e:
#        st.error(f"Failed to save to Google Sheets: {str(e)}")
#        return False




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


# --- Core Chatbot Functions ---
# def validate_followup(user_question: str, question_id: str, options: List[str]) -> str:
#    try:
#        user_embedding = get_embedding(user_question)
#        referenced_option = extract_referenced_option(user_question, options)
    
#        history = []
    
#        if st.session_state.last_recommendation:
#            history.append((f"Original survey question: {question_text}", st.session_state.last_recommendation))




#        history.append((f"Follow-up: {user_question}", ""))




#        if referenced_option:
#            history.append((f"The user mentioned: {referenced_option}", "Acknowledged."))




#        for source in data["general_followups"] + data["questions"]:
#            if source.get("embedding") and (source.get("question_id") == question_id or "question_id" not in source):
#                score = cosine_similarity(user_embedding, source["embedding"])
#                if score >= 0.70:
#                    return get_gpt_recommendation(user_question, options=options, history=history)




#        return "Please ask a question related to the current survey topic."
#    except Exception as e:
#        st.error(f"Follow-up validation failed: {str(e)}")


#------------


# def validate_followup(user_question: str, question_id: str, options: List[str]) -> str:
#     try:
#         user_embedding = get_embedding(user_question)
#         referenced_option = extract_referenced_option(user_question, options)
      
#         # First check general followups (greetings, etc.)
#         general_scores = []
#         for source in data["general_followups"]:
#             if source.get("embedding"):
#                 score = cosine_similarity(user_embedding, source["embedding"])
#                 general_scores.append((score, source))
      
#         # Get the highest scoring general followup if above threshold
#         if general_scores:
#             max_score, best_match = max(general_scores, key=lambda x: x[0])
#             if max_score >= 0.85:  # Higher threshold for general followups
#                 if "followup_text" in best_match:
#                     # For greetings, return a simple response
#                     if best_match["followup_text"].lower() in ["hi", "hello", "hey"]:
#                         return "Hello! How can I help you with your survey question?"
#                     # For other general followups, return their response
#                     return best_match.get("response", "How can I help you?")
      
#         # Only proceed with recommendation logic if not a general followup
#         history = []
#         if st.session_state.last_recommendation:
#             history.append((f"Original survey question: {question_text}", st.session_state.last_recommendation))


#         history.append((f"Follow-up: {user_question}", ""))


#         if referenced_option:
#             history.append((f"The user mentioned: {referenced_option}", "Acknowledged."))


#         # Check question-specific followups
#         question_scores = []
#         for source in data["questions"]:
#             if source.get("embedding") and source.get("question_id") == question_id:
#                 score = cosine_similarity(user_embedding, source["embedding"])
#                 question_scores.append((score, source))
      
#         if question_scores:
#             max_score, best_match = max(question_scores, key=lambda x: x[0])
#             if max_score >= 0.70:
#                 return get_gpt_recommendation(user_question, options=options, history=history)


#         return "Please ask a question related to the current survey topic."
#     except Exception as e:
#         st.error(f"Follow-up validation failed: {str(e)}")


# ***********
# def validate_followup(user_question: str, question_id: str, options: List[str]) -> str:
#     try:
#         # First check for direct option references
#         option_ref = extract_referenced_option(user_question, options)
      
#         if option_ref:
#             option_num, option_text = option_ref
#             # Format the question to include the full option text
#             formatted_question = f"{user_question} (Referring to: {option_text})"
#             user_embedding = get_embedding(formatted_question)
#         else:
#             user_embedding = get_embedding(user_question)


#         # Rest of your existing validation logic
#         history = []
      
#         if st.session_state.last_recommendation:
#             history.append((f"Original survey question: {question_text}", st.session_state.last_recommendation))


#         history.append((f"Follow-up: {user_question}", ""))


#         if option_ref:
#             history.append((f"The user is asking about option {option_ref[0]}: {option_ref[1]}", ""))


#         for source in data["general_followups"] + data["questions"]:
#             if source.get("embedding") and (source.get("question_id") == question_id or "question_id" not in source):
#                 score = cosine_similarity(user_embedding, source["embedding"])
#                 if score >= 0.70:
#                     return get_gpt_recommendation(
#                         user_question,
#                         options=options,
#                         history=history,
#                         referenced_option=option_ref
#                     )


#         return "Please ask a question related to the current survey topic."
#     except Exception as e:
#         st.error(f"Follow-up validation failed: {str(e)}")
#&&&
def validate_followup(user_question: str, question_id: str, options: List[str]) -> str:
    try:
        # Clean input
        user_question = user_question.strip()
        if not user_question:
            return "Please enter a valid question."

        # Handle greetings
        greetings = {"hi", "hello", "hey", "greetings"}
        if user_question.lower().rstrip('!?.,') in greetings:
            st.session_state.last_recommendation = None
            return "Hello! Please ask about the survey options."

        # Extract referenced option using the improved function
        referenced_option = extract_referenced_option(user_question, options)
        
        # Prepare conversation history
        history = []
        if st.session_state.last_recommendation:
            history.append(("Original question", st.session_state.last_recommendation))
        history.append(("Follow-up", user_question))
        
        # If option found, get GPT response
        if referenced_option:
            option_num = options.index(referenced_option) + 1
            history.append(("Referenced option", f"Option {option_num}: {referenced_option}"))
            
            return get_gpt_recommendation(
                user_question,
                options=options,
                history=history,
                is_followup=True
            )

        # Fallback to semantic matching only if no option referenced
        user_embedding = get_embedding(user_question)
        if not user_embedding:
            return "Sorry, I couldn't process your question. Please try again."

        # 6. Check against general followups with high threshold
        general_threshold = 0.50
        general_scores = []
        for source in data.get("general_followups", []):
            if source.get("embedding"):
                score = cosine_similarity(user_embedding, source["embedding"])
                if score >= general_threshold:
                    general_scores.append((score, source))
        
        if general_scores:
            best_score, best_match = max(general_scores, key=lambda x: x[0])
            return get_gpt_recommendation(
                user_question,
                options=options,
                history=history,
                is_followup=True
            )
            

        # 7. Check against question-specific followups
        question_threshold = 0.70
        question_scores = []
        for source in data.get("questions", []):
            if (source.get("embedding") and 
                source.get("question_id", "") == question_id):
                score = cosine_similarity(user_embedding, source["embedding"])
                if score >= question_threshold:
                    question_scores.append((score, source))
        
        # Only proceed to GPT if we have question-specific matches
        if question_scores:
            return get_gpt_recommendation(
                user_question,
                options=options,
                history=history,
                is_followup=True
            )
        
        # 8. Final fallback if no matches found
        return "Please ask a question about the survey."

    except Exception as e:
        st.error(f"Error in followup validation: {str(e)}")
        return "Sorry, I encountered an error processing your question."
# Modify the validate_followup function to replace option references:


# def get_gpt_recommendation(question: str, options: List[str] = None, history: List[Tuple[str, str]] = None) -> str:
#    try:
#        messages = []
#        followup_mode = False




#        if history:
#            for q, a in history:
#                if "Follow-up:" in q:
#                    followup_mode = True
#                if q.strip():
#                    messages.append({"role": "user", "content": q})
#                if a.strip():
#                    messages.append({"role": "assistant", "content": a})




#        if followup_mode:
#            prompt = f"""The user has asked a follow-up question about a survey recommendation.
# You must use prior context and reasoning to answer concisely in under 50 words.




# Respond in this format:
# "Answer: <your answer to the follow-up question>"
# """
#        else:
#            options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]) if options else ""
#            prompt = f"""Survey Question: {question}
# Available Options:
# {options_text}




# Please recommend the best option with reasoning. Limit your response to 50 words.




# Respond in this format:
# "Recommended option: <text>"


# "Reason: <short explanation>"
# """




#        messages.append({"role": "user", "content": prompt})




#        response = client.chat.completions.create(
#            model="gpt-3.5-turbo",
#            messages=messages,
#            temperature=0.7
#        )




#        result = response.choices[0].message.content
#        st.session_state.last_recommendation = result
#        return result
#    except Exception as e:
#        st.error(f"Recommendation generation failed: {str(e)}")
#        return "Sorry, I couldn't generate a recommendation due to an error."


#*******
# def get_gpt_recommendation(
#     question: str,
#     options: List[str] = None,
#     history: List[Tuple[str, str]] = None,
#     referenced_option: Tuple[str, str] = None
# ) -> str:
#     try:
#         messages = []
#         followup_mode = False


#         if history:
#             for q, a in history:
#                 if "Follow-up:" in q:
#                     followup_mode = True
#                 if q.strip():
#                     messages.append({"role": "user", "content": q})
#                 if a.strip():
#                     messages.append({"role": "assistant", "content": a})


#         if followup_mode:
#             prompt = f"""The user has asked a follow-up question about a survey recommendation.
# Context:
# - Original question: {question_text}
# - Options: {chr(10).join(options)}
# {f"- Referenced option: {referenced_option[1]}" if referenced_option else ""}


# Please answer concisely in under 50 words, focusing on the specific option mentioned if applicable.


# Respond in this format:
# "Answer: <your answer>"
# """
#         else:
#             options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]) if options else ""
#             prompt = f"""Survey Question: {question}
# Available Options:
# {options_text}


# Please recommend the best option with reasoning. Limit your response to 50 words.


# Respond in this format:
# "Recommended option: <text>"


# "Reason: <short explanation>"
# """


#         messages.append({"role": "user", "content": prompt})


#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=messages,
#             temperature=0.7
#         )


#         result = response.choices[0].message.content
#         st.session_state.last_recommendation = result
#         return result
#     except Exception as e:
#         st.error(f"Recommendation generation failed: {str(e)}")
#         return "Sorry, I couldn't generate a recommendation due to an error."


###&&&
def get_gpt_recommendation(
   question: str,
   options: List[str] = None,
   history: List[Tuple[str, str]] = None,
   is_followup: bool = False,
   referenced_option: Optional[str] = None
) -> str:
   try:
       messages = []
      
       # Include conversation history if provided
       if history:
           for q, a in history:
               if q.strip():
                   messages.append({"role": "user", "content": q})
               if a.strip():
                   messages.append({"role": "assistant", "content": a})


       # Different processing for follow-up vs initial recommendation
       if is_followup:
           # Follow-up question mode - focused analysis
           prompt = f"""The user has asked a follow-up question about a survey recommendation.
Context:
- Original question: {question_text}
- Options: {chr(10).join(options)}
{f"- Referenced option: {referenced_option}" if referenced_option else ""}


The user has asked a follow-up question about a survey recommendation.
You must answer the question or use prior context and reasoning to answer concisely in under 50 words.


Respond in this format:
"Answer: <your answer>"
"""
       else:
           # Initial recommendation mode - full recommendation
           options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]) if options else ""
           prompt = f"""Survey Question: {question}
Available Options:
{options_text}


Please recommend the best option with reasoning. Limit your response to 50 words.


Respond in this format:ge
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

# def display_conversation():
#   if 'conversation' not in st.session_state:
#       st.session_state.conversation = []




#   if len(st.session_state.conversation) > 0:
#       role, message = st.session_state.conversation[-1]
#       if role != "user":
#           st.markdown(f"**Chatbot:** {message}")

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
  # if not st.session_state.first_load:
  #     save_progress()










# if st.button("Get Recommendation"):
#     update_interaction_time()
#     recommendation = get_gpt_recommendation(question_text, options)
#     st.session_state.conversation.append(("assistant", recommendation))
  
#     # Update usage data
#     st.session_state.usage_data.update({
#         'chatbot_used': True,
#         'questions_asked': st.session_state.usage_data.get('questions_asked', 0) + 1,
#         'get_recommendation': True,
#         'total_time': st.session_state.total_interaction_time
#     })
  
#     save_session_data()  # Single save point


  #save_progress()
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




# if user_input := st.text_input("Ask a follow-up question:"):
#     update_interaction_time()
#     st.session_state.conversation.append(("user", user_input))
#     response = validate_followup(user_input, question_id, options)
#     st.session_state.conversation.append(("assistant", response))
  
#     # Update usage data
#     st.session_state.usage_data.update({
#         'chatbot_used': True,
#         'questions_asked': st.session_state.usage_data.get('questions_asked', 0) + 1,
#         'followup_used': True,
#         'total_time': st.session_state.total_interaction_time
#     })
  
#     save_session_data()


  #save_progress()


user_input = st.text_input("Ask a follow-up question:")
if user_input:
   update_interaction_time()
   st.session_state.conversation.append(("user", user_input))
   response = validate_followup(user_input, question_id, options)
   st.session_state.conversation.append(("assistant", response))
  
   end_interaction_and_accumulate_time()


   st.session_state.usage_data.update({
       'chatbot_used': True,
       'followup_used': True,
       'questions_asked': st.session_state.usage_data.get('questions_asked', 0) + 1,
       'total_time': st.session_state.total_interaction_time  # Make sure this is included
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
