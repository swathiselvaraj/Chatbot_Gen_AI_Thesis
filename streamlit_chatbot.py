
# import streamlit as st
# from openai import OpenAI
# import time
# import json
# import pandas as pd
# import numpy as np
# from urllib.parse import unquote
# import gspread
# from gspread_dataframe import set_with_dataframe
# import re
# import uuid
# from typing import List, Dict, Optional, Tuple

# # Initialize OpenAI client
# client = OpenAI(api_key=st.secrets["openai"]["api_key"])
# st.set_page_config(page_title="Survey Chatbot", layout="wide")


# query_params = st.query_params
# question_id = query_params.get("qid", "Q1")
# question_text = query_params.get("qtext", "What is your decision?")
# #&options_raw = query_params.get("opts", "Option A|Option B|Option C")


# #&options = options_raw.split("|")
# options_raw = query_params.get("opts", "Option 1|Option 2|Option 3|Option 4")  # Default now has 4 options
# options = options_raw.split("|")
# ##&
# while len(options) < 4:
#    options.append("")

# option_mapping = {f"option {i+1}": options[i] for i in range(4)}
# option_mapping.update({f"option{i+1}": options[i] for i in range(4)})  # Also handle "option1" format
# # Ensure we have exactly 4 options, pad with empty strings if needed
# participant_id = query_params.get("pid", str(uuid.uuid4()))
# # --- Session State Initialization ---
# if 'conversation' not in st.session_state:
#   st.session_state.conversation = []
# if 'last_recommendation' not in st.session_state:
#   st.session_state.last_recommendation = None
# if 'last_question_id' not in st.session_state:
#   st.session_state.last_question_id = None
# if 'first_load' not in st.session_state:
#   st.session_state.first_load = True
# if 'sheet_initialized' not in st.session_state:
#   st.session_state.sheet_initialized = False
# if 'already_saved' not in st.session_state:  # New flag to track saves
#   st.session_state.already_saved = False
# if 'original_recommendation' not in st.session_state:
#     st.session_state.original_recommendation = None

# if "original_options" not in st.session_state:
#     st.session_state.original_options = options
#     st.session_state.option_mapping = {
#         f"option{i+1}": options[i] for i in range(len(options))
#     }


# if 'usage_data' not in st.session_state:
#    st.session_state.usage_data = {
#        'participant_id': participant_id,
#        'question_id': question_id,
#        'chatbot_used': False,
#        'questions_asked': 0,
#        'get_recommendation': False,
#        'followup_used': False,
#        'start_time': None,
#        'total_time': 0  # This will accumulate all interaction time
#    }



# if 'interaction_active' not in st.session_state:
#    st.session_state.interaction_active = False
# if 'total_interaction_time' not in st.session_state:
#    st.session_state.total_interaction_time = 0
# if 'last_interaction_time' not in st.session_state:
#    st.session_state.last_interaction_time = None
# if 'get_recommendation_used' not in st.session_state:
#   st.session_state.get_recommendation_used = False
# if 'followup_used' not in st.session_state:
#   st.session_state.followup_used = False

# # --- Data Loading ---
# @st.cache_resource
# def load_embedding_data():
#   try:
#       with open("data/followup_embeddings_list.json", "r") as f:
#           return json.load(f)
#   except Exception as e:
#       st.error(f"Failed to load embeddings: {str(e)}")
#       return {"general_followups": [], "questions": []}

# data = load_embedding_data()

# # --- Utility Functions ---
# def get_embedding(text: str) -> List[float]:
#   try:
#       response = client.embeddings.create(
#           input=text,
#           model="text-embedding-3-small"
#       )
#       return response.data[0].embedding
#   except Exception as e:
#       st.error(f"Embedding generation failed: {str(e)}")
#       return []

# def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
#   try:
#       a = np.array(vec1)
#       b = np.array(vec2)
#       return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
#   except Exception as e:
#       st.error(f"Similarity calculation failed: {str(e)}")
#       return 0.0


# def extract_referenced_option(user_input: str, options: List[str]) -> Optional[str]:
#     """
#     Extracts referenced survey option from user input with improved matching
#     Handles formats: 
#     - 'option X', 'optionX', 'option [text]'
#     - 'X' (just the number)
#     - 'why not option X', 'why option X'
#     - Direct mentions of option text
#     """
#     if not user_input or not options:
#         return None

#     user_input_lower = user_input.lower()
    
#     # Check for direct number references (1, 2, 3, 4)
#     for i in range(len(options)):
#         # Match: "1", "option 1", "option1", "why not option 1"
#         if (re.search(rf'(^|\b)(option\s*)?({i+1})\b', user_input_lower) or
#             re.search(rf'(why\s+(not\s+)?option\s*)?({i+1})\b', user_input_lower)):
#             return options[i]

#     # Check for direct text matches (if user quotes part of the option)
#     for option in options:
#         # Simple text matching (case insensitive)
#         option_lower = option.lower()
#         if len(option_lower) > 5 and option_lower in user_input_lower:
#             return option

#     return None

# # Add these near your other utility functions
# def get_contextual_prompt(question_type: str, user_input: str, referenced_option: str = None) -> str:
#     options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options) if opt.strip()])
    
#     base_context = f"""
#     ## Supermarket Dashboard Context ##
#     Role: Analyze LimeSurvey options with dashboard insights.

#     Current Options:
#     {options_text}

#     Key Metrics:
#     - Sales trends (last 3 months)
#     - Customer preference scores
#     - Inventory turnover rates
#     """

#     if referenced_option:
#         # Handle ALL option-related questions
#         return f"""{base_context}
        
#         User Question: "{user_input}"
#         Referenced Option: {referenced_option}

#         Task:
#         1. For "why not" qif refeuestions:
#            - List 1-2 drawbacks of '{referenced_option}'
#            - Suggest better alternatives
#         2. For "explain" questions:
#            - Describe pros/cons of '{referenced_option}'
#            - Reference dashboard data
#         3. For "compare" questions:
#            - Contrast with other options
#            - Use metrics (e.g., "Option 3 has 15% higher...")
#         4. Always:
#            - Be specific to THESE options
#            - Avoid generic phrases
#         """
#     else:
#         # Non-option questions (keep your existing logic)
#         return f"{base_context}\n\nUser Question: \"{user_input}\"\n\nTask: Answer concisely."

# def classify_question(user_input: str) -> str:
#     """Determine question type for routing"""
#     user_input = user_input.lower()
    
#     if any(word in user_input for word in ["what is", "define", "explain", "meaning of"]):
#         return "definition"
#     elif any(word in user_input for word in ["recommend", "suggest", "should i", "which is better"]):
#         return "recommendation"
#     elif any(word in user_input for word in ["how to", "improve", "increase", "reduce"]):
#         return "actionable"
#     else:
#         return "general"

# def update_interaction_time():
#    now = time.time()
#    if not st.session_state.interaction_active:
#        st.session_state.interaction_start_time = now
#        st.session_state.interaction_active = True
#    st.session_state.last_interaction_time = now


# def end_interaction_and_accumulate_time():
#    if st.session_state.interaction_active and st.session_state.interaction_start_time:
#        now = time.time()
#        duration = now - st.session_state.interaction_start_time
#        st.session_state.total_interaction_time += duration
#        st.session_state.interaction_active = False
#        st.session_state.interaction_start_time = None


# def initialize_gsheet():
#    """Initialize the Google Sheet with proper unique headers"""
#    try:
#        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
#        sheet = gc.open("Chatbot Usage Log")
      
#        try:
#            worksheet = sheet.worksheet("Logs")
#        except:
#            worksheet = sheet.add_worksheet(title="Logs", rows=1000, cols=20)
      
#        # Define and verify headers - ensure all are unique
#        expected_headers = [
#            "participant_id", "question_id", "chatbot_used",
#            "questions_asked", "total_time_seconds",
#            "got_recommendation", "asked_followup", "record_timestamp"
#        ]
      
#        current_headers = worksheet.row_values(1)
      
#        # Only update headers if they don't match exactly
#        if not current_headers or set(current_headers) != set(expected_headers):
#            worksheet.clear()
#            worksheet.append_row(expected_headers)
      
#        return worksheet
      
#    except Exception as e:
#        st.error(f"Google Sheets initialization failed: {str(e)}")
#        return None



# def save_session_data():
#    try:
#        # Use total_interaction_time instead of calculating fresh
#        data = {
#            "participant_id": participant_id,
#            "question_id": question_id,
#            "chatbot_used": "yes" if (st.session_state.usage_data['chatbot_used'] or
#                                     st.session_state.usage_data['followup_used']) else "no",
#            "questions_asked": st.session_state.usage_data['questions_asked'],
#            "total_time_seconds": round(st.session_state.total_interaction_time, 2),
#            "got_recommendation": "yes" if st.session_state.usage_data['get_recommendation'] else "no",
#            "asked_followup": "yes" if st.session_state.usage_data['followup_used'] else "no",
#            "record_timestamp": pd.Timestamp.now().isoformat()
#        }


#        if save_to_gsheet(data):
#            st.session_state.already_saved = True
#            return True
#        return False
#    except Exception as e:
#        st.error(f"Session save failed: {str(e)}")
#        return False


# def save_to_gsheet(data_dict: Dict) -> bool:
#    try:
#        worksheet = initialize_gsheet()
#        if not worksheet:
#            return False


#        # Get all records with expected headers to avoid duplicates
#        records = worksheet.get_all_records(expected_headers=[
#            "participant_id", "question_id", "chatbot_used",
#            "questions_asked", "total_time_seconds",
#            "got_recommendation", "asked_followup", "record_timestamp"
#        ])
      
#        # Find existing record
#        row_index = None
#        for i, record in enumerate(records):
#            pid_match = str(record.get("participant_id", "")).strip() == str(data_dict.get("participant_id", "")).strip()
#            qid_match = str(record.get("question_id", "")).strip() == str(data_dict.get("question_id", "")).strip()
#            if pid_match and qid_match:
#                row_index = i + 2  # +2 to account for header row and 1-based indexing
#                break


#        # Prepare complete data row
#        headers = worksheet.row_values(1)
#        row_data = {k: data_dict.get(k, "") for k in headers}
      
#        if row_index:
#            # Update existing row
#            worksheet.update(
#                f"A{row_index}:{chr(65 + len(headers) - 1)}{row_index}",
#                [[row_data.get(h, "") for h in headers]]
#            )
#        else:
#            # Add new row
#            worksheet.append_row([row_data.get(h, "") for h in headers])
      
#        return True


#    except Exception as e:
#        st.error(f"Failed to save to Google Sheets: {str(e)}")
#        return False

# def validate_followup(user_input: str, question_id: str, options: List[str]) -> str:
#     try:
#         user_input = user_input.strip()
#         if not user_input:
#             return "Please enter a valid question."

#         # Handle greetings
#         greetings = {"hi", "hello", "hey", "greetings"}
#         if user_input.lower().rstrip('!?.,') in greetings:
#             st.session_state.last_recommendation = None
#             return "Hello! I can help with survey questions. What would you like to know?"

#         # Extract referenced option if any
#         referenced_option = extract_referenced_option(user_input, options)
        
#         # Handle all option-related questions
#         if referenced_option or any(x in user_input.lower() for x in ["option", "1", "2", "3", "4"]):
#             # Get the option number
#             option_num = options.index(referenced_option) + 1 if referenced_option else None
            
#             # Determine question type
#             is_why_not = "why not" in user_input.lower()
#             is_why = "why" in user_input.lower() and not is_why_not
#             is_explain = "explain" in user_input.lower()
            
#             # Case 1: Asking about why NOT a specific option
#             if is_why_not and referenced_option:
#                 if (st.session_state.original_recommendation and 
#                     referenced_option in st.session_state.original_recommendation['text']):
#                     return f"Actually, {referenced_option} WAS recommended because: {st.session_state.original_recommendation['reasoning']}"
                
#                 if not st.session_state.original_recommendation:
#                     return "Please first get a recommendation before asking why an option wasn't chosen."
                
#                 prompt = f"""Explain why this option wasn't recommended:
                
#                 Survey Question: {question_text}
#                 Recommended Option: {st.session_state.original_recommendation['text']}
#                 Option Being Questioned: Option {option_num} ({referenced_option})

#                 Provide 1-2 specific reasons comparing to the recommended option.
#                 Be concise (1-2 sentences max).
#                 """
#                 return get_gpt_response_with_context(prompt)
            
#             # Case 2: Asking WHY an option (or general option question)
#             elif (is_why or is_explain or referenced_option) and referenced_option:
#                 prompt = f"""Analyze this survey option:
                
#                 Survey Question: {question_text}
#                 Option: {referenced_option} (Option {option_num})
#                 User Question: {user_input}

#                 Provide a concise analysis (1-2 sentences) focusing on:
#                 - Key advantages/disadvantages
#                 - How it compares to other options
#                 - Specific metrics if available
#                 """
#                 return get_gpt_response_with_context(prompt)
            
#             # Case 3: General option question without specific reference
#             else:
#                 return get_gpt_recommendation(
#                     f"{user_input} (about these options: {', '.join(options)})",
#                     options=options,
#                     is_followup=True
#                 )

#         # Rest of your existing logic for non-option questions...
#         # (keep the embedding/similarity checks here)
        
#         # If we get here, use general GPT response
#         return get_gpt_recommendation(
#             user_input,
#             options=options,
#             is_followup=True
#         )

#     except Exception as e:
#         st.error(f"Error in followup validation: {str(e)}")
#         return "Sorry, I encountered an error processing your question."

# def get_gpt_response_with_context(prompt: str) -> str:
#     """Get GPT response with enhanced context handling"""
#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "system", "content": prompt}],
#             temperature=0.5,
#             max_tokens=150
#         )
#         # Clean up the response
#         result = response.choices[0].message.content
#         return result.split("Answer:")[-1].strip() if "Answer:" in result else result
    
#     except Exception as e:
#         st.error(f"GPT response failed: {str(e)}")
#         return "I couldn't generate a response. Please try again."

# def get_gpt_recommendation(
#     question: str,
#     options: List[str] = None,
#     history: List[Tuple[str, str]] = None,
#     is_followup: bool = False,
#     referenced_option: Optional[str] = None
# ) -> str:
#     try:
#         messages = []
        
#         # Include conversation history if provided
#         if history:
#             for q, a in history:
#                 if q.strip():
#                     messages.append({"role": "user", "content": q})
#                 if a.strip():
#                     messages.append({"role": "assistant", "content": a})

#         # Different processing for follow-up vs initial recommendation
#         if is_followup:
#             prompt = f"""The user has asked a follow-up question about a survey recommendation.
# Context:
# - Original question: {question}
# - Options: {chr(10).join(options)}
# {f"- Referenced option: {referenced_option}" if referenced_option else ""}

# The user has asked a follow-up question about a survey recommendation.
# You must answer the question or use prior context and reasoning to answer concisely in under 50 words.

# Respond in this format:
# "Answer: <your answer>"
# """
#         else:
#             # Initial recommendation mode - full recommendation
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
        
#         # Store original recommendation with reasoning
#         if not is_followup:
#             if "Recommended option:" in result and "Reason:" in result:
#                 rec_text = result.split("Recommended option:")[1].split("Reason:")[0].strip()
#                 reasoning = result.split("Reason:")[1].strip()
#             else:
#                 rec_text = result
#                 reasoning = "Based on overall analysis of options and dashboard trends."
            
#             st.session_state.original_recommendation = {
#                 'text': rec_text,
#                 'reasoning': reasoning,
#                 'options': options.copy(),
#                 'timestamp': time.time()
#             }
        
#         st.session_state.last_recommendation = result
#         return result
    
#     except Exception as e:
#         st.error(f"Recommendation generation failed: {str(e)}")
#         return "Sorry, I couldn't generate a recommendation."

# def log_interaction(question: str, response: str, question_type: str):
#     """Log interactions for continuous improvement"""
#     try:
#         log_entry = {
#             "timestamp": pd.Timestamp.now().isoformat(),
#             "participant_id": participant_id,
#             "question_id": question_id,
#             "question": question,
#             "response": response,
#             "question_type": question_type,
#             "similarity_score": None,  # Can be calculated later
#             "feedback": None  # Can be added if you implement feedback
#         }
        
#         # Initialize logs if not exists
#         if 'interaction_logs' not in st.session_state:
#             st.session_state.interaction_logs = []
        
#         st.session_state.interaction_logs.append(log_entry)
        
#     except Exception as e:
#         print(f"Failed to log interaction: {str(e)}")
# def display_conversation():
#   if 'conversation' not in st.session_state:
#       st.session_state.conversation = []

#   if len(st.session_state.conversation) > 0:
#       role, message = st.session_state.conversation[-1]
#       if role != "user":
#           st.markdown(f"**Chatbot:** {message}")




# def save_progress():
#    """Save or update progress in Google Sheets"""
#    if st.session_state.already_saved:
#        return True


#    if not st.session_state.usage_data.get('start_time'):
#        return False


#    try:
#        # Ensure timing variables are defined
#        start_time = st.session_state.get("interaction_start_time")
#        end_time = st.session_state.get("interaction_end_time")
#        total_time = round(end_time - start_time, 2) if start_time and end_time else 0


#        usage_data = {
#            "participant_id": participant_id,
#            "question_id": question_id,
#            "chatbot_used": "yes" if (st.session_state.get("get_recommendation_used") or st.session_state.get("followup_used")) else "no",
#            "questions_asked_to_chatbot": st.session_state.usage_data.get('followups_asked', 0),
#            "total_chatbot_time_seconds": total_time,
#            "get_recommendation": "yes" if st.session_state.get("get_recommendation_used") else "no",
#            "further_question_asked": "yes" if st.session_state.get("followup_used") else "no",
#            "timestamp": pd.Timestamp.now().isoformat()
#        }


#        if save_to_gsheet(usage_data):
#            st.session_state.usage_data['start_time'] = time.time()
#            st.session_state.already_saved = True
#            return True


#        return False  # Only one save attempt


#    except Exception as e:
#        st.error(f"Progress save failed: {str(e)}")
#        return False



# # --- Main App Logic ---
# # Get query parameters


# # Initialize Google Sheet on first load
# if st.session_state.first_load and not st.session_state.sheet_initialized:
#   initialize_gsheet()
#   st.session_state.sheet_initialized = True




# # Track question changes
# if question_id != st.session_state.get('last_question_id'):
#   st.session_state.conversation = []
#   st.session_state.last_question_id = question_id
#   st.session_state.already_saved = False  # Reset saved flag for new question


# if st.button("Get Recommendation"):
#    update_interaction_time()
#    recommendation = get_gpt_recommendation(question_text, options)
#    st.session_state.conversation.append(("assistant", recommendation))
#    end_interaction_and_accumulate_time()
  
#    # Update usage data
#    st.session_state.usage_data.update({
#        'chatbot_used': True,
#        'questions_asked': st.session_state.usage_data.get('questions_asked', 0) + 1,
#        'get_recommendation': True,
#        'total_time': st.session_state.total_interaction_time
#    })
#    save_session_data()


# user_input = st.text_input("Ask a follow-up question:")
# # In your main app section, modify the user input handling:
# if user_input:
#     update_interaction_time()
#     st.session_state.conversation.append(("user", user_input))
    
#     # First check if it's a simple command
#     if user_input.lower().strip() in ['help', '?']:
#         response = "I can help with:\n- Explaining dashboard terms\n- Analyzing trends\n- Making recommendations\nAsk me anything about the supermarket data!"
#     else:
#         response = validate_followup(user_input, question_id, options)
    
#     st.session_state.conversation.append(("assistant", response))
#     end_interaction_and_accumulate_time()

#     # Update usage data
#     st.session_state.usage_data.update({
#         'chatbot_used': True,
#         'followup_used': True,
#         'questions_asked': st.session_state.usage_data.get('questions_asked', 0) + 1,
#         'total_time': st.session_state.total_interaction_time
#     })

#     save_session_data()

# # Display conversation
# display_conversation()

# # Debug information
# if query_params.get("debug", "false") == "true":
#   st.write("### Debug Information")
#   st.write("Query Parameters:", query_params)
#   st.write("Current Question ID:", question_id)
#   st.write("Participant ID:", participant_id)
#   st.write("Session State:", {
#       k: v for k, v in st.session_state.items()
#       if k not in ['conversation', '_secrets']
#   })

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

# Get query parameters
query_params = st.query_params
question_id = query_params.get("qid", "Q1")
question_text = query_params.get("qtext", "What is your decision?")
options_raw = query_params.get("opts", "Option 1|Option 2|Option 3|Option 4")
options = [opt.strip() for opt in options_raw.split("|") if opt.strip()]
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
if 'already_saved' not in st.session_state:
    st.session_state.already_saved = False
if 'original_recommendation' not in st.session_state:
    st.session_state.original_recommendation = None
if 'original_options' not in st.session_state:
    st.session_state.original_options = options
    st.session_state.option_mapping = {f"option{i+1}": options[i] for i in range(len(options))}
    st.session_state.number_mapping = {str(i+1): options[i] for i in range(len(options))}

if 'usage_data' not in st.session_state:
    st.session_state.usage_data = {
        'participant_id': participant_id,
        'question_id': question_id,
        'chatbot_used': False,
        'questions_asked': 0,
        'get_recommendation': False,
        'followup_used': False,
        'start_time': None,
        'total_time': 0
    }

# --- Improved Option Extraction ---
def extract_referenced_option(user_input: str, options: List[str]) -> Optional[str]:
    """Extracts referenced option with better handling of different formats"""
    if not user_input or not options:
        return None

    user_input_lower = user_input.lower()
    number_map = {'one': '1', 'two': '2', 'three': '3', 'four': '4'}
    
    # Check for direct number references (1, 2, 3, 4)
    for num in ['1', '2', '3', '4']:
        # Match formats: "1", "option 1", "option1", "why not option 1"
        if (re.search(rf'(^|\b)(option\s*)?({num})\b', user_input_lower) or
            re.search(rf'(why\s+(not\s+)?option\s*)?({num})\b', user_input_lower)):
            if int(num) <= len(options):
                return options[int(num)-1]
    
    # Check for word numbers ("one", "two", etc.)
    for word, num in number_map.items():
        if re.search(rf'\b{word}\b', user_input_lower):
            if int(num) <= len(options):
                return options[int(num)-1]
    
    # Check for direct text matches
    for option in options:
        option_lower = option.lower()
        if len(option_lower) > 5 and option_lower in user_input_lower:
            return option
    
    return None

# --- Enhanced Recommendation Handling ---
def get_gpt_recommendation(question: str, options: List[str]) -> Dict:
    """Get recommendation with structured output"""
    try:
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        prompt = f"""Survey Question: {question}
Available Options:
{options_text}

Please analyze these options and recommend the best one. Provide:

1. Recommended option (number and text)
2. Clear reasoning (2-3 sentences)
3. Key advantages
4. Potential drawbacks of other options

Format your response as JSON:
{{
    "recommended_option": "Option X: [text]",
    "reasoning": "[detailed reasoning]",
    "advantages": ["advantage1", "advantage2"],
    "drawbacks": {{
        "Option 1": "[drawback]",
        "Option 2": "[drawback]",
        ...
    }}
}}"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        result['options'] = options.copy()  # Store original options
        
        # Store the full recommendation
        st.session_state.original_recommendation = result
        return result
        
    except Exception as e:
        st.error(f"Recommendation generation failed: {str(e)}")
        return {
            "recommended_option": "Option 1: " + options[0],
            "reasoning": "Default recommendation due to error",
            "advantages": [],
            "drawbacks": {}
        }

# --- Improved Followup Handling ---
def validate_followup(user_input: str, question_id: str, options: List[str]) -> str:
    try:
        user_input = user_input.strip()
        if not user_input:
            return "Please enter a valid question."

        # Handle greetings
        greetings = {"hi", "hello", "hey", "greetings"}
        if user_input.lower().rstrip('!?.,') in greetings:
            return "Hello! I can help analyze the survey options. What would you like to know?"

        referenced_option = extract_referenced_option(user_input, options)
        is_why_not = "why not" in user_input.lower()
        is_why = "why" in user_input.lower() and not is_why_not

        # Handle option-specific questions
        if referenced_option or any(x in user_input.lower() for x in ["option", "1", "2", "3", "4"]):
            if not st.session_state.original_recommendation:
                return "Please first get a recommendation before asking follow-up questions."
            
            rec = st.session_state.original_recommendation
            rec_option = rec['recommended_option']
            
            # Case 1: Asking about why NOT a specific option
            if is_why_not and referenced_option:
                if referenced_option in rec_option:
                    return f"Actually, {referenced_option} WAS recommended because: {rec['reasoning']}"
                
                # Find drawbacks for the referenced option
                drawback = rec['drawbacks'].get(referenced_option, 
                    "It wasn't the optimal choice based on the current analysis.")
                return f"Option {referenced_option} wasn't recommended because: {drawback}"
            
            # Case 2: Asking about the recommended option
            elif referenced_option and referenced_option in rec_option:
                return (f"Recommended Option: {rec_option}\n\n"
                       f"Reasoning: {rec['reasoning']}\n\n"
                       f"Advantages: {', '.join(rec['advantages'])}")
            
            # Case 3: General option question
            elif referenced_option:
                return (f"Analysis of {referenced_option}:\n\n"
                       f"Compared to recommended option {rec_option}, this option "
                       f"{rec['drawbacks'].get(referenced_option, 'has some limitations')}")
            
            # Case 4: General question about options
            else:
                return get_gpt_response_with_context(
                    f"User asked: {user_input}\n\n"
                    f"Current recommendation: {rec_option}\n"
                    f"Options: {', '.join(options)}\n\n"
                    "Provide a concise answer (1-2 sentences) relating to the recommendation."
                )

        # Handle recommendation explanation requests
        explanation_phrases = [
            "why this recommendation", "explain your suggestion",
            "justify your answer", "how did you decide"
        ]
        if any(phrase in user_input.lower() for phrase in explanation_phrases):
            if st.session_state.original_recommendation:
                rec = st.session_state.original_recommendation
                return (f"Recommendation Analysis:\n\n"
                       f"Chosen Option: {rec['recommended_option']}\n"
                       f"Reasoning: {rec['reasoning']}\n\n"
                       f"Options considered:\n"
                       + "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]))
            else:
                return "Please first get a recommendation before asking for an explanation."

        # Default response for other questions
        return get_gpt_response_with_context(
            f"Survey Question: {question_text}\n"
            f"Current Options: {', '.join(options)}\n"
            f"User Question: {user_input}\n\n"
            "Provide a concise answer (1-2 sentences) based on the available options."
        )

    except Exception as e:
        st.error(f"Error in followup validation: {str(e)}")
        return "Sorry, I encountered an error processing your question."

# --- Main App Logic ---
if st.session_state.first_load and not st.session_state.sheet_initialized:
    initialize_gsheet()
    st.session_state.sheet_initialized = True

if question_id != st.session_state.get('last_question_id'):
    st.session_state.conversation = []
    st.session_state.last_question_id = question_id
    st.session_state.already_saved = False

# Get Recommendation Button
if st.button("Get Recommendation"):
    update_interaction_time()
    recommendation = get_gpt_recommendation(question_text, options)
    response = (
        f"Recommended Option: {recommendation['recommended_option']}\n\n"
        f"Reasoning: {recommendation['reasoning']}"
    )
    st.session_state.conversation.append(("assistant", response))
    end_interaction_and_accumulate_time()
    
    st.session_state.usage_data.update({
        'chatbot_used': True,
        'questions_asked': st.session_state.usage_data.get('questions_asked', 0) + 1,
        'get_recommendation': True,
        'total_time': st.session_state.total_interaction_time
    })
    save_session_data()

# Follow-up Question Handling
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
        'total_time': st.session_state.total_interaction_time
    })
    save_session_data()

# Display conversation
if st.session_state.conversation:
    for role, message in st.session_state.conversation:
        if role == "user":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Chatbot:** {message}")

# Debug information
if query_params.get("debug", "false") == "true":
    st.write("### Debug Information")
    st.write("Options:", options)
    st.write("Option Mapping:", st.session_state.option_mapping)
    st.write("Current Recommendation:", st.session_state.original_recommendation)