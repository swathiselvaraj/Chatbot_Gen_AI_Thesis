import openai
import json
from openai import OpenAI

# Initialize the client
client = OpenAI( api_key = "key")
# === Step 1: Load existing JSON file ===
with open("data/followup_embeddings_list.json", "r") as f:
    data = json.load(f)

# === Step 2: Load your dashboard JSON file ===
with open("data/dashboard_data.json", "r") as f:
    dashboard_json = json.load(f)

# === Step 3: Generate questions from GPT ===
prompt = f"""
You are an AI assistant analyzing a business dashboard described in the JSON data below.

Your task is to generate a list of realistic, diverse questions a user might ask about this dashboard when interacting with a chatbot. These should include questions about:

1. Trends over time (e.g., increasing/decreasing values)
2. Comparisons between actual and target values
3. Operational suggestions (e.g., open more cash desks?)
4. Time-specific questions (e.g., peak hours, changes by hour)
5. Sales analysis and forecasts
6. Customer traffic patterns
7. Product performance and anomalies

Make sure the questions reflect curiosity, decision-making needs, or requests for clarification. Keep them clear and natural, as if a user were typing them to a chatbot.

Here is the dashboard data:
{json.dumps(dashboard_json)}

Please return 30 to 40 unique, relevant, and well-phrased questions.

"""

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.4,
    max_tokens=2200
)

# To access the response text:
generated_text = response.choices[0].message.content

# === Step 4: Extract and clean questions ===
questions = [q.strip("1234567890. ").strip() for q in generated_text.split("\n") if q.strip()]

# === Step 5: Generate embeddings for each question ===
batch_size = 10
new_followups = []

for i in range(0, len(questions), batch_size):
    batch = questions[i:i + batch_size]
    embed_response = client.embeddings.create(  # Note the change here
        input=batch,
        model="text-embedding-ada-002"
    )
    for question, embedding in zip(batch, embed_response.data):
        new_followups.append({
            "followup_text": question,
            "embedding": embedding.embedding  # Note the change here
        })

# === Step 6: Append to "dashboard_followups" list in your JSON ===
if "dashboard_followups" not in data:
    data["dashboard_followups"] = []

data["dashboard_followups"].extend(new_followups)

# === Step 7: Save the updated file ===
with open("data/followup_embeddings_list.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Added {len(new_followups)} questions to 'dashboard_followups'")