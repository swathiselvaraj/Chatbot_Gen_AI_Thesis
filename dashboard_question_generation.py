import openai
import json
from openai import OpenAI

# Initialize the client
client = OpenAI(api_key="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")  # Replace with your actual API key

# === Step 1: Load existing JSON file ===
with open("data/followup_embeddings_list.json", "r") as f:
    data = json.load(f)

# === Step 2: Load your dashboard JSON file ===
with open("data/dashboard_data.json", "r") as f:
    dashboard_json = json.load(f)

# === Step 3: Generate questions from GPT ===
prompt = f"""
You are a helpful assistant.

Based on the following store dashboard data in JSON format, generate 50 realistic and varied questions a human might ask. 
Keep them concise and relevant to store management, customer traffic, waiting time, staffing, and sales trends.

Dashboard JSON:
{json.dumps(dashboard_json)}
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