import openai
import json
from PIL import Image
import base64

# Set your OpenAI API key
openai.api_key = 'open ai key'

# Load and encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Define the image path
image_path = "dashboard_image.png"  # Replace with your image filename
base64_image = encode_image(image_path)

# Send image to GPT-4 Vision
response = openai.chat.completions.create(
    model="gpt-4o",
    response_format={ "type": "json_object" },  # ← Force JSON output
    messages=[
        {
            "role": "system",
            "content": "You are a precise data extraction tool that returns JSON from business dashboards. Only return the raw structured data in JSON format."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all textual and numerical information from this store management dashboard and return as a structured JSON."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }
    ],
    max_tokens=2000
)

# Extract and parse GPT response
extracted_text = response.choices[0].message.content.strip()

# Save to a JSON file
output_file = "dashboard_data.json"
with open(output_file, "w") as f:
    try:
        json_data = json.loads(extracted_text)
        json.dump(json_data, f, indent=4)
        print(f"✅ Data saved to {output_file}")
    except json.JSONDecodeError:
        print("⚠️ The response was not valid JSON. Here's the raw output:\n")
        print(extracted_text)
