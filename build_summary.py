import os
from google import genai
from dotenv import load_dotenv
from pypdf import PdfReader

# Load API key
load_dotenv()
API_KEY = os.getenv("API_KEY")

client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-2.0-flash"

DOCS_FOLDER = "./data/"
OUTPUT_FILE = "kb_summary.txt"


def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


all_text = ""

for file in os.listdir(DOCS_FOLDER):
    path = os.path.join(DOCS_FOLDER, file)

    if file.endswith(".pdf"):
        all_text += extract_text_from_pdf(path)
    elif file.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            all_text += f.read()

if not all_text.strip():
    raise ValueError("No text found in documents.")

print("Generating knowledge base summary...")

prompt = f"""
You are an insurance domain expert.

Summarize the following product documents into a structured knowledge base.

For each product include:
- Product Name
- Target customer
- Key benefits
- Coverage highlights
- Ideal use case

Keep it concise and structured.

DOCUMENTS:
{all_text}
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(response.text)

print("Knowledge base summary saved to kb_summary.txt")
