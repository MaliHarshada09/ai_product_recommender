import streamlit as st
import os
import time
import json
from google import genai
import datetime
from dotenv import load_dotenv
import subprocess

# -------------------------------
# LOAD API KEY NeW
# -------------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not found in .env file")

# -------------------------------
# CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Insurance Product AI",
    page_icon="🛡️",
    layout="centered"
)

client = genai.Client(api_key=API_KEY)

DOCS_FOLDER = "./data/"
KB_FILE = "kb_summary.txt"

if not os.path.exists(DOCS_FOLDER):
    os.makedirs(DOCS_FOLDER)

# -------------------------------
# RATE LIMIT CONTROL
# -------------------------------
MAX_RETRIES = 3
RETRY_DELAY = 6
REQUEST_COOLDOWN = 10


def generate_with_retry(call_fn):
    """Retry Gemini call on 429 errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return call_fn()
        except Exception as e:
            if "429" in str(e):
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise Exception(
                        "API quota reached. Please wait a few minutes and try again."
                    )
            else:
                raise e


def check_cooldown():
    """Prevent rapid repeated requests."""
    last_time = st.session_state.get("last_request_time", 0)
    current_time = time.time()

    if current_time - last_time < REQUEST_COOLDOWN:
        remaining = int(REQUEST_COOLDOWN - (current_time - last_time))
        st.warning(f"Please wait {remaining} seconds before running again.")
        return False

    st.session_state["last_request_time"] = current_time
    return True


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def load_kb_summary():
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return None



def format_insurance_output(data):
    output = []

    # Customer Profile
    profile = data.get("customer_profile", {})
    output.append("CUSTOMER PROFILE")
    output.append("-" * 30)
    output.append(f"Age: {profile.get('age', 'N/A')}")
    output.append(f"Gender: {profile.get('gender', 'N/A')}")
    output.append(f"Location: {profile.get('location', 'N/A')}")
    output.append(f"Health Conditions: {profile.get('health_conditions', 'N/A')}")
    output.append("")

    # Insurance Needs
    output.append("INSURANCE NEEDS")
    output.append("-" * 30)
    for need in data.get("insurance_needs", []):
        output.append(f"- {need}")
    output.append("")

    # Recommended Products
    output.append("RECOMMENDED PRODUCTS")
    output.append("=" * 40)

    for i, product in enumerate(data.get("recommended_products", []), start=1):
        output.append(f"\n{i}) {product.get('Product Name', 'N/A')}")
        output.append("-" * 40)
        output.append(f"Target Customer: {product.get('Target Customer', 'N/A')}")
        output.append("")

        # Key Benefits
        output.append("Key Benefits:")
        for benefit in product.get("Key Benefits", []):
            output.append(f"  • {benefit}")

        # Coverage Highlights
        output.append("\nCoverage Highlights:")
        for highlight in product.get("Coverage Highlights", []):
            output.append(f"  • {highlight}")

        # Ideal Use Case
        output.append("\nIdeal Use Case:")
        output.append(f"  {product.get('Ideal Use Case', 'N/A')}")
        output.append("")

    return "\n".join(output)


# -------------------------------
# APP UI
# -------------------------------
st.title("🛡️ Insurance Product Recommendation AI")
st.markdown("""
This tool analyzes customer calls and recommends insurance products.

**Steps:**
1. Place product PDFs or text files in the `data` folder.
2. Click **Rebuild Knowledge Base**.
3. Upload a customer audio call and run analysis.
""")

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🔄 Rebuild Knowledge Base"):
        with st.spinner("Building knowledge base summary..."):
            try:
                subprocess.run(["python", "build_summary.py"], check=True)
                st.success("Knowledge base rebuilt successfully.")
            except Exception as e:
                st.error(f"Failed to build knowledge base: {e}")

    if os.path.exists(KB_FILE):
        st.success("Knowledge base ready.")
    else:
        st.warning("Knowledge base not built yet.")


# -------------------------------
# MAIN FLOW (NO COLUMNS)
# -------------------------------
uploaded_audio = st.file_uploader(
    "Upload Customer Call",
    type=['mp3', 'wav', 'm4a', 'aac']
)

if uploaded_audio:
    st.audio(uploaded_audio)

    kb_summary = load_kb_summary()

    if not kb_summary:
        st.warning("⚠️ Please rebuild the knowledge base first.")
    else:
        if st.button("🚀 Run AI Analysis"):

            if not check_cooldown():
                st.stop()

            temp_path = f"temp_{uploaded_audio.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())

            try:
                with st.status("Analyzing...", expanded=True) as status:
                    st.write("Uploading audio...")
                    customer_audio = client.files.upload(file=temp_path)

                    current_date = datetime.date.today().strftime("%B %d, %Y")

                    full_prompt = f"""TODAY'S DATE: {current_date}
                KNOWLEDGE BASE: {kb_summary}
                
                You are an AI-powered insurance advisor assistant.

Your task is to analyze a customer conversation transcript (converted from audio) 
and recommend suitable insurance products in a safe, compliant, and transparent manner.

You must strictly follow the guardrails and instructions below.

-------------------------
CORE INSTRUCTIONS
-------------------------

1. Extract only the information that is explicitly mentioned in the transcript.
2. Do NOT guess, hallucinate, or invent any customer details.
3. If important information is missing, list it under "missing_information".
4. Make recommendations based only on:
   - Customer profile
   - Stated needs
   - Risk appetite
   - Affordability (if known)

5. Recommend a maximum of 3 products from KNOWLEDGE BASE .

-------------------------
GUARDRAILS
-------------------------

SAFETY & COMPLIANCE
- Do not provide legal, tax, or medical advice.
- Do not make guaranteed return claims.
- Do not promise financial outcomes.
- Use neutral, advisory language (e.g., "may be suitable", "could help").
- If the transcript includes sensitive or inappropriate content, ignore it.

HALLUCINATION CONTROL
- Only use information present in the transcript.
- If age, income, or goals are not mentioned, do not assume values.
- Clearly state any assumptions separately.

AFFORDABILITY CHECK
- If income is mentioned:
  - Do not recommend premium-heavy or high-investment products that seem unrealistic.
- If income is not mentioned:
  - Avoid suggesting specific premium amounts.
  - Use general coverage guidance only.

RISK PROFILE HANDLING
- If customer shows low risk tolerance:
  - Avoid investment-linked or high-risk plans.
- If risk appetite is unclear:
  - Recommend protection-first products (e.g., term or health insurance).

TRANSPARENCY
- Clearly explain why each product is recommended.
- Use simple, customer-friendly language.

-------------------------
TASK STEPS
-------------------------

1. Extract key customer details:
   - Age
   - Marital status
   - Dependents
   - Occupation
   - Income level
   - Financial goals
   - Risk appetite
   - Existing insurance
   - Key concerns

2. Identify primary insurance needs.

3. Recommend up to 3 suitable insurance product types from KNOWLEDGE BASE.

4. For each recommendation, provide:
   - Product type
   - Reason for recommendation
   - Key benefits
   - Suggested coverage approach (not exact premium unless income is known)

5. List:
   - Missing information
   - Assumptions (if any)
"""

                    response = generate_with_retry(
                        lambda: client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[full_prompt, customer_audio],
                            config={
                                "temperature": 0,
                                "top_p": 1,
                                "top_k": 1,
                                "max_output_tokens": 800
                                
                            }
                        )
                    )

                    status.update(
                        label="Analysis Complete!",
                        state="complete",
                        expanded=False
                    )

                st.subheader("📋 AI Recommendation Report")

                # formatted_output = format_insurance_output(response.text)
                
                
                st.markdown(response.text)

            except Exception as e:
                st.error(f"Analysis Error: {e}")

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)


# -------------------------------
# FOOTER
# -------------------------------
if os.path.exists(KB_FILE):
    st.divider()
    st.caption("Knowledge base loaded from local summary file.")
