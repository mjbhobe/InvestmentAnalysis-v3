"""
llm.py - module where we load our API keys and crete our LLM
    (you can add more LLMs here if you want)

Author: Manish Bhobe
My experiments with Python, ML and Generative AI.
Code is meant for illustration purposes ONLY. Use at your own risk!
Author is not liable for any damages arising from direct/indirect use of this code.
"""

import os
from dotenv import load_dotenv

from agno.models.google import Gemini
import google.generativeai as genai

# load API key from st.secrets or .env file
if os.environ.get("STREAMLIT_CLOUD"):
    # when deploying to streamlit, read from st.secrets
    os.environ["GOOGLE_API_KEY"] = st.secrets("GOOGLE_API_KEY")
else:
    # running locally - load from .env file
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

ONE_K = 1024

# this is the LLM we'll use across all our agents
google_gemini_llm = Gemini(
    id="gemini-2.0-flash", temperature=0.0, max_output_tokens=5 * ONE_K
)
