"""
peers_comparison_agent.py - an agent that does a comparative analysis
    of all performance ratios between a company & its peers

Author: Manish Bhobé

My experiments with AI, ML and Generative AI
Code is meant to be used for educational purposes only!

**WARNING**
- This code is intended for educational purposes only!
- It is not meant to replace professional financial advisors.
- The author does not endorse using this as a replacement for sound financial
  advise from an experienced financial advisor.
"""

import os
from dotenv import load_dotenv
from textwrap import dedent
import yaml
import pathlib
import streamlit as st

from agno.agent import Agent
from agno.models.google import Gemini
import google.generativeai as genai

from tools.financial_analysis_tools import FinancialAnalysisTools
from tools.peer_comparison_tools import PeerComparisonTools

from utils.llm import google_gemini_llm

# Load environment variables and configure Gemini
if os.environ.get("STREAMLIT_CLOUD"):
    # when deploying to streamlit, read from st.secrets
    os.environ["GOOGLE_API_KEY"] = st.secrets("GOOGLE_API_KEY")
else:
    # running locally - load from .env file
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

config_file_path = (
    pathlib.Path(__file__).parent.parent / "config/peer_comparison_prompts.yaml"
)
assert (
    config_file_path.exists()
), f"FATAL ERROR: configuration file {config_file_path} does not exist!"

# load prompts from config/prompts.yaml
# externalizing the prompts from code.
config = None
with open(str(config_file_path), "r") as f:
    config = yaml.safe_load(f)

if config is None:
    raise RuntimeError(
        f"FATAL ERROR: unable to read from configuration file at {config_file_path}"
    )


peers_comparison_agent = Agent(
    name="Peers Comparison Agent",
    model=Gemini(id="gemini-2.0-flash", temperature=0.3),
    # model=google_gemini_llm,
    tools=[
        # use just the company info tool from Financial Analysis toolkit
        FinancialAnalysisTools(liquidity_ratios=False, company_info=True),
        PeerComparisonTools(),
    ],
    goal=dedent(
        """
        Compare the latest financial performance of the company with its peers as 
        well as the industry benchmarks and come up with a comparison analysis of 
        how the company stands viz-a-viz its competitors and within the industry.
        """
    ),
    description=dedent(config["prompts"]["system_prompt"]),
    instructions=dedent(config["prompts"]["peer_comparison_instructions"]),
    expected_output=dedent(config["prompts"]["expected_output_format"]),
    markdown=True,
    show_tool_calls=True,
    debug_mode=True,
)

