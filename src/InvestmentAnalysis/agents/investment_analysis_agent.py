"""
investment_analysis_agent.py - an agent that co-ordinates the work of financial
    analysis, sentiment analysis etc. agents to come up with an overall recommendation
    of the long term investment potential of a company.

Author: Manish Bhobé

My experiments with AI, ML and Generative AI
Code is meant to be used for educational purposes only!

**WARNING**
At no point is this code/to be used as a replacement for sound
financial investment advise from a Financial expert.
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

from agents.financial_analysis_agent import financial_analysis_agent
from agents.sentiment_analysis_agent import sentiment_analysis_agent


# Load environment variables and configure Gemini
if os.environ.get("STREAMLIT_CLOUD"):
    # when deploying to streamlit, read from st.secrets
    os.environ["GOOGLE_API_KEY"] = st.secrets("GOOGLE_API_KEY")
else:
    # running locally - load from .env file
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

config_file_path = (
    pathlib.Path(__file__).parent.parent / "config/investment_analysis_prompts.yaml"
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


investment_analysis_agent = Agent(
    name="Investment Analysis Agent",
    model=Gemini(id="gemini-2.0-flash"),
    team=[financial_analysis_agent, sentiment_analysis_agent],
    goal=dedent(
        """
        Based on the financial analysis, sentiment analysis 
        come up with an overall recommendation for the long term investment potential
        of a company to potential investors.
    """
    ),
    description=dedent(config["prompts"]["system_prompt"]),
    instructions=dedent(config["prompts"]["investment_analysis_instructions"]),
    expected_output=dedent(config["prompts"]["expected_output_format"]),
    markdown=True,
    show_tool_calls=True,
    debug_mode=True,
)
