"""
sentiment_analysis_agent.py - an agent that does an overall sentiment analysis
    of recent market news for a company stock and reverts with overall sentiment
    (one of Positive or Negative or Neutral) and an average sentiment score.

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
from agno.utils.log import logger
from agno.models.google import Gemini
import google.generativeai as genai

from tools.sentiment_analysis_tools import SentimentAnalysisTools

# Load environment variables and configure Gemini
if os.environ.get("STREAMLIT_CLOUD"):
    # when deploying to streamlit, read from st.secrets
    logger.debug(
        "Detected that I am running in Streamlit cloud. Fetching API key from st.secrets()"
    )
    os.environ["GOOGLE_API_KEY"] = st.secrets("GOOGLE_API_KEY")
else:
    # running locally - load from .env file
    logger.debug("Detected stand-alone app. Fetching API key from .env file")
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

config_file_path = (
    pathlib.Path(__file__).parent.parent / "config/sentiment_analysis_prompts.yaml"
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


sentiment_analysis_agent = Agent(
    name="Sentiment Analysis Agent",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[SentimentAnalysisTools()],
    goal=dedent(
        """
        Analyse latest company market news and determine the the overall market sentiment, 
        which will serve as an input for a recommendation of the long term investment potential 
        of the company.
    """
    ),
    description=dedent(config["prompts"]["system_prompt"]),
    instructions=dedent(config["prompts"]["sentiment_analysis_prompt"]),
    expected_output=dedent(config["prompts"]["expected_output_format"]),
    markdown=True,
    show_tool_calls=True,
    debug_mode=True,
)
