"""
investment_analysis_streamlit.py - analysis of a company's financial & stock performance
    using Yahoo Finance and then leveraging an LLM to analyze performance viz-a-viz
    its peers and make a final investment recommendation.
    Supports LLMs from OpenAI (paid), Anthropic (paid), and Gemini.
    Groq is also supported, but it has issues with context window size when making
    final recommendations. This is the streamlit based front-end for the analysis

Author: Manish Bhobe
My experiments with Python, ML and Generative AI.
Code is meant for illustration purposes ONLY. Use at your own risk!
Author is not liable for any damages arising from direct/indirect use of this code.
"""

import numpy as np
import streamlit as st
import yfinance as yf
from typing import Iterator

from agno.agent import RunResponse
from agno.utils.log import logger
from agents.investment_analysis_agent import investment_analysis_agent

# Page configuration
st.set_page_config(
    page_title="Investment Analyzer - Ver 3.0", page_icon="📊", layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    .stock-input {
        max-width: 400px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "analysis_generated" not in st.session_state:
    st.session_state.analysis_generated = False


def is_valid_stock_symbol(symbol: str) -> bool:
    # try:
    #     ticker = yf.Ticker(symbol.upper())
    #     # try to get info, should raise exception if invalid
    #     _ = ticker.info
    #     return True
    # except Exception as e:
    #     logger.fatal(f"ERROR: {symbol.upper()} is not a valid stock symbol.")
    #     return False
    ticker = yf.Ticker(symbol.upper())
    # try to download 1 days stock price
    hist = ticker.history(period="1d")
    # if I get some history, symbol is valid!
    if hist.empty:
        logger.fatal(f"ERROR: {symbol.upper()} is not a valid stock symbol.")
        return False
    return True


def generate_investment_analysis(symbol: str, agent):
    prompt = f"Generate investment analysis for {symbol}"
    # return agent.print_response(prompt, stream=True)
    response: RunResponse = agent.run(prompt, markdown=True)
    return response.content, response.metrics
    # response_stream : Iterator[RunResponse] = agent.run(prompt, markdown=True, stream=True)


# Main UI
st.markdown(
    "<h1 class='main-header'>Investment Analyzer - Ver 3.0📈</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <center>
    This multi-agent team based application does an investment analysis for publicly 
    traded companies and comes up with an overall recommendation on the long term 
    investment potential.
    <br/>
    <div style='color: #888;'>     
    <small>   
    This version combines financial analysis, peer comparison and sentiment analysis to
    come up with an overall recommendation on the long term investment potential of the company. 
    </small>
    </div>
    </center>
    <p/>
""",
    unsafe_allow_html=True,
)

# Stock symbol input
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        stock_symbol = st.text_input(
            "Enter Stock Symbol to begin (should be same as used on Yahoo! Finance website):",
            placeholder="e.g., TCS.NS",
            key="stock_input",
        )
    with col2:
        col2.markdown(f"<div style='height: 28px;'></div>", unsafe_allow_html=True)
        analyze_button = st.button("Analyze", type="primary")

# Analysis section
if analyze_button and stock_symbol:
    # check if user has entered a valid stock symbol
    if not is_valid_stock_symbol(stock_symbol):
        st.markdown(
            f"""<div style='color: red;'>
            Invalid stock symbol: {stock_symbol.upper()}. Please enter valid symbol to proceed!
            </div>""",
            unsafe_allow_html=True,
        )
        st.stop()

    try:
        stock_symbol = stock_symbol.upper()
        company_name = yf.Ticker(stock_symbol).info.get("longName")
        with st.spinner(
            f"Generating investment analysis for {company_name} ({stock_symbol})..."
        ):
            analysis, metrics = generate_investment_analysis(
                stock_symbol, investment_analysis_agent
            )

        st.success("Analysis completed!")

        # Display analysis in an expandable container
        with st.expander("View Detailed Analysis", expanded=True):
            # metrics is a dict like this
            # metrics = {
            #   "input_tokens":[input tokens per agent],
            #   "output_tokens":[input tokens per agent],
            #   "total_tokens":[input tokens + output tokens per agent],
            #   "time":[total time per agent],
            #   }
            input_tokens = np.array(metrics["input_tokens"]).sum()
            output_tokens = np.array(metrics["output_tokens"]).sum()
            total_tokens = np.array(metrics["total_tokens"]).sum()
            total_time = np.array(metrics["time"]).sum()
            # st.markdown(f"**Metrics**: {metrics}")
            st.markdown(
                f"**Token Count** -> Input: {input_tokens:5d} - Output: {output_tokens:5d} - Total: {total_tokens:5d} | **Time Taken**: {total_time:2f}s"
            )
            st.markdown(analysis)

        st.session_state.analysis_generated = True

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <small><b>Developed by:</b> Manish Bhobé • <b>Powered by:</b> yfinance | Agno (for agents) | Google Gemini (LLM) | Streamlit (UI) • <b>For educational purposes only!</b></small>
    </div>
    """,
    unsafe_allow_html=True,
)
