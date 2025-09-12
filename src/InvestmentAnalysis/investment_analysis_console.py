"""
investment_analysis_console.py - analysis of a company's financial & stock performance
    using Yahoo Finance and then leveraging an LLM to analyze performance viz-a-viz
    its peers and make a final investment recommendation.
    Supports LLMs from OpenAI (paid), Anthropic (paid), and Gemini.
    Groq is also supported, but it has issues with context window size when making
    final recommendations. This is the console based driver program.

Author: Manish Bhobe
My experiments with Python, ML and Generative AI.
Code is meant for illustration purposes ONLY. Use at your own risk!
Author is not liable for any damages arising from direct/indirect use of this code.
"""

from rich.console import Console
import yfinance as yf
from textwrap import dedent

from agno.utils.log import logger

from agents.investment_analysis_agent import investment_analysis_agent


def generate_investment_analysis(symbol: str):
    prompt = f"Generate investment analysis for {symbol}"
    return investment_analysis_agent.print_response(prompt, stream=True)


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


console = Console()

# try for various companies (some sample tickers below)
# refer to the Yahoo! Finance website for ticker symbols
# -- on NY Stock Exchange (NYSE)
# AAPL - Apple
# JPM - JP Morgan Chase
# LLY - Eli Lilly
# XOM - Exxon Mobil
# -- on NSE
# RELIANCE.NS - Reliance Industries
# TCS.NS - TCS
# HDFCBANK.NS - HDFC Bank
# -- on London Stock Exchange (LSE)
# OA4J.L - AstraZeneca Plc
# ULVR.L - Unilever Plc
# VOD.L - Vodafone Group Plc
while True:
    stock_symbol = console.input(
        "[green]Enter stock symbol (as on Yahoo! Finance):[/green] "
    )
    stock_symbol = stock_symbol.strip().upper()
    if stock_symbol.lower() in ["bye", "quit", "exit"]:
        break
    if not is_valid_stock_symbol(stock_symbol):
        console.print(f"[red]{stock_symbol} does not appear to be a valid symbol!")
        continue

    generate_investment_analysis(stock_symbol.upper())
