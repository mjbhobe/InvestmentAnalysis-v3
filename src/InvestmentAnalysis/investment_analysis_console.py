from rich.console import Console
from textwrap import dedent
from agents.investment_analysis_agent import investment_analysis_agent


def generate_investment_analysis(symbol: str):
    prompt = dedent(
        f"""
        Generate investment analysis for {symbol}
    """
    )
    return investment_analysis_agent.print_response(prompt, stream=True)


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
    stock_symbol = stock_symbol.strip()
    if stock_symbol.lower() in ["bye", "quit", "exit"]:
        break
    generate_investment_analysis(stock_symbol.upper())
