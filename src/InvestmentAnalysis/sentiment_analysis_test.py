from rich.console import Console
from textwrap import dedent
from agents.sentiment_analysis_agent import sentiment_analysis_agent


def generate_sentiment_analysis(symbol: str):
    prompt = dedent(
        f"""
        Generate sentiment analysis for {symbol}
    """
    )
    return sentiment_analysis_agent.print_response(prompt, stream=True)


console = Console()

# try for various companies
# AAPL - Apple
# AMZN - Amazon
# MSFT - Microsoft Corp
# RELIANCE.NS - Reliance Industries
# TCS.NS - TCS
while True:
    stock_symbol = console.input(
        "[green]Enter stock symbol (as on Yahoo! Finance):[/green] "
    )
    stock_symbol = stock_symbol.strip()
    if stock_symbol.lower() in ["bye", "quit", "exit"]:
        break
    generate_sentiment_analysis(stock_symbol.upper())
