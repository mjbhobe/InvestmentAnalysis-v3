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
    generate_investment_analysis(stock_symbol.upper())
