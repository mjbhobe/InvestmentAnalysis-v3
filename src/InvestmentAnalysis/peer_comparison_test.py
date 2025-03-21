from rich.console import Console
from textwrap import dedent
from agents.peer_comparison_agent import peers_comparison_agent


def generate_peer_comparison(symbol: str):
    prompt = dedent(
        f"""
        Generate peer comparison for {symbol}
    """
    )
    return peers_comparison_agent.print_response(prompt, stream=True)


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
    generate_peer_comparison(stock_symbol.upper())
