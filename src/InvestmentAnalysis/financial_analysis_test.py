from rich.console import Console
from rich.markdown import Markdown
from textwrap import dedent
import yfinance as yf
from agents.financial_analysis_agent import financial_analysis_agent
from tools.ratios import (
    get_liquidity_ratios,
    get_profitability_ratios,
    get_efficiency_ratios,
    get_valuation_ratios,
    get_leverage_ratios,
    get_performance_and_growth_metrics,
)
from agno.utils.log import logger


def generate_financial_analysis(symbol: str):
    prompt = f"Generate financial analysis for {symbol}"
    return financial_analysis_agent.print_response(prompt, stream=True)


console = Console()


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
    if not is_valid_stock_symbol(stock_symbol):
        console.print(f"[red]{stock_symbol} does not appear to be a valid symbol!")
        continue

    # generate_financial_analysis(stock_symbol.upper())
    # let's do direct ratios calculation
    liquidity_ratios = get_liquidity_ratios(stock_symbol.upper())
    console.print("[green]Liquidity Ratios:[/green]")
    console.print(Markdown(liquidity_ratios))

    profitability_ratios = get_profitability_ratios(stock_symbol.upper())
    console.print("[green]Profitability Ratios:[/green]")
    console.print(Markdown(profitability_ratios))

    efficiency_ratios = get_efficiency_ratios(stock_symbol.upper())
    console.print("[green]Efficiency Ratios:[/green]")
    console.print(Markdown(efficiency_ratios))

    valuation_ratios = get_valuation_ratios(stock_symbol.upper())
    console.print("[green]Valuation Ratios:[/green]")
    console.print(Markdown(valuation_ratios))

    leverage_ratios = get_leverage_ratios(stock_symbol.upper())
    console.print("[green]Leverage Ratios:[/green]")
    console.print(Markdown(leverage_ratios))

    performance_and_growth_metrics = get_performance_and_growth_metrics(
        stock_symbol.upper()
    )
    console.print("[green]Performance and Growth Metrics:[/green]")
    console.print(Markdown(performance_and_growth_metrics))
