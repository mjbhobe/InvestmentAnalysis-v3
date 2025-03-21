import json
import yfinance
import numpy as np
import pandas as pd
from typing import Dict, List

from agno.tools import Toolkit
from agno.utils.log import logger


try:
    import yfinance as yf
except ImportError:
    raise ImportError(
        "`yfinance` not installed. Please install using `pip install yfinance`."
    )

pd.set_option("future.no_silent_downcasting", True)


# created similar to Agno's YFinanceTools
class PeersAnalysisTools(Toolkit):
    def __init__(
        self,
    ):
        super().__init__(name="peers_analyis_tools")

        # register functions
        logger.debug("Registering get_performance_ratios function")
        self.register(self.get_performance_ratios)

    def __calculate_performance_ratios(self, symbol: str) -> Dict[str, float]:
        """
        Use this function to get all performance ratios for a company for
        the latest financial year, which can be used for comparison with peers
        (internal helper function, used by registered get_performance_ratios function).

        Args:
            symbol (str): The stock symbol.

        Returns:
            str: JSON containing company profile and overview.
        """
        try:
            logger.debug("Calculaying performance ratios for {symbol}")

            ticker = yf.Ticker(symbol)

            balance_sheet = ticker.balance_sheet.transpose().sort_index(ascending=True)
            financials = ticker.financials.transpose().sort_index(ascending=True)
            income_stmt = ticker.income_stmt.transpose().sort_index(ascending=True)
            cash_flow = ticker.cash_flow.transpose().sort_index(ascending=True)

            # get numbers for latest financial year
            current_assets = balance_sheet["Current Assets"].iloc[-1]
            current_liabilities = balance_sheet["Current Liabilities"].iloc[-1]
            # some companies do not report inventory (e.g. Reliance does, Persistent may not)
            inventory_fields_exist = "Inventory" in ticker.balance_sheet.index

            revenue = financials["Total Revenue"].iloc[-1]
            operating_income = financials["Operating Income"].iloc[-1]
            net_income = financials["Net Income"].iloc[-1]
            total_assets = balance_sheet["Total Assets"].iloc[-1]
            shareholder_equity = balance_sheet["Stockholders Equity"].iloc[-1]
            ebit = income_stmt["EBIT"].iloc[-1]
            current_liabilities = balance_sheet["Current Liabilities"].iloc[-1]

            cost_of_goods_sold = financials["Cost Of Revenue"].iloc[-1]

            market_cap = ticker.info["marketCap"]
            shareholder_equity = balance_sheet["Stockholders Equity"].iloc[-1]
            ebidta = financials.get(
                "EBIDTA",
                financials["Operating Income"]
                + financials.get("Depreciation & Amortization", 0),
            ).iloc[-1]
            total_debt = balance_sheet["Total Debt"].iloc[-1]
            cash_equivalents = balance_sheet["Cash And Cash Equivalents"].iloc[-1]
            ev = market_cap + total_debt - cash_equivalents

            total_debt = balance_sheet["Total Debt"].iloc[-1]
            interest_expense = financials["Interest Expense"].iloc[-1]

            ratios = {}

            # liquidity ratios
            ratios["Current Ratio"] = current_assets / current_liabilities
            ratios["Quick Ratio"] = (
                (current_assets - balance_sheet["Inventory"].iloc[-1])
                / current_liabilities
                if inventory_fields_exist
                else np.nan
            )
            ratios["Cash Ratio"] = (
                balance_sheet["Cash And Cash Equivalents"].iloc[-1]
                / balance_sheet["Current Liabilities"].iloc[-1]
            )

            # profitability ratios
            ratios["Return on Equity (RoE)"] = net_income / shareholder_equity
            ratios["Return on Assets (RoA)"] = net_income / total_assets
            ratios["Return on Capital Employed (RoCE)"] = ebit / (
                total_assets - current_liabilities
            )
            ratios["Net Profit Margin"] = net_income / revenue
            ratios["Operating Margin"] = operating_income / revenue

            # efficiency ratios
            ratios["Asset Turnover"] = revenue / total_assets

            if inventory_fields_exist:
                inventory = balance_sheet["Inventory"]
                # instead of rolling mean, we get just the mean
                # average_inventory = inventory.rolling(2).mean()
                average_inventory = inventory.mean()

            ratios["Inventory Turnover"] = (
                cost_of_goods_sold / average_inventory
                if inventory_fields_exist
                else np.nan
            )

            # valuation ratios
            ratios["Price-to-Earnings (P/E)"] = ticker.info["trailingPE"]
            ratios["Price-to-Sales (P/S)"] = market_cap / revenue
            ratios["Price-to-Book (P/B)"] = market_cap / shareholder_equity
            ratios["EV/EBIDTA"] = ev / ebidta

            # leverage ratios
            ratios["Debt-to-Equity (D/E)"] = total_debt / shareholder_equity
            ratios["Interest Coverage"] = ebit / interest_expense

            # performance & growth metrics
            ratios["Revenue Growth (%)"] = (
                financials["Total Revenue"].pct_change().iloc[-1] * 100.0
            )
            ratios["EBIT Growth (%)"] = financials["EBIT"].pct_change().iloc[-1] * 100.0
            ratios["Net Profit Margin (%)"] = (
                financials["Net Income"] / financials["Total Revenue"]
            ).iloc[-1] * 100.0
            shares_outstanding = balance_sheet["Ordinary Shares Number"]
            eps = financials["Net Income"] / shares_outstanding
            ratios["EPS Growth (%)"] = eps.pct_change().iloc[-1] * 100.0

            ratios["EPS"] = (financials["Net Income"] / shares_outstanding).iloc[-1]
            ratios["Debt-to-Equity"] = (
                balance_sheet["Total Debt"] / balance_sheet["Stockholders Equity"]
            ).iloc[-1]

            ratios["Free Cash Flow"] = cash_flow["Free Cash Flow"].iloc[-1]
            ratios["FCF Growth (%)"] = (
                cash_flow["Free Cash Flow"].pct_change().iloc[-1] * 100.0
            )

            return ratios
        except Exception as e:
            return f"Error fetching company profile for {symbol}: {e}"

    def get_performance_ratios(self, symbols: List[str]) -> str:
        """
        Use this function to get all performance ratios for a company for
        the latest financial year, which can be used for comparison with peers.

        Args:
            symbols (List[str]): List of stock symbols for which comparison is needed

        Returns:
            str: JSON containing company profile and overview.
        """
        try:
            logger.debug("Fetching performance ratios for {symbols}")

            ratios = {}
            for symbol in symbols:
                ratios[symbol] = self.__calculate_performance_ratios(symbol)

            # return json.dumps(ratios)
            return f"\n{pd.DataFrame(ratios).to_markdown()}\n"
        except Exception as e:
            return f"Error fetching company profile for {symbols}: {e}"


# --- test harness ---
if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    # create an instance of the class
    peers_analysis_tools = PeersAnalysisTools()

    # test the function
    # assume TCS.NS is the company we are analyzing & the rest are peers
    peers = ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "PERSISTENT.NS"]

    console.print(peers_analysis_tools.get_performance_ratios(peers))
