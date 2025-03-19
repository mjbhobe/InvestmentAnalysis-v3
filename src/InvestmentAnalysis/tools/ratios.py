"""
ratios.py - calculates various financial ratios for a company

These utility functions can be used by a financial analyst to analyze
a company stock as a potential investment target.

Author: Manish Bhobe
My experiments with Python, AI and Generative AI
Code is meant for learning purposes ONLY!
"""

import warnings

warnings.filterwarnings("ignore")

import os
import pathlib
import numpy as np
import pandas as pd
import json
import yfinance as yf
from agno.utils.log import logger


# display tweaks
# Set Pandas to display float values with 4 decimal places
pd.options.display.float_format = "{:.4f}".format
pd.options.display.max_colwidth = 1024  # make wide before wrapping
pd.options.display.max_rows = 100  # before truncation

# Set NumPy to display float values with 4 decimal places
np.set_printoptions(precision=4, suppress=True)


def is_valid_ticker(symbol: str) -> bool:
    """
    Checks if symbol is a valid ticker symbol. For it to be valud, yf.Ticker(symbol).info
    should not raise an exception!

    Args:
        symbol(str): a ticker symbol (such as "AAPL" or "PERSISTENT.NS")
        (please visit Yahoo Finance website to get valid symbol of company)
    """
    try:
        ticker = yf.Ticker(symbol)
        return "shortName" in ticker.info
    except Exception as e:
        return False


def get_liquidity_ratios(symbol: str) -> pd.DataFrame:
    """
    Use this function to get the end-of-financial-year values for liquidity ratios for a given symbol.
        - Current Ratio = Current Assets / Current Liabilities
        - Quick Ratio = (Current Assets - Inventory) / Current Liabilities
        - Cash Ratio = (Cash & Equivalents) / Current Liabilities

    Liquidity ratios help determine short-term stability of a company. These can be interpreted
    as follows (though the interpretation could differ person-to-person)
        - Current Ratio: > 1.5 is good (indicates ability to cover short-term liabilities). < 1 risky!
        - Quick Ratio: > 1 is good (measures liquidity excluding inventory) [conservative number]
        - Cash Ratio: Company's pure cash position/pile - higher the better.

    Args:
        symbol (string) - the stock symbol

    Returns:
        str: markdown version of the liquidity ratios with rows ordered by date
           and columns having values for each of the liquidity ratio
    """
    
    logger.debug(f"Calculatig liquidity ratios for {symbol}")

    ticker = yf.Ticker(symbol)
    balance_sheet = ticker.balance_sheet.transpose().sort_index(ascending=True)

    current_assets = balance_sheet["Current Assets"]
    current_liabilities = balance_sheet["Current Liabilities"]
    # some companies may not report inventory (e.g. Reliance does, Persistent does not)
    inventory_fields_exist = "Inventory" in ticker.balance_sheet.index

    ratios = {}
    ratios["Current Ratio"] = current_assets / current_liabilities
    if inventory_fields_exist:
        ratios["Quick Ratio"] = (
            current_assets - balance_sheet["Inventory"]
        ) / current_liabilities

    ratios["Cash Ratio"] = (
        balance_sheet["Cash And Cash Equivalents"]
        / balance_sheet["Current Liabilities"]
    )


    ret =  f"\n{pd.DataFrame(ratios).to_markdown()}\n"
    logger.debug(ret)
    return ret


def get_profitability_ratios(symbol: str) -> pd.DataFrame:
    """
    Use this function to get the end-of-financial-year values for profitability ratios for a given symbol.
        - Return on Equity (RoE) = Net Income / Shareholder's Equity
        - Return on Assets (RoA) = Net Income / Total Assets
        - Return on Capital Employed (RoCE) = EBIT / (Total Assets - Current Liabilities)
        - Net Profit Margin = Net Income / Revenue'
        - Operating Margin = Operating Income / Revenue

    Profitability ratios help earnings of and returns from a company. These can be interpreted
    as follows (though the interpretation could differ person-to-person)
        - Return on Equity (RoE) - >15% is good (higher return on shareholder investment)
        - Return on Assets (RoA) - >5% is good (shows how efficiently assets generate capital)
        - Return on Capital Employed (RoCE) - >12% preferred, measures efficiency in capital utilization
        - Net Profit Margin - >10% is strong, measures final profitability per dollar of revenue
        - Operating Margin - >20% is Excellent (strong profitability & cost efficiency),
            10-20% is Good (healthy operations & stable profits), 5-10% (Low, businesses is operational, but struggles with cost), <5% is weak (thin margins & potential weakness), negative (loss making business!!)

    Args:
        symbol (string) - the stock symbol

    Returns:
        str: markdown version of the profitability ratios with rows ordered by date
           and columns having values for each of the profitability ratio
    """
    ticker = yf.Ticker(symbol)
    balance_sheet = ticker.balance_sheet.transpose().sort_index(ascending=True)
    financials = ticker.financials.transpose().sort_index(ascending=True)
    income_stmt = ticker.income_stmt.transpose().sort_index(ascending=True)

    revenue = financials["Total Revenue"]
    operating_income = financials["Operating Income"]
    net_income = financials["Net Income"]
    total_assets = balance_sheet["Total Assets"]
    shareholder_equity = balance_sheet["Stockholders Equity"]
    ebit = income_stmt["EBIT"]
    current_liabilities = balance_sheet["Current Liabilities"]

    ratios = {}
    ratios["Return on Equity (RoE)"] = net_income / shareholder_equity
    ratios["Return on Assets (RoA)"] = net_income / total_assets
    ratios["Return on Capital Employed (RoCE)"] = ebit / (
        total_assets - current_liabilities
    )
    ratios["Net Profit Margin"] = net_income / revenue
    ratios["Operating Margin"] = operating_income / revenue

    ret =  f"\n{pd.DataFrame(ratios).to_markdown()}\n"
    logger.debug(ret)
    return ret


def get_efficiency_ratios(symbol: str) -> pd.DataFrame:
    """
    Use this function to get the end-of-financial-year values for efficiency ratios for a given symbol.
        - Asset Turnover Ratio = Revenue / Total Assets
        - Inventory Turnover = Cost of Goods Sold (COGS) / Inventory

    Financial ratios measure operational efficiency of a company. These can be interpreted
    as follows (though the interpretation could differ person-to-person)
        - Asset Turnover Ratio - measures how effectively assets generate sales. Higher is better
        - Inventory Turnover - >5 means fast-moving stocks/inventory, lower values indicate excess inventory
            (NOTE: not all companies report inventory!)

    Args:
        symbol (string) - the stock symbol

    Returns:
        str: markdown version of the efficiency ratios with rows ordered by date
           and columns having values for each of the efficiency ratio
    """
    ticker = yf.Ticker(symbol)
    balance_sheet = ticker.balance_sheet.transpose().sort_index(ascending=True)
    financials = ticker.financials.transpose().sort_index(ascending=True)

    ratios = {}

    revenue = financials["Total Revenue"]
    cost_of_goods_sold = financials["Cost Of Revenue"]
    total_assets = balance_sheet["Total Assets"]

    # Inventory Data
    # NOTE: inventory may or may not get reported. For example,
    # Tata Motors reports it, Persisteny Systems does not
    inventory_fields_exist = "Inventory" in ticker.balance_sheet.index
    if inventory_fields_exist:
        inventory = balance_sheet["Inventory"]
        average_inventory = inventory.rolling(2).mean()

    ratios["Asset Turnover"] = revenue / total_assets
    if inventory_fields_exist:
        ratios["Inventory Turnover"] = cost_of_goods_sold / average_inventory

    ret =  f"\n{pd.DataFrame(ratios).to_markdown()}\n"
    logger.debug(ret)
    return ret


def get_valuation_ratios(symbol: str) -> pd.DataFrame:
    """
    Use this function to get the end-of-financial-year values for valuation ratios for a given symbol.
        - Price-to-Earnings Ratio (P/E) - Price per Share / Earnings per Share (EPS)
        - Price-to-Sales Ratio (P/S) - Market Capitalization / Revenue
        - Price-to-Book Ratio (P/B) - Market Capitalization / Book Value of Equity
        - EV/EBIDTA ratio - (Market Cap + Debt - Cash & Equivalents) / EBIDTA
            where EV = Enterprise value & EBIDTA = Earnings before interest, taxes,
            depreciation and amortization

    Valuation ratios are intended to measure stock price "fairness" - generally we should
    look for lower values, which means stock price is undervalues, so should be a good investement.
    These can be interpreted as follows (though the interpretation could differ person-to-person)
        - Price-to-Earnings Ratio (P/E) - <20 is reasonable, lower means undervalued (so good for
            investment), higher suggests overvalues (to be avoided)
        - Price-to-Sales Ratio (P/S) - <2 means potentially undervalued (should target), 2-4 fair valued,
            and >4 potentially overvalues (should be avoided)
        - Price-to-Book Ratio (P/B) - <3 is good. Implies undervalued compared to book value (so
            good as an investment target)
        - EV/EBIDTA ratio - <10 is preferred (= cheap or undervalued, so good investment target),
            10-15 (= fair value - neutral) and >15 (= expensive or overvalued - avoid!)

    Args:
        symbol (string) - the stock symbol

    Returns:
        str: markdown version of the valuation ratios with rows ordered by date
           and columns having values for each of the valuation ratio
    """
    ticker = yf.Ticker(symbol)
    balance_sheet = ticker.balance_sheet.transpose().sort_index(ascending=True)
    financials = ticker.financials.transpose().sort_index(ascending=True)

    ratios = {}

    market_cap = ticker.info["marketCap"]
    revenue = financials["Total Revenue"]
    shareholder_equity = balance_sheet["Stockholders Equity"]
    ebidta = financials.get(
        "EBIDTA",
        financials["Operating Income"]
        + financials.get("Depreciation & Amortization", 0),
    )
    total_debt = balance_sheet["Total Debt"]
    cash_equivalents = balance_sheet["Cash And Cash Equivalents"]
    ev = market_cap + total_debt - cash_equivalents

    ratios["Price-to-Earnings (P/E)"] = ticker.info["trailingPE"]
    ratios["Price-to-Sales (P/S)"] = market_cap / revenue
    ratios["Price-to-Book (P/B)"] = market_cap / shareholder_equity
    ratios["EV/EBIDTA"] = ev / ebidta

    ret =  f"\n{pd.DataFrame(ratios).to_markdown()}\n"
    logger.debug(ret)
    return ret


def get_leverage_ratios(symbol: str) -> pd.DataFrame:
    """
    Use this function to get the end-of-financial-year values for leverage ratios for a given symbol.
        - Debt-to-Equity Ratio (D/E) - Total Debt / Shareholders Equity
        - Interest Coverage - EBIT / Interest Expense

    Leverage ratios are intended to measure debt risk of a company. These can be interpreted
    as follows (though the interpretation could differ person-to-person)
        - Debt-to-Equity Ratio (D/E) - <1 is good. Higher values means higher debt risk
        - Interest Coverage - >3 is safe, lower values means higher risk of default

    Args:
        symbol (string) - the stock symbol

    Returns:
        str: markdown version of the leverage ratios with rows ordered by date
           and columns having values for each of the leverage ratio
    """
    ticker = yf.Ticker(symbol)
    balance_sheet = ticker.balance_sheet.transpose().sort_index(ascending=True)
    financials = ticker.financials.transpose().sort_index(ascending=True)

    total_debt = balance_sheet["Total Debt"]
    shareholder_equity = balance_sheet["Stockholders Equity"]
    ebit = financials["EBIT"]
    interest_expense = financials["Interest Expense"]

    ratios = {}
    ratios["Debt-to-Equity (D/E)"] = total_debt / shareholder_equity
    ratios["Interest Coverage"] = ebit / interest_expense

    ret =  f"\n{pd.DataFrame(ratios).to_markdown()}\n"
    logger.debug(ret)
    return ret


def get_performance_and_growth_metrics(symbol: str) -> pd.DataFrame:
    """
    Use this function to get the end-of-financial-year values for performance & growth metrics for a given symbol.
        - Revenue Growth (%) = (Current Year Revenue - Previous Year Revenue) / Previous Year Revenue
        - EBIT Growth (%) = (Current Year EBIT - Previous Year EBIT) / Previous Year EBIT
        - EPS Growth (%) = (Current Year EPS - Previous Year EPS) / Previous Year EPS
        - FCF Growth (%) = (Current Year FCF - Prev Year FCF) / Prev Year FCF [FCF = Free Cash Flow]
        - Net Profit Margin (%) = Net Income / Total Revenue
        - Earnings Per Share (EPS) = (Net Income - Preferred Dividends) / Shares Outstanding
        - Debt to Equity (D/E) = Total Debt / Total Shareholders' Equity
        - Free Cash Flow = Operating Cash Flow - Capital Expenditure

    Revenue & growth metrics are intended to measure business growth, profitability, earnings potential, 
    financial stability, and cash flow generation of a company.
    These can be interpreted as follows (though the interpretation could differ person-to-person)
        - Revenue Growth (%) - measures revenue growth across years. >10% is strong growth,
            5-10% is moderate growth, <5% is slow growth, and negative is revenue decline (avoid!)
        - EBIT Growth (%) - measures profitabity growth across years. >10% is strong operational efficiency &
            profitability growth, 5-10% is moderate growth, <5% is slow growth and negative means declining
            profitability (avoid!). Positive values indicate company is becoming more profitable.
        - EPS Growth (%) - measures how fast the company's earnings are increasing on a per-share basis.
            >15% means strong EPS growth (great for investors), 5-15% means moderate but stable growth,
            0-5% means slow earnings expansion and <0% (negative) means declining EPS (avoid!)
        - FCF Growth (%) - shows whether company is generating more "excess cash flow" over time.
            >15% means strong cash flows (company can invest & pay dividends), 5-15% means moderate growth & stable cash generation, 0-5% means low growth (cash flow stagnation) and negative means declining FCF (bad for investment - avoid!)
        - Net Profit Margin (%) - measures the profitability of a company.
            >10% means healthy profitability, 5-10% means sustainable profitability, 0-5% means thin margins, and negative means loss making business (avoid!)
        - Earnings Per Share (EPS) - measures profit earned by the company per share.
            Increasing EPS means company is growing profits per share - look for Y-on-Y values.
            Declining EPS means EPS is shrinking & this company should be avoided.
        - Debt to Equity (D/E) - measures debt-burden risk for a company
            <1.0 means healthy leverage (debt burden is low, hence good)
            1.0 - 2.0 usually means moderate risk, but varies by industry
            > 2.0 implies high debt burden (risky & should be avoided!)
        - Free Cash Flow - measures cash flows in & out of the company
            a positive value indicates good financial health (income > expenses)
            a negative value indicates excessive spending or low profitability of company

    Args:
        symbol (string) - the stock symbol

    Returns:
        str: markdown version of the performance & growth metrics with rows ordered by date
           and columns having values for each of the performance & growth metric
    """
    ticker = yf.Ticker(symbol)
    balance_sheet = ticker.balance_sheet.transpose().sort_index(ascending=True)
    financials = ticker.financials.transpose().sort_index(ascending=True)
    cash_flow = ticker.cash_flow.transpose().sort_index(ascending=True)

    ratios = pd.DataFrame(index=financials.index)
    ratios["Revenue Growth (%)"] = financials["Total Revenue"].pct_change() * 100.0
    ratios["EBIT Growth (%)"] = financials["EBIT"].pct_change() * 100.0
    ratios["Net Profit Margin (%)"] = (
        financials["Net Income"] / financials["Total Revenue"]
    ) * 100.0
    shares_outstanding = balance_sheet["Ordinary Shares Number"]
    eps = financials["Net Income"] / shares_outstanding
    ratios["EPS Growth (%)"] = eps.pct_change() * 100.0

    ratios["EPS"] = financials["Net Income"] / shares_outstanding
    ratios["Debt-to-Equity"] = (
        balance_sheet["Total Debt"] / balance_sheet["Stockholders Equity"]
    )
    ratios["Free Cash Flow"] = cash_flow["Free Cash Flow"]
    ratios["FCF Growth (%)"] = cash_flow["Free Cash Flow"].pct_change() * 100.0

    # NOTE: unlike other functions, ratios is a DataFrame
    ret =  f"\n{ratios.to_markdown()}\n"
    logger.debug(ret)
    return ret
