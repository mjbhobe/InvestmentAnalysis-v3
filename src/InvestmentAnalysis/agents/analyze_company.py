"""
analyze_company.py - analysis of a company's financial & stock performance
    using Yahoo Finance and then leveraging an LLM to analyze performance viz-a-viz
    its peers and make a final investment recommendation.
    Supports LLMs from OpenAI (paid), Anthropic (paid), and Gemini.
    Groq is also supported, but it has issues with context window size when making
    final recommendations.

Author: Manish Bhobe
My experiments with Python, ML and Generative AI.
Code is meant for illustration purposes ONLY. Use at your own risk!
Author is not liable for any damages arising from direct/indirect use of this code.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import os, sys
from dotenv import load_dotenv, find_dotenv
import pathlib
from datetime import datetime

# for supported LLMs
from openai import OpenAI
from groq import Groq
from anthropic import Anthropic
import google.generativeai as genai

# load env variables from .env file
_ = load_dotenv(find_dotenv())

SYS_PROMPT = f"""
You are an expert at Finance, Financial Markets and Financial market ratio calculations and 
analysis of companies and can give expert and detailed recommendations about a company, its 
Financial performance and its potential as a good investment target
"""

PROVIDER_AND_MODEL = {
    "OpenAI": "gpt-4o-mini",
    "Groq": "llama-3.3-70b-versatile",
    "Anthropic": "claude-3-5-sonnet-20241022",
    "Gemini": "gemini-1.5-flash",
}
MODEL_NAME = "unk"  # initial value, gets set when get_chat_model() is called


def get_chat_model(provider: str):
    """create an instance of the LLM (chat client) depending on the provider
        chosen. Value of provider comes from dropdown displayed on UI
    Params:
        provider: string [one of OpenAI, Groq etc - see PROVIDER_AND_MODEL.keys()]
    Returns:
        instance of appropriate Chat client for respective provider
    """
    global PROVIDER_AND_MODEL, MODEL_NAME
    assert (
        provider in PROVIDER_AND_MODEL.keys()
    ), f"FATAL: {provider} is not a supported model!"

    MODEL_NAME = PROVIDER_AND_MODEL[provider]

    if provider == "OpenAI":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif provider == "Groq":
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    elif provider == "Anthropic":
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    elif provider == "Gemini":
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        client = genai.GenerativeModel(MODEL_NAME)

    return client


def get_model_completion(chat_client, prompt: str) -> str:
    """
    gets the chat model to make a completion for prompt provided.
    This function handles the API variations across all the supported models

    Params:
        chat_client - instance of chat client created previously
        prompt (str) - the prompt for which you want a completion (response)
            from instance of chat client
    Returns:
        Text (or Markdown) response from the chat client (Gemini usually returns
        markdown, rest of models return plain text)
    """
    global MODEL_NAME, SYS_PROMPT

    if chat_client.__class__.__name__ in ["OpenAI", "Groq"]:
        # these use the same API
        completion = chat_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )
        return completion.choices[0].message.content
    elif chat_client.__class__.__name__ in ["Anthropic"]:
        completion = chat_client.messages.create(
            model=MODEL_NAME,
            max_tokens=2048,
            temperature=0,
            system=SYS_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        )
        return completion.content[0].text
    elif chat_client.__class__.__name__ in ["GenerativeModel"]:
        # Google Gemini
        # Gemini does not have concept of System prompt
        # so we concatenate SYS_PROMPT with our prompt and pass
        # the concatenation as our overall prompt
        chat_prompt = SYS_PROMPT + "\n\n" + prompt
        completion = chat_client.generate_content(
            chat_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=2048,
                temperature=0,
            ),
        )
        return completion.text
    else:
        raise ValueError(f"{chat_client.__class__.__name__} is not a supported LLM!")


def is_valid_ticker(symbol: str) -> bool:
    """
        Checks if symbol is a valid ticker symbol. For it to be valud, yf.Ticker(symbol).info
        should not raise an exception!
    Params:
        symbol(str): a ticker symbol (such as "AAPL" or "PERSISTENT.NS")
        (please visit Yahoo Finance website to get valid symbol of company)
    """
    try:
        ticker = yf.Ticker(symbol)
        return "shortName" in ticker.info
    except Exception as e:
        return False


# Function to fetch data
def fetch_data(symbol: str):
    """
        fetches ticker data, financials, balance sheet and cash flows
        for the past 5 years from today
    Params:
        symbol (str): valid ticker symbol (as used by Yahoo Finance!)
    Returns:
        Tuple of 4 values (ticker, financials, balance_sheet, cash_flow)
        ticker is a yf.Ticker() instance, rest are pandas Dataframes.
    """
    ticker = yf.Ticker(symbol)
    financials = ticker.financials.transpose()
    balance_sheet = ticker.balance_sheet.transpose()
    cash_flow = ticker.cashflow.transpose()

    # NOTE: financials, balance_sheet and cash_flow are by default
    # reverse sorted by date-time index (i.e. have the
    # most recent year on top). This screws up all calculations
    # We'll fix that by sorting dataframes in ascending data order
    # (i.e. latest year is the last in the dataframe)
    financials = financials.sort_index(ascending=True)
    balance_sheet = balance_sheet.sort_index(ascending=True)
    cash_flow = cash_flow.sort_index(ascending=True)

    return ticker, financials, balance_sheet, cash_flow


def get_peer_companies(chat_client, ticker: yf.Ticker) -> dict:
    """
        Get top 5 peer companies of ticker, which operate in the same industry
        as ticker and whose stocks trade on the same primary stock exchange as ticker
        This is an LLM assisted function - the LLM fetches this data
    Params:
        chat_client: instance of LLM we are using
        ticker(str): valid ticker symbol of company (as used by Yahoo Finance!). E.g. "AAPL", "PERSISTENT.NS"
    Returns:
        A Python dict object, with 5 entries, each with ticker symbol as key and company name as value
        For example:
        {"TCS.NS":"Tata Consultancy Services", "INFY.NS":"Infosys Ltd", ...} (5 entries)
    """
    # Create the prompt to retrieve the peer companies
    peer_cos_prompt = f"""
    I want to retrieve the top 10 peer companies of {ticker.info['symbol']} which operate in the 
    {ticker.info['industryDisp']} industry and trade on the primary stock exchange on which 
    {ticker.info['symbol']} trades. The peer company symbols should not include {ticker.info['symbol']}!
    Please provide your response in the form of a Python dictionary with the key being the stock symbol of 
    the company and the value being the full name of the company. 
    Return the response as a string formatted as a Python dict with no surrounding text or markdown.
    """

    if chat_client.__class__.__name__ in ["GenerativeModel"]:
        # Google Gemini generates un-necessary markdown in it's response
        # hoping this additional prompt will omit that
        peer_cos_prompt += (
            "\n"
            + "NOTE: Do not generate any markdown text in your response. Return just plan text"
        )

    completion = get_model_completion(chat_client, peer_cos_prompt)

    # Extract the response
    # peers = response.choices[0].text.strip()
    peers = completion  # .choices[0].message.content
    print(f"Peers: {peers}")

    # Convert the response to a Python dict
    try:
        # presumably the LLM returns the 10 closest peers sorted by closeness
        peers_dict = eval(peers)
        # sort by key
        peers_dict = {
            key: peers_dict[key]
            for key in peers_dict.keys()
            # sometimes LLM returns wrong ticker symbol!
            if is_valid_ticker(key)
        }
        # select the top 5 valid symbols & descriptions
        peers_dict = dict(list(peers_dict.items())[:5])
        # sort keys in ascending order
        peers_dict = {key: peers_dict[key] for key in sorted(peers_dict.keys())}
    except Exception as e:
        peers_dict = {"error": str(e)}

    return peers_dict


# Function to calculate financial ratios
def calculate_ratios(
    ticker: yf.Ticker,
    financials: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
) -> pd.DataFrame:
    """
        Calculates the following financial ratios:
            1. Liquidity Ratios:
                - Current Ratio = Current Assets / Current Liabilities
                - Quick Ratio = (Current Assets - Inventory) / Current Liabilities
                - Cash Ratio = (Cash & Equivalents) / Current Liabilities
            2. Profitability Ratios:
                - Return on Equity (RoE) = Net Income / Shareholder's Equity
                - Return on Assets (RoA) = Net Income / Total Assets
                - Return on Capital Employed (RoCE) = EBIT / (Current Assets - Current Liabilities)
                - Net Profit Margin = Net Income / Revenue
                - Operating Margin = Operating Income / Revenue
            3. Leverage Ratios:
                - Debt-to-Equity Ratio (DoE) = Total Debt / Shareholder's Equity
                - Interest Coverage Ratio = EBIT / Interest Expense
            4. Efficiency Ratios:
                - Asset Turnover Ratio = Revenue / Total Assets
                - Inventory Turnover Ratio = Cost of Goods Sold / Average Inventory (Optional)
            5. Valuation Ratios:
                - Price-to-Earnings Ratio (P/E) = Price per Share / Earnings per Share (EPS)
                - Price-to-Sales Ratio (P/S) = Market Capitalization / Revenue
                - Price-to-Book Ratio (P/B) = Market Capitalization / Book Value of Equity
            6. Performance and Growth Metrics (%, means reported as percentage [multiply by 100])
                - Revenue Growth (%) = (Current Year Revenue - Previous Year Revenue) / Previous Year Revenue
                - EBIT Growth (%) = (Current Year EBIT - Previous Year EBIT) / Previous Year EBIT
                - EPS Growth (%) = (Current Year EPS - Previous Year EPS) / Previous Year EPS
                - FCF Growth (%) = (Current Year FCF - Prev Year FCF) / Prev Year FCF [FCF = Free Cash Flow]
                - Net Profit Margin (%) = Net Income / Total Revenue
                - Earnings Per Share (EPS) = (Net Income - Preferred Dividends) / Shares Outstanding
                - Debt to Equity (D/E) = Total Debt / Total Shareholders' Equity
                - Free Cash Flow = Operating Cash Flow - Capital Expenditure
    Params:
        ticker (yf.Ticker): an instance of Ticker object
        financials, balance_sheet and cash_flow: all pandas Dataframe instances
        All the above params are returned from `fetch_data(...)` call.
        So you can call `calculate_ratios(**fetch_data("AAPL"))` for Apple's ratios
    Returns:
        Python dict, with ratio as key and value of ratio as the value
        Example:
            ratios = {
                "Current Ratio" : 3.456,
                "Earnings Per Share" : 44.35,
                ... and so on
            }
    """
    ratios = {}

    info = ticker.info

    try:
        # Balance Sheet
        current_assets = balance_sheet["Current Assets"].iloc[-1]
        current_liabilities = balance_sheet["Current Liabilities"].iloc[-1]
        total_assets = balance_sheet["Total Assets"].iloc[-1]
        shareholder_equity = balance_sheet["Stockholders Equity"].iloc[-1]
        total_debt = balance_sheet["Total Debt"].iloc[-1]

        # Financials
        revenue = financials["Total Revenue"].iloc[0]
        operating_income = financials["Operating Income"].iloc[0]
        net_income = financials["Net Income"].iloc[0]
        cost_of_goods_sold = financials["Cost Of Revenue"].iloc[0]
        ebit = financials["EBIT"].iloc[0]

        # Inventory Data
        # NOTE: inventory may or may not get reported. For example,
        # Tata Motors reports it, Persisteny Systems does not
        inventory_fields_exist = "Inventory" in ticker.balance_sheet.index
        if inventory_fields_exist:
            inventory_current = balance_sheet["Inventory"].iloc[-1]
            inventory_previous = balance_sheet["Inventory"].iloc[-2]
            average_inventory = (inventory_current + inventory_previous) / 2

        # Interest Expense
        interest_expense = financials["Interest Expense"].iloc[0]

        # Ratios -------------------------------

        ratios["Revenue Growth"] = financials["Total Revenue"].pct_change()
        ratios["Earnings-per-share (EPS)"] = (
            financials["Net Income"] / balance_sheet["Common Stock"]
        )

        # Liquidity Ratios
        ratios["Current Ratio"] = current_assets / current_liabilities
        if inventory_fields_exist:
            ratios["Quick Ratio"] = (
                current_assets - balance_sheet["Inventory"].iloc[0]
            ) / current_liabilities
        # Profitability Ratios
        ratios["Net Profit Margin"] = net_income / revenue
        ratios["Operating Margin"] = operating_income / revenue
        ratios["Return on Assets (RoA)"] = net_income / total_assets
        ratios["Return on Equity (RoE)"] = net_income / shareholder_equity
        # Leverage Ratios
        ratios["Debt-to-Equity (D/E)"] = total_debt / shareholder_equity
        ratios["Interest Coverage"] = ebit / interest_expense

        # Efficiency ratios
        ratios["Asset Turnover"] = revenue / total_assets
        if inventory_fields_exist:
            ratios["Inventory Turnover"] = cost_of_goods_sold / average_inventory

        # Valuation Ratios
        market_cap = info["marketCap"]
        ratios["Price-to-Earnings (P/E)"] = info["trailingPE"]
        ratios["Price-to-Sales (P/S)"] = market_cap / revenue
        ratios["Price-to-Book (P/B)"] = market_cap / shareholder_equity

        return pd.DataFrame(ratios)
    except KeyError as e:
        print(f"KeyError: missing data for {e}")
        raise e


def compare_with_peers(symbol: str, peers: dict) -> pd.DataFrame:
    """
        Compares financial ratios of company with those of peers
    Params:
        symbol(str): ticker symbol of company (as used by Yahoo Finance!)
        peers (dict): top-5 peers from get_peer_companies() call
    Returns:
        Dataframe with ratios of symbol + those of it's 5 peer companies
    """
    peer_ratios = {}

    for peer in peers.keys():
        ticker, financials, balance_sheet, cash_flow = fetch_data(peer)
        peer_ratios[peer] = calculate_ratios(
            ticker, financials, balance_sheet, cash_flow
        ).mean()

    # Convert to DataFrame
    return pd.DataFrame(peer_ratios)


# Function to get recommendation using OpenAI API
def get_recommendation(chat_client, symbol: str, report: str, peers=None) -> str:
    """
        Gets recommendation from LLM based on financial performance, ratios and peer-comparison
    Params:
        chat_client: instance of LLM to use
        symbol (str): valid ticker symbol (as used by Yahoo Finance!)
        report (str): string representation of Financial performance of company
        peers (dict) [optional]: dict of top 5 peers
    Returns:
        Recommendation from LLM as a string (could contain Markdown, especially if LLM is Gemini!)
    """
    # Set your OpenAI API key in the environment variables
    reco_prompt = (
        f"Given the following financial performance report:\n\n{report}\n\n "
        f"First, give a commentary and your analysis of {symbol} performance\n"
    )
    if peers is not None:
        reco_prompt += f"Next, give a commentary and your analysis of how {symbol} has fared viz-a-viz peers {peers}\n"

    reco_prompt += f"Finally, What is your recommendation on this company's long-term investment potential?"
    completion = get_model_completion(chat_client, reco_prompt)

    return completion


# Main application
def main():
    st.title("Financial Analysis of Company")

    # create a 2 x 2 layout
    col1, col2 = st.columns(2)

    with col1:
        # Row 1, col 1
        # choose LLM provider
        provider = st.selectbox(
            "Select LLM Provider",
            ["Select"] + list(PROVIDER_AND_MODEL.keys()),
        )
    with col2:
        # Row 1, col 2
        # Input company symbol
        symbol = st.text_input("Enter the Company Ticker (e.g AAPL, PERSISTENT.NS):")

    if provider and symbol:
        # provider selected & symbol entered

        # first check if entered symbol is valid or not
        if not is_valid_ticker(symbol):
            st.error(
                f"{symbol} appears to be an invalid ticker symbol. Please enter a valid ticker symbol"
            )
            st.stop()

        # create our chat model
        client = get_chat_model(provider)
        print(f"You have chosen {provider} LLM")

        this_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        markdown_file_dir = pathlib.Path(__file__).parent / "reports"
        if not markdown_file_dir.exists():
            markdown_file_dir.mkdir()

        markdown_file_path = (
            markdown_file_dir / f"{symbol}_report_{provider}_{this_time}.md"
        )
        print(f"Report will be generated in : {markdown_file_path}")
        # sys.exit(-1)

        with open(str(markdown_file_path), "w", encoding="utf-8") as repo_file:
            ticker, financials, balance_sheet, cash_flow = fetch_data(symbol)
            report = ""
            NL2 = "\n\n"
            repo_file.write(
                f"# Financial Report, Analysis and AI Recommendation for {symbol}{NL2}"
            )

            basic_info = f"#### Basic Info for {symbol}"
            st.markdown(basic_info)
            report += basic_info + NL2
            repo_file.write(basic_info + NL2)

            company_name = f"**Company Name:** {ticker.info['longName']}"
            st.markdown(company_name)
            report += company_name + NL2
            repo_file.write(company_name + NL2)

            business_summary = f"**Business Summary:**"
            st.markdown(business_summary)
            report += business_summary + NL2
            repo_file.write(business_summary + NL2)

            long_business_summary = f"{ticker.info['longBusinessSummary']}"
            st.markdown(long_business_summary)
            report += long_business_summary + NL2
            repo_file.write(long_business_summary + NL2)

            peers = get_peer_companies(client, ticker)
            top5_peer_cos = f"**Peers (top {len(peers)})**"
            st.markdown(top5_peer_cos)
            report += top5_peer_cos + NL2
            repo_file.write(top5_peer_cos + NL2)

            # st.markdown(f"```python\n{peers}\n```")
            peers_md = "| Symbol | Company Name|\n"
            peers_md += "|--------|-------------|\n"
            for key, value in peers.items():
                peers_md += f"|{key}|{value}|\n"
                st.markdown(f"- {value} ({key})")
            # for sym, descr in peers.items():
            #     st.markdown(f"- {descr} ({sym})")
            report += peers_md + NL2
            repo_file.write(peers_md + NL2)

            financials_title = f"#### Financials"
            st.markdown(financials_title)
            report += financials_title + NL2
            repo_file.write(financials_title + NL2)

            st.dataframe(financials)
            fin_str = financials.to_markdown()
            report += fin_str + NL2
            repo_file.write(fin_str + NL2)

            balance_sheet_title = f"#### Balance Sheet"
            st.markdown(balance_sheet_title)
            report += balance_sheet_title + NL2
            repo_file.write(balance_sheet_title + NL2)

            st.dataframe(balance_sheet)
            bal_str = balance_sheet.to_markdown()
            report += bal_str + NL2
            repo_file.write(bal_str + NL2)

            cash_flow_title = f"#### Cash Flows"
            st.markdown(cash_flow_title)
            report += cash_flow_title + NL2
            repo_file.write(cash_flow_title + NL2)

            st.dataframe(cash_flow)
            cash_str = cash_flow.to_markdown()
            report += cash_str + NL2
            repo_file.write(cash_str + NL2)

            ratios = calculate_ratios(ticker, financials, balance_sheet, cash_flow)

            markdown_title = f"### Financial Ratios for {symbol}"
            st.markdown(markdown_title)
            report += markdown_title + NL2
            repo_file.write(markdown_title + NL2)

            st.dataframe(ratios)
            ratios_str = ratios.to_markdown()
            report += ratios_str + NL2
            repo_file.write(ratios_str + NL2)

            if peers:
                # create dataframe of symbol ratios
                symbol_df = pd.DataFrame(ratios.mean().T, columns=[f"{symbol}"])
                # and those of peers
                peers_df = compare_with_peers(symbol, peers)
                # which you wanna concatenate
                # peer_comparison[symbol] = ratios.mean().T
                comparison_df = pd.concat([symbol_df, peers_df], axis=1)

                peer_comparison_title = "### Peer Comparison"
                st.markdown(peer_comparison_title)
                report += peer_comparison_title + NL2
                repo_file.write(peer_comparison_title + NL2)

                st.dataframe(comparison_df)
                comparison_str = comparison_df.to_markdown()
                report += comparison_str + NL2
                repo_file.write(comparison_str + NL2)

                # get recommendation from LLM
                recommendation_title = "### AI Recommendation"
                st.write(recommendation_title)
                repo_file.write(recommendation_title + NL2)

                recommendation = get_recommendation(client, symbol, report, peers)
                st.write(recommendation)
                repo_file.write(recommendation)

                repo_file.flush()


if __name__ == "__main__":
    main()