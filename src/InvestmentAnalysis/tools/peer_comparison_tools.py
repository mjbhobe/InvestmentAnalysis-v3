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
class PeerComparisonTools(Toolkit):
    def __init__(
        self,
    ):
        super().__init__(name="peers_analyis_tools")

        # register functions
        logger.debug("Registering get_performance_ratios function")
        self.register(self.get_peer_comparison_and_industry_benchmarks)

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
            logger.debug(f"Calculating performance ratios for {symbol}")

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

            # ------------ calculate the ratios -----------------------------------------

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
            logger.debug(f"   Ratios for {symbol}:\n {json.dumps(ratios, indent=2)}\n")
            return ratios
        except Exception as e:
            return f"Error fetching company profile for {symbol}: {e}"

    def get_peer_comparison_and_industry_benchmarks(self, symbols: List[str]) -> str:
        """
        Use this function to get a comparison table of all key performance ratios
        of the company as well as its peers and industry benchmarks for the same.
        These are calulated for the latest financial year.

        Args:
            symbols (List[str]): List of stock symbols for which comparison is needed.

        Returns:
            str: pandas Dataframe in markdown format. The dataframe has all the key
                metrics as the index and company symbols as the columns. The last column
                of this table holds the industry benchmark (which is basically the row-wise)
                mean of all the metrics.
                Example output generated (assuming you are analyzing )
                    |                                   |        TCS.NS |      INFY.NS |      WIPRO.NS |   HCLTECH.NS |      TECHM.NS |   PERSISTENT.NS |   Industry Benchmark |
                    |:----------------------------------|--------------:|-------------:|--------------:|-------------:|--------------:|----------------:|---------------------:|
                    | Current Ratio                     |    2.45063    |    2.30531   |   2.57731     |    2.61071   |    1.8567     |     1.88776     |          2.2814      |
                    | Quick Ratio                       |    2.45003    |  nan         |   2.57372     |    2.60257   |    1.85373    |   nan           |          2.37001     |
                    | Cash Ratio                        |    0.195363   |    0.381208  |   0.384036    |    0.415427  |    0.344554   |     0.303326    |          0.337319    |
                    | Return on Equity (RoE)            |    0.507332   |    0.299934  |   0.147292    |    0.230022  |    0.0884084  |     0.220564    |          0.248925    |
                    | Return on Assets (RoA)            |    0.313474   |    0.191672  |   0.0958403   |    0.157371  |    0.0542977  |     0.148298    |          0.160159    |
                    | Return on Capital Employed (RoCE) |    0.625592   |    0.370788  |   0.177513    |    0.279296  |    0.106323   |     0.287956    |          0.307911    |
                    | Net Profit Margin                 |    0.190574   |    0.170617  |   0.123052    |    0.142858  |    0.0453462  |     0.111335    |          0.130631    |
                    | Operating Margin                  |    0.246686   |    0.207359  |   0.148986    |    0.182208  |    0.0658172  |     0.132685    |          0.163957    |
                    | Asset Turnover                    |    1.64489    |    1.1234    |   0.778859    |    1.10159   |    1.1974     |     1.33199     |          1.19636     |
                    | Inventory Turnover                | 6327.19       |  nan         | 562.205       |  390.88      | 1235.8        |   nan           |       2129.02        |
                    | Price-to-Earnings (P/E)           |   26.5654     |   23.3033    |  22.3604      |   24.888     |   33.4104     |    61.9341      |         32.077       |
                    | Price-to-Sales (P/S)              |    5.37412    |  355.461     |   3.08001     |    3.86322   |    2.39799    |     8.28009     |         63.076       |
                    | Price-to-Book (P/B)               |   14.3066     |  624.876     |   3.68675     |    6.22032   |    4.67519    |    16.4035      |        111.695       |
                    | EV/EBIDTA                         |   21.7687     | 1714.03      |  21.1794      |   21.0183    |   35.905      |    62.2418      |        312.69        |
                    | Debt-to-Equity (D/E)              |    0.0886406  |    0.0948953 |   0.219566    |    0.0843209 |    0.0951165  |     0.0909955   |          0.112256    |
                    | Interest Coverage                 |   80.6877     |   78.6071    |  12.728       |   38.915     |   64.0998     |    31.9801      |         51.1696      |
                    | Revenue Growth (%)                |    6.84606    |    1.92181   |  -0.803757    |    8.33563   |   -2.42953    |    17.6155      |          5.24761     |
                    | EBIT Growth (%)                   |    8.8219     |    5.81731   |   1.28571     |    8.46228   |  -49.5386     |    16.0024      |         -1.52484     |
                    | Net Profit Margin (%)             |   19.0574     |   17.0617    |  12.3052      |   14.2858    |    4.53462    |    11.1335      |         13.0631      |
                    | EPS Growth (%)                    |   10.1568     |    6.14809   |   2.14068     |    5.70583   |  -51.3418     |    16.6603      |         -1.75502     |
                    | EPS                               |  126.885      |    0.764985  |  10.5813      |   57.984     |   26.7166     |    71.7851      |         49.1194      |
                    | Debt-to-Equity                    |    0.0886406  |    0.0948953 |   0.219566    |    0.0843209 |    0.0951165  |     0.0909955   |          0.112256    |
                    | Free Cash Flow                    |    4.1664e+11 |    2.882e+09 |   1.65706e+11 |    2.14e+11  |    5.5853e+10 |     9.37391e+09 |          1.44076e+11 |
                    | FCF Growth (%)                    |    7.20185    |   13.7332    |  43.1375      |   30.9029    |   22.5707     |    79.4177      |         32.8273      |
        """
        try:
            logger.debug("Fetching performance ratios for {symbols}")

            ratios = {}
            for symbol in symbols:
                ratios[symbol] = self.__calculate_performance_ratios(symbol)

            df = pd.DataFrame(ratios)
            # calculate industry benchmarks - average across rows
            df["Industry Benchmark"] = df.mean(axis=1)
            logger.debug(f"Returning peer comparison table\n{df.to_markdown()}")
            # return json.dumps(ratios)
            return f"\n{df.to_markdown()}\n"
        except Exception as e:
            return f"Error fetching company profile for {symbols}: {e}"
