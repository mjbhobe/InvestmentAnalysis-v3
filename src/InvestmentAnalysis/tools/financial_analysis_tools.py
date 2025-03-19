import json
import yfinance

from agno.tools import Toolkit
from agno.utils.log import logger


try:
    import yfinance as yf
except ImportError:
    raise ImportError("`yfinance` not installed. Please install using `pip install yfinance`.")
from .ratios import (
    get_liquidity_ratios,
    get_profitability_ratios,
    get_efficiency_ratios,
    get_valuation_ratios,
    get_leverage_ratios,
    get_performance_and_growth_metrics,
)

# created similar to Agno's YFinanceTools
class FinancialAnalysisTools(Toolkit):
    def __init__(self,
        liquidity_ratios=True,
        profitability_ratios=False,
        efficiency_ratios=False,
        valuation_ratios=False,
        leverage_ratios=False,
        performance_and_growth_metrics=False,
        company_info=False,
        enable_all=False,
    ):
        super().__init__(name="finanalysis_tools")

        # register functions
        if liquidity_ratios or enable_all:
            logger.debug("Registering get_liquidity_ratios function")
            self.register(get_liquidity_ratios)
        if profitability_ratios or enable_all:
            logger.debug("Registering get_profitability_ratios function")
            self.register(get_profitability_ratios)
        if efficiency_ratios or enable_all:
            logger.debug("Registering get_efficiency_ratios function")
            self.register(get_efficiency_ratios)
        if valuation_ratios or enable_all:
            logger.debug("Registering get_valuation_ratios function")
            self.register(get_valuation_ratios)
        if leverage_ratios or enable_all:
            logger.debug("Registering get_leverage_ratios function")
            self.register(get_leverage_ratios)
        if performance_and_growth_metrics or enable_all:
            logger.debug("Registering get_performance_and_growth_metrics function")
            self.register(get_performance_and_growth_metrics)
        if company_info or enable_all:
            logger.debug("Registering get_company_info function")
            self.register(self.get_company_info)

    
    def get_company_info(self, symbol:str) -> str:
        """Use this function to get company information and overview for a given stock symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            str: JSON containing company profile and overview.
        """
        try:
            company_info_full = yf.Ticker(symbol).info
            if company_info_full is None:
                return f"Could not fetch company info for {symbol}"

            logger.debug(f"Fetching company info for {symbol}")

            company_info_cleaned = {
                "Name": company_info_full.get("shortName"),
                # fields added
                "LongName": company_info_full.get("longName"),
                "Business Summary": company_info_full.get("longBusinessSummary"),
                "Symbol": company_info_full.get("symbol"),
                "Current Stock Price": f"{company_info_full.get('regularMarketPrice', company_info_full.get('currentPrice'))} {company_info_full.get('currency', 'USD')}",
                "Market Cap": f"{company_info_full.get('marketCap', company_info_full.get('enterpriseValue'))} {company_info_full.get('currency', 'USD')}",
                "Sector": company_info_full.get("sector"),
                "Industry": company_info_full.get("industry"),
                "Address": company_info_full.get("address1"),
                "City": company_info_full.get("city"),
                "State": company_info_full.get("state"),
                "Zip": company_info_full.get("zip"),
                "Country": company_info_full.get("country"),
                "EPS": company_info_full.get("trailingEps"),
                "P/E Ratio": company_info_full.get("trailingPE"),
                "52 Week Low": company_info_full.get("fiftyTwoWeekLow"),
                "52 Week High": company_info_full.get("fiftyTwoWeekHigh"),
                "50 Day Average": company_info_full.get("fiftyDayAverage"),
                "200 Day Average": company_info_full.get("twoHundredDayAverage"),
                "Website": company_info_full.get("website"),
                "Summary": company_info_full.get("longBusinessSummary"),
                "Analyst Recommendation": company_info_full.get("recommendationKey"),
                "Number Of Analyst Opinions": company_info_full.get("numberOfAnalystOpinions"),
                "Employees": company_info_full.get("fullTimeEmployees"),
                "Total Cash": company_info_full.get("totalCash"),
                "Free Cash flow": company_info_full.get("freeCashflow"),
                "Operating Cash flow": company_info_full.get("operatingCashflow"),
                "EBITDA": company_info_full.get("ebitda"),
                "Revenue Growth": company_info_full.get("revenueGrowth"),
                "Gross Margins": company_info_full.get("grossMargins"),
                "Ebitda Margins": company_info_full.get("ebitdaMargins"),
            }
            return json.dumps(company_info_cleaned, indent=2)
        except Exception as e:
            return f"Error fetching company profile for {symbol}: {e}"


