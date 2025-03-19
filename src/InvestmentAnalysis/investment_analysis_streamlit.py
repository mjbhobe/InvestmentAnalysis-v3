import streamlit as st
import yfinance as yf

from agno.agent import RunResponse
from agno.utils.log import logger
from agents.investment_analysis_agent import investment_analysis_agent

# Page configuration
st.set_page_config(
    page_title="Investment Analyzer - Ver 2.0", page_icon="📊", layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    .stock-input {
        max-width: 400px;
        margin: 0 auto;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "analysis_generated" not in st.session_state:
    st.session_state.analysis_generated = False


def is_valid_stock_symbol(symbol: str) -> bool:
    try:
        ticker = yf.Ticker(symbol.upper())
        # try to get info, should raise exception if invalid
        _ = ticker.info
        return True
    except Exception as e:
        logger.fatal(f"ERROR: {symbol.upper()} is not a valid stock symbol.")
        return False


def generate_investment_analysis(symbol: str, agent):
    prompt = f"Generate investment analysis for {symbol}"
    # return agent.print_response(prompt, stream=True)
    response: RunResponse = agent.run(prompt, markdown=True)
    return response.content


# Main UI
st.markdown(
    "<h1 class='main-header'>Investment Analyzer - Ver 2.0📈</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    This tool provides investment analysis for publicly traded companies and comes up with an overall
    recommendation on the long term investment potential.
    <br/>
    <div style='color: #777;'>     
    <small>   
    In this version we combine Financial analysis and sentiment analysis to come up with an overall
    recommendation on the long term investment potential of the company. </small>
    </div>
    <p/>
""",
    unsafe_allow_html=True,
)

# st.markdown(
#     """
#     <div style='color: #777;'>
#     <small>
#     In this version we combine Financial analysis and sentiment analysis to come up with an overall
#     recommendation on the long term investment potential of the company. </small>
#     </div>
# """,
#     unsafe_allow_html=True,
# )

# Stock symbol input
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        stock_symbol = st.text_input(
            "Enter Stock Symbol to begin (should be same as used on Yahoo! Finance website):",
            placeholder="e.g., TCS.NS",
            key="stock_input",
        )
    with col2:
        col2.markdown(f"<div style='height: 28px;'></div>", unsafe_allow_html=True)
        analyze_button = st.button("Analyze", type="primary")

# Analysis section
if analyze_button and stock_symbol:
    # check if user has entered a valid stock symbol
    if not is_valid_stock_symbol(stock_symbol):
        st.markdown(
            f"""<div style='color: red;'>
            Invalid stock symbol: {stock_symbol.upper()}. Please enter valid symbol to proceed!
            </div>""",
            unsafe_allow_html=True,
        )
        st.stop()

    try:
        stock_symbol = stock_symbol.upper()
        with st.spinner(f"Generating investment analysis for {stock_symbol}..."):
            analysis = generate_investment_analysis(
                stock_symbol, investment_analysis_agent
            )

        st.success("Analysis completed!")

        # Display analysis in an expandable container
        with st.expander("View Detailed Analysis", expanded=True):
            st.markdown(analysis)

        st.session_state.analysis_generated = True

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #777;'>
        <small>Developed by Manish Bhobé • using yfinance • Agno (Agents) • Google Gemini • Streamlit</small>
        <small>For educational purposes only!</small>
    </div>
""",
    unsafe_allow_html=True,
)
