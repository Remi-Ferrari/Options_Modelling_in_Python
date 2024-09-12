import streamlit as st

st.set_page_config(
    page_title="Options Modelling in Python: Dashboard",
    layout="wide",
    initial_sidebar_state="expanded")

# session state init for navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home():
    st.session_state.page = "home"

def go_to_project1():
    st.session_state.page = "project1"

def go_to_project2():
    st.session_state.page = "project2"

def go_to_project3():
    st.session_state.page = "project3"

# function to run Project 1 (BS_Pricing)
def run_project1():

    import streamlit as st
    import scipy
    from scipy.stats import norm
    import numpy as np
    from numpy import log, sqrt, exp
    import matplotlib.pyplot as plt
    import seaborn as sns

    # CSS
    st.markdown("""
    <style>
    
    

    .FB1 {
        display: flex;
        margin-top: -55px; 
        margin-bottom: 8px;
        width: 80%;
        text-align: left;
        box-sizing: border-box;
        padding: 0.1em 0.22em;
        font-size: 22px;
        font-weight: bold;
        color: white;
        background-color: #090A0B;
        border: 1px solid #090A0B;
        border-radius: 4px;
        cursor: default;
        text-decoration: none;
    }
    .FB2 {
        display: flex;
        margin-top: -22px; 
        margin-bottom: 28px;
        width: 100%;
        text-align: left;
        box-sizing: border-box;
        padding: 0.1em 0.22em;
        font-size: 22px;
        font-weight: bold;
        color: white;
        background-color: #090A0B;
        border: 1px solid #090A0B;
        border-radius: 4px;
        cursor: default;
        text-decoration: none;
    }

    .stAlert { background-color: #353C41; border-radius: 4px; }  

    #black-scholes-eu-options-pricing-model > span:nth-child(1) > a:nth-child(1),
    #call-price-heatmap > span:nth-child(1) > a:nth-child(1),
    #put-price-heatmap > span:nth-child(1) > a:nth-child(1) {
        display: none !important;}

    .metric-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 8px; 
        width: auto; 
        margin: 0 auto; 
    }

    .metric-call {
        background-color: #90ee90; 
        color: black; 
        margin-right: 10px; 
        border-radius: 10px; 
    }

    .metric-put {
        background-color: #ffcccb; 
        color: black; 
        border-radius: 10px; 
    }

    .metric-value {
        font-size: 1.5rem; 
        font-weight: bold;
        margin: 0; 
    }

    .metric-label {
        font-size: 1rem; 
        margin-bottom: 4px; 
    }

    </style>
    """, unsafe_allow_html=True)

    # Black-Scholes Model
    class BlackScholes:
        def __init__(
                self,
                time_to_maturity: float,
                strike: float,
                current_price: float,
                volatility: float,
                interest_rate: float,
        ):
            self.time_to_maturity = time_to_maturity
            self.strike = strike
            self.current_price = current_price
            self.volatility = volatility
            self.interest_rate = interest_rate

        def calculate_prices(
                self,
        ):
            time_to_maturity = self.time_to_maturity
            strike = self.strike
            current_price = self.current_price
            volatility = self.volatility
            interest_rate = self.interest_rate

            d1 = (log(current_price / strike) + (interest_rate + 0.5 * volatility ** 2) * time_to_maturity) / (
                        volatility * sqrt(time_to_maturity))

            d2 = d1 - volatility * sqrt(time_to_maturity)

            call_price = current_price * norm.cdf(d1) - (
                        strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2))

            put_price = (strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)) - current_price * norm.cdf(
                -d1)

            self.call_price = call_price
            self.put_price = put_price

            return call_price, put_price

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="FB1">Model Parameters </div >', unsafe_allow_html=True)
        current_price = st.number_input("Underlying Asset Price", value=100.0)
        strike = st.number_input("Option Strike Price", value=100.0)
        time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
        volatility = st.number_input("Expected Volatility of Underlying Asset", value=0.2)
        interest_rate = st.number_input("Risk-Free Rate", value=0.05)

        st.markdown("---")

        st.markdown('<div class="FB2">Heatmap Parameters </div >', unsafe_allow_html=True)
        spot_min = st.slider('Min Spot Price', min_value=0.01, value=current_price * 0.8, step=0.01)
        spot_max = st.slider('Max Spot Price', min_value=0.01, value=current_price * 1.2, step=0.01)
        vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility * 0.5, step=0.01)
        vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility * 1.5, step=0.01)

        spot_range = np.linspace(spot_min, spot_max, 10)
        vol_range = np.linspace(vol_min, vol_max, 10)

        st.markdown("---")


    def plot_heatmap(bs_model, spot_range, vol_range, strike):
        call_prices = np.zeros((len(vol_range), len(spot_range)))
        put_prices = np.zeros((len(vol_range), len(spot_range)))

        for i, vol in enumerate(vol_range):
            for j, spot in enumerate(spot_range):
                bs_temp = BlackScholes(
                    time_to_maturity=bs_model.time_to_maturity,
                    strike=strike,
                    current_price=spot,
                    volatility=vol,
                    interest_rate=bs_model.interest_rate
                )
                bs_temp.calculate_prices()
                call_prices[i, j] = bs_temp.call_price
                put_prices[i, j] = bs_temp.put_price

        # Call Price Heatmap
        fig_call, ax_call = plt.subplots(figsize=(10, 8))
        sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True,
                    fmt=".2f", cmap="viridis", ax=ax_call)
        ax_call.set_title('CALL')
        ax_call.set_xlabel('Underlying Asset Price')
        ax_call.set_ylabel('Expected Volatility')

        # Put Price Heatmap
        fig_put, ax_put = plt.subplots(figsize=(10, 8))
        sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True,
                    fmt=".2f", cmap="viridis", ax=ax_put)
        ax_put.set_title('PUT')
        ax_put.set_xlabel('Spot Price')
        ax_put.set_ylabel('Volatility')

        return fig_call, fig_put

    # Main Page
    st.title("Black-Scholes EU Options Pricing Model")

    # Calculate Call and Put values
    bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
    call_price, put_price = bs_model.calculate_prices()

    st.markdown("")

    st.info(
        "Visualize potential Call and Put Option values based on the Black-Scholes model using the interactive heatmap. Model parameters include **Underlying Asset Price**, **Option Strike Price**, **Time to Maturity**, **Volatility** and **Risk-Free Rate**.")

    # Interactive Heatmaps
    col1, col2 = st.columns([1, 1], gap="small")

    with col1:
        st.subheader("Call Price Heatmap")
        heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, strike)
        st.pyplot(heatmap_fig_call)

    with col2:
        st.subheader("Put Price Heatmap")
        _, heatmap_fig_put = plot_heatmap(bs_model, spot_range, vol_range, strike)
        st.pyplot(heatmap_fig_put)

    col1, col2 = st.columns([1, 1], gap="small")

    # CALL & PUT Value boxes

    with col1:
        st.markdown(f"""
            <div class="metric-container metric-call">
                <div>
                    <div class="metric-label">CALL Value</div>
                    <div class="metric-value">${call_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-container metric-put">
                <div>
                    <div class="metric-label">PUT Value</div>
                    <div class="metric-value">${put_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)


# Function to run Project 2 (BS_PNL_Pricing)
def run_project2():

    import numpy as np
    import scipy
    from scipy.stats import norm
    from numpy import log, sqrt, exp
    import matplotlib.pyplot as plt
    import seaborn as sns

    # CSS
    st.markdown("""
    <style>

    #black-scholes-options-p-l-scenarios-interactive-heatmap > span:nth-child(1) > a:nth-child(1), 
    #call-option-p-l-heatmap > span:nth-child(1) > a:nth-child(1),
    #put-option-p-l-heatmap > span:nth-child(1) > a:nth-child(1) {
        display: none !important;}

    .metric-container {

        display: flex;
        justify-content: center;
        align-items: center;
        padding: 6px;
        width: 100%;
        margin-top: 10px;
        margin bottom: 5px;
        border-radius: 5px;
    }

    .metric-call { background-color: #90ee90; color: black; margin-top: 10px; }

    .metric-put { background-color: #ffcccb; color: black; margin-top: 10px; }

    .metric-value { font-size: 1.5rem; font-weight: bold; margin: 0; }

    .metric-label { font-size: 1rem; margin-bottom: 4px; }

    .stAlert { background-color: #353C41; border-radius: 4px; }  

    .FB1 {
        display: flex;
        margin-top: -50px; 
        margin-bottom: 8px;
        width: 80%;
        text-align: left;
        box-sizing: border-box;
        padding: 0.1em 0.22em;
        font-size: 22px;
        font-weight: bold;
        color: white;
        background-color: #090A0B;
        border: 1px solid #090A0B;
        border-radius: 4px;
        cursor: default;
        text-decoration: none;
    }

    .FB2 {
        display: flex;
        margin-top: -22px; 
        margin-bottom: 28px;
        width: 100%;
        text-align: left;
        box-sizing: border-box;
        padding: 0.1em 0.22em;
        font-size: 22px;
        font-weight: bold;
        color: white;
        background-color: #090A0B;
        border: 1px solid #090A0B;
        border-radius: 4px;
        cursor: default;
        text-decoration: none;
    }

    low-tooltip {
        top-margin: 5px;
    }

    </style>
    """, unsafe_allow_html=True)

    # Black-Scholes Pricing Model Class
    class BlackScholes:
        def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
            self.time_to_maturity = time_to_maturity
            self.strike = strike
            self.current_price = current_price
            self.volatility = volatility
            self.interest_rate = interest_rate

        def calculate_prices(self):
            d1 = (log(self.current_price / self.strike) + (
                    self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (
                         self.volatility * sqrt(self.time_to_maturity))
            d2 = d1 - self.volatility * sqrt(self.time_to_maturity)

            call_price = self.current_price * norm.cdf(d1) - (
                    self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2))
            put_price = (self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(
                -d2)) - self.current_price * norm.cdf(-d1)

            return call_price, put_price

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="FB1">Model Parameters </div >', unsafe_allow_html=True)
        current_price = st.number_input("Current Underlying Asset Price", value=100.0)
        strike = st.number_input("Strike Price", value=100.0)
        time_to_maturity = st.number_input("Time to Maturity (in Years)", value=1.0)
        volatility = st.number_input("Volatility (σ)", value=0.2)
        interest_rate = st.number_input("Risk-Free Rate", value=0.05)
        user_call_price = st.number_input("User Call Option Price", value=10.0, help="Provide call option price")
        user_put_price = st.number_input("User Put Option Price", value=10.0, help="Provide put option price")

        st.markdown("---")

        st.markdown('<div class="FB2">Heatmap Parameters</div>', unsafe_allow_html=True)
        spot_min = st.slider('Min Spot Price', min_value=0.01, value=current_price * 0.8, step=0.01)
        spot_max = st.slider('Max Spot Price', min_value=0.01, value=current_price * 1.2, step=0.01)
        vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility * 0.5,
                            step=0.01)
        vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility * 1.5,
                            step=0.01)

        spot_range = np.linspace(spot_min, spot_max, 10)
        vol_range = np.linspace(vol_min, vol_max, 10)

        st.markdown("---")

    # P&L Scenarios Heatmaps Function
    def plot_heatmap(bs_model, spot_range, vol_range, strike, user_option_price):
        pnl_call = np.zeros((len(vol_range), len(spot_range)))
        pnl_put = np.zeros((len(vol_range), len(spot_range)))

        for i, vol in enumerate(vol_range):
            for j, spot in enumerate(spot_range):
                bs_temp = BlackScholes(
                    time_to_maturity=bs_model.time_to_maturity,
                    strike=strike,
                    current_price=spot,
                    volatility=vol,
                    interest_rate=bs_model.interest_rate
                )
                call_price, put_price = bs_temp.calculate_prices()
                pnl_call[i, j] = call_price - user_option_price  # PnL for Call
                pnl_put[i, j] = put_price - user_option_price  # PnL for Put

        # Absolute color scale: green for positive P&L, red for negative P&L
        vmin = min(np.min(pnl_call), np.min(pnl_put))
        vmax = max(np.max(pnl_call), np.max(pnl_put))

        # Plotting Call PnL Heatmap
        fig_call, ax_call = plt.subplots(figsize=(10, 8))
        sns.heatmap(pnl_call, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True,
                    fmt=".2f", cmap="RdYlGn", center=0, vmin=vmin, vmax=vmax, ax=ax_call)
        ax_call.set_title(f'P&L - ${user_call_price:.2f} CALL')
        ax_call.set_xlabel('Spot Price')
        ax_call.set_ylabel('Volatility')

        # Plotting Put PnL Heatmap
        fig_put, ax_put = plt.subplots(figsize=(10, 8))
        sns.heatmap(pnl_put, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True,
                    fmt=".2f", cmap="RdYlGn", center=0, vmin=vmin, vmax=vmax, ax=ax_put)
        ax_put.set_title(f'P&L - ${user_put_price:.2f} PUT')
        ax_put.set_xlabel('Spot Price')
        ax_put.set_ylabel('Volatility')

        return fig_call, fig_put

    # Call/put value calculations
    bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
    call_price, put_price = bs_model.calculate_prices()

    st.title("Black-Scholes Options P&L Scenarios - Interactive Heatmap")

    st.info(
        "Visualize potential P&L scenarios based on the Black-Scholes model using the interactive heatmap.  \n"
        "The user-provided option price serves as the reference for P&L calculations."
    )

    # Heatmaps for PnL Scenarios
    col1, col2 = st.columns([1, 1], gap="small")

    with col1:

        st.subheader("Call Option P&L Heatmap")
        heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, strike, user_call_price)
        st.pyplot(heatmap_fig_call)

    with col2:

        st.subheader("Put Option P&L Heatmap")
        _, heatmap_fig_put = plot_heatmap(bs_model, spot_range, vol_range, strike, user_put_price)
        st.pyplot(heatmap_fig_put)

    with col1:
        st.markdown(f"""
            <div class="metric-container metric-call">
                <div>
                    <div class="metric-label">CALL Value</div>
                    <div class="metric-value">${call_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-container metric-put">
                <div>
                    <div class="metric-label">PUT Value</div>
                    <div class="metric-value">${put_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Calculate difference between Options Value and Price
    Call_PnL = call_price - user_call_price
    Put_PnL = put_price - user_put_price

    # Colour scheme based on P&L value
    def get_color(value, max_value):

        if max_value == 0:
            max_value = 1

        # Normalization
        normalized_value = value / max_value

        # Colour Gradient calculation
        if normalized_value >= 0:
            r = 255 - int(255 * normalized_value)
            g = int(255 * normalized_value)
        else:
            r = 255
            g = int(255 * (1 + normalized_value))

        return f"rgb({r}, {g}, 0)"

    # Calculate max absolute value
    max_pnl = max(abs(Call_PnL), abs(Put_PnL))

    # Get dynamic colors for each metric
    call_color = get_color(Call_PnL, max_pnl)
    put_color = get_color(Put_PnL, max_pnl)

    # P&L metrics with dynamic bg colour
    with col1:
        st.markdown(f"""
            <div class="metric-container" style="background-color: {call_color};">
                <div>
                    <div class="metric-label">CALL P&L</div>
                    <div class="metric-value">${Call_PnL:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("", help="User CALL Option Price - CALL Value")

    with col2:
        st.markdown(f"""
            <div class="metric-container" style="background-color: {put_color};">
                <div>
                    <div class="metric-label">PUT P&L
                         </div>
                    <div class="metric-value">${Put_PnL:.2f}</div>                       
                </div>
            </div>

        """, unsafe_allow_html=True)

        st.markdown("", help="User PUT Option Price - PUT Value")

# Function to run Project 3 (Basic_PNL)
def run_project3():

    import numpy as np
    import matplotlib.pyplot as plt

    # CSS

    st.markdown("""
    <style>
    .FB1 {
        display: flex;
        margin-top: -50px; 
        margin-bottom: 8px;
        width: 85%;
        text-align: left;
        box-sizing: border-box;
        padding: .06em 0.250em;
        font-size: 22px;
        font-weight: bold;
        color: white;
        background-color: #090A0B;
        border: 0.75px solid #090A0B;
        border-radius: 4px;
        cursor: default;
        text-decoration: none;
    }
    .FB2 {
        display: flex;
        margin-top: -22px; 
        margin-bottom: 28px;
        width: 100%;
        text-align: left;
        box-sizing: border-box;
        padding: 0.1em 0.22em;
        font-size: 22px;
        font-weight: bold;
        color: white;
        background-color: #090A0B;
        border: 1px solid #090A0B;
        border-radius: 4px;
        cursor: default;
        text-decoration: none;
    }
    .stAlert { background-color: #353C41; border-radius: 4px; }  

    #options-p-l-heatmaps-and-graph > span:nth-child(1) > a:nth-child(1),
    #call-option-analysis > span:nth-child(1) > a:nth-child(1),
    #put-option-analysis > span:nth-child(1) > a:nth-child(1)
     {display: none !important;}

    </style>
    """, unsafe_allow_html=True)

    # User Inputs for Hyperparameters
    with st.sidebar:
        # Global Params
        st.markdown('<div class="FB1">Global Parameters </div >', unsafe_allow_html=True)

        # Separate inputs for call and put option prices
        call_option_price = st.number_input('Call Option Price', min_value=1.0, max_value=100.0, value=10.0, key='call_option_price')
        put_option_price = st.number_input('Put Option Price', min_value=1.0, max_value=100.0, value=10.0, key='put_option_price')

        st.markdown("---")

        # Heatmap Params
        st.markdown('<div class="FB2">Heatmap Parameters </div >', unsafe_allow_html=True)
        spot_price_range = st.slider('Underlying Asset Price Range', min_value=50, max_value=200, value=(80, 150), step=1, key='spot_price_range')
        strike_price_range = st.slider('Strike Price Range', min_value=50, max_value=200, value=(70, 160), step=1, key='strike_price_range')
        tick_spacing = st.number_input('Tick Spacing', min_value=4, max_value=20, value=5, key='tick_spacing')

        st.markdown("---")

        # Graph params
        st.markdown('<div class="FB2">Graph Parameters </div >', unsafe_allow_html=True)
        spot_price_range2 = st.slider('Underlying Asset Price Range for Graph', min_value=0, max_value=200, value=(80, 150), step=1, key='spot_price_range2')
        pnl2 = st.slider('P&L Range', min_value=((max(call_option_price, put_option_price)) * -1) - 5, max_value=200.00, value=(min(call_option_price, put_option_price) * -1 - 2, 100.00), step=0.1, key='pnl2')
        call_strike_price2 = st.number_input('Call Strike Price for Graph', min_value=1.0, max_value=100.0, value=10.0, key='call_strike_price2')
        put_strike_price2 = st.number_input('Put Strike Price for Graph', min_value=1.0, max_value=100.0, value=10.0, key='put_strike_price2')
        graph_tick_spacing = st.sidebar.number_input('Tick Spacing for Graphs', min_value=1, max_value=25, value=10, key='graph_tick_spacing')

        st.markdown("---")

    st.title("Theoretical Option P&L Heatmap and Graph")

    st.info("""
        Use the interactive heatmap and graph below to explore the profit and loss (P&L) of different options scenarios based on the **option price**, **strike price**, and **underlying spot price**.
        """)

    # Calculate Spot and Strike Prices
    spot_prices = np.arange(spot_price_range[0], spot_price_range[1] + 1, 1)
    strike_prices = np.arange(strike_price_range[0], strike_price_range[1] + 1, 1)

    # Create the P&L grids
    call_pnl_grid = np.zeros((len(strike_prices), len(spot_prices)))
    put_pnl_grid = np.zeros((len(strike_prices), len(spot_prices)))

    # Populate the P&L grids
    for i, strike in enumerate(strike_prices):
        for j, spot in enumerate(spot_prices):
            # Call Option PnL
            call_pnl = max(0, spot - strike) - call_option_price
            call_pnl_grid[i, j] = call_pnl

            # Put Option PnL
            put_pnl = max(0, strike - spot) - put_option_price
            put_pnl_grid[i, j] = put_pnl

    # Create two columns for call and put options
    col1, col2 = st.columns(2)

    with col1:
        st.header("Call Option Analysis")

        # Call Option Heatmap
        fig_call_heatmap, ax_call_heatmap = plt.subplots(figsize=(7, 5))
        cax1 = ax_call_heatmap.imshow(call_pnl_grid, cmap='RdYlGn', interpolation='bilinear', aspect='auto', extent=[spot_prices.min(), spot_prices.max(), strike_prices.min(), strike_prices.max()], origin='lower')
        ax_call_heatmap.set_title('Call Option P&L Heatmap')
        ax_call_heatmap.set_xlabel('Price of Underlying (USD)')
        ax_call_heatmap.set_ylabel('Strike Price (USD)')
        ax_call_heatmap.set_xticks(np.arange(spot_prices.min(), spot_prices.max() + tick_spacing, tick_spacing))
        ax_call_heatmap.set_yticks(np.arange(strike_prices.min(), strike_prices.max() + tick_spacing, tick_spacing))
        fig_call_heatmap.colorbar(cax1, ax=ax_call_heatmap, label='PnL (USD)')
        st.pyplot(fig_call_heatmap)

        # Call Option graph
        spot_prices2 = np.arange(spot_price_range2[0], spot_price_range2[1] + 1, 1)
        call_values = np.maximum(spot_prices2 - call_strike_price2, 0) - call_option_price

        fig_call, ax_call = plt.subplots(figsize=(7, 5))
        ax_call.plot(spot_prices2, call_values, label='Call Option Value', color='green')
        ax_call.fill_between(spot_prices2, 0, call_values, where=(call_values > 0), color='green', alpha=0.3, label='Call Profit Area')
        ax_call.fill_between(spot_prices2, call_values, 0, where=(call_values < 0), color='red', alpha=0.3, label='Call Loss Area')
        ax_call.set_ylim(pnl2[0], pnl2[1])
        ax_call.axhline(0, color='black', linewidth=1)
        ax_call.set_xlabel('Spot Price of Underlying (USD)')
        ax_call.set_xticks(np.arange(spot_prices2.min(), spot_prices2.max() + graph_tick_spacing, graph_tick_spacing))
        ax_call.set_ylabel('Profit and Loss')
        ax_call.set_title('Theoretical Value of Call Option')
        ax_call.legend()
        st.pyplot(fig_call)

    with col2:
        st.header("Put Option Analysis")

        # Put Option Heatmap
        fig_put_heatmap, ax_put_heatmap = plt.subplots(figsize=(7, 5))
        cax2 = ax_put_heatmap.imshow(put_pnl_grid, cmap='RdYlGn', interpolation='bilinear', aspect='auto', extent=[spot_prices.min(), spot_prices.max(), strike_prices.min(), strike_prices.max()], origin='lower')
        ax_put_heatmap.set_title('Put Option P&L Heatmap')
        ax_put_heatmap.set_xlabel('Price of Underlying (USD)')
        ax_put_heatmap.set_ylabel('Strike Price (USD)')
        ax_put_heatmap.set_xticks(np.arange(spot_prices.min(), spot_prices.max() + tick_spacing, tick_spacing))
        ax_put_heatmap.set_yticks(np.arange(strike_prices.min(), strike_prices.max() + tick_spacing, tick_spacing))
        fig_put_heatmap.colorbar(cax2, ax=ax_put_heatmap, label='PnL (USD)')
        st.pyplot(fig_put_heatmap)

        # Put Option graph
        put_values = np.maximum(put_strike_price2 - spot_prices2, 0) - put_option_price

        fig_put, ax_put = plt.subplots(figsize=(7, 5))
        ax_put.plot(spot_prices2, put_values, label='Put Option Value', color='red')
        ax_put.fill_between(spot_prices2, 0, put_values, where=(put_values > 0), color='green', alpha=0.3, label='Put Profit Area')
        ax_put.fill_between(spot_prices2, put_values, 0, where=(put_values < 0), color='red', alpha=0.3, label='Put Loss Area')
        ax_put.set_ylim(pnl2[0], pnl2[1])
        ax_put.axhline(0, color='black', linewidth=1)
        ax_put.set_xlabel('Spot Price of Underlying (USD)')
        ax_put.set_xticks(np.arange(spot_prices2.min(), spot_prices2.max() + graph_tick_spacing, graph_tick_spacing))
        ax_put.set_ylabel('Profit and Loss')
        ax_put.set_title('Theoretical Value of Put Option')
        ax_put.legend()
        st.pyplot(fig_put)

# Main Page or Home page
if st.session_state.page == "home":

    # Home Page Content
    st.title("Options Pricing Dashboard")
    # Line
    st.markdown(
        """<hr style="height:1px;width:45%;margin-top:-12px;margin-bottom:-40px;border:none;color:#333;background-color:#333;" /> """,
        unsafe_allow_html=True)

    st.markdown(
        """
        <p style='font-size: 12px; margin-top: -15px; margin-bottom: -10px;'>Double-Click a tool below:</p>
        """,
        unsafe_allow_html=True
    )

    css = """
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(132deg, 
                #00000A 38%,
                #0F0F16 45%,
                #1D1D21 50%,    
                #2F2F2F 85%,       
                #7F462C 92%,
                #844B31 100%
                  
            );
            background-attachment: fixed;
        }
        
        body {
      margin: 0;
      padding: 0;
    }
    
    .logo-container {
      position: fixed;
      bottom: 20px; 
      right: 20px; 
      display: flex;
      gap: 10px; 
    }
    
    .logo-container img {
      vertical-align: middle;
    }
    
    .footer-container {
      position: fixed;
      bottom: 50px; 
      right: 20px; 
      text-align: right; 
    }
    
    .footer-text1 {
      font-size: 11px; 
      color: #ffffff; 
      margin-bottom: -4px; 
    }
    .footer-text2 {
      font-size: 13px; 
      color: #ffffff; 
    
    .stAlert { background-color: #FF0000; border-radius: 4px; }  

    
        </style>
        """

    st.markdown(css, unsafe_allow_html=True)


linkedin_url = "https://www.linkedin.com/in/r%C3%A9mi-ferrari-137032293/"
github_url = "https://github.com/Remi-Ferrari"
if st.session_state.page == "home":
    st.markdown('''
    <div class="footer-container">
        <div class="footer-text1">
            A project by:
        </div>
        <div class="footer-text2">
            Rémi Ferrari
        </div>
    </div>
    <div class="logo-container">
        <a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="25" height="25">
        </a>
        <a href="{github_url}" target="_blank" style="text-decoration: none; color: inherit;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" height="25">
        </a>
    </div>
    '''.format(linkedin_url=linkedin_url, github_url=github_url), unsafe_allow_html=True)

if st.session_state.page == "home":

    # Buttons to navigate to each project
    if st.button("Black-Scholes EU Option Pricing Model"):
        go_to_project1()

    if st.button("Black-Scholes Options P&L Scenarios - Interactive Heatmap"):
        go_to_project2()

    if st.button("Theoretical Option P&L Heatmap and Graph"):
        go_to_project3()

elif st.session_state.page == "project1":
    run_project1()
    st.sidebar.button("Home", on_click=go_home)

elif st.session_state.page == "project2":
    run_project2()
    st.sidebar.button("Home", on_click=go_home)

elif st.session_state.page == "project3":
    run_project3()
    st.sidebar.button("Home", on_click=go_home)


if st.session_state.page == "home":

    col1, col2 = st.columns([6, 9])
    st.write("")
    with col1:
        st.write("")
        st.write("")

        st.info("""
                        This platform is dedicated to showcasing projects focused on the topic of options pricing.
                        Dive into the world of options and explore their pricing mechanisms and the impact on
                        profit and loss through the projects listed below. Whether you are looking to understand 
                        the fundamentals, visualize basic options pricing, or analyze P&L charts, 
                        you will find a range of tools built to develop intuition. 
                        Delve into more advanced models like the Black-Scholes options pricing model 
                        and discover how various factors influence options pricing and their outcomes.

                        The project was built using Python, CSS & html coding languages. 
                        Python framework, Streamlit, was utilized for displaying this project. 
                        Python libraries include: scipy, numpy, seaborn and matplotlib. 

                        Much credit goes to the prior work of [Prudhvi Reddy M](https://github.com/prudhvi-reddy-m) 
                        and [Tim Freiberg](https://github.com/tmfreiberg) within this area.



                        """)

