import numpy as np
import pandas as pd
from numpy import log, exp, sqrt
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.style.use('fivethirtyeight')

class BlackScholes:
    """
    Black-Scholes model for calculating the price of a European call option.
    
    The class calculates the call option price using the Black-Scholes formula 
    based on the given parameters such as stock price, strike price, volatility, 
    risk-free rate, and time to maturity.

    Args:
        S (float) : Fixed Initial stock price (to simplify the interpretation)
        X (float) : Strike price
        r (float) : Risk-free interest rate
        sigma (float) : Volatility of the stock
        T (float) : Time to maturity
    """
    
    def __init__(self, S, X, r, sigma, T):
        """
        Initialize the Black-Scholes model with the given parameters.
        
        Args:
            S (float) : Fixed Initial stock price (to simplify the interpretation)
            X (float) : Strike price
            r (float) : Risk-free interest rate
            sigma (float) : Volatility of the stock
            T (float) : Time to maturity
        """
        self.S = S  # Stock price
        self.X = X  # Strike price
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility
        self.T = T  # Time to maturity

    def call_price(self, t):
        """
        Calculate the Black-Scholes call option price at time t.
        
        This method computes the price of a European call option at a given 
        time using the Black-Scholes model.

        Args:
            t (float) : Current time (between 0 and T)

        Returns:
            float : The call option price at time t
        """
        # Compute the d1 and d2 terms used in the Black-Scholes formula
        d1 = (log(self.S / self.X) + (self.r + (self.sigma**2) / 2) * (self.T - t)) / (self.sigma * sqrt(self.T - t))
        d2 = d1 - self.sigma * sqrt(self.T - t)
        
        # Black-Scholes formula for the call option price
        return self.S * stats.norm.cdf(d1) - self.X * exp(-self.r * (self.T - t)) * stats.norm.cdf(d2)

    def get_call_prices_for_times(self, t_range):
        """
        Get the call option prices for a given time range.
        
        Args:
            t_range (array-like) : Array of time values (between 0 and T)
        
        Returns:
            list : List of call option prices at the given times
        """
        return [self.call_price(t) for t in t_range]

    def get_call_prices_for_strikes(self, t_range, strikes):
        """
        Get a DataFrame of call option prices for a range of strikes and times to maturity.
        
        Args:
            t_range (array-like) : Array of time values (between 0 and T)
            strikes (array-like) : List of strike prices
            
        Returns:
            pd.DataFrame : DataFrame with time values as index and strike prices as columns
        """
        data = {}
        for strike in strikes:
            bs = BlackScholes(self.S, strike, self.r, self.sigma, self.T)
            data[strike] = bs.get_call_prices_for_times(t_range)
        return pd.DataFrame(data, index=t_range)

    def plot_call_price_vs_time(self, df):
        """
        Plot the call option price as a function of time to maturity for different strike prices.
        
        Args:
            df (pd.DataFrame) : DataFrame where each column represents a strike price
                                 and rows correspond to call prices for each time value.
        """
        # Create a new figure for plotting
        plt.figure(figsize=(10, 8))

        sns.lineplot(data=df, dashes=False, palette="Set1")  # Ensure solid lines and color differentiation  # "Set1" provides a good set of colors
        
        # Add labels and legend to the plot
        plt.xlabel('t')  # Time to maturity
        plt.ylabel('C')   # Call option price
        plt.ylim(-0.5, 10) # Set y-axis limits for better readability
        plt.legend(title='Strike Price', loc='upper left')      # Show legend for different strikes
        plt.show()        # Display the plot

    def plot_call_price_vs_stock(self, s_range, t):
        """
        Plot the call option price as a function of stock price at a fixed time.
        
        Args:
            s_range (array-like) : Array of stock prices to evaluate the option price at
            t (float) : Fixed time value to evaluate the option price at
        """
        # Create a new figure for plotting
        plt.figure(figsize=(10, 8))
        
        # Calculate the call prices for all stock prices in s_range at the fixed time t
        sns.lineplot(x=s_range, y=[BlackScholes(s, self.X, self.r, self.sigma, self.T).call_price(t) for s in s_range], label=f'X = {self.X}')
        
        # Add labels and legend to the plot
        plt.xlabel('S')  # Stock price
        plt.ylabel('C(t=T)')  # Call option price at time T
        plt.legend()         # Show legend for the strike price
        plt.show()           # Display the plot


if __name__ == "__main__":
    """
    Example usage of the Black-Scholes model.
    
    In this section, we create instances of the Black-Scholes class with different
    parameters and plot the results using the `plot_call_price_vs_time` and 
    `plot_call_price_vs_stock` methods.
    """
    # Create a Black-Scholes model for a stock price of 100, strike price of 100, risk-free rate 0.06, volatility 0.3, and time to maturity 1 year
    bs_model = BlackScholes(S=100, X=100, r=0.06, sigma=0.3, T=1)

    # Calculate the call option price at time t = 0.999 for the strike price 95
    print(f"Call option price at t = 0.999: {bs_model.call_price(0.999)}")
    
    # Define the range of time values from 0.75 to 1.0 with a step size of 0.0001
    t_values = np.arange(0.75, 1.0, 0.0001)
    # Define the range of stock prices from 95 to 105 with a step size of 0.01 [Covers in-the-money, at-the-money, and out-of-the-money cases.]
    s_values = np.arange(95, 105, 0.01)

    # Get the DataFrame of call prices for the different strikes
    strikes = [95, 98, 100, 105]
    df = bs_model.get_call_prices_for_strikes(t_values, strikes)

    print(df.head())  # Print the first few rows of the DataFrame
    # Plot the call prices using the returned DataFrame
    bs_model.plot_call_price_vs_time(df)
    
    # Plot the call option prices as a function of stock price at time t=0.999
    bs_model.plot_call_price_vs_stock(s_values, t=0.99999)
