import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.option_pricing.Black_Scholes import BlackScholes
from src.option_pricing.Monte_Carlo import BS_Call_MC
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def plot_black_scholes_prices():
    """
    This function generates plots for the Black-Scholes option pricing model.
    It generates two plots:
    1. Call price vs. time to maturity
    2. Call price vs. stock price at a given time before maturity.
    """

    # Define the time and stock price ranges for plotting
    t_values = np.arange(0.75, 1.0, 0.0001)  # Time range (from 0.75 to 1.0)
    s_values = np.arange(
        95, 105, 0.01
    )  # Stock price range (from 95 to 105) [Covers in-the-money, at-the-money, and out-of-the-money cases.]

    # Create an instance of the Black-Scholes model with specified parameters
    bs_model = BlackScholes(S=100, X=100, r=0.06, sigma=0.3, T=1)

    # Get the DataFrame of call prices for the different strikes
    strikes = [95, 98, 100, 105]
    df = bs_model.get_call_prices_for_strikes(t_values, strikes)

    # Plot the call option price vs. time to maturity (t)
    bs_model.plot_call_price_vs_time(df)

    # Plot the call option price vs. stock price (S) for t = 0.99999 (near maturity)
    bs_model.plot_call_price_vs_stock(s_values, t=0.99999)


def compare_option_prices():
    """
    This function compares option prices from the Monte Carlo simulation and the Black-Scholes formula
    for two scenarios: Strike 95 and Time-to-Maturity (T-t = 0.25 and T-t = 0.75).
    """
    # Scenario 1: Monte Carlo and exact Black-Scholes for T-t = 0.25
    print(
        "Monte Carlo (Strike 95, t = 0.999):",
        BS_Call_MC(100, 95, 0.06, 0.3, 1, 0.999, 10).calculate_option_price(),
    )
    print(
        "Exact (Strike 95, t = 0.999):",
        BlackScholes(S=100, X=95, r=0.06, sigma=0.3, T=1).call_price(0.999),
    )

    # Scenario 2: Monte Carlo and exact Black-Scholes for T-t = 0.75
    print(
        "Monte Carlo (Strike 95, t = 0.75):",
        BS_Call_MC(100, 95, 0.06, 0.3, 1, 0.75, 100000).calculate_option_price(),
    )
    print(
        "Exact (Strike 95, t = 0.75):",
        BlackScholes(S=100, X=95, r=0.06, sigma=0.3, T=1).call_price(0.75),
    )


def plot_monte_carlo_vs_exact(curr_time):
    """
    This function compares the option price calculated by Monte Carlo simulations
    for varying number of iterations against the exact Black-Scholes price at a given time to maturity (curr_time).

    Args:
        curr_time (float): The current time for option pricing.
    """
    # Create an empty DataFrame to store Monte Carlo results for different iteration values
    df = pd.DataFrame(columns=["Iter", "BSc"])

    # Run Monte Carlo simulation for different numbers of iterations and store results
    for i in range(1, 100000, 500):  # Iterating from 1 to 100000 in steps of 500
        new_row = pd.DataFrame(
            {
                "Iter": [i],
                "BSc": [
                    BS_Call_MC(
                        100, 95, 0.06, 0.3, 1, curr_time, i
                    ).calculate_option_price()
                ],
            }
        )
        df = pd.concat([df, new_row], ignore_index=True)

    # Plotting the results of Monte Carlo simulations
    plt.figure(figsize=(10, 8))

    # Plot the exact Black-Scholes price as a horizontal red dotted line
    plt.hlines(
        BlackScholes(S=100, X=95, r=0.06, sigma=0.3, T=1).call_price(curr_time),
        xmin=0,
        xmax=100000,
        linestyle="dotted",
        colors="red",
        label="Exact",
    )

    # Plot Monte Carlo simulation results
    plt.plot(df.set_index("Iter"), lw=1.5, label="Monte Carlo")

    # Add title and labels to the plot
    plt.title("S=100, X=95, T-t=0.25")
    plt.xlabel("Iterations")
    plt.ylabel("Call Price")

    # Set y-axis limits based on exact Black-Scholes price
    plt.ylim(
        BlackScholes(S=100, X=95, r=0.06, sigma=0.3, T=1).call_price(curr_time) - 1,
        BlackScholes(S=100, X=95, r=0.06, sigma=0.3, T=1).call_price(curr_time) + 1,
    )

    # Show legend and plot
    plt.legend()
    plt.show()


def plot_monte_carlo_vs_exact_for_strikes():
    """
    This function compares the option prices calculated using Monte Carlo simulations
    and the Black-Scholes model for different strike prices (95, 98, 100, 105) over time to maturity.
    """
    # DataFrame to store Monte Carlo results for different strikes and times
    df = pd.DataFrame(columns=["t", "95", "98", "100", "105"])

    # Define time range (from 0.75 to 1.0)
    t = np.arange(0.75, 1.0, 0.001)

    # Run Monte Carlo simulations for different strikes and times
    for i in t:
        new_row = pd.DataFrame(
            {
                "t": [i],
                "95": [
                    BS_Call_MC(
                        100, 95, 0.06, 0.3, 1, i, 100000
                    ).calculate_option_price()
                ],
                "98": [
                    BS_Call_MC(
                        100, 98, 0.06, 0.3, 1, i, 100000
                    ).calculate_option_price()
                ],
                "100": [
                    BS_Call_MC(
                        100, 100, 0.06, 0.3, 1, i, 100000
                    ).calculate_option_price()
                ],
                "105": [
                    BS_Call_MC(
                        100, 105, 0.06, 0.3, 1, i, 100000
                    ).calculate_option_price()
                ],
            }
        )
        df = pd.concat([df, new_row], ignore_index=True)

    # Plotting results for varying strikes and times
    plt.figure(figsize=(10, 8))

    # Plot exact Black-Scholes solution for each strike price
    plt.plot(
        t,
        BlackScholes(S=100, X=95, r=0.06, sigma=0.3, T=1).call_price(t),
        alpha=0.6,
        label="X = 95",
    )
    plt.plot(
        t,
        BlackScholes(S=100, X=98, r=0.06, sigma=0.3, T=1).call_price(t),
        alpha=0.6,
        label="X = 98",
    )
    plt.plot(
        t,
        BlackScholes(S=100, X=100, r=0.06, sigma=0.3, T=1).call_price(t),
        alpha=0.6,
        label="X = 100",
    )
    plt.plot(
        t,
        BlackScholes(S=100, X=105, r=0.06, sigma=0.3, T=1).call_price(t),
        alpha=0.6,
        label="X = 105",
    )

    # Plot Monte Carlo results for each strike price
    plt.plot(df["t"], df["95"], lw=2, c="r")
    plt.plot(df["t"], df["98"], lw=2, c="r")
    plt.plot(df["t"], df["100"], lw=2, c="r")
    plt.plot(df["t"], df["105"], lw=2, c="r")

    # Add labels, legend, and display the plot
    plt.legend()
    plt.xlabel("Time to Maturity (t)")
    plt.ylabel("Call Price")
    plt.show()


def main():
    """
    Main function that calls all other functions to generate plots and compare results.
    """

    # Plot Black-Scholes prices for different stock price and time-to-maturity ranges
    plot_black_scholes_prices()

    # Compare option prices using Monte Carlo vs exact Black-Scholes pricing for different strike prices and times
    compare_option_prices()

    # Plot Monte Carlo vs exact pricing for different iterations at T-t = 0.25 and T-t = 0.99
    plot_monte_carlo_vs_exact(0.75)
    plot_monte_carlo_vs_exact(0.99)

    # Plot Monte Carlo vs exact pricing for different strikes and times
    plot_monte_carlo_vs_exact_for_strikes()


if __name__ == "__main__":
    main()
