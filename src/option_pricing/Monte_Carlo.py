import numpy as np
from numpy import log, exp, sqrt

class BS_Call_MC:
    """
    Black-Scholes Monte Carlo simulation for pricing a European call option.
    
    This class uses the Monte Carlo method to estimate the price of a European 
    call option using the Black-Scholes model. The simulation models the stock 
    price at maturity, calculates the payoff, and discounts it to the present.

    Args:
        S (float) : Initial stock price
        X (float) : Strike price
        r (float) : Risk-free interest rate
        sigma (float) : Volatility of the stock
        T (float) : Time to maturity
        t (float) : Current time (0 <= t <= T)
        I (int) : Number of simulations for Monte Carlo method
    """
    
    def __init__(self, S, X, r, sigma, T, t, I):
        """
        Initialize the parameters for the Black-Scholes Monte Carlo simulation.
        
        Args:
            S (float) : Initial stock price
            X (float) : Strike price
            r (float) : Risk-free interest rate
            sigma (float) : Volatility of the stock
            T (float) : Time to maturity
            t (float) : Current time
            I (int) : Number of simulations for Monte Carlo method
        """
        self.S = S        # Initial stock price
        self.X = X        # Strike price
        self.r = r        # Risk-free interest rate
        self.sigma = sigma # Volatility of the stock
        self.T = T        # Time to maturity
        self.t = t        # Current time
        self.I = I        # Number of simulations

    def calculate_option_price(self):
        """
        Calculate the price of the European call option using Monte Carlo simulation.
        
        This method simulates the stock price at maturity, calculates the payoff 
        of the option, and discounts it back to the present using the risk-free rate.
        
        Returns:
            float : The estimated price of the European call option
        """
        # Create an array to store the simulated stock prices and payoff data
        data = np.zeros((self.I, 2))
        
        # Generate random numbers from a normal distribution (standard normal)
        z = np.random.normal(0, 1, [1, self.I])
        
        # Simulate the stock price at maturity using the Black-Scholes formula
        ST = self.S * exp((self.T - self.t) * (self.r - 0.5 * self.sigma**2) + 
                          self.sigma * sqrt(self.T - self.t) * z)
        
        # Calculate the payoff (ST - X) for each simulation
        data[:, 1] = ST - self.X
        
        # Ensure no negative payoff by taking the maximum of each value (option payoff cannot be negative)
        # Then average the payoffs over all simulations
        average = np.sum(np.amax(data, axis=1)) / float(self.I)
        
        # Discount the average payoff to the present value using the risk-free rate
        return np.exp(-self.r * (self.T - self.t)) * average

if __name__ == "__main__":
    """
    Example usage of the BS_Call_MC class.
    
    This section demonstrates how to create an instance of the BS_Call_MC class,
    calculate the European call option price using the Monte Carlo method, 
    and print the result.
    """
    # Initialize the BS_Call_MC class with parameters: stock price, strike price, 
    # risk-free rate, volatility, time to maturity, current time, and number of simulations.
    option = BS_Call_MC(100, 95, 0.06, 0.3, 1, 0.999, 100000)

    # Call the method to calculate the option price using Monte Carlo simulation
    price = option.calculate_option_price()

    # Print the calculated option price
    print(f"The option price is: {price}")
