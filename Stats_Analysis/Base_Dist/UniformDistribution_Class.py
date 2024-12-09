import numpy as np
from scipy.stats import uniform
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Truncation is required for defining a valid probability distribution so it is built into the start rather than an optional extra
# inherently defined over a finite interval [lower_bound, upper_bound]
class UniformDistribution:
    """
    Uniform distribution probability distribution.

    This class supports computation of the PDF and CDF for scalar and array inputs, defined over a finite interval [lower_bound, upper_bound].

    Parameters
    ----------
    lower_bound : float
        The lower bound of the uniform distribution.
    upper_bound : float
        The upper bound of the uniform distribution.

    Raises
    ------
    ValueError
        If lower_bound >= upper_bound.
    """

    def __init__(self, lower_bound, upper_bound):
        """
        Initialize the uniform distribution over the interval [lower_bound, upper_bound].
        """
        if lower_bound >= upper_bound:
            raise ValueError("Lower bound must be less than upper bound.")
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Scale parameter for scipy.stats.uniform
        self.scipy_scale = upper_bound - lower_bound

        # Exploit scipys uniform  distribution as underlying distribution
        self.dist = uniform(loc=lower_bound, scale=self.scipy_scale)

    def pdf(self, X):
        """
        Calculate the Probability Density Function (PDF).

        Parameters
        ----------
        X : float or np.ndarray
            The value(s) at which to evaluate the PDF.

        Returns
        -------
        float or np.ndarray
            The normalized PDF value(s) which are 0 for X outside [lower_bound, upper_bound].
        """
        return self.dist.pdf(X)

    # In the case of the uniform there is nothing to fit/optimise so the PDF is the same
    def pdf_fitting(self, X): 
        """
        Calculate the Probability Density Function (PDF) for a fit.

        Parameters
        ----------
        X : float or np.ndarray
            The value(s) at which to evaluate the PDF.

        Returns
        -------
        float or np.ndarray
            The normalized PDF value(s) which are 0 for X outside [lower_bound, upper_bound].
        """
        return self.dist.pdf(X)
    
    def cdf(self, X):
        """
        Compute the Cumulative Distribution Function (CDF).

        Parameters
        ----------
        X : float or np.ndarray
            The value(s) at which to evaluate the CDF.

        Returns
        -------
        float or np.ndarray
            The CDF value(s).
        """
        return self.dist.cdf(X)
    
    def cdf_fitting(self, X): 
        """
        Calculate the Probability Density Function (PDF) for a fit.

        Parameters
        ----------
        X : float or np.ndarray
            The value(s) at which to evaluate the PDF.

        Returns
        -------
        float or np.ndarray
            The normalized PDF value(s) which are 0 for X outside [lower_bound, upper_bound].
        """
        return self.dist.cdf(X)


    def normalisation_check(self):
        """
        Perform a numerical integration using scipy.integrate.quad to check the normalization of the PDF.

        If the PDF has been truncated:
        It is first performed over the region the PDF is defined [lower_bound, upper_bound]

        It is then performed over the entire real line (-infinity to infinity).

        Prints the results of the numerical integrations.
        """
        
        if self.lower_bound is None:
            lower_bound = -np.inf
        else:
            lower_bound = self.lower_bound

        if self.upper_bound is None:
            upper_bound = np.inf
        else:
            upper_bound = self.upper_bound

        if self.lower_bound is not None or self.upper_bound is not None:
            print(f"Normalisation over the region the PDF is defined/truncated: [{lower_bound},{upper_bound}]")
            integral_bounded, error_bounded = quad(lambda x: self.pdf(x), lower_bound, upper_bound)
            print(f"Integral: {integral_bounded} \u00B1 {error_bounded}")

        print(f"Normalisation over the whole real line: [infinity to infinity]")
        integral_inf, error_inf = quad(lambda x: self.pdf(x), -np.inf, np.inf)
        print(f"Integral: {integral_inf} \u00B1 {error_inf}")


    def plot_dist(self):
        """
        Plot the PDF and CDF for the Crystal Ball distribution.

        If both lower and upper bounds are set:
        The PDF and CDF are plotted between[lower_bound, upper_bound].

        If both lower and upper bounds are not set:
        The PDF and CDF is plotted between [mu - 5*sigma, mu + 5*sigma]
        """

        X = np.linspace(self.lower_bound-0.1*(self.upper_bound-self.lower_bound), self.upper_bound+0.1*(self.upper_bound-self.lower_bound), 1000)


        # LHS Plot: the PDF
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(X, self.pdf(X), color='black', linestyle='-', label='PDF')
        plt.xlim(X[0], X[-1])
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Normal PDF(X)', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=14)
        plt.legend()

        # RHS plot: the CDF
        plt.subplot(1, 2, 2)
        plt.plot(X, self.cdf(X), color='black', linestyle='-', label='CDF')
        plt.xlim(X[0], X[-1])
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Normal CDF(X)', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=14)
        plt.legend()

        plt.tight_layout()
        plt.show()