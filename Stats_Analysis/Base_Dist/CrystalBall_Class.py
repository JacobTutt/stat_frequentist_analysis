import numpy as np
from scipy.stats import crystalball
from scipy.integrate import quad
import matplotlib.pyplot as plt

class CrystalBall:
    """
    Crystal Ball probability distribution.

    The Crystal Ball distribution is lower_bound modified Gaussian distribution with lower_bound power-law tail on one side. 
    This class supports computation of the PDF and CDF for scalar and array inputs, with optional truncation.

    Parameters
    ----------
    mu : float
        The mean of the Crystal Ball distribution.
    sigma : float
        The standard deviation of the Crystal Ball distribution.
    beta : float
        The threshold value where the distribution transitions from Gaussian to power-law. 
        Must be beta > 0.
    m : float
        The power-law tail exponent. Controls the steepness of the power-law tail. 
        Must be m > 1.
    lower_bound : float, optional
        The lower bound. Default is None, meaning no lower bound is applied.
    upper_bound : float, optional
        The upper bound. Default is None, meaning no upper bound is applied.
    
    Raises
    ------
    ValueError
        If beta <= 0.
        If m <= 1.
        If lower_bound >= upper_bound.
    """

    def __init__(self, mu, sigma, beta, m, lower_bound = None, upper_bound = None):
        """
        Initialize the Crystal Ball distribution with optional truncation.
        """

        self.mu = mu
        self.sigma = sigma

        # Checks for beta > 0 and m > 1 before initialising
        if beta <= 0:
            raise ValueError("beta must be greater than 0")
        if m <= 1:
            raise ValueError("m must be greater than 1")
        
        self.beta = beta
        self.m = m
        
        # Exploit scipys crystalball distribution as underlying distribution
        self.dist = crystalball(beta, m, loc=mu, scale=sigma)

        # If both lower and upper bounds are declared, check that lower_bound < upper_bound, ie the correct way around
        # Also catches for the case where the bounds are equal
        if lower_bound is not None and upper_bound is not None and lower_bound >= upper_bound:
            raise ValueError("lower_bound must be less than upper_bound")

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # This allows truncation from the start
        # Also allows only one bound to be set - ie upper limit only
        if lower_bound is not None or upper_bound is not None:
            self.untrunc_cdf_upper, self.untrunc_cdf_lower = self._calc_trunc_fact()
            self.truncation_factor = self.untrunc_cdf_upper - self.untrunc_cdf_lower
        else:
            self.untrunc_cdf_upper = 1
            self.untrunc_cdf_lower = 0
            self.truncation_factor = 1


    def _calc_trunc_fact(self):
        """
        Calculate the normalisation factor if the distribution has been truncated. ie limits applied
        """
        # Calculate the untruncated CDF at the lower bound if it is set
        if self.lower_bound is not None:
            untrunc_cdf_lower = self.dist.cdf(self.lower_bound)
        else:
            untrunc_cdf_lower = 0
        
        # Calculate the untruncated CDF at the upper bound if it is set
        if self.upper_bound is not None:
            untrunc_cdf_upper = self.dist.cdf(self.upper_bound)
        else:
            untrunc_cdf_upper = 1

        # Calculate the area under the curve between the bounds, ie the area that need to be normalised to 1
        return untrunc_cdf_upper, untrunc_cdf_lower
    

    def pdf(self, X):
        """
        Calculate the Probability Density Function (PDF), automatically including truncation if bounds are set.

        Parameters
        ----------
        X : float or np.ndarray
            The value(s) at which to evaluate the PDF.

        Returns
        -------
        float or np.ndarray
            The normalized PDF value(s), accounting for optional truncation.

        Notes
        -----
        - For `Z > -beta`, the PDF is defined by lower_bound Gaussian core.
        - For `Z <= -beta`, the PDF transitions to a power-law tail.
        - If truncation bounds are provided, the PDF is zero outside the truncation range.
        """

        # Calculate the untruncated PDF using scipy.stats.crystalball
        pdf_non_norm = self.dist.pdf(X)

        # If a bound has been set, set the PDF is set to 0 outside the bounds
        # If X is outside the bounds - condition is met and set to 0
        # If X is within the bounds - condition is not met and set to pdf_non_norm
        if self.lower_bound is not None or self.upper_bound is not None:
            pdf = np.where(np.logical_or(X < self.lower_bound, X > self.upper_bound), 0, pdf_non_norm)

        # If no bounds are set, the PDF is untouched
        else:
            pdf = pdf_non_norm

        # Apply the truncation factor calculated earlier (for non truncated is simply 1)
        return pdf / self.truncation_factor
    
    def pdf_fitting(self, X, mu, sigma, beta, m):
        """
        Calculate the Probability Density Function (PDF) with no set parameters, automatically including truncation if bounds are set, for use in MLE fitting.

        """
        # Calculate the untruncated PDF using scipy.stats.crystalball
        pdf_non_norm = crystalball.pdf(X, beta, m, mu, sigma)

        # If a bound has been set, set the PDF is set to 0 outside the bounds
        # If X is outside the bounds - condition is met and set to 0
        # If X is within the bounds - condition is not met and set to pdf_non_norm
        if self.lower_bound is not None or self.upper_bound is not None:
            pdf = np.where(np.logical_or(X < self.lower_bound, X > self.upper_bound), 0, pdf_non_norm)
            truncation_factor_fit = crystalball.cdf(self.upper_bound, beta, m, mu, sigma) - crystalball.cdf(self.lower_bound, beta, m, mu, sigma)
        # If no bounds are set, the PDF is untouched
        else:
            pdf = pdf_non_norm
            truncation_factor_fit = 1 

        # Apply the truncation factor calculated earlier (for non truncated is simply 1)
        return pdf / truncation_factor_fit


    def cdf(self, X):
        """
        Calculate the Cumulative Distribution Function (CDF).

        Parameters
        ----------
        X : float or np.ndarray
            The value(s) at which to evaluate the truncated CDF.

        Returns
        -------
        float or np.ndarray
            The truncated CDF value(s).

        Notes
        -----
        - For values less than lower_bound, the CDF equals 0.
        - For values greater than upper_bound, the CDF equals 1
        - Within the bounds, the CDF is scaled by the truncation factor.
        """
        
        # Calculate the untruncated CDF using scipy.stats.crystalball
        cdf = self.dist.cdf(X)

        # If a lower bound has been declared, set the CDF to be 0 when lower than
        # All contributions are before the lower bound are removed
        if self.lower_bound is not None:
            cdf = np.where(X < self.lower_bound, 0, cdf - self.untrunc_cdf_lower)

        # If an upper bound has been declared, set the CDF to be 1 - any probability not accounted for 
        if self.upper_bound is not None:
            cdf = np.where(X > self.upper_bound, self.untrunc_cdf_upper - self.untrunc_cdf_lower, cdf)

        # Apply the truncation factor to the CDF
        return cdf / self.truncation_factor
    
    def cdf_fitting(self, X, mu, sigma, beta, m):
        """
        Calculate the Cumulative Distribution Function (CDF) with no set parameters, automatically including truncation if bounds are set, for use in Binned MLE fitting.

        Parameters
        ----------
        X : float or np.ndarray
            The value(s) at which to evaluate the truncated CDF.

        Returns
        -------
        float or np.ndarray
            The truncated CDF value(s).

        Notes
        -----
        - For values less than lower_bound, the CDF equals 0.
        - For values greater than upper_bound, the CDF equals 1
        - Within the bounds, the CDF is scaled by the truncation factor.
        """
        
        # Calculate the untruncated CDF using scipy.stats.crystalball
        cdf = crystalball.cdf(X, beta, m, mu, sigma)
        if self.lower_bound is not None or self.upper_bound is not None:
                    untrunc_cdf_lower = crystalball.cdf(self.lower_bound, beta, m, mu, sigma)
                    untrunc_cdf_upper = crystalball.cdf(self.upper_bound, beta, m, mu, sigma)
                    cdf = np.where(X < self.lower_bound, 0, cdf - untrunc_cdf_lower)
                    cdf = np.where(X > self.upper_bound, untrunc_cdf_upper - untrunc_cdf_lower, cdf)
                    truncation_factor_fit = untrunc_cdf_upper - untrunc_cdf_lower
                    return cdf / truncation_factor_fit
        else:
            return cdf

    
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

        # Set the range of the plot
        if self.lower_bound is None and self.upper_bound is None:
            X = np.linspace(self.mu - 5*self.sigma, self.mu + 5*self.sigma, 1000)

        if self.lower_bound is not None and self.upper_bound is None:
            X = np.linspace(self.lower_bound, self.lower_bound + self.sigma*10, 1000)

        if self.lower_bound is None and self.upper_bound is not None:
            X = np.linspace(self.upper_bound - self.sigma*10, self.upper_bound, 1000)

        else:
            X = np.linspace(self.lower_bound-0.1*(self.upper_bound-self.lower_bound), self.upper_bound+0.1*(self.upper_bound-self.lower_bound), 1000)

        # LHS Plot: the PDF
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(X, self.pdf(X), color='black', linestyle='-', label='PDF')
        plt.xlim(X[0], X[-1])
        plt.axvline(self.mu, color='r', linestyle='--', label=r'Mean: $\mu$')
        plt.axvline(self.mu-self.sigma*self.beta, color='b', linestyle='--', label=r'Transition: $Z = -\beta$')
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Crystal Ball PDF(X)', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=14)
        plt.legend()

        # RHS plot: the CDF
        plt.subplot(1, 2, 2)
        plt.plot(X, self.cdf(X), color='black', linestyle='-', label='CDF')
        plt.xlim(X[0], X[-1])
        plt.axvline(self.mu, color='r', linestyle='--', label=r'Mean: $\mu$')
        plt.axvline(self.mu-self.sigma*self.beta, color='b', linestyle='--', label=r'Transition: $Z = -\beta$')
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Crystal Ball CDF(X)', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=14)
        plt.legend()

        plt.tight_layout()
        plt.show()