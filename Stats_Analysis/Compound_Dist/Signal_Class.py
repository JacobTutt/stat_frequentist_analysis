from ..Base_Dist.ExponentialDecay_Class import ExponentialDecay
from ..Base_Dist.CrystalBall_Class import CrystalBall

import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Signal:
    """
    Signal probability distribution defined by S(X, Y) = CrystalBall(X) * ExponentialDecay(Y).

    This class supports computation of the joint PDF and joint CDF for scalar and array inputs,
    with optional truncation for both X and Y.

    Parameters
    ----------
    mu : float
        The mean of the CrystalBall distribution in the X dimension.
    sigma : float
        The standard deviation of the CrystalBall distribution in the X dimension.
    beta : float
        The threshold value of the CrystalBall distribution in the X dimension.
        Must be beta > 0.
    m : float
        The power-law tail exponent of the CrystalBall distribution in the X dimension.
        Must be m > 1.
    lamb : float
        The decay constant (rate) of the ExponentialDecay distribution in the Y dimension.
        Must be lamb > 0.
    lower_bound_X : float, optional
        The lower bound for the CrystalBall distribution. Default is None.
    upper_bound_X : float, optional
        The upper bound for the CrystalBall distribution. Default is None.
    lower_bound_Y : float, optional
        The lower bound for the ExponentialDecay distribution. Default is None.
    upper_bound_Y : float, optional
        The upper bound for the ExponentialDecay distribution. Default is None.

    Raises
    ------
    ValueError
        If beta <= 0.
        If m <= 1.
        If lamb <= 0.
        If lower_bound_X >= upper_bound_X.
        If lower_bound_Y >= upper_bound_Y.
    """

    def __init__(self, mu, sigma, beta, m, lamb, lower_bound_X=None, upper_bound_X=None, lower_bound_Y=None, upper_bound_Y=None):
        """
        Initialize the Signal distribution with optional truncation by defining the CrystalBall and ExponentialDecay distributions in the X and Y dimensions, respectively.
        """

        # Errors are automatically raised by the underlying distributions
        try:
            self.X = CrystalBall(mu, sigma, beta, m, lower_bound_X, upper_bound_X)
            self.Y = ExponentialDecay(lamb, lower_bound_Y, upper_bound_Y)
        except ValueError as e:
            raise ValueError(f"Error when initialising Signal distribution: {e}")
        
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.m = m
        self.lamb = lamb
        self.lower_bound_X = lower_bound_X
        self.upper_bound_X = upper_bound_X
        self.lower_bound_Y = lower_bound_Y
        self.upper_bound_Y = upper_bound_Y

    def pdf(self, X, Y):
        """
        Calculate the joint Probability Density Function (PDF).

        The Joint PDF is defined as:
        S(X, Y) = CrystalBall_PDF(X) * Exponential_PDF(Y)

        Parameters
        ----------
        X : float or np.ndarray
            The value(s) of X at which to evaluate the PDF.
        Y : float or np.ndarray
            The value(s) of Y at which to evaluate the PDF.

        Returns
        -------
        float or np.ndarray
            The normalized joint PDF value(s)
            0 if X is outside [lower_bound_X, upper_bound_X] or Y is outside [lower_bound_Y, upper_bound_Y].
        """
        return self.X.pdf(X) * self.Y.pdf(Y)
    
    def pdf_fitting(self, X, Y, mu, sigma, beta, m, lamb):
        """
        Calculate the Probability Density Function (PDF) for a given set of parameters, for use with MLE fitting.
        """
        return self.X.pdf_fitting(X, mu, sigma, beta, m) * self.Y.pdf_fitting(Y, lamb)

    def cdf(self, X, Y):
        """
        Compute the joint Cumulative Distribution Function (CDF).

        The Joint CDF is defined as:
        C(X, Y) = Integral of S(X,Y) from 0, X and 0, Y

        As the distributions are independent, the joint CDF is the product of the individual CDFs:
        C(X, Y) = CrystalBall_CDF(X) * Exponential_CDF(Y)

        Parameters
        ----------
        X : float or np.ndarray
            The value(s) of X at which to evaluate the CDF.
        Y : float or np.ndarray
            The value(s) of Y at which to evaluate the CDF.

        Returns
        -------
        float or np.ndarray
            The joint CDF value(s),
            0 if X is outside [lower_bound_X, upper_bound_X] or Y is outside [lower_bound_Y, upper_bound_Y].
        """
        return self.X.cdf(X) * self.Y.cdf(Y)
    
    def normalisation_check(self):
        """
        Perform a numerical integration using scipy.integrate.dblquad to check the normalization of the joint PDF.

        If the PDF is truncated:
        It is first performed over the region the PDF is defined [lower_bound_X, upper_bound_X] for X and [lower_bound_Y, upper_bound_Y] for Y. 
        
        It is then performed over the entire real plane [-infinity, infinity] for X and Y.

        Prints the results of the numerical integrations.
        """

        # Set the limits for the integration, if None remains it will not pass into integration
        if self.lower_bound_X is not None:
            lower_bound_X = self.lower_bound_X
        else:
            lower_bound_X = -np.inf
        
        if self.upper_bound_X is not None:
            upper_bound_X = self.upper_bound_X
        else:
            upper_bound_X = np.inf

        if self.lower_bound_Y is not None:
            lower_bound_Y = self.lower_bound_Y
        else:
            lower_bound_Y = -np.inf

        if self.upper_bound_Y is not None:
            upper_bound_Y = self.upper_bound_Y
        else:
            upper_bound_Y = np.inf



        if (self.lower_bound_X is not None) or (self.upper_bound_X is not None) or (self.lower_bound_Y is not None) or (self.upper_bound_Y is not None):
            print(f"Normalisation over the region the PDF is defined/truncated: [{lower_bound_X}, {upper_bound_X}] in X, [{lower_bound_Y}, {upper_bound_Y}] in Y")
            # have to use constant `lambda` functions for the y limits of integration due to format of 
            integral_bounds, error_bounds = dblquad(lambda y, x: self.pdf(x, y), lower_bound_X, upper_bound_X, lambda x: lower_bound_Y, lambda x: upper_bound_Y)
            print(f"Integral: {integral_bounds} \u00B1 {error_bounds}")

        print(f"Normalisation over the whole real plane: X in [-infinity, infinity], Y in [-infinity, infinity]")
        integral_inf, error_inf = dblquad(lambda y, x: self.pdf(x, y), -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
        print(f"Integral: {integral_inf} \u00B1 {error_inf}")
    
    def plot_dist(self):
        """
        3D plots of the signal PDF.
        """
        # The X values the PDF will span over - slight overextension to show zero probability regions
        X = np.linspace(self.lower_bound_X - 0.05*(self.upper_bound_X - self.lower_bound_X), self.upper_bound_X + 0.05*(self.upper_bound_X-self.lower_bound_X), 1000)

        # All the cases in which the Y values will span over
        # Account for case where no bounds, one bound or both bounds are given
        # slight overextension to show zero probability regions
        if self.lower_bound_Y is not None and self.upper_bound_Y is not None:
            Y = np.linspace(self.lower_bound_Y - 0.05*(self.upper_bound_Y - self.lower_bound_Y), self.upper_bound_Y + 0.05*(self.upper_bound_Y-self.lower_bound_Y), 1000)

        elif self.lower_bound_Y is not None and self.upper_bound_Y is None:
            Y = np.linspace(self.lower_bound_Y, self.lower_bound_Y + 6*self.sigma_b, 1000)
                            
        elif self.lower_bound_Y is None and self.upper_bound_Y is not None:
            Y = np.linspace(self.upper_bound_Y - 6*self.sigma_b, self.upper_bound_Y, 1000)

        else:
            Y = np.linspace(self.mu_b - 3*self.sigma_b, self.mu_b + 3*self.sigma_b, 1000)

        # Meshgrid for the 3D plot
        X, Y = np.meshgrid(X, Y)
        Z = self.pdf(X, Y)

        fig = plt.figure(figsize=(16, 7))
        spec = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        label_fontsize = 20
        tick_fontsize = 18
        colorbar_fontsize = 14

        # LHS: Contour plot with Colour Bar
        ax1 = fig.add_subplot(spec[0, 0])
        contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
        ax1.set_xlabel('X', fontsize=label_fontsize, labelpad=12)
        ax1.set_ylabel('Y', fontsize=label_fontsize, labelpad=12)
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
        colorbar = fig.colorbar(contour, ax=ax1, shrink=0.8)
        colorbar.ax.tick_params(labelsize=colorbar_fontsize)

        # RHS: 3D surface plot
        ax2 = fig.add_subplot(spec[0, 1], projection='3d')
        ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
        ax2.set_xlabel('X', fontsize=label_fontsize, labelpad=15)
        ax2.set_ylabel('Y', fontsize=label_fontsize, labelpad=15)
        ax2.set_zlabel('Total PDF', fontsize=label_fontsize, labelpad=15)
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
        ax2.view_init(elev=30, azim=60)

        plt.tight_layout()
        plt.show()

    


    
