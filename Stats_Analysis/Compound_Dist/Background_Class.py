from ..Base_Dist.NormalDistribution_Class import NormalDistribution
from ..Base_Dist.UniformDistribution_Class import UniformDistribution

import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Background:
    """
    Background probability distribution defined by B(X, Y) = UnfiromDistribution(X) * NormalDistrution(Y).

    This class supports computation of the joint PDF and joint CDF for scalar and array inputs,
    with optional truncation for Y (X's Uniform must inherently be truncated).

    Parameters
    ----------
    mu_b : float
        The mean of the Normal distribution in the Y dimension.
    sigma_b : float
        The standard deviation of the Normal distribution in the Y dimension.
        Must be sigma_b > 0.
    lower_bound_X : float
        The lower bound of the Uniform distribution in the X dimension.
    upper_bound_X : float
        The upper bound of the Uniform distribution in the X dimension.
        Must be lower_bound_X < upper_bound_X.
    lower_bound_Y : float, optional
        The lower bound for truncation of the Normal distribution in the Y dimension. Default is None.
    upper_bound_Y : float, optional
        The upper bound for truncation of the Normal distribution in the Y dimension. Default is None.

    Raises
    ------
    ValueError
        If sigma_b <= 0.
        If lower_bound_X >= upper_bound_X.
        If lower_bound_Y >= upper_bound_Y.


    """

    def __init__(self, mu_b, sigma_b, lower_bound_X, upper_bound_X, lower_bound_Y=None, upper_bound_Y=None):
        """
        Initialize the Background distribution with optional truncation by defining the Uniform and Normal distributions in the X and Y dimensions, respectively.
        """

        # Errors are automatically raised by the underlying distributions
        try:
            self.X = UniformDistribution(lower_bound_X, upper_bound_X)
            self.Y = NormalDistribution(mu_b, sigma_b, lower_bound_Y, upper_bound_Y)
        except ValueError as e:
            raise ValueError(f"Error when initilasing Signal distribution: {e}")
        
        self.mu_b = mu_b
        self.sigma_b = sigma_b
        self.lower_bound_X = lower_bound_X
        self.upper_bound_X = upper_bound_X
        self.lower_bound_Y = lower_bound_Y
        self.upper_bound_Y = upper_bound_Y
            
    def pdf(self, X, Y):
        """
        Calculate the joint Probability Density Function (PDF).

        The Joint PDF is defined as:
        B(X, Y) = Uniform_PDF(X) * Normal_PDF(Y)

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
    
    def pdf_fitting(self, X, Y, mu_b, sigma_b):
        """
        Calculate the Probability Density Function (PDF) for a given set of parameters, for use with MLE fitting.
        """
        return self.X.pdf_fitting(X) * self.Y.pdf_fitting(Y, mu_b, sigma_b)

    def cdf(self, X, Y):
        """
        Compute the joint Cumulative Distribution Function (CDF).

        The Joint CDF is defined as:
        C(X, Y) = Integral of B(X,Y) from 0, X and 0, Y

        As the distributions are independent, the joint CDF is the product of the individual CDFs:
        C(X, Y) = Unifrom_CDF(X) * Normal_CDF(Y)

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
        """
        return self.X.cdf(X) * self.Y.cdf(Y)
    
    def cdf_fitting(self, X, Y, mu_b, sigma_b):
        """
        Calculate the Cumulative Density Function (CDF) for a given set of parameters, for use with Binned MLE fitting.
        """
        return self.X.cdf_fitting(X) * self.Y.cdf_fitting(Y, mu_b, sigma_b)

    def normalisation_check(self, over_whole_plane=False):
            """
            Check the normalization of the joint Probability Density Function (PDF) using numerical integration.

            This method performs numerical integration with `scipy.integrate.dblquad` to ensure that the joint PDF 
            integrates to 1. It supports both truncated and untruncated cases.

            Parameters
            ----------
            over_whole_plane : bool, optional
                If True, perform integration over the entire real plane (-infinity to infinity) for both X and Y.
                Default is False, in which case integration is only performed over the defined/truncated region.

            Resturns
            ------
            Normalisation results for:
                - The defined/truncated region: [lower_bound_X, upper_bound_X] for X and [lower_bound_Y, upper_bound_Y] for Y.
                - The entire real plane: X in [-infinity, infinity] and Y in [-infinity, infinity] (only if `over_whole_plane` is True).

            Notes
            -----
            - If the PDF is truncated, the method integrates over the truncated region defined by the bounds 
            (`lower_bound_X`, `upper_bound_X`, `lower_bound_Y`, `upper_bound_Y`).
            - If no bounds are defined, the truncated region defaults to the entire real plane.
            - The integration over the entire real plane is computationally intensive and can be skipped by setting `over_whole_plane` to False.
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
            if over_whole_plane:
                print(f"Normalisation over the whole real plane: X in [-infinity, infinity], Y in [-infinity, infinity]")
                integral_inf, error_inf = dblquad(lambda y, x: self.pdf(x, y), -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
                print(f"Integral: {integral_inf} \u00B1 {error_inf}")

    
    def plot_dist(self):
        """
        2D plots of the background PDF.
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
        ax2.set_zlabel('Total PDF', fontsize=label_fontsize, labelpad=30)
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
        ax2.tick_params(axis='z', pad=13) 
        ax2.view_init(elev=30, azim=60)

        plt.tight_layout()
        plt.show()

    

    



