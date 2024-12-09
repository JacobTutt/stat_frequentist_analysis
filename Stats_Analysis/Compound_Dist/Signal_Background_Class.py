from .Background_Class import Background
from .Signal_Class import Signal
import os
import re
import shutil  
import numpy as np  
from scipy.optimize import minimize
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL, BinnedNLL
from tabulate import tabulate
from tqdm import tqdm
from tqdm import trange
from scipy.stats import norm
from sweights import SWeight

class Signal_Background:
    """
    Combined Signal and Background probability distribution.

    This class models a mixture distribution defined by:
    S(X, Y) = f * Signal(X, Y) + (1-f) * Background(X, Y)

    The Signal distribution is defined as the product of a CrystalBall distribution (X dimension)
    and an ExponentialDecay distribution (Y dimension).

    The Background distribution is defined as the product of a Uniform distribution (X dimension)
    and a Normal distribution (Y dimension).

    Parameters
    ----------
    mu : float
        The mean of the CrystalBall distribution in the X dimension for the Signal component.
    sigma : float
        The standard deviation of the CrystalBall distribution in the X dimension for the Signal component.
    beta : float
        The threshold value of the CrystalBall distribution in the X dimension for the Signal component.
        Must be beta > 0.
    m : float
        The power-law tail exponent of the CrystalBall distribution in the X dimension for the Signal component.
        Must be m > 1.
    lamb : float
        The decay constant (rate) of the ExponentialDecay distribution in the Y dimension for the Signal component.
        Must be lamb > 0.
    mu_b : float
        The mean of the Normal distribution in the Y dimension for the Background component.
    sigma_b : float
        The standard deviation of the Normal distribution in the Y dimension for the Background component.
        Must be sigma_b > 0.
    f : float
        The fraction of the Signal distribution in the mixture. Must be in the range [0, 1].
    lower_bound_X : float
        The lower bound of the Uniform distribution in the X dimension for the Background component
        and the truncation of the CrystalBall distribution in the Signal component.
    upper_bound_X : float
        The upper bound of the Uniform distribution in the X dimension for the Background component
        and the truncation of the CrystalBall distribution in the Signal component.
        Must satisfy lower_bound_X < upper_bound_X.
    lower_bound_Y : float, optional
        The lower bound of both the ExponentialDecay distribution (Signal component)
        and the Normal distribution (Background component) in the Y dimension. Default is None.
    upper_bound_Y : float, optional
        The upper bound of both the ExponentialDecay distribution (Signal component)
        and the Normal distribution (Background component) in the Y dimension. Default is None.

    Raises
    ------
    ValueError 
        If f is not in the range [0, 1].
        If beta <= 0.
        If m <= 1.
        If lamb <= 0.
        If sigma_b <= 0.
        If lower_bound_X >= upper_bound_X.
        If lower_bound_Y >= upper_bound_Y.
    """


    def __init__(self, mu, sigma, beta, m, lamb,  mu_b, sigma_b, f, lower_bound_X, upper_bound_X, lower_bound_Y=None, upper_bound_Y=None):

        # Defined the fraction between the signal and background contributions
        if f < 0 or f > 1:
            raise ValueError("f must be in the range [0, 1]")
        
        self.f = f
        
        # Other errors are automatically raised by the underlying distributions
        try:
            self.Signal = Signal(mu, sigma, beta, m, lamb, lower_bound_X, upper_bound_X, lower_bound_Y, upper_bound_Y)
            self.Background = Background(mu_b, sigma_b, lower_bound_X, upper_bound_X, lower_bound_Y, upper_bound_Y)
        except ValueError as e:
            raise ValueError(f"Error when initialising Signal_Background distribution: {e}")
        
        self.true_params = [mu, sigma, beta, m, lamb, mu_b, sigma_b, f]
        self.samples = None

        self.lower_bound_X = lower_bound_X 
        self.upper_bound_X = upper_bound_X
        self.lower_bound_Y = lower_bound_Y
        self.upper_bound_Y = upper_bound_Y

        # If both lower and upper bounds are defined, find the maximum value of the PDF in this region from the start
        if self.lower_bound_X is not None and self.upper_bound_X is not None:
            self.max_pdf = self._find_max_pdf()


    def pdf(self, X, Y):
        """
        Calculate the joint Probability Density Function (PDF).

        The Joint PDF is defined as:
        S_B(X, Y) = f * Signal_PDF(X, Y) + (1-f) * Background_PDF(X, Y)

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
        return self.f * self.Signal.pdf(X, Y) + (1 - self.f) * self.Background.pdf(X, Y)


    def cdf(self, X, Y):
        """
        Compute the joint Cumulative Distribution Function (CDF).

        The Joint CDF is defined as:
        C(X, Y) = Integral of S_B(X,Y) from 0, X and 0, Y

        As the distributions are independent, the joint CDF is the product of the individual CDFs:
        C(X, Y) = f * Signal_CDF(X, Y) + (1-f) * Background_CDF(X, Y)

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

        return self.f * self.Signal.cdf(X, Y) + (1 - self.f) * self.Background.cdf(X, Y)


    def _find_max_pdf(self):
        """
        Finds the maximum value of the joint PDF by first using a rough grid search and then a local optimisation.
        Only performed if both the lower and upper bounds are defined.

        Returns
        -------
        float
            The maximum value of the joint PDF within the defined bounds.
        """
        # Coarse grid search
        # Make use of numpy meshgrid to create a grid of points
        # Allows the use of vectorized operations
        x_vals = np.linspace(self.lower_bound_X, self.upper_bound_X, 10)
        y_vals = np.linspace(self.lower_bound_Y, self.upper_bound_Y, 10)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = self.pdf(X, Y)

        # Find the maximum value of the PDF over the grid
        max_index = np.unravel_index(np.argmax(Z), Z.shape)
        # Returns the X and Y values of the maximum point to use as the starting point for the local optimization
        best_start = (X[max_index], Y[max_index])

        # Local optimization starting from the best grid point
        def neg_pdf(point):
            x, y = point
            return -self.pdf(x, y)
        
        # Use scipy.optimize.minimize to find the minimium value of the negetive PDF
        # Limited-memory Broyden–Fletcher–Goldfarb–Shanno with Box constraint
        # Quick and precise for solving bounded optimization problems
        min_result = minimize(neg_pdf,best_start, bounds=[(self.lower_bound_X, self.upper_bound_X), (self.lower_bound_Y, self.upper_bound_Y)], method="L-BFGS-B")

        if min_result.success:
            max_pdf = -min_result.fun
            print(f"Maximum PDF value found: {max_pdf}")
        else:
            raise RuntimeError("Optimisation failed")
        
        return max_pdf
    
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

        Returns
        -------
        Normalisation results for:
            - The defined/truncated region: [lower_bound_X, upper_bound_X] for X and [lower_bound_Y, upper_bound_Y] for Y.
            - The entire real plane: X in [-infinity, infinity] and Y in [-infinity, infinity] (only if `over_whole_plane` is True).

        Notes
        -----
        - If the PDF is truncated, the method integrates over the truncated region defined by the bounds (`lower_bound_X`, `upper_bound_X`, `lower_bound_Y`, `upper_bound_Y`).
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
        Visualize the joint Probability Density Function (PDF) in 2D and 3D.

        LHS: A 2D contour plot with a color bar to represent the PDF values.
        RHS: A 3D surface plot to show the PDF as a function of X and Y.

        The X and Y ranges for the plots are determined based on the bounds provided for the PDF, with slight overextensions to display regions of zero probability.
        """
        # The X values the PDF will span over - slight overextension to show zero probability regions
        X = np.linspace(self.lower_bound_X - 0.05*(self.upper_bound_X - self.lower_bound_X), self.upper_bound_X + 0.05*(self.upper_bound_X-self.lower_bound_X), 1000)

        # All the cases in which the Y values will span over
        # Account for case where no bounds, one bound or both bounds are given
        # slight overextension to show zero probability regions
        if self.lower_bound_Y is not None and self.upper_bound_Y is not None:
            Y = np.linspace(self.lower_bound_Y - 0.05*(self.upper_bound_Y - self.lower_bound_Y), self.upper_bound_Y + 0.05*(self.upper_bound_Y-self.lower_bound_Y), 1000)

        elif self.lower_bound_Y is not None and self.upper_bound_Y is None:
            Y = np.linspace(self.lower_bound_Y, self.lower_bound_Y + 6*self.true_params[6], 1000)
                            
        elif self.lower_bound_Y is None and self.upper_bound_Y is not None:
            Y = np.linspace(self.upper_bound_Y - 6*self.true_params[6], self.upper_bound_Y, 1000)

        else:
            Y = np.linspace(self.true_params[5] - 3*self.true_params[6], self.true_params[5] + 3*self.true_params[6], 1000)

        # Meshgrid for the 3D plot
        X, Y = np.meshgrid(X, Y)
        Z = self.pdf(X, Y)

        fig = plt.figure(figsize=(16, 7))
        spec = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        label_fontsize = 20
        tick_fontsize = 18
        colorbar_fontsize = 14

        # LHS: Contour plot with a color bar to s
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


    def marginal_pdf_x(self, X): 
        """
        Calculate the marginal Probability Density Function (PDF) in the X dimension.
        Integral of the joint PDF over the Y dimension, to remove the Y dependence.

        The marginal PDF is defined as:
        S_B(X) = Integral of S_B(X, Y) wrt Y over [-infinity, infinity]

        Parameters
        ----------
        X : float or np.ndarray
            The value(s) of X at which to evaluate the PDF.

        Returns
        -------
        tuple of (float or np.ndarray, float or np.ndarray, float or np.ndarray)
            A tuple containing:
            - The signal component of the marginal PDF.
            - The background component of the marginal PDF.
            - The total marginal PDF (signal + background).
        """
        # Use pre defined X component of the Signal and Background PDFs to calculate the marginal PDF
        signal_cont_pdf = self.f*self.Signal.X.pdf(X)
        background_cont_pdf = (1 - self.f)*self.Background.X.pdf(X)
        # Return the signal, background and total PDF values seperately for plotting
        return signal_cont_pdf, background_cont_pdf, signal_cont_pdf + background_cont_pdf
    
    def marginal_cdf_x(self, X):
        """
        Calculate the marginal Cumulative Distribution Function (CDF) in the X dimension.
        Integral of the joint CDF over the Y dimension, to remove the Y dependence.

        The marginal CDF is defined as:
        C(X) = Integral of C_B(X, Y) wrt Y over [-infinity, infinity]

        Parameters
        ----------
        X : float or np.ndarray
            The value(s) of X at which to evaluate the CDF.

        Returns
        -------
        tuple of (float or np.ndarray, float or np.ndarray, float or np.ndarray)
            A tuple containing:
            - The signal component of the marginal CDF.
            - The background component of the marginal CDF.
            - The total marginal CDF (signal + background).
        """
        # Use pre defined X component of the Signal and Background CDFs to calculate the marginal CDF
        signal_cont_cdf = self.f*self.Signal.X.cdf(X)
        background_cont_cdf = (1 - self.f)*self.Background.X.cdf(X)
        # Return the signal, background and total PDF values seperately for plotting
        return signal_cont_cdf, background_cont_cdf, signal_cont_cdf + background_cont_cdf
    
    def marginal_pdf_y(self, Y):
        """
        Calculate the marginal Probability Density Function (PDF) in the Y dimension.
        Integral of the joint PDF over the X dimension, to remove the X dependence.

        The marginal PDF is defined as:
        S_B(Y) = Integral of S_B(X, Y) wrt X over [-infinity, infinity]

        Parameters
        ----------
        Y : float or np.ndarray
            The value(s) of Y at which to evaluate the PDF.

        Returns
        -------
        tuple of (float or np.ndarray, float or np.ndarray, float or np.ndarray)
            A tuple containing:
            - The signal component of the marginal PDF.
            - The background component of the marginal PDF.
            - The total marginal PDF (signal + background).
        """
        # Use pre defined Y component of the Signal and Background PDFs to calculate the marginal PDF
        signal_cont_pdf = self.f*self.Signal.Y.pdf(Y)
        background_cont_pdf = (1 - self.f)*self.Background.Y.pdf(Y)
        # Return the signal, background and total PDF values seperately for plotting
        return signal_cont_pdf, background_cont_pdf, signal_cont_pdf + background_cont_pdf
    
    def marginal_cdf_y(self, Y):
        """
        Calculate the marginal Cumulative Distribution Function (CDF) in the Y dimension.
        Integral of the joint CDF over the X dimension, to remove the X dependence.

        The marginal CDF is defined as:
        C(Y) = Integral of C_B(X, Y) wrt X over [-infinity, infinity]

        Parameters
        ----------
        Y : float or np.ndarray
            The value(s) of Y at which to evaluate the CDF.

        Returns
        -------
        tuple of (float or np.ndarray, float or np.ndarray, float or np.ndarray)
            A tuple containing:
            - The signal component of the marginal CDF.
            - The background component of the marginal CDF.
            - The total marginal CDF (signal + background).
        """
        # Use pre defined Y component of the Signal and Background CDFs to calculate the marginal CDF
        signal_cont_cdf = self.f*self.Signal.Y.cdf(Y)
        background_cont_cdf = (1 - self.f)*self.Background.Y.cdf(Y)
        # Return the signal, background and total PDF values seperately for plotting
        return signal_cont_cdf, background_cont_cdf, signal_cont_cdf + background_cont_cdf

    def plot_marginal(self):
        """
        Create a 2x2 grid of plots:
        - Top left: marginal_pdf_x with signal and background contributions and overall
        - Top right: marginal_cdf_x with signal and background contributions and overall
        - Bottom left: marginal_pdf_y with signal and background contributions and overall
        - Bottom right: marginal_cdf_y with signal and background contributions and overall
        """
        X = np.linspace(self.lower_bound_X - 0.05*(self.upper_bound_X - self.lower_bound_X), self.upper_bound_X + 0.05*(self.upper_bound_X-self.lower_bound_X), 1000)

        # given all options to support when only one bound is given
        if self.lower_bound_Y is not None and self.upper_bound_Y is not None:
            Y = np.linspace(self.lower_bound_Y - 0.05*(self.upper_bound_Y - self.lower_bound_Y), self.upper_bound_Y + 0.05*(self.upper_bound_Y-self.lower_bound_Y), 1000)

        elif self.lower_bound_Y is not None and self.upper_bound_Y is None:
            Y = np.linspace(self.lower_bound_Y, self.lower_bound_Y + 6*self.true_params[6], 100)
                            
        elif self.lower_bound_Y is None and self.upper_bound_Y is not None:
            Y = np.linspace(self.upper_bound_Y - 6*self.true_params[6], self.upper_bound_Y, 100)

        else:
            Y = np.linspace(self.true_params[5] - 3*self.true_params[6], self.true_params[5] + 3*self.true_params[6], 1000)

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))

        # Predetermined font sizes and line widths
        label_fontsize = 16
        tick_fontsize = 14
        legend_fontsize = 14
        line_width = 2
        alpha_value = 0.8

        # Top left: marginal_pdf_x
        signal_pdf_x, background_pdf_x, total_pdf_x = self.marginal_pdf_x(X)
        axs[0, 0].plot(X, signal_pdf_x, label='Signal PDF: Crystal Ball', color='green', linewidth=line_width, alpha=alpha_value)
        axs[0, 0].plot(X, background_pdf_x, label='Background PDF: Uniform', color='blue', linewidth=line_width, alpha=alpha_value)
        axs[0, 0].plot(X, total_pdf_x, label='Total PDF', color='red', linewidth=line_width, alpha=alpha_value)
        axs[0, 0].set_xlabel('X', fontsize=label_fontsize)
        axs[0, 0].set_ylabel(f'Marginal PDF in X (f = {self.f})', fontsize=label_fontsize)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        axs[0, 0].legend(fontsize=legend_fontsize)
        axs[0, 0].grid(linestyle='--', alpha=0.6)

        # Top right: marginal_cdf_x
        signal_cdf_x, background_cdf_x, total_cdf_x = self.marginal_cdf_x(X)
        axs[0, 1].plot(X, signal_cdf_x, label='Signal CDF: Crystal Ball', color='green', linewidth=line_width, alpha=alpha_value)
        axs[0, 1].plot(X, background_cdf_x, label='Background CDF: Uniform', color='blue', linewidth=line_width, alpha=alpha_value)
        axs[0, 1].plot(X, total_cdf_x, label='Total CDF', color='red', linewidth=line_width, alpha=alpha_value)
        axs[0, 1].set_xlabel('X', fontsize=label_fontsize)
        axs[0, 1].set_ylabel(f'Marginal CDF in X (f = {self.f})', fontsize=label_fontsize)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        axs[0, 1].legend(fontsize=legend_fontsize)
        axs[0, 1].grid(linestyle='--', alpha=0.6)

        # Bottom left: marginal_pdf_y
        signal_pdf_y, background_pdf_y, total_pdf_y = self.marginal_pdf_y(Y)
        axs[1, 0].plot(Y, signal_pdf_y, label='Signal PDF: Exponential', color='green', linewidth=line_width, alpha=alpha_value)
        axs[1, 0].plot(Y, background_pdf_y, label='Background PDF: Normal', color='blue', linewidth=line_width, alpha=alpha_value)
        axs[1, 0].plot(Y, total_pdf_y, label='Total PDF', color='red', linewidth=line_width, alpha=alpha_value)
        axs[1, 0].set_xlabel('Y', fontsize=label_fontsize)
        axs[1, 0].set_ylabel(f'Marginal PDF in Y (f = {self.f})', fontsize=label_fontsize)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        axs[1, 0].legend(fontsize=legend_fontsize)
        axs[1, 0].grid(linestyle='--', alpha=0.6)

        # Bottom right: marginal_cdf_y
        signal_cdf_y, background_cdf_y, total_cdf_y = self.marginal_cdf_y(Y)
        axs[1, 1].plot(Y, signal_cdf_y, label='Signal CDF: Exponential', color='green', linewidth=line_width, alpha=alpha_value)
        axs[1, 1].plot(Y, background_cdf_y, label='Background CDF: Normal', color='blue', linewidth=line_width, alpha=alpha_value)
        axs[1, 1].plot(Y, total_cdf_y, label='Total CDF', color='red', linewidth=line_width, alpha=alpha_value)
        axs[1, 1].set_xlabel('Y', fontsize=label_fontsize)
        axs[1, 1].set_ylabel(f'Marginal CDF in Y (f = {self.f})', fontsize=label_fontsize)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        axs[1, 1].legend(fontsize=legend_fontsize)
        axs[1, 1].grid(linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

    def accept_reject_sample(self, desired_samples=100000, init_batch_size=1000, max_batch_size=2000000, poisson = False, save_to_class=False):
        """
        Generate random samples from the joint Signal-Background distribution
        using the accept-reject method with dynamic batch sizing.

        This method uses an initial batch to estimate the acceptance rate and 
        dynamically adjusts the batch size to efficiently generate the required number 
        of samples.

        Parameters
        ----------
        desired_samples : int, optional
            Total number of samples to generate (default: 100,000).
        init_batch_size : int, optional
            Batch size for the initial sampling to estimate the acceptance rate (default: 1,000).
        max_batch_size : int, optional
            Maximum batch size for iterations to prevent overloading memory (default: 2,000,000).
            May need adjusting for devices with limited memory.
        poisson : bool, optional
            If True, the total number of samples (`desired_samples`) will be drawn 
            from a Poisson distribution with a mean of `desired_samples` (default: False).
        save_to_class : bool, optional
            If True, the generated samples will be saved as an attribute of the class 
            (`self.samples`) for later use (default: False).

        Returns
        -------
        np.ndarray
            Array of shape (desired_samples, 2) containing the sampled (X, Y) points.

        Raises
        ------
        ValueError
            If any of the bounds are not defined, this is required to generate sample

        Notes
        -----
        - The initial batch estimates the acceptance rate as:
          acceptance_rate = (Number of Accepted Samples in Initial Batch) / (Initial Batch Size)
        - Subsequent batch sizes are calculated dynamically based on the acceptance rate and
          the number of remaining desired samples, with a 10% buffer.
        - A maximum batch size (`max_batch_size`) is enforced to ensure memory efficiency.
        """
        if self.lower_bound_X is None or self.upper_bound_X is None or self.lower_bound_Y is None or self.upper_bound_Y is None:
            raise ValueError("To preform Accept-reject sampling both X and Y limits must be set.")

            # Apply Poisson variation if enabled
        if poisson:
            actual_samples = np.random.poisson(desired_samples)
        else:
            actual_samples = desired_samples
        
        # Pre-allocate space for samples
        samples = np.empty((actual_samples, 2))
        sample_count = 0

        # Run an inital batch to estimate the acceptance rate the dynamic batch size on
        # Generates smaller batches: X, Y, Uniform Random, True Values
        batch_X = np.random.uniform(self.lower_bound_X, self.upper_bound_X, size=init_batch_size)
        batch_Y = np.random.uniform(self.lower_bound_Y, self.upper_bound_Y, size=init_batch_size)
        batch_uniform = np.random.uniform(0, self.max_pdf, size=init_batch_size)
        batch_true_vals = self.pdf(batch_X, batch_Y)

        # Compare Uniform Random to find the accepted samples, where within pdf
        # Each index is a boolean value
        batch_accept_index = batch_uniform <= batch_true_vals

        # Collect x and y samples that are accepted
        accepted_samples = np.array([batch_X[batch_accept_index], batch_Y[batch_accept_index]]).T

        # Add samples to overall score the samples array with the accepted samples
        num_accepted = len(accepted_samples)
        samples[:num_accepted] = accepted_samples
        sample_count += num_accepted


        # Estimate acceptance rate
        # Use np.sum to count the number of True values in the array
        acceptance_rate = num_accepted/ init_batch_size


        # While samples is less than desired number - produce samples to account for acceptance rate
        # Generate samples in batches until the desired number of samples is reached - ideally in 1 batch
        while sample_count < actual_samples:
            # Calculate the remaining desired samples
            remain_actual_samples = actual_samples - sample_count
            # Calculate the batch size to generate based on remaining samples and acceptance rate
            # A maximum batch size, 500000, is set to prevent overloading memory
            # A 10% buffer is added to the batch size to ensure enough samples are generated
            batch_size = min(int(1.1*(remain_actual_samples / acceptance_rate)), max_batch_size)

            # Generate a new batch of proposals using the same method as above
            batch_X = np.random.uniform(self.lower_bound_X, self.upper_bound_X, size=batch_size)
            batch_Y = np.random.uniform(self.lower_bound_Y, self.upper_bound_Y, size=batch_size)
            batch_uniform = np.random.uniform(0, self.max_pdf, size=batch_size)
            batch_true_vals = self.pdf(batch_X, batch_Y)
            batch_accept_index = batch_uniform <= batch_true_vals
            accepted_samples = np.array([batch_X[batch_accept_index], batch_Y[batch_accept_index]]).T

            num_accepted = len(accepted_samples)
            remaining_space = actual_samples - sample_count

            # Ensure the number of items added does not exceed the remaining space
            if num_accepted > remaining_space:
                samples[sample_count:] = accepted_samples[:remaining_space]
                sample_count += remaining_space
            else:
                samples[sample_count:sample_count + num_accepted] = accepted_samples
                sample_count += num_accepted

        if save_to_class:
            # Store to the class 
            self.samples = samples

        return samples
    
    def plot_samples(self, samples = None):
        """
        Plot the results of the sampled data in a 2x2 grid:
        - Top-left: 3D histogram of the joint distribution.
        - Top-right: Surface plot of the joint PDF.
        - Bottom-left: Histogram of sampled X values vs marginal PDF.
        - Bottom-right: Histogram of sampled Y values vs marginal PDF.

        Parameters
        ----------
        samples : np.ndarray, optional
            Array of shape (N, 2) containing the sampled data points (X, Y).
            If not provided, the method attempts to use `self.samples`.
            If neither is available, a ValueError is raised.

        Raises
        ------
        ValueError
            If no samples are provided and no samples are stored in `self.samples`.
        """
        if samples is None and self.samples is None:
            raise ValueError("No samples have been generated. Please run the `accept_reject_sample` method first.")
        if samples is None:
            samples = self.samples
        
        # Define ranges for X and Y based on bounds
        X = np.linspace(self.lower_bound_X - 0.1*(self.upper_bound_X - self.lower_bound_X), self.upper_bound_X+ 0.1*(self.upper_bound_X - self.lower_bound_X), 1000)
        Y = np.linspace(self.lower_bound_Y- 0.1*(self.upper_bound_Y - self.lower_bound_Y), self.upper_bound_Y+ 0.1*(self.upper_bound_Y - self.lower_bound_Y), 1000)

        # Compute the PDF grid for plotting
        X_grid, Y_grid = np.meshgrid(X, Y)
        Z = self.pdf(X_grid, Y_grid)

        # Set consistent font sizes
        label_fontsize = 18
        title_fontsize = 18
        tick_fontsize = 16

        fig = plt.figure(figsize=(14, 14))
        # Create a 2x2 grid of plots
        spec = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.1])

        # Top left: 3D histogram of samples, using similiar colour map scheme to pdf plot
        ax1 = fig.add_subplot(spec[0, 0], projection='3d')
        hist, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=30, 
                                            range=[[self.lower_bound_X, self.upper_bound_X], 
                                                    [self.lower_bound_Y, self.upper_bound_Y]])
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        xpos, ypos = xpos.ravel(), ypos.ravel()
        zpos = np.zeros_like(xpos)
        dx = dy = (xedges[1] - xedges[0])
        dz = hist.ravel()

        # Apply colormap
        cmap = get_cmap('viridis')
        norm = Normalize(vmin=dz.min(), vmax=dz.max())
        colors = cmap(norm(dz))

        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=colors)
        ax1.set_xlabel("X", fontsize=label_fontsize)
        ax1.set_ylabel("Y", fontsize=label_fontsize)
        ax1.set_zlabel("Frequency", fontsize=label_fontsize, labelpad = 20)
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
        ax1.view_init(elev=30, azim=60)
        ax1.set_title("3D Histogram of Joint Distribution", fontsize=title_fontsize)

        # Top-right: Surface plot of the joint PDF for comparison with histogram in top-left
        ax2 = fig.add_subplot(spec[0, 1], projection='3d')
        ax2.plot_surface(X_grid, Y_grid, Z, cmap='viridis', alpha=0.9)
        ax2.set_xlabel("X", fontsize=label_fontsize)
        ax2.set_ylabel("Y", fontsize=label_fontsize, labelpad = 10)
        ax2.set_zlabel("PDF Value", fontsize=label_fontsize, labelpad = 20)
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
        ax2.view_init(elev=30, azim=60)
        ax2.set_title("Surface Plot of True Joint PDF", fontsize=title_fontsize)

        # Bottom-left: Histogram of X values vs marginal PDF in X
        ax3 = fig.add_subplot(spec[1, 0])
        # Gives bin edges and counts
        hist_X, bins_X, _ = ax3.hist(samples[:, 0], bins=30, density=True, alpha=0.5, color='navy', label="Sampled X")
        # Calculate the bin centers
        bin_centers_X = 0.5 * (bins_X[:-1] + bins_X[1:])
        # Can ignore the signal and background contribution given (_) as they are not used
        _, _, marginal_total_X = self.marginal_pdf_x(X)
        ax3.plot(X, marginal_total_X, label="Marginal PDF X", color="red", linewidth=4)
        ax3.set_xlim(X[0], X[-1])
        ax3.set_xlabel("X", fontsize=label_fontsize)
        ax3.set_ylabel("Density", fontsize=label_fontsize)
        ax3.tick_params(axis='both', labelsize=tick_fontsize)
        ax3.legend(fontsize=label_fontsize)
        ax3.set_title("Histogram of X with Marginal PDF of X", fontsize=title_fontsize)

        # Bottom-right: Histogram of samples Y values vs marginal PDF in Y
        ax4 = fig.add_subplot(spec[1, 1])
        # Gives bin edges and counts
        hist_Y, bins_Y, _ = ax4.hist(samples[:, 1], bins=30, density=True, alpha=0.5, color='navy', label="Sampled Y")
        # Calculate the bin centers
        bin_centers_Y = 0.5 * (bins_Y[:-1] + bins_Y[1:])
        # Can ignore the signal and background contribution given (_) as they are not used
        _, _, marginal_total_Y = self.marginal_pdf_y(Y)
        ax4.plot(Y, marginal_total_Y, label="Marginal PDF Y", color="red", linewidth=4)
        ax4.set_xlim(Y[0], Y[-1])
        ax4.set_xlabel("Y", fontsize=label_fontsize)
        ax4.set_ylabel("Density", fontsize=label_fontsize)
        ax4.tick_params(axis='both', labelsize=tick_fontsize)
        ax4.legend(fontsize=label_fontsize)
        ax4.set_title("Histogram of Y with Marginal PDF of X", fontsize=title_fontsize)

        plt.tight_layout()
        plt.show()
    
    
    def pdf_fitting(self, X, Y, mu, sigma, beta, m, lamb, mu_b, sigma_b, f):
        """
        Calculate the joint Probability Density Function (PDF) for given parameters.

        Parameters
        ----------
        X, Y : float or np.ndarray
            Values where the PDF is evaluated.
        mu, sigma, beta, m, lamb, mu_b, sigma_b, f : float
            Parameters to use for the calculation.

        Returns
        -------
        np.ndarray
            PDF values for the input X, Y.
        """
        # Provides an undefined pdf which can then be adjusted during MLE fitting
        return f * self.Signal.pdf_fitting(X, Y, mu, sigma, beta, m, lamb) + (1 - f) * self.Background.pdf_fitting(X, Y, mu_b, sigma_b)
    
    def cdf_fitting(self, X, Y, mu, sigma, beta, m, lamb, mu_b, sigma_b, f):
        """
        Calculate the joint Probability Density Function (PDF) for given parameters.

        Parameters
        ----------
        X, Y : float or np.ndarray
            Values where the PDF is evaluated.
        mu, sigma, beta, m, lamb, mu_b, sigma_b, f : float
            Parameters to use for the calculation.

        Returns
        -------
        np.ndarray
            PDF values for the input X, Y.
        """
        # Provides an undefined pdf which can then be adjusted during MLE fitting
        return f * self.Signal.cdf_fitting(X, Y, mu, sigma, beta, m, lamb) + (1 - f) * self.Background.cdf_fitting(X, Y, mu_b, sigma_b)
        
    
    def fit_params(self, initial_params, samples = None, print_results = False, save_to_class = False):
        """
        Perform an extended maximum likelihood fit using `iminuit`.

        This method fits the parameters of a model to the given data by minimizing the 
        negative log-likelihood using the `iminuit` package. The fit is based on 
        Extended Unbinned Maximum Likelihood (EUMLE).

        Parameters
        ----------
        initial_params : list of float
            Initial guesses for the model parameters in the order:
            [mu, sigma, beta, m, lamb, mu_b, sigma_b, f, N_expected].

        samples : np.ndarray, optional
            Observed data points of shape (N, 2), where each row represents a pair of (X, Y) values.
            If not provided, the method attempts to use samples stored in the instance (`self.samples`).

        print_results : bool, optional
            If True, prints the `iminuit` results to the console. Default is False.

        save_to_class : bool, optional
            If True, saves the resulting `Minuit` object (`self.mi`) to the instance for later use.
            Default is False.

        Returns
        -------
        tuple
            A tuple containing:
            - `mi.values` : A dictionary of fitted parameter values.
            - `mi.errors` : A dictionary of parameter uncertainties.

        Raises
        ------
        ValueError
            If no data samples are provided or stored in the instance (`self.samples`).
        RuntimeError
            If the minimization process fails to converge (`migrad` is invalid).

        Notes
        -----
        - Parameter limits are set based on physical significance and distribution constraints:
        - The fit includes an error analysis using the `Hesse` algorithm to estimate parameter uncertainties.
        - The `iminuit` object provides detailed information about the fit, including parameter correlations.
        """

        # Allow for the samples to be passed in, if not try use the samples already generated and stored` class
        if samples is None:
            samples = self.samples
            if samples is None:
                raise ValueError("No samples have been provided or generated by the `accept_reject_sample` method first")
            
        # Split the samples into X and Y so  correct passing in
        samples_x = samples[:, 0]
        samples_y = samples[:, 1]

        # Define the density function - required by iminuit Extended Unbinned Likelihood
        def density(samples_in, mu, sigma, beta, m, lamb, mu_b, sigma_b, f, N):
            samples_x, samples_y = samples_in
            pdf_vals = self.pdf_fitting(samples_x, samples_y, mu, sigma, beta, m, lamb, mu_b, sigma_b, f)
            return N, N * pdf_vals

        # Create the negative log-likelihood function
        neg_log_likelihood = ExtendedUnbinnedNLL((samples_x, samples_y), density)

        print((samples_x, samples_y))

        # Create the Minuit object with initial guesses
        mi = Minuit(neg_log_likelihood, mu=initial_params[0],sigma=initial_params[1], beta=initial_params[2], 
                    m=initial_params[3], lamb=initial_params[4], mu_b=initial_params[5], sigma_b=initial_params[6],
                    f=initial_params[7], N=initial_params[8])

        # Set parameter limits based on each paramaters restrictions and physical significance
        # sigma > 0
        mi.limits["sigma"] = (1e-3, None)
        # beta > 0
        mi.limits["beta"] = (1e-2, None)
        # m > 1
        mi.limits["m"] = (1.01, None)
        # lambda > 0
        mi.limits["lamb"] = (1e-3, None)
        # sigma_b > 0
        mi.limits["sigma_b"] = (1e-3, None)
        # f in [0, 1]
        mi.limits["f"] = (0, 1)
        # N_expected > 0
        mi.limits["N"] = (1e-3, None)

        # Run the minimisation
        mi.migrad()

        if not mi.valid:
            raise RuntimeError("Minimisation did not converge")

        # Run the error analysis
        mi.hesse()

        if save_to_class:
            # Save to class
            self.mi = mi

        if print_results:
            print(mi)
        
        # Return the Minuit Values and Errors
        return mi.values, mi.errors
    

    def fit_params_results(self):
        """
        Visualise the results of the parameter fitting process.

        Returns
        -------
        1. Parameter Summary Table
        2. Correlation Heatmap

        Raises
        ------
        ValueError
            If the `Minuit` object (`self.mi`) is not available. Ensure that the `fit_params` method is run before calling this method.

        Notes
        -----
        - True values must be stored in `self.true_params` for comparison in the table.
        - Fitted values and uncertainties are extracted from the `Minuit` object (`self.mi`).
        - The correlation matrix is calculated from the `Minuit` covariance matrix.
        """

        # check data has actually been fitted to and results are available
        if not hasattr(self, "mi"):
            raise ValueError("Minuit object not available. Please run fit_params first.")

        # Define parameter parent distributions for the table
        # stored in the list format (parent distribution(group), name, fitted value, error, true value)=
        params_table = [
            # Crystal Ball parameters
            ("Crystal Ball (Signal)", "mu", self.mi.values["mu"], self.mi.errors["mu"], self.true_params[0]),
            ("Crystal Ball (Signal)", "sigma", self.mi.values["sigma"], self.mi.errors["sigma"], self.true_params[1]),
            ("Crystal Ball (Signal)", "beta", self.mi.values["beta"], self.mi.errors["beta"], self.true_params[2]),
            ("Crystal Ball (Signal)", "m", self.mi.values["m"], self.mi.errors["m"], self.true_params[3]),
            # Exponential Decay parameter
            ("Exponential (Signal)", "lamb", self.mi.values["lamb"], self.mi.errors["lamb"], self.true_params[4]),
            # Normal parameters
            ("Normal (Background)", "mu_b", self.mi.values["mu_b"], self.mi.errors["mu_b"], self.true_params[5]),
            ("Normal (Background)", "sigma_b", self.mi.values["sigma_b"], self.mi.errors["sigma_b"], self.true_params[6]),
            # Overall parameters
            ("Overall", "f", self.mi.values["f"], self.mi.errors["f"], self.true_params[7]),
            ("Overall", "N", self.mi.values["N"], self.mi.errors["N"], None), 
        ]

        # Create a dictionary to store the fitting results within the class simultaneously
        self.fit_results = {}


        # Add columns for "Value ± Error" and "Number of Standard Errors Away"
        formatted_table = []
        for group, name, value, error, true_val in params_table:

            # Save the results to the overall class dictionary - one at a time
            self.fit_results[name] = {"value": value, "error": error, "distribution": group}
            # String of the value with error
            value_with_error = f"{value:.4f} ± {error:.4f}"
            # Calculate the number of standard errors away from the true value
            if true_val is not None:
                num_std_errs = abs(value - true_val) / error
            else:
                num_std_errs = "N/A"
            # Append the formatted row to the table
            formatted_table.append([group, name, value_with_error, true_val, num_std_errs])

        # Set the column headers
        headers = ["Distribution", "Parameter", "Value ± Error", "True Value", "Std Errors Away"]

        print(tabulate(formatted_table, headers=headers, floatfmt=".4f", tablefmt="pretty"))

        # Get the correlation matrix and parameter names using mi from minuit object
        correlation_matrix = self.mi.covariance.correlation()
        parameters = self.mi.parameters

        fig, ax = plt.subplots(figsize=(8, 8))  
        # Plot the heatmap
        cax = ax.matshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)

        # Add a colorbar
        cbar = plt.colorbar(cax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=14)  

        ax.set_xticks(range(len(parameters)))
        ax.set_yticks(range(len(parameters)))
        ax.set_xticklabels(parameters, fontsize=14, rotation=45) 
        ax.set_yticklabels(parameters, fontsize=14)

        # Put the values in the centre of the cells for the heat map
        for (i, j), val in np.ndenumerate(correlation_matrix):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=12)

        plt.tight_layout()
        plt.show()


    def plot_profiled_likelihoods(self):
        """
        Plot the profiled log-likelihoods for all parameters in a 3x3 grid with improved formatting.
        
        Returns
        -------
        A  3x3 grid of plots, where each subplot corresponds to a parameter's profiled log-likelihood (`-2 ln L`) as a function of its value. 
        The plots highlight the 1σ and 2σ confidence intervals for each parameter.

        Raises
        ------
        ValueError
            If the `Minuit` object (`self.mi`) is not available. Ensure that the `fit_params` method has been run successfully before calling this method.
    """
        # Check if the class has a Minuit object stored
        if not hasattr(self, "mi"):
            raise ValueError("Fit has not been performed. Run fit_params() first.")

        # LaTeX labels for parameters - for use on plot axis
        param_labels = {
            "mu": r"$\mu$",
            "sigma": r"$\sigma$",
            "beta": r"$\beta$",
            "m": r"$m$",
            "lamb": r"$\lambda$",
            "mu_b": r"$\mu_b$",
            "sigma_b": r"$\sigma_b$",
            "f": r"$f$",
            "N": r"$N$"
        }

        # Plot standard preferences
        plot_config = {
            "xlabel_fontsize": 14,
            "ylabel_fontsize": 14,
            "title_fontsize": 16,
            "tick_fontsize": 12,
            "text_fontsize": 12,
            "legend_fontsize": 12,
            "line_width": 2,
            "grid": True,
        }

        # Recall parameter names
        params = self.mi.parameters

        # Create a 3x3 grid for plotting
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()

        # For all parameters a seperate subplot is created
        for i, param in enumerate(params):
            ax = axes[i]
            print(f'Plotting profile for {param}')
            # Get the likelihood profile data using iminuit's `mnprofile`
            # scan_values: where the nll is evaluated
            # nll_values: the nll values
            # success_flags: whether the minimization was successful
            scan_values, nll_values, success_flags = self.mi.mnprofile(param, bound=2, size=50)

            # Only inlclude points in which the minimization was successful
            scan_values = scan_values[success_flags]
            nll_values = nll_values[success_flags]

            # Shift nll values so is it standardised with minimum value at 0
            nll_values = nll_values - np.min(nll_values)

            # Find the 1 sigma and 2 sigma confidence intervals
            # The best fit/ minimium point
            min_value = scan_values[np.argmin(nll_values)] 
            # where the nll values are less than 0.5 and 2.0
            lhs_1sigma = scan_values[np.where(nll_values <= 0.5)[0][0]]
            rhs_1sigma = scan_values[np.where(nll_values <= 0.5)[0][-1]]
            lhs_2sigma = scan_values[np.where(nll_values <= 2.0)[0][0]]
            rhs_2sigma = scan_values[np.where(nll_values <= 2.0)[0][-1]]

            # Calculate the difference from the minimium value - ie where it will be on graph
            lhs_1sigma_diff = lhs_1sigma - min_value
            rhs_1sigma_diff = rhs_1sigma - min_value
            lhs_2sigma_diff = lhs_2sigma - min_value
            rhs_2sigma_diff = rhs_2sigma - min_value

            # Plot the likelihood profile
            ax.plot(scan_values, nll_values, color="blue", linewidth=plot_config["line_width"], label=r"$-2\ln\mathcal{L}$")

            # Shade the 1 sigma interval under the curve in green
            ax.fill_between(
                scan_values,
                0, nll_values,
                where=((scan_values >= lhs_1sigma) & (scan_values <= rhs_1sigma)),
                color="green",
                alpha=0.3,
                label=r"$1\sigma$ interval"
            )

            # Shade the 2 sigma interval under the curve in red
            ax.fill_between(
                scan_values,
                0, nll_values,
                where=((scan_values >= lhs_2sigma) & (scan_values <= lhs_1sigma)) |
                    ((scan_values >= rhs_1sigma) & (scan_values <= rhs_2sigma)),
                color="red",
                alpha=0.3,
                label=r"$2\sigma$ interval"
            )

            # Add horizontal lines for delta log like = 0.5 and 2 (ie 1 sigma and 2 sigma)
            ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
            ax.axhline(2.0, color="black", linestyle="--", linewidth=1)

            # Put text on the horizontal lines
            ax.text(
                scan_values[0], 0.5 + 0.1, r"$\Delta\ln\mathcal{L} = 0.5$",
                fontsize=plot_config["text_fontsize"], color="black", verticalalignment="bottom")
            ax.text(
                scan_values[0], 2.0 + 0.1, r"$\Delta\ln\mathcal{L} = 2.0$",
                fontsize=plot_config["text_fontsize"], color="black", verticalalignment="bottom")

            # Add vertical dashed lines for 1 sigma and 2 sigma intervals
            ax.axvline(lhs_1sigma, color="gray", linestyle="--")
            ax.axvline(rhs_1sigma, color="gray", linestyle="--")
            ax.axvline(lhs_2sigma, color="gray", linestyle="--", alpha=0.7)
            ax.axvline(rhs_2sigma, color="gray", linestyle="--", alpha=0.7)

            # Display the left and right confidence intervals for both 1 sigma and 2 sigma in the top right
            ax.text(
                0.98, 0.98,
                f"$1\sigma$: [{lhs_1sigma_diff:.3f}, {rhs_1sigma_diff:.3f}]\n"
                f"$2\sigma$: [{lhs_2sigma_diff:.3f}, {rhs_2sigma_diff:.3f}]",
                transform=ax.transAxes,
                fontsize=plot_config["text_fontsize"],
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray")
            )

            # set title and labels
            ax.set_xlabel(param_labels.get(param, param), fontsize=plot_config["xlabel_fontsize"])
            ax.set_ylabel(r"$-2\ln\mathcal{L}$", fontsize=plot_config["ylabel_fontsize"])
            ax.tick_params(axis="both", labelsize=plot_config["tick_fontsize"])

            # Add legend 
            handles, labels = ax.get_legend_handles_labels()
            filtered_handles_labels = [(h, l) for h, l in zip(handles, labels) if l in [r"$-2\ln\mathcal{L}$", r"$1\sigma$ interval", r"$2\sigma$ interval"]]
            handles, labels = zip(*filtered_handles_labels)
            ax.legend(handles, labels, fontsize=plot_config["legend_fontsize"], loc="upper left")

            # Add grid look
            if plot_config["grid"]:
                ax.grid(True)

        plt.tight_layout()
        plt.show()



    def param_bootstrap_samples(self, num_samples, sample_sizes, output_directory="Bootstrap/Samples", poisson=False):
        """
        Generate bootstrap samples with different sizes using the accept-reject method and save them as single files.
        Starts fresh by deleting the existing directory.

        Parameters
        ----------
        num_samples : int
            The number of bootstrap samples to generate for each sample size.
        sample_sizes : list of int
            The base sizes of each bootstrap sample. Each size will have a separate output file.
        output_directory : str, optional
            Directory to save the generated samples as .npy files (default: "Bootstrap/Samples").
        poisson : bool, optional
            Whether to apply Poisson variation to the sample sizes (default: False).

        Notes
        -----
        - Each file will be named "Samples_No_<num_samples>_Size_<sample_size>.npy".
        - Each file contains a list of arrays, where each array corresponds to a sample with its actual size.
        """

        # Ensure the output directory exists and is empty
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        os.makedirs(output_directory)

        # Loop over each sample size, storing each set in a separate file
        for size in sample_sizes:
            print(f"Generating {num_samples} samples with base size {size}...")

            # List to store samples - must be list to must store variable sizes due to Poisson
            all_samples = []
            
            # Generate the desired number of samples
            for i in range(num_samples):
                # Use Poisson distribution if poisson = True otherwise use the base size
                if poisson:
                    actual_size = np.random.poisson(size)
                else:
                    actual_size = size
                
                # Generate a sample with the determined size
                sample = self.accept_reject_sample(desired_samples=actual_size)
                # Add to overall list
                all_samples.append(sample)

             # Save list of samples as list of arrays
            file_name = f"Samples_No_{num_samples}_BaseSize_{size}.npy"
            file_path = os.path.join(output_directory, file_name)
            np.save(file_path, np.array(all_samples, dtype=object), allow_pickle=True)

            print(f"Bootstrap samples with base size {size} saved to {file_path}")



    def param_bootstrap_fit(self,initial_params , input_directory="Bootstrap/Samples", output_directory="Bootstrap/Param_Results"):
        """
        Perform parameter fitting using `fit_params` method on bootstrap samples and save the results.

        Parameters
        ----------
        initial_params : list
            Initial guesses for the parameters [mu, sigma, beta, m, lamb, mu_b, sigma_b, f].
        input_directory : str, optional
            Path to the directory containing bootstrap sample files (default: "Bootstrap/Samples").
            Each file should follow the naming convention "Samples_No_<num_samples>_BaseSize_<sample_size>.npy".
        output_directory : str, optional
            Path to the directory where the fitting results will be saved (default: "Bootstrap/Param_Results").
            Each result file will be named "ParamResults_No_<num_samples>_BaseSize_<sample_size>.npy".

        Returns
        -------
        For each sample file:
        - A `.npy` file containing an array of shape (2, num_samples, num_params):
            - First dimension (`values`): Fitted parameter values for each sample.
            - Second dimension (`errors`): Corresponding errors for each parameter.
        - Prints the number of non-converging samples and the output file path.

        Raises
        ------
        ValueError
            If the `fit_params` method encounters invalid inputs or fitting fails for unexpected reasons.

        """

        # Ensure the output directory exists and is empty
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        os.makedirs(output_directory)

        # Repeat for all files in the input directory
        for file_name in os.listdir(input_directory):
            # Ensure it is a numpy file - ie the samples
            if file_name.endswith(".npy"):
                # Extract the sample size and number samples from the file name
                match = re.search(r"Samples_No_(\d+)_BaseSize_(\d+).npy", file_name)
                if not match:
                    continue
                num_samples = int(match.group(1))
                sample_size = int(match.group(2))

                print(f"Processing bootstrap samples of size {sample_size}...")

                # Load the bootstrap samples data from the file
                file_path = os.path.join(input_directory, file_name)
                samples = np.load(file_path, allow_pickle=True) 

                # Add the Base sample size to the initial parameters as the expected number of events
                initial_params_samples = initial_params.copy()
                initial_params_samples.append(sample_size)
                num_params = len(initial_params_samples)
                
                # Pre define arrays to store the results for faster computation
                values = np.full((num_samples, num_params), np.nan)
                errors = np.full((num_samples, num_params), np.nan)
                non_converged_count = 0

                # Repeat parameter fitting for each sample in the file
                for i, sample in enumerate(samples):
                    try:
                        # Perform the parameter fitting using `fit_params`
                        fit_results, fit_errors = self.fit_params(initial_params_samples, sample)
                        values[i] = fit_results
                        errors[i] = fit_errors
                    except Exception as e:
                        # If fitting fails, log the failure and continue
                        print(f"Sample {i + 1} (size {sample_size}) did not converge")
                        non_converged_count += 1

                # Combine results into a single array
                results = np.array([values, errors])

                # Save the results to the file
                output_file_name = f"ParamResults_No_{num_samples}_BaseSize_{sample_size}.npy"
                output_file_path = os.path.join(output_directory, output_file_name)
                np.save(output_file_path, results, allow_pickle=True)

                print(f"In total {non_converged_count} samples did not converge out of {num_samples}.")
                print(f"Results saved to {output_file_path}\n ")


    def param_bootstrap_analysis(self, input_directory="Bootstrap/Param_Results", output_directory="Bootstrap/Plots"):
        """
        Analyse and visualise parameter fitting results from bootstrap sample, storing plots in the output directory.

        Parameters
        ----------
        input_directory : str, optional
            Path to the directory containing bootstrap fitting result files (default: "Bootstrap/Param_Results").
            Each file is expected to follow the naming convention:
            "ParamResults_No_<num_samples>_BaseSize_<sample_size>.npy".
        output_directory : str, optional
            Path to the directory where plots will be saved (default: "Bootstrap/Plots").
            Subdirectories for histograms, trends, and pull plots are created or cleared before use.

        Returns
        -------
        This method generates the following plots:
        1. Histograms for each parameter (value, error, and pull) across bootstrap samples.
        2. Bias and Error trends vs. sample size for each parameter.
        4. Pull distributions for each parameter across all samples.

        results : dict
            A dictionary storing computed metrics for each sample size. Keys are sample sizes, and values are dictionaries with:
            - "Values_Mean", "Values_Std", "Values_Bias", "Errors_Mean", "Errors_Std", "Pull_Mean", "Pull_Std", "Pull_Mean_Error", "Pull_Std_Error".

        Raises
        ------
        FileNotFoundError
            If the input directory does not exist or contains no valid files.
        """

        # LaTeX labels for parameters - for use on plot axis
        param_labels = {
            "mu": r"$\mu$",
            "sigma": r"$\sigma$",
            "beta": r"$\beta$",
            "m": r"$m$",
            "lamb": r"$\lambda$",
            "mu_b": r"$\mu_b$",
            "sigma_b": r"$\sigma_b$",
            "f": r"$f$",
            "N": r"$N$"
        }

        # Standard preferences for plots
        plot_config = {
            "xlabel_fontsize": 16,
            "ylabel_fontsize": 16,
            "title_fontsize": 18,
            "tick_fontsize": 14,
            "suptitle_fontsize": 18,
            "legend_fontsize": 14,
            "text_fontsize": 14,
            "line_width": 2,
            "bar_alpha": 0.6,  # Add this key for bar transparency
            "region_alpha": 0.3,  # Transparency for shaded regions
            "text_boxstyle": dict(boxstyle="round", facecolor="white", edgecolor="gray"),  # Box for μ, σ text
        }

        # Clear and recreate the output directories for plots
        for data_label in ["Value", "Error", "Pull"]:
            sub_dir = f"{output_directory}/{data_label}_Histograms"
            if os.path.exists(sub_dir):
                shutil.rmtree(sub_dir)  
            os.makedirs(sub_dir) 
            
        sub_dir = f"{output_directory}/Trends_with_Samples_Size"
        if os.path.exists(sub_dir):
            shutil.rmtree(sub_dir) 
        os.makedirs(sub_dir)  

        sub_dir = f"{output_directory}/Pull_Plots"
        if os.path.exists(sub_dir):
            shutil.rmtree(sub_dir) 
        os.makedirs(sub_dir) 

        # Initialise the results dictionary
        results = {} 

        # Loop over each file in the input directory - storing the results of paraemter fitting
        for filename in os.listdir(input_directory):
            if filename.endswith(".npy"):
                match = re.search(r"ParamResults_No_(\d+)_BaseSize_(\d+).npy", filename)
                if match:
                    sample_size = int(match.group(2))
                    print(f"Processing sample size: {sample_size}")

                # Load the data from the file
                filepath = os.path.join(input_directory, filename)
                data = np.load(filepath)
                Values = data[0]
                Errors = data[1]

                # Calculate the true values for the sample size - using class stored and the samples size from file name
                Truth = self.true_params.copy()
                Truth.append(sample_size)
                Truth = np.array(Truth)
                Pull = (Values - Truth) / Errors

                # Calculate the mean and standard deviation of the values, errors, and pulls
                calc_values = {
                    "Values_Mean": np.nanmean(Values, axis=0),
                    "Values_Std": np.nanstd(Values, axis=0),
                    "Values_Bias": np.nanmean(Values, axis=0) - Truth,
                    "Errors_Mean": np.nanmean(Errors, axis=0),
                    "Errors_Std": np.nanstd(Errors, axis=0),
                    "Pull_Mean": np.nanmean(Pull, axis=0),
                    "Pull_Std": np.nanstd(Pull, axis=0),
                    "Pull_Mean_Error": np.nanstd(Pull, axis=0) / np.sqrt(Pull.shape[0]),  # SEM of the pull mean
                    "Pull_Std_Error": np.nanstd(Pull, axis=0) / np.sqrt(2 * Pull.shape[0]),  # Error on pull std
                }

                # Add the calculated results to the results dictionary under key of sample size
                results[sample_size] = calc_values

                # Dictionary to map labels to data arrays for plotting efficiency
                data_types = {
                    "Value": Values,
                    "Error": Errors,
                    "Pull": Pull,}
                                
                # Create Pull Distribution Plots

                # Loop over data types: Values, Errors, and Pulls
                for data_label, data_array in data_types.items():
                    # Each are saves to a separate subdirectory
                    sub_dir = f"{output_directory}/{data_label}_Histograms"
                    num_params = data_array.shape[1]

                    # Create a 3x3 grid for plotting histograms
                    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
                    axes = axes.flatten()
                    
                    # Loop over each parameter and plot its histogram in each subplot in the grid
                    for i in range(num_params):
                        param_values = data_array[:, i]
                        param_values = param_values[~np.isnan(param_values)]

                        # Remove the last 2% and top 2% of values
                        lower_limit = np.percentile(param_values, 2)
                        upper_limit = np.percentile(param_values, 98)
                        param_values = param_values[(param_values >= lower_limit) & (param_values <= upper_limit)]

                        # Find the histogram characterists and the error on each bar
                        bins = np.histogram_bin_edges(param_values, bins=15)
                        hist, bin_edges = np.histogram(param_values, bins=bins)
                        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                        bin_widths = bin_edges[1:] - bin_edges[:-1]
                        hist_errors = np.sqrt(hist)

                        # Plot the histogram
                        ax = axes[i]
                        ax.bar(bin_centers, hist, width=bin_widths, alpha=0.6, label='Histogram', color='blue', edgecolor='black')
                        ax.errorbar(bin_centers, hist, yerr=hist_errors, fmt='k.', capsize=3, label='Error Bars')

                        # Calculate mean and std for gaussian
                        mu = np.nanmean(param_values)
                        std = np.nanstd(param_values)

                        # Compute Gaussian PDF
                        x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
                        pdf = norm.pdf(x, mu, std)

                        # Scale Gaussian PDF to match the histogram
                        pdf_scaled = pdf * np.sum(hist) * bin_widths[0]
                        ax.plot(x, pdf_scaled, 'r-', label='Gaussian Fit')

                        # Add labels using param_labels dictioanry
                        param_name = list(param_labels.keys())[i]  
                        param_label = param_labels.get(param_name, f"Param {i + 1}")  


                        # Display mu and sigma of plot in the top-left corner
                        ax.text(
                            0.02, 0.98, f"{data_label}: {param_label}\n {mu:.2f}$\pm${std:.2f}",
                            transform=ax.transAxes, fontsize=plot_config["text_fontsize"],
                            verticalalignment='top', horizontalalignment='left',
                            bbox=plot_config["text_boxstyle"]
                        )

                        # Set the plot labels and legend
                        ax.set_xlabel(f"{data_label} : {param_label}", fontsize=plot_config["xlabel_fontsize"])
                        ax.set_ylabel("Counts", fontsize=plot_config["ylabel_fontsize"])
                        ax.tick_params(axis='both', which='major', labelsize=plot_config["tick_fontsize"])
                        ax.legend(loc='upper right', fontsize=plot_config["legend_fontsize"])

                    # Set the overall title of grid
                    fig.suptitle(
                        f"Histograms of Parameter {data_label} - Sample Size {sample_size}",
                        fontsize=plot_config["suptitle_fontsize"])
                    plt.tight_layout(rect=[0, 0, 1, 0.96])

                    # Save the plot
                    save_path = f"{sub_dir}/{data_label}_histograms_{sample_size}.png"
                    plt.savefig(save_path)
                    plt.close(fig)  

                    print(f"Histogram of {data_label} saved in {sub_dir}")



        # Create Bias and Error against sample size summary plots
        # Sort the sample sizes so the plots are in order
        sample_sizes = sorted(results.keys())
        num_params = len(param_labels)

        # Plot Bias vs. Sample Size
        fig, axes = plt.subplots(3, 3, figsize=(13, 9))
        axes = axes.flatten()
        
        # Plot for all parameters bar N as this is the expected number of events and not comparible
        for i in range(num_params-1):
            # Determine Absolute Bias for each parameter
            biases = [abs(results[sample_size]["Values_Bias"][i]) for sample_size in sample_sizes]
            ax = axes[i]
            ax.plot(sample_sizes, biases, marker='o', label='Bias')
            ax.set_xlabel("Sample Size", fontsize=plot_config["xlabel_fontsize"])
            ax.set_ylabel(f"Abs(Bias) in {list(param_labels.values())[i]}", fontsize=plot_config["ylabel_fontsize"])
            ax.tick_params(axis='both', which='major', labelsize=plot_config["tick_fontsize"])
            ax.grid(True)
            # ax.set_xscale('log')


        # Remove unused subplots, make bottom-right blank
        for j in range(num_params, 9):
            if j == 8:  # Bottom-right subplot
                axes[j].set_axis_off()  # Turn off the axis to leave it blank
            else:
                fig.delaxes(axes[j])  # Remove unused axes

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save Plot
        plt.savefig(f"{output_directory}/Trends_with_Samples_Size/Bias_vs_Sample_Size.png")
        plt.show() 
        plt.close(fig)
        print(f"Bias vs Sample Size plot saved in {output_directory}/Trends_with_Samples_Size")

        # Plot Values_Std vs. Sample Size
        fig, axes = plt.subplots(3, 3, figsize=(13, 9))
        axes = axes.flatten()

        # Plot for all parameters bar N as this is the expected number of events and not comparible
        for i in range(num_params - 1): 
            # Determine the uncertainty for each parameter value
            errors_mean = [results[sample_size]["Values_Std"][i] for sample_size in sample_sizes]
            ax = axes[i]
            ax.plot(sample_sizes, errors_mean, marker='o', label='Error Mean')
            ax.set_xlabel("Sample Size", fontsize=plot_config["xlabel_fontsize"])
            ax.set_ylabel(f"Uncertainty in {list(param_labels.values())[i]}", fontsize=plot_config["ylabel_fontsize"])
            ax.tick_params(axis='both', which='major', labelsize=plot_config["tick_fontsize"])
            ax.grid(True)
            # ax.set_xscale('log')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save Plot
        plt.savefig(f"{output_directory}/Trends_with_Samples_Size/Errors_Mean_vs_Sample_Size.png")
        plt.show()  # Display in Jupyter Notebook
        plt.close(fig)
        print(f"Error vs Sample Size plot saved in {output_directory}/Trends_with_Samples_Size")

        # Plot the Pull Distribution for each parameter - with a seperate plot for each sample size
        for sample_size, calc_values in results.items():
            num_params = len(param_labels)
            fig, ax = plt.subplots(figsize=(10, 6))

            # Add light grey shading for Pull between 1 and 2
            ax.axvspan(-2, -1, color="lightgrey", alpha=0.4)
            ax.axvspan(1, 2, color="lightgrey", alpha=0.4)

            # Loop over parameters and plot horizontal bars
            for i, (param, label) in enumerate(param_labels.items()):
                pull_mean = calc_values["Pull_Mean"][i]
                pull_std = calc_values["Pull_Std"][i]
                pull_mean_error = calc_values["Pull_Mean_Error"][i]
                pull_std_error = calc_values["Pull_Std_Error"][i]

                # Light blue bar: between the centers of the two red bars
                ax.barh(
                    y=i,
                    width=2 * pull_std,
                    left=pull_mean - pull_std,
                    height=0.4,
                    color="lightblue",
                    alpha=plot_config["bar_alpha"],
                    edgecolor="black",
                )

                # Dark blue region: centered at Pull Mean, between Pull Mean +/- Pull Mean Error
                ax.barh(
                    y=i,
                    width=pull_mean_error * 2,
                    left=pull_mean - pull_mean_error,
                    height=0.4,
                    color="darkblue",
                    alpha=plot_config["bar_alpha"],
                    edgecolor="black",
                )

                # Red regions: centered at Pull Mean ± Pull Std, with widths defined by Pull Std Error
                ax.barh(
                    y=i,
                    width=pull_std_error * 2,
                    left=pull_mean - pull_std - pull_std_error,
                    height=0.4,
                    color="red",
                    alpha=plot_config["bar_alpha"],
                    edgecolor="black",
                )
                ax.barh(
                    y=i,
                    width=pull_std_error * 2,
                    left=pull_mean + pull_std - pull_std_error,
                    height=0.4,
                    color="red",
                    alpha=plot_config["bar_alpha"],
                    edgecolor="black",
                )

            # Add labels and grid
            ax.set_yticks(range(num_params))
            ax.set_yticklabels(list(param_labels.values()), fontsize=plot_config["ylabel_fontsize"])
            ax.set_xlabel("Pull", fontsize=plot_config["xlabel_fontsize"])
            ax.set_title(f"Pull Distributions for Sample Size {sample_size}", fontsize=plot_config["title_fontsize"])
            ax.tick_params(axis="both", which="major", labelsize=plot_config["tick_fontsize"])
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axvline(-2, color="brown", linestyle="dashdot", linewidth=1)
            ax.axvline(2, color="brown", linestyle="dashdot", linewidth=1)
            ax.grid(True, axis="x")

            # Save the plot
            save_path = os.path.join(output_directory, f"Pull_Plots/Pull_Distributions_{sample_size}.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)

            print(f"Saved pull distribution plot for sample size {sample_size} in {save_path}")

        # Return the results dictionary
        return results

    def fit_params_sWeights(self, initial_params, samples = None, print_results = False, norm_check = True):
        """
        Fit parameters using the sWeights method.

        This method performs the following steps:
        1. Fits the X dimension using an extended unbinned likelihood method.
        2. Calculates signal and background weights using the sWeights method.
        3. Projects the signal distribution to the Y dimension.
        4. Fits the Y dimension using a binned maximum likelihood estimation.

        Parameters
        ----------
        initial_params : list or array-like
            Initial guesses for the parameters [mu, sigma, beta, m, f, lamb, N].
        samples : np.ndarray, optional
            The samples to be used for fitting. If not provided, the samples generated by the `accept_reject_sample` method will be used.
        print_results : bool, optional
            If True, prints the fitting results and plots the intermediate steps. Default is False.
        norm_check : bool, optional
            If True, performs normalization checks during the sWeights calculation. Default is True.

        Returns
        -------
        mi_total_values : dict
            A dictionary containing the fitted parameter values.
        mi_total_errors : dict
            A dictionary containing the errors of the fitted parameters.

        Raises
        ------
        ValueError
            If no samples are provided or generated by the `accept_reject_sample` method.
        RuntimeError
            If the minimization does not converge.
        """
        # Allow for the samples to be passed in, if not try use the samples already generated in class
        if samples is None:
            samples = self.samples
            if samples is None:
                raise ValueError("No samples have been provided or generated by the `accept_reject_sample` method first")
            
        # Get the indices that would sort the first column
        sorted_indices = np.argsort(samples[:, 0])

        # Sort the entire array in the X dimension for easy plotting
        samples = samples[sorted_indices]
        samples_x = samples[:, 0]
        samples_y = samples[:, 1]

        # Define the density function - required by iminuit Extended Unbinned Likelihood
        def density_x(samples_in, mu, sigma, beta, m, f, N):
            samples_x = samples_in
            pdf_vals = f * self.Signal.X.pdf_fitting(samples_x, mu, sigma, beta, m) + (1-f) * self.Background.X.pdf_fitting(samples_x)
            return N, N * pdf_vals

        # Create the negative log-likelihood function - to be used only for x dimension
        neg_log_likelihood_x = ExtendedUnbinnedNLL(samples_x, density_x)

        # Create the Minuit object with initial guesses
        mi_x = Minuit(neg_log_likelihood_x, mu=initial_params[0],sigma=initial_params[1], beta=initial_params[2], 
                    m=initial_params[3], f=initial_params[4], N=initial_params[6])

        # Set parameter limits based on each paramaters restrictions and physical significance
        # sigma > 0
        mi_x.limits["sigma"] = (1e-3, None)
        # beta > 0
        mi_x.limits["beta"] = (1e-2, None)
        # m > 1
        mi_x.limits["m"] = (1.01, None)
        # f in [0, 1]
        mi_x.limits["f"] = (0, 1)
        # N_expected > 0
        mi_x.limits["N"] = (1e-3, None)

        # Run the minimisation
        mi_x.migrad()

        if not mi_x.valid:
            raise RuntimeError("Minimisation did not converge")

        # Run the error analysis
        mi_x.hesse()

        if print_results:
            print("Step 1 - Iminuit Extended MLE: X Dimension Fitting Results")
            print(mi_x)

        def signal_x_pdf_fit(x):
            return self.Signal.X.pdf_fitting(x, mi_x.values["mu"], mi_x.values["sigma"], mi_x.values["beta"], mi_x.values["m"])
    
        def background_x_pdf_fit(x): 
            return self.Background.X.pdf_fitting(x)
        
        
        # Calculate the signal and background weights using the SWeight method
        signal_count  = mi_x.values["N"] * mi_x.values["f"]
        background_count = mi_x.values["N"] * (1 - mi_x.values["f"])
        xrange = (self.lower_bound_X, self.upper_bound_X)
        sweighter = SWeight( samples_x, pdfs=[signal_x_pdf_fit,background_x_pdf_fit], yields=[signal_count,background_count], discvarranges=((xrange),), checks = norm_check)
        signal_weight = sweighter.get_weight(0,samples_x)
        background_weight = sweighter.get_weight(1,samples_x)
    
        # Display the results
        if print_results:
            print("Step 2 - Sweights: Determine Signal and Background Weights from X")
            fig, ax = plt.subplots()
            ax.plot(samples_x, signal_weight, 'r--', label='Signal Weight')
            ax.plot(samples_x, background_weight, 'b--', label='Background Weight')
            ax.plot(signal_weight + background_weight, 'k-', label='Total Weight')
            ax.legend()
            ax.set_xlim(self.lower_bound_X, self.upper_bound_X)
            ax.set_xlabel('$X$')
            ax.set_ylabel('Weight')
            plt.show()

        # Project the distribution to the Y dimension by applying the weight
        # Effectively giving signal distribution in Y
        yrange = (self.lower_bound_Y, self.upper_bound_Y)
        ysw, ye = np.histogram( samples_y, bins=50, range=yrange, weights=signal_weight)
        ysw2, ye = np.histogram( samples_y, bins=50, range=yrange, weights=signal_weight**2)
        cy = 0.5*(ye[1:]+ye[:-1])

        if print_results:
            print("Step 3 - Sweights: Project to Signal Distribution in Y")

            # Plot to show the SWeighted data in Y compared to the true signal distribution
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot the weighted histogram with error bars
            ax.errorbar(cy, ysw, yerr=ysw2**0.5, fmt='rx', label='All SWeighted Data in Y', markersize=5, capsize=4, capthick=1)
            ax.hist(cy, bins=len(cy), weights=ysw, density=False, alpha=0.3, color='red')

            # Plot the expected PDF curve
            ax.plot(cy, (ye[2] - ye[1]) * mi_x.values["N"] * mi_x.values["f"] * self.Signal.Y.pdf(cy), label='Expected Signal Counts \n From True PDF', color='blue', linewidth=2)


            ax.grid(visible=True, linestyle='--', linewidth=0.5)
            ax.set_xlabel("Y", fontsize=14)
            ax.set_ylabel("Counts", fontsize=14)
            ax.legend(fontsize=12)

            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=10)

            plt.tight_layout()
            plt.show()

        # Define the CDF function for the signal in Y dimension for Binned MLE
        def cdf_signal_y(x, lamb):
            return self.Signal.Y.cdf_fitting(x, lamb)
        
        # Perform Binned MLE for the signal in the Y dimension witht the whole data set
        neg_log_likelihood_signal_y_binned = BinnedNLL(ysw, ye, cdf_signal_y)
        mi_signal_y = Minuit(neg_log_likelihood_signal_y_binned, lamb=initial_params[5])

        # Limit lamb > 0
        mi_signal_y.limits["lamb"] = (1e-3, None)

        mi_signal_y.migrad()
        if not mi_signal_y.valid:
            raise RuntimeError("Minimisation in Y did not converge")
        
        # Run the error analysis
        mi_signal_y.hesse()

        if print_results:
            print("Step 4 - Binned MLE: Signal in Y Dimension Fitting Results")
            print(mi_signal_y)

        # Explicitly construct dictionaries from ValueView objects
        mi_x_values_dict = {key: mi_x.values[key] for key in mi_x.parameters}
        mi_signal_y_values_dict = {key: mi_signal_y.values[key] for key in mi_signal_y.parameters}
        mi_x_errors_dict = {key: mi_x.errors[key] for key in mi_x.parameters}
        mi_signal_y_errors_dict = {key: mi_signal_y.errors[key] for key in mi_signal_y.parameters}
        mi_total_values = {**mi_x_values_dict, **mi_signal_y_values_dict}
        mi_total_errors = {**mi_x_errors_dict, **mi_signal_y_errors_dict}

        if print_results:
            print("\n Final summary of Results")
            # Print table of results 
            true_values = {
                "mu": self.true_params[0],
                "sigma": self.true_params[1],
                "beta": self.true_params[2],
                "m": self.true_params[3],
                "f": self.true_params[7],
                "N": "N/A", 
                "lamb": self.true_params[4],
            }
            table_data = [
                [
                    param, 
                    f"{round(value, 4)} ± {round(mi_total_errors[param], 4)}", 
                    true_values.get(param, "N/A")  # Use "N/A" if no true value is provided
                ]
                for param, value in mi_total_values.items()
            ]
            headers = ["Parameter", "Value ± Error", "True Value"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

        return mi_total_values, mi_total_errors

    def param_bootstrap_sWeights_fit(self,initial_params, norm_check = True,  input_directory="Bootstrap/Samples", output_directory="Bootstrap/sWeights/Results"):
        """
        Perform parameter fitting using sWeights for signal and background PDFs for all bootstrap samples.

        This function fits the parameters of signal and background probability density functions (PDFs)
        using extended unbinned maximum likelihood for the X dimension. It then calculates the signal
        and background weights using the sWeight method and fits the signal PDF in the Y dimension using
        binned maximum likelihood.

        Parameters
        ----------
        initial_params : list
            Initial guesses for the parameters. The list must contain:
            [mu, sigma, beta, m, f, lamb, N].
        samples : numpy.ndarray, optional
            Array of samples for fitting. If None, it uses samples generated or stored
            in the class. Defaults to None.
        print_results : bool, optional
            If True, prints detailed fitting results and plots. Defaults to False.
        norm_check : bool, optional
            If True, enables normalization checks in the sWeight method. Defaults to True.

        Returns
        -------
        tuple
            Two dictionaries containing the fitted parameter values and their associated errors:
                - mi_total_values (dict): Fitted parameter values.
                - mi_total_errors (dict): Fitted parameter errors.

        Raises
        ------
        ValueError
            If no samples are provided or available for fitting.
        RuntimeError
            If minimization for X or Y dimension does not converge.
        """
        # Ensure the output directory exists and is empty
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        os.makedirs(output_directory)

        # Repeat for all files in the input directory
        for file_name in os.listdir(input_directory):
            # Ensure it is a numpy file - ie the samples
            if file_name.endswith(".npy"):
                # Extract the sample size and number samples from the file name
                match = re.search(r"Samples_No_(\d+)_BaseSize_(\d+).npy", file_name)
                if not match:
                    continue
                num_samples = int(match.group(1))
                sample_size = int(match.group(2))

                print(f"Processing bootstrap samples of size {sample_size} using sWeights...")

                # Load the bootstrap samples data from the file
                file_path = os.path.join(input_directory, file_name)
                samples = np.load(file_path, allow_pickle=True) 

                # Add the Base sample size to the initial parameters as the expected number of events
                initial_params_samples = initial_params.copy()
                initial_params_samples.append(sample_size)
                num_params = len(initial_params_samples)
                
                # Pre define arrays to store the results for faster computation
                values = np.full((num_samples, num_params), np.nan)
                errors = np.full((num_samples, num_params), np.nan)
                non_converged_count = 0

                # Repeat parameter fitting for each sample in the file
                for i, sample in enumerate(samples):
                    try:
                        # Perform the parameter fitting using `fit_params_sWeights`
                        fit_results, fit_errors = self.fit_params_sWeights(initial_params_samples, sample, norm_check = norm_check)
                        values[i] = np.array(list(fit_results.values()))
                        errors[i] = np.array(list(fit_errors.values()))

                    except Exception as e:
                        # If fitting fails, log the failure and continue
                        print(f"Sample {i + 1} (size {sample_size}) did not converge")
                        print(e)
                        non_converged_count += 1

                # Combine results into a single array
                results = np.array([values, errors])

                # Save the results to the file 
                output_file_name = f"ParamResults_No_{num_samples}_BaseSize_{sample_size}.npy"
                output_file_path = os.path.join(output_directory, output_file_name)
                np.save(output_file_path, results, allow_pickle=True)

                print(f"In total {non_converged_count} samples did not converge out of {num_samples}.")
                print(f"Results saved to {output_file_path}\n ")


    def param_bootstrap_sWeights_analysis(self, input_directory="Bootstrap/sWeights/Results", output_directory="Bootstrap/sWeights/Plots"):
        """
        Perform a parametric bootstrap analysis using sWeights and generate plots.

        This method performs the following steps:
        1. Loads the bootstrap results from the specified input directory.
        2. Calculates the mean, standard deviation, bias, and pull for each parameter.
        3. Generates histograms for the values, errors, and pulls of each parameter.
        4. Creates summary plots for bias and error against sample size.
        5. Generates pull distribution plots for each parameter and sample size.

        Parameters
        ----------
        input_directory : str, optional
            The directory containing the bootstrap results (.npy files). Default is "Bootstrap/sWeights/Results".
        output_directory : str, optional
            The directory where the plots will be saved. Default is "Bootstrap/sWeights/Plots".

        Returns
        -------
        This method generates the following plots:
        1. Histograms for each parameter (value, error, and pull) across bootstrap samples.
        2. Bias and Error trends vs. sample size for each parameter.
        4. Pull distributions for each parameter across all samples.

        results : dict
            A dictionary storing computed metrics for each sample size. Keys are sample sizes, and values are dictionaries with:
            - "Values_Mean", "Values_Std", "Values_Bias", "Errors_Mean", "Errors_Std", "Pull_Mean", "Pull_Std", "Pull_Mean_Error", "Pull_Std_Error".

        Raises
        ------
        FileNotFoundError
            If the input directory does not exist or contains no valid files.
        """

        # LaTeX labels for parameters - for use on plot axis
        param_labels = {
            "mu": r"$\mu$",
            "sigma": r"$\sigma$",
            "beta": r"$\beta$",
            "m": r"$m$",
            "f" : r"$f$",
            "N": r"$N$",
            "lamb": r"$\lambda$",
        }

        # Standard preferences for plots
        plot_config = {
            "xlabel_fontsize": 16,
            "ylabel_fontsize": 16,
            "title_fontsize": 18,
            "tick_fontsize": 14,
            "suptitle_fontsize": 18,
            "legend_fontsize": 14,
            "text_fontsize": 14,
            "line_width": 2,
            "bar_alpha": 0.6,  # Add this key for bar transparency
            "region_alpha": 0.3,  # Transparency for shaded regions
            "text_boxstyle": dict(boxstyle="round", facecolor="white", edgecolor="gray"),  # Box for μ, σ text
        }

        # Clear and recreate the output directories for plots
        for data_label in ["Value", "Error", "Pull"]:
            sub_dir = f"{output_directory}/{data_label}_Histograms"
            if os.path.exists(sub_dir):
                shutil.rmtree(sub_dir)  
            os.makedirs(sub_dir) 
            
        sub_dir = f"{output_directory}/Trends_with_Samples_Size"
        if os.path.exists(sub_dir):
            shutil.rmtree(sub_dir) 
        os.makedirs(sub_dir)  

        sub_dir = f"{output_directory}/Pull_Plots"
        if os.path.exists(sub_dir):
            shutil.rmtree(sub_dir) 
        os.makedirs(sub_dir) 

        # Initialise the results dictionary
        results = {} 

        # Loop over each file in the input directory - storing the results of paraemter fitting
        for filename in os.listdir(input_directory):
            if filename.endswith(".npy"):
                match = re.search(r"ParamResults_No_(\d+)_BaseSize_(\d+).npy", filename)
                if match:
                    sample_size = int(match.group(2))
                    print(f"Processing sWeights fitted sample size: {sample_size}")

                # Load the data from the file
                filepath = os.path.join(input_directory, filename)
                data = np.load(filepath)
                Values = data[0]
                Errors = data[1]


                # Calculate the true values for the sample size - using class stored and the samples size from file name
                Truth = [self.true_params[i] for i in [0, 1, 2, 3, 7]]
                Truth.append(sample_size)
                Truth.append(self.true_params[4])
                Truth = np.array(Truth)
                Pull = (Values - Truth) / Errors

                # Calculate the mean and standard deviation of the values, errors, and pulls
                calc_values = {
                    "Values_Mean": np.nanmean(Values, axis=0),
                    "Values_Std": np.nanstd(Values, axis=0),
                    "Values_Bias": np.nanmean(Values, axis=0) - Truth,
                    "Errors_Mean": np.nanmean(Errors, axis=0),
                    "Errors_Std": np.nanstd(Errors, axis=0),
                    "Pull_Mean": np.nanmean(Pull, axis=0),
                    "Pull_Std": np.nanstd(Pull, axis=0),
                    "Pull_Mean_Error": np.nanstd(Pull, axis=0) / np.sqrt(Pull.shape[0]),  # SEM of the pull mean
                    "Pull_Std_Error": np.nanstd(Pull, axis=0) / np.sqrt(2 * Pull.shape[0]),  # Error on pull std
                }

                # Add the calculated results to the results dictionary under key of sample size
                results[sample_size] = calc_values

                # Dictionary to map labels to data arrays for plotting efficiency
                data_types = {
                    "Value": Values,
                    "Error": Errors,
                    "Pull": Pull,}
                                
                # Create Pull Distribution Plots

                # Loop over data types: Values, Errors, and Pulls
                for data_label, data_array in data_types.items():
                    # Each are saves to a separate subdirectory
                    sub_dir = f"{output_directory}/{data_label}_Histograms"
                    num_params = data_array.shape[1]

                    # Create a 3x3 grid for plotting histograms
                    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
                    axes = axes.flatten()
                    
                    # Loop over each parameter and plot its histogram in each subplot in the grid
                    for i in range(num_params):
                        param_values = data_array[:, i]
                        param_values = param_values[~np.isnan(param_values)]

                        # Remove the last 2% and top 2% of values
                        lower_limit = np.percentile(param_values, 2)
                        upper_limit = np.percentile(param_values, 98)
                        param_values = param_values[(param_values >= lower_limit) & (param_values <= upper_limit)]

                        # Find the histogram characterists and the error on each bar
                        bins = np.histogram_bin_edges(param_values, bins=15)
                        hist, bin_edges = np.histogram(param_values, bins=bins)
                        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                        bin_widths = bin_edges[1:] - bin_edges[:-1]
                        hist_errors = np.sqrt(hist)

                        # Plot the histogram
                        ax = axes[i]
                        ax.bar(bin_centers, hist, width=bin_widths, alpha=0.6, label='Histogram', color='blue', edgecolor='black')
                        ax.errorbar(bin_centers, hist, yerr=hist_errors, fmt='k.', capsize=3, label='Error Bars')

                        # Calculate mean and std for gaussian
                        mu = np.nanmean(param_values)
                        std = np.nanstd(param_values)

                        # Compute Gaussian PDF
                        x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
                        pdf = norm.pdf(x, mu, std)

                        # Scale Gaussian PDF to match the histogram
                        pdf_scaled = pdf * np.sum(hist) * bin_widths[0]
                        ax.plot(x, pdf_scaled, 'r-', label='Gaussian Fit')

                        # Add labels using param_labels dictioanry
                        param_name = list(param_labels.keys())[i]  
                        param_label = param_labels.get(param_name, f"Param {i + 1}")  


                        # Display mu and sigma of plot in the top-left corner
                        ax.text(
                            0.02, 0.98, f"{data_label}: {param_label}\n {mu:.2f}$\pm${std:.2f}",
                            transform=ax.transAxes, fontsize=plot_config["text_fontsize"],
                            verticalalignment='top', horizontalalignment='left',
                            bbox=plot_config["text_boxstyle"]
                        )

                        # Set the plot labels and legend
                        ax.set_xlabel(f"{data_label} : {param_label}", fontsize=plot_config["xlabel_fontsize"])
                        ax.set_ylabel("Counts", fontsize=plot_config["ylabel_fontsize"])
                        ax.tick_params(axis='both', which='major', labelsize=plot_config["tick_fontsize"])
                        ax.legend(loc='upper right', fontsize=plot_config["legend_fontsize"])

                    # Set the overall title of grid
                    fig.suptitle(
                        f"Histograms of Parameter {data_label}- Using sWeights - Sample Size {sample_size}",
                        fontsize=plot_config["suptitle_fontsize"])
                    plt.tight_layout(rect=[0, 0, 1, 0.96])

                    # Save the plot
                    save_path = f"{sub_dir}/{data_label}_histograms_{sample_size}.png"
                    plt.savefig(save_path)
                    plt.close(fig)  

                    print(f"Histogram of {data_label} saved in {sub_dir}")



        # Create Bias and Error against sample size summary plots
        # Sort the sample sizes so the plots are in order
        sample_sizes = sorted(results.keys())
        num_params = len(param_labels)

        # Plot Bias vs. Sample Size
        fig, axes = plt.subplots(3, 2, figsize=(13, 9))
        axes = axes.flatten()
        
        # Plot for all parameters bar N as this is the expected number of events and not comparible
        for i in [0,1,2,3,4,6]:
            # Determine Absolute Bias for each parameter
            biases = [abs(results[sample_size]["Values_Bias"][i]) for sample_size in sample_sizes]
            if i != 6:
                ax = axes[i]
            else:
                ax = axes[5]
            ax.plot(sample_sizes, biases, marker='o', label='Bias')
            ax.set_xlabel("Sample Size", fontsize=plot_config["xlabel_fontsize"])
            ax.set_ylabel(f"Abs(Bias) in {list(param_labels.values())[i]}", fontsize=plot_config["ylabel_fontsize"])
            ax.tick_params(axis='both', which='major', labelsize=plot_config["tick_fontsize"])
            ax.grid(True)
            # ax.set_xscale('log')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save Plot
        plt.savefig(f"{output_directory}/Trends_with_Samples_Size/Bias_vs_Sample_Size.png")
        plt.show() 
        plt.close(fig)
        print(f"Bias vs Sample Size plot saved in {output_directory}/Trends_with_Samples_Size")

        # Plot Values_Std vs. Sample Size
        fig, axes = plt.subplots(3, 2, figsize=(13, 9))
        axes = axes.flatten()

        # Plot for all parameters bar N as this is the expected number of events and not comparible
        for i in [0,1,2,3,4,6]: 
            # Determine the uncertainty for each parameter value
            errors_mean = [results[sample_size]["Values_Std"][i] for sample_size in sample_sizes]
            if i != 6:
                ax = axes[i]
            else:
                ax = axes[5]
            ax.plot(sample_sizes, errors_mean, marker='o', label='Error Mean')
            ax.set_xlabel("Sample Size", fontsize=plot_config["xlabel_fontsize"])
            ax.set_ylabel(f"Uncertainty in {list(param_labels.values())[i]}", fontsize=plot_config["ylabel_fontsize"])
            ax.tick_params(axis='both', which='major', labelsize=plot_config["tick_fontsize"])
            ax.grid(True)
            # ax.set_xscale('log')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save Plot
        plt.savefig(f"{output_directory}/Trends_with_Samples_Size/Errors_Mean_vs_Sample_Size.png")
        plt.show()  # Display in Jupyter Notebook
        plt.close(fig)
        print(f"Error vs Sample Size plot saved in {output_directory}/Trends_with_Samples_Size")

        # Plot the Pull Distribution for each parameter - with a seperate plot for each sample size
        for sample_size, calc_values in results.items():
            num_params = len(param_labels)
            fig, ax = plt.subplots(figsize=(10, 6))

            # Add light grey shading for Pull between 1 and 2
            ax.axvspan(-2, -1, color="lightgrey", alpha=0.4)
            ax.axvspan(1, 2, color="lightgrey", alpha=0.4)

            # Loop over parameters and plot horizontal bars
            for i, (param, label) in enumerate(param_labels.items()):
                pull_mean = calc_values["Pull_Mean"][i]
                pull_std = calc_values["Pull_Std"][i]
                pull_mean_error = calc_values["Pull_Mean_Error"][i]
                pull_std_error = calc_values["Pull_Std_Error"][i]

                # Light blue bar: between the centers of the two red bars
                ax.barh(
                    y=i,
                    width=2 * pull_std,
                    left=pull_mean - pull_std,
                    height=0.4,
                    color="lightblue",
                    alpha=plot_config["bar_alpha"],
                    edgecolor="black",
                )

                # Dark blue region: centered at Pull Mean, between Pull Mean +/- Pull Mean Error
                ax.barh(
                    y=i,
                    width=pull_mean_error * 2,
                    left=pull_mean - pull_mean_error,
                    height=0.4,
                    color="darkblue",
                    alpha=plot_config["bar_alpha"],
                    edgecolor="black",
                )

                # Red regions: centered at Pull Mean ± Pull Std, with widths defined by Pull Std Error
                ax.barh(
                    y=i,
                    width=pull_std_error * 2,
                    left=pull_mean - pull_std - pull_std_error,
                    height=0.4,
                    color="red",
                    alpha=plot_config["bar_alpha"],
                    edgecolor="black",
                )
                ax.barh(
                    y=i,
                    width=pull_std_error * 2,
                    left=pull_mean + pull_std - pull_std_error,
                    height=0.4,
                    color="red",
                    alpha=plot_config["bar_alpha"],
                    edgecolor="black",
                )

            # Add labels and grid
            ax.set_yticks(range(num_params))
            ax.set_yticklabels(list(param_labels.values()), fontsize=plot_config["ylabel_fontsize"])
            ax.set_xlabel("Pull", fontsize=plot_config["xlabel_fontsize"])
            ax.set_title(f"Pull Distributions for Sample Size {sample_size}", fontsize=plot_config["title_fontsize"])
            ax.tick_params(axis="both", which="major", labelsize=plot_config["tick_fontsize"])
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axvline(-2, color="brown", linestyle="dashdot", linewidth=1)
            ax.axvline(2, color="brown", linestyle="dashdot", linewidth=1)
            ax.grid(True, axis="x")

            # Save the plot
            save_path = os.path.join(output_directory, f"Pull_Plots/Pull_Distributions_{sample_size}.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)

            print(f"Saved pull distribution plot for sample size {sample_size} in {save_path}")

        # Return the results dictionary
        return results