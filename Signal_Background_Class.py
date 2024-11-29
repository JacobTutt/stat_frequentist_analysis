import os
import numpy as np  
from Background_Class import Background
from Signal_Class import Signal
from scipy.optimize import minimize
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL
from tabulate import tabulate
import pandas as pd
from tqdm import tqdm

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
        
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.m = m
        self.lamb = lamb
        self.mu_b = mu_b
        self.sigma_b = sigma_b
        self.lower_bound_X = lower_bound_X
        self.upper_bound_X = upper_bound_X
        self.lower_bound_Y = lower_bound_Y
        self.upper_bound_Y = upper_bound_Y
        self.samples = None

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
        Finds the maximum value of the joint PDF by first using a rough grid search and the a specialised local optimization approach.
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
            # have to use constant `lambda` functions for the y limits of integration due to format of dblquad
            integral_bounds, error_bounds = dblquad(lambda y, x: self.pdf(x, y), lower_bound_X, upper_bound_X, lambda x: lower_bound_Y, lambda x: upper_bound_Y)
            print(f"Integral: {integral_bounds} \u00B1 {error_bounds}")

        print(f"Normalisation over the whole real plane: X in [-infinity, infinity], Y in [-infinity, infinity]")
        integral_inf, error_inf = dblquad(lambda y, x: self.pdf(x, y), -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
        print(f"Integral: {integral_inf} \u00B1 {error_inf}")
    

    def plot_dist(self):
        """
        3D plots of the joint PDF.
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
        float or np.ndarray
            The normalized marginal PDF value(s)
        """
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
        float or np.ndarray
            The marginal CDF value(s)
        """
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
        float or np.ndarray
            The normalized marginal PDF value(s)
        """
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
        float or np.ndarray
            The marginal CDF value(s)
        """
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
            Y = np.linspace(self.lower_bound_Y, self.lower_bound_Y + 6*self.sigma_b, 100)
                            
        elif self.lower_bound_Y is None and self.upper_bound_Y is not None:
            Y = np.linspace(self.upper_bound_Y - 6*self.sigma_b, self.upper_bound_Y, 100)

        else:
            Y = np.linspace(self.mu_b - 3*self.sigma_b, self.mu_b + 3*self.sigma_b, 1000)

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Top left: marginal_pdf_x
        signal_pdf_x, background_pdf_x, total_pdf_x = self.marginal_pdf_x(X)
        axs[0, 0].plot(X, signal_pdf_x, label='Signal PDF: Crystal Ball', color='green')
        axs[0, 0].plot(X, background_pdf_x, label='Background PDF: Uniform', color='blue')
        axs[0, 0].plot(X, total_pdf_x, label='Total PDF', color='red')
        axs[0, 0].set_xlabel('X', fontsize=14)
        axs[0, 0].set_ylabel(f'Marginal PDF in X, f={self.f}', fontsize=14)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=12)
        axs[0, 0].legend(fontsize=14)

        # Top right: marginal_cdf_x
        signal_cdf_x, background_cdf_x, total_cdf_x = self.marginal_cdf_x(X)
        axs[0, 1].plot(X, signal_cdf_x, label='Signal CDF: Crytsal Ball', color='green')
        axs[0, 1].plot(X, background_cdf_x, label='Background CDF: Uniform', color='blue')
        axs[0, 1].plot(X, total_cdf_x, label='Total CDF', color='red')
        axs[0, 1].set_xlabel('X', fontsize=14)
        axs[0, 1].set_ylabel(f'Marginal CDF in X, f={self.f}', fontsize=14)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=12)
        axs[0, 1].legend(fontsize=14)

        # Bottom left: marginal_pdf_y
        signal_pdf_y, background_pdf_y, total_pdf_y = self.marginal_pdf_y(Y)
        axs[1, 0].plot(Y, signal_pdf_y, label='Signal PDF: Exponential', color='green')
        axs[1, 0].plot(Y, background_pdf_y, label='Background PDF: Normal', color='blue')
        axs[1, 0].plot(Y, total_pdf_y, label='Total PDF', color='red')
        axs[1, 0].set_xlabel('Y', fontsize=14)
        axs[1, 0].set_ylabel(f'Marginal PDF in Y, f={self.f}', fontsize=14)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=12)
        axs[1, 0].legend(fontsize=14)

        # Bottom right: marginal_cdf_y
        signal_cdf_y, background_cdf_y, total_cdf_y = self.marginal_cdf_y(Y)
        axs[1, 1].plot(Y, signal_cdf_y, label='Signal CDF: Exponential', color='green')
        axs[1, 1].plot(Y, background_cdf_y, label='Background CDF: Normal', color='blue')
        axs[1, 1].plot(Y, total_cdf_y, label='Total CDF', color='red')
        axs[1, 1].set_xlabel('Y', fontsize=14)
        axs[1, 1].set_ylabel(f'Marginal CDF in Y, f={self.f}', fontsize=14)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=12)
        axs[1, 1].legend(fontsize=14)

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
        print(f"The initial batch acceptance rate is: {acceptance_rate}")


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
            
            print(f"Accepted {num_accepted} out of {batch_size} samples in batch, total: {sample_count}")

        if save_to_class:
            # Store to the class 
            self.samples = samples

        return samples
    
    def plot_samples(self):
        """
        Plot the results of the sampled data in a 2x2 grid:
        - Top-left: 3D histogram of the joint distribution
        - Top-right: Surface plot of the joint PDF
        - Bottom-left: Histogram of sampled X vs marginal PDF
        - Bottom-right: Histogram of sampled Y vs marginal PDF
        """
        if self.samples is None:
            raise ValueError("No samples have been generated. Please run the `accept_reject_sample` method first.")
        
        # Define ranges for X and Y based on bounds
        X = np.linspace(self.lower_bound_X - 0.1*(self.upper_bound_X - self.lower_bound_X), self.upper_bound_X+ 0.1*(self.upper_bound_X - self.lower_bound_X), 1000)
        Y = np.linspace(self.lower_bound_Y- 0.1*(self.upper_bound_Y - self.lower_bound_Y), self.upper_bound_Y+ 0.1*(self.upper_bound_Y - self.lower_bound_Y), 1000)

        # Compute the PDF grid for plotting
        X_grid, Y_grid = np.meshgrid(X, Y)
        Z = self.pdf(X_grid, Y_grid)

        # Import samples from the class
        samples = self.samples

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
        hist_X, bins_X, _ = ax3.hist(samples[:, 0], bins=30, density=True, alpha=0.5, color='navy', label="Sampled X")
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
        hist_Y, bins_Y, _ = ax4.hist(samples[:, 1], bins=30, density=True, alpha=0.5, color='navy', label="Sampled Y")
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
        return f * self.Signal.pdf_fitting(X, Y, mu, sigma, beta, m, lamb) + (1 - f) * self.Background.pdf_fitting(X, Y, mu_b, sigma_b)
    
    
    def fit_params(self, initial_params, samples = None, print_results = False, save_to_class = False):
        """
        Perform an extended maximum likelihood fit using `iminuit`.

        Parameters
        ----------
        data : np.ndarray
            Observed data points of shape (N, 2), where each row is (x, y).
        initial_params : list
            Initial guesses for the parameters [mu, sigma, beta, m, lamb, mu_b, sigma_b, f, N_expected].

        Returns
        -------
        Minuit
            The `Minuit` object containing the fit results.
        """

        # Allow for the samples to be passed in, if not try use the samples already generated in class
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

        # Create the Minuit object with initial guesses
        mi = Minuit(neg_log_likelihood, mu=initial_params[0],sigma=initial_params[1], beta=initial_params[2], 
                    m=initial_params[3], lamb=initial_params[4], mu_b=initial_params[5], sigma_b=initial_params[6],
                    f=initial_params[7], N=initial_params[8])

        # Set parameter limits based on each paramaters restrictions and physical significance
        # sigma > 0
        mi.limits["sigma"] = (1e-3, None)
        # beta > 0
        mi.limits["beta"] = (1e-3, None)
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
            raise RuntimeError("Minimization did not converge")

        # Run the error analysis
        mi.hesse()
        if save_to_class:
            # Save to class
            self.mi = mi
        if print_results:
            print(mi)
        return mi.values, mi.errors
    

    def fit_params_results(self):
        """
        Print the results of the fit along with the true values and how many standard errors
        the fitted values are away from the true values.

        Parameters
        ----------
        mi : Minuit
            The Minuit object containing the fit results.
        """

        # check data has actually been fitted to and results are available
        if not hasattr(self, "mi"):
            raise ValueError("Minuit object not available. Please run fit_params first.")

        # Define parameter parent distributions for the table
        # stored in the list format (parent distribution(group), name, fitted value, error, true value)=
        params_table = [
            # Crystal Ball parameters
            ("Crystal Ball (Signal)", "mu", self.mi.values["mu"], self.mi.errors["mu"], self.mu),
            ("Crystal Ball (Signal)", "sigma", self.mi.values["sigma"], self.mi.errors["sigma"], self.sigma),
            ("Crystal Ball (Signal)", "beta", self.mi.values["beta"], self.mi.errors["beta"], self.beta),
            ("Crystal Ball (Signal)", "m", self.mi.values["m"], self.mi.errors["m"], self.m),
            # Exponential Decay parameter
            ("Exponential (Signal)", "lamb", self.mi.values["lamb"], self.mi.errors["lamb"], self.lamb),
            # Normal parameters
            ("Normal (Background)", "mu_b", self.mi.values["mu_b"], self.mi.errors["mu_b"], self.mu_b),
            ("Normal (Background)", "sigma_b", self.mi.values["sigma_b"], self.mi.errors["sigma_b"], self.sigma_b),
            # Overall parameters
            ("Overall", "f", self.mi.values["f"], self.mi.errors["f"], self.f),
            ("Overall", "N", self.mi.values["N"], self.mi.errors["N"], None), 
        ]

        # Create a dictionary to store the fitting results within the class simultaneously
        self.fit_results = {}


        # Add columns for "Value ± Error" and "Number of Standard Errors Away"
        formatted_table = []
        for group, name, value, error, true_val in params_table:

            # Save the results to the overall class dictionary
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

        # Declare the table's headers
        headers = ["Distribution", "Parameter", "Value ± Error", "True Value", "Std Errors Away"]

        # Print the table
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

        # Add labels
        ax.set_xticks(range(len(parameters)))
        ax.set_yticks(range(len(parameters)))
        ax.set_xticklabels(parameters, fontsize=14, rotation=45) 
        ax.set_yticklabels(parameters, fontsize=14)

        # Display values in the cells of the coloured heat map
        for (i, j), val in np.ndenumerate(correlation_matrix):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=12)

        plt.tight_layout()
        plt.show()

 
    # def plot_profiled_likelihoods(self):
    #     """
    #     Plot the profiled log-likelihood for each parameter in a 3x3 grid.
    #     The x-axis is scanned over the fitted parameter value ± 2.5σ, and the y-axis is the -2 log-likelihood.
    #     Store the 1σ confidence intervals (lhs and rhs deviations) for each parameter in the class.
    #     """

    #     if not hasattr(self, "mi"):
    #         raise ValueError("Fit has not been performed. Run fit_params() first.")

    #     # Initialize storage for profiled errors
    #     self.profiled_errors = {}

    #     # Extract parameter values, errors, and names
    #     params = self.mi.parameters
    #     values = self.mi.values
    #     errors = self.mi.errors

    #     # Define LaTeX labels for each parameter
    #     param_labels = {
    #         "mu": r"$\mu$",
    #         "sigma": r"$\sigma$",
    #         "beta": r"$\beta$",
    #         "m": r"$m$",
    #         "lamb": r"$\lambda$",
    #         "mu_b": r"$\mu_b$",
    #         "sigma_b": r"$\sigma_b$",
    #         "f": r"$f$",
    #         "N": r"$N$"
    #     }

    #     # Create a 3x3 grid for plotting
    #     fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    #     axes = axes.flatten()

    #     # Loop over all parameters
    #     for i, (param, value, error) in enumerate(zip(params, values, errors)):
    #         ax = axes[i]

    #         # Define the range to scan: ±2.5σ
    #         scan_range = np.linspace(value - 2.5 * error, value + 2.5 * error, 25)
    #         nlls = []

    #         # Compute the profiled likelihood
    #         for scan_value in scan_range:
    #             self.mi.values[param] = scan_value
    #             self.mi.fixed[param] = True  # Fix the parameter being profiled
    #             self.mi.migrad()  # Run minimization
    #             nlls.append(self.mi.fval)

    #         # Reset parameter to be free
    #         self.mi.fixed[param] = False
    #         self.mi.values[param] = value

    #         # Shift NLL values so the minimum is 0
    #         nlls = np.array(nlls) - np.min(nlls)

    #         # Plot the likelihood
    #         ax.plot(scan_range, nlls, color='blue', label=r"$-2\ln\mathcal{L}$", linewidth=2)

    #         # Find 1σ (ΔlnL = 0.5) confidence intervals
    #         lhs_1sigma = scan_range[np.where(nlls <= 0.5)[0][0]]
    #         rhs_1sigma = scan_range[np.where(nlls <= 0.5)[0][-1]]

    #         # Store the deviations (distances) from the central value
    #         self.profiled_errors[param] = {
    #             "lhs_1sigma": lhs_1sigma - value,  # Difference on the left
    #             "rhs_1sigma": rhs_1sigma - value  # Difference on the right
    #         }

    #         # Plot shaded regions for 1σ and 2σ
    #         ax.fill_between(
    #             scan_range,
    #             0, 0.5,
    #             where=(nlls <= 0.5),
    #             color='green',
    #             alpha=0.3,
    #             label=r"$1\sigma$ interval"
    #         )
    #         ax.fill_between(
    #             scan_range,
    #             0, 2.0,
    #             where=(nlls <= 2.0),
    #             color='red',
    #             alpha=0.3,
    #             label=r"$2\sigma$ interval"
    #         )

    #         # Add horizontal lines for ΔlnL = 0.5 and 2
    #         ax.axhline(0.5, color='black', linestyle='--', linewidth=1)
    #         ax.axhline(2.0, color='black', linestyle='--', linewidth=1)

    #         # Add text for the horizontal lines
    #         ax.text(scan_range[0], 0.5, r"$\Delta\ln\mathcal{L} = 0.5$", fontsize=10, verticalalignment='bottom')
    #         ax.text(scan_range[0], 2.0, r"$\Delta\ln\mathcal{L} = 2.0$", fontsize=10, verticalalignment='bottom')

    #         # Set labels and font sizes
    #         ax.set_xlabel(param_labels.get(param, param), fontsize=14)
    #         ax.set_ylabel(r"$-2\ln\mathcal{L}$", fontsize=14)
    #         ax.tick_params(axis='both', which='major', labelsize=12)

    #         # Adjust x-ticks for better visibility (especially for "N")
    #         if param == "N":
    #             ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))
    #             for label in ax.get_xticklabels():
    #                 label.set_fontsize(10)

    #     # Remove unused subplots
    #     for j in range(len(params), len(axes)):
    #         fig.delaxes(axes[j])

    #     # Add a single legend to the first subplot
    #     axes[0].legend(fontsize=12)

    #     # Adjust layout
    #     plt.tight_layout()
    #     plt.show()

    def param_bootstrap_data(self, n_samples, n_iterations, initial_params):
        """
        Perform a parametric bootstrap analysis for multiple sample sizes.

        Parameters
        ----------
        n_samples : list
            Array of sample sizes (e.g., [250, 500, 1000, ...]).
        n_iterations : int
            Number of bootstrap iterations for each sample size.
        initial_params : list
            Initial guesses for the parameters [mu, sigma, beta, m, lamb, mu_b, sigma_b, f, N].

        Returns
        -------
        None
            Saves combined results (values and errors, including N) to `.npy` files for each sample size.
        """
        # The resultant data from this method is going to be stored in the `Bootstrap_Results` directory
        # Create the directory if it doesn't exist
        if not os.path.exists("Bootstrap_Results"):
            os.makedirs("Bootstrap_Results")
        # If the directory already exists, remove all files and subdirectories
        else:
            for filename in os.listdir("Bootstrap_Results"):
                file_path = os.path.join("Bootstrap_Results", filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  

        # Define the number of parameters (including N)
        num_params = len(initial_params) + 1

        for n in tqdm(n_samples, desc="Sample Sizes"):
            # Generate all toy datasets at once
            toys = [ self.accept_reject_sample(desired_samples=n, poisson=True, save_to_class=False) for _ in range(n_iterations) ]

            # Add the actual number of samples (N) as the last value in initial_params
            current_initial_params = initial_params.copy()
            current_initial_params.extend(n)

            # Allocate a 3D array for results: (n_iterations, 2, num_params)
            # Axis 0: n_iterations
            # Axis 1: 0 for values, 1 for errors
            # Axis 2: num_params [mu, sigma, beta, m, lamb, mu_b, sigma_b, f, N]
            results = np.empty((n_iterations, 2, num_params))  

            # Loop over all bootstrap toys 
            for i, toy in tqdm(enumerate(toys), desc=f"Bootstrap Iterations for n={n}", leave=False, total=n_iterations):
                # Fit parameters using `fit_params`
                fit_values, fit_errors = self.fit_params(initial_params=current_initial_params, samples=toy)
                # Store values in results[i, 0, :]
                results[i, 0, :] = list(fit_values.values())
                # Store errors in results[i, 1, :]
                results[i, 1, :] = list(fit_errors.values())

            # Save the combined results to a `.npy` file in 
            np.save(f"Samples/Bootstrap_results_{n}.npy", results)
            print(f"Saved combined results for n_samples={n} to .npy file.")
