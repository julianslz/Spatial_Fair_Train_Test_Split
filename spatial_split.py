# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:44:15 2022

@author: TODL
"""

import numpy as np

# from cython_kriging import covariance
import cython_kriging


def sample_to_match(indices, function, target_data, num_samples):
    """Sample `num_samples` from a dataset given by `function(indices)` without
    replacement, so that the samples match the distribution of `target_data`.

    The idea to use the target data to create a PDF p(x),
    here using a histogram (a kernel density estimate could also be used).
    Then we sample from the dataset using p(x_i) as a weight for data point
    x_i. We perform many samples without replacement in turn.

    Parameters
    ----------
    indices : container of integers
        A list or numpy array of integers, such as dataset = function(indices).
        The purpose of using indices and a callable, instead of just a dataset,
        is to be able to update the dataset as elements are removed from it.
    function : callable
        A callable function such that dataset = function(indices).
    target_data : container
        A dataset whose distribution the sampling will attempt to match.
    num_samples : integer
        Number of samples to draw from the data set.

    Yields
    -------
    index : integer
        Index of a chosen data point in the data set.

    Examples
    --------
    >>> import numpy as np
    >>> target_data = np.random.beta(3, 2, 1000)
    >>> samples_data = np.random.beta(1.2, 1.3, 1000)
    >>> indices = np.arange(len(samples_data))
    >>> def function(indices):
    ...     return samples_data[indices]
    >>> chosen_indices = list(sample_to_match(indices, function, target_data, 250))
    >>> chosen_data = [samples_data[i] for i in chosen_indices]


    """

    # Bin the target data to create a target distribution
    hist, edges = np.histogram(target_data, bins="fd")

    # Update the dataset
    remaining_indices = np.array(indices, dtype=int)
    initital_dataset_size = len(indices)  # Used for verify loop invariant

    # Append and prepend a 0. If the a dataset point fall outside of the bins
    # of the target distribution, we assign zero weight to it
    hist_modified_zeros = np.zeros(len(hist) + 2)
    hist_modified_zeros[1:-1] = hist

    # Loop over each desired sample
    for sample in range(num_samples):

        # Update the dataset using the remaining indices
        dataset = function(remaining_indices)
        assert len(remaining_indices) == len(dataset)

        # Map the dataset to the bins defined by the target distribution
        # If 0 is returned, the sample is to the left of the leftmost bin
        # If len(bins) is returned, the sample is to the right of the rightmost bin
        dataset_bin_idx = np.digitize(dataset, bins=edges, right=False)

        # Map each sample in the dataset to it's probability (bin-height)
        # in the target distribution
        probability = hist_modified_zeros[dataset_bin_idx]

        # The sum is zero if there is no overlap in the distributions,
        # in which case we sample randomly from the dataset
        # We could prefer sampling more in the direction of the real-world
        # dataset on the real line, but it's hard to say how much
        if np.sum(probability) == 0:
            probability = np.ones(len(hist) + 2)

        assert np.sum(probability) > 0

        # Normalize the weights, needed for np.random.choice
        probability = probability / np.sum(probability)

        # Draw a weighted sample, update the remaining indices and
        index = np.random.choice(remaining_indices, p=probability, size=1)[0]
        remaining_indices = remaining_indices[remaining_indices != index]

        # Verify loop invariant
        assert sample + 1 + len(remaining_indices) == initital_dataset_size
        yield index


class SpatialSplit:

    verbose = True

    def __init__(self, x_dataset, y_dataset, x_realworld, y_realworld):
        """


        Parameters
        ----------
        x_dataset : TYPE
            DESCRIPTION.
        y_dataset : TYPE
            DESCRIPTION.
        x_realworld : TYPE
            DESCRIPTION.
        y_realworld : TYPE
            DESCRIPTION.
        covariance_function : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.x_dataset = np.array(x_dataset)
        self.y_dataset = np.array(y_dataset)
        assert len(self.x_dataset) == len(self.y_dataset)

        self.x_realworld = np.array(x_realworld)
        self.y_realworld = np.array(y_realworld)
        assert len(self.x_realworld) == len(self.y_realworld)

        # Step 1: Compute simple kriging variance for the planned real world use
        # of the model, conditioned on the data points in the data set
        self.rw_variances = cython_kriging.get_test_set_variances(
            x_dataset, y_dataset, x_realworld, y_realworld
        )
        assert len(self.rw_variances) == len(self.x_realworld)

        # Step 2: Bin the real world variances
        # self.rw_variance_hist, self.rw_variance_bin_edges = np.histogram(self.rw_variances, bins="fd")

        # if self.verbose:
        #    print(f"Binned {len(self.rw_variances)} real world variances into {len(self.rw_variance_hist)} bins.")

    def draw_test_train_split(self, train_size=0.25, random_state=None):

        # If the train size is a fraction, scale it
        if 0 < train_size < 1:
            n_samples = int(len(self.x_dataset) * train_size)
        else:
            n_samples = train_size

        indices = np.arange(len(self.x_dataset))

        def eval_kriging_var(indices):
            return cython_kriging.get_variances_loo(
                self.x_dataset[indices], self.y_dataset[indices]
            )

        chosen_indices = list(
            sample_to_match(indices, eval_kriging_var, self.rw_variances, n_samples)
        )
        return chosen_indices, eval_kriging_var(indices)
