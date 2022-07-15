# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:44:15 2022

@author: TODL
"""

import numpy as np

# from cython_kriging import covariance
import cython_kriging
from sklearn.utils.validation import check_random_state


def sample_to_match(indices, function, target_data, num_samples, random_state=None):
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

    rng = check_random_state(random_state)

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
            probability = np.ones(len(probability))

        assert np.sum(probability) > 0

        # Normalize the weights, needed for np.random.choice
        probability = probability / np.sum(probability)

        # Draw a weighted sample, update the remaining indices and
        index = rng.choice(remaining_indices, p=probability, size=1)[0]
        remaining_indices = remaining_indices[remaining_indices != index]

        # Verify loop invariant
        assert sample + 1 + len(remaining_indices) == initital_dataset_size
        yield index


class SpatialSplit:
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


        Examples
        --------
        >>> import numpy as np
        >>> x_dataset = np.array([180., 780., 180., 330., 100.])
        >>> y_dataset = np.array([789., 429., 709., 229., 500.])
        >>> x_rw = np.array([460., 500., 540., 500., 680.])
        >>> y_rw =  np.array([469., 300., 129., 719.,  29.])
        >>> splitter = SpatialSplit(x_dataset, y_dataset, x_rw, y_rw)
        >>> indices = splitter.draw_test_indices(2, random_state=42)
        >>> indices
        [4, 3]

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

    def variance_dataset_loo(self):
        """Get the leave-one-out simple kriging dataset variance.

        Index i contains the simple kriging variance at data point i,
        based on a model trained on all other data points.


        Returns
        -------
        array
            Array of variances, one for each point in the dataset.

        """
        return cython_kriging.get_variances_loo(self.x_dataset, self.y_dataset)

    def variance_realworld(self):
        """Get the simple kriging realworld variance.

        Index i contains the simple kriging variance at real world data point i,
        based on a model trained on all dataset data (but no real world data).


        Returns
        -------
        array
            Array of variances, one for each point in the dataset.

        """
        return self.rw_variances

    def draw_test_indices(self, test_size=0.25, random_state=None):
        """Draw indices in the dataset to be used for testing.

        Parameters
        ----------
        test_size : number, optional
           If float, should be between 0.0 and 1.0 and represent the proportion
           of the dataset to include in the test split. If int, represents the
           absolute number of test samples.
        random_state : int, RandomState instance or None, default=None
           Controls the random state of the sampling.
           Pass an int for reproducible output across multiple function calls.

        Returns
        -------
        list
            List of indices that are selected as test data.

        """

        # If the train size is a fraction, scale it
        if 0 < test_size < 1:
            n_samples = int(len(self.x_dataset) * test_size)
        else:
            n_samples = test_size

        indices = np.arange(len(self.x_dataset))

        def eval_kriging_var(indices):
            return cython_kriging.get_variances_loo(
                self.x_dataset[indices], self.y_dataset[indices]
            )

        chosen_indices = list(
            sample_to_match(
                indices,
                eval_kriging_var,
                self.rw_variances,
                n_samples,
                random_state=random_state,
            )
        )
        return chosen_indices
