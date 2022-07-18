# -*- coding: utf-8 -*-
"""
An implementation of the algorithm described in the paper
   "Fair train-test split in machine learning: 
    Mitigating spatial autocorrelation for improved prediction accuracy"

"""

import numpy as np
from sklearn.utils.validation import check_random_state
from scipy.linalg import cho_factor, cho_solve, inv


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
    (index, function(indicies)[index]): tuple
        Index of a chosen data point in the data set, as well as dataset value
        at that interation.

    Examples
    --------
    >>> import numpy as np
    >>> target_data = np.random.beta(3, 2, 1000)
    >>> samples_data = np.random.beta(1.2, 1.3, 1000)
    >>> indices = np.arange(len(samples_data))
    >>> def function(indices):
    ...     return samples_data[indices]
    >>> chosen_indices = list(i for (i, _) in sample_to_match(indices, function, target_data, 250))
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

        # Verify loop invariant
        assert sample + len(remaining_indices) == initital_dataset_size

        # If the indices are e.g. [4, 6, 8, 11], and index 8 was chosen,
        # then we want to return the variance of dataset[2], since 8 is at
        # index 2 in `remaining_indices`
        dataset_idx = np.where(remaining_indices == index)
        yield index, dataset[dataset_idx]

        remaining_indices = remaining_indices[remaining_indices != index]


def simple_kriging_variances(C, rhs):
    """Given matrix of covariances C with shape (n, n), and a matrix of right
    hand side (rhs) vectors of shape (n, k), solve the simple kriging system
    of equations for each k.


    Parameters
    ----------
    C : np.ndarray
        A symmetric, positive definite covariance matrix of shape (n, n).
    rhs : np.ndarray
        A matrix of right hand sides of shape (n, k).

    Returns
    -------
    variances : np.ndarray
        The k'th entry is the kriging variance for point rhs[:, k].

    """
    assert C.shape[0] == C.shape[1] == rhs.shape[0]

    # The number of test points
    num_test_points = rhs.shape[1]

    # Factor the symmetric positive definite covariance matrix
    # Note that while the simple kriging matrix is positive symmetric definite
    # and may be solved by Cholesky, the same is not true for the ordinary
    # kriging matrix. If we want to solve that sytem, we should use LU instead
    L, low = cho_factor(C)

    # Create a matrix of kriging variances
    # The kriging variance at point k is var(estimate_k - true_value_k)
    # and reflects the uncertainty of the value at the point.
    # See Chapter 3 in "Multivateiate Geostatistics" by Hans Wackernagel
    variances = np.empty(num_test_points)
    for k in range(num_test_points):

        # Solve for optimal weights that minimize estimation variance
        weights = cho_solve((L, low), rhs[:, k])
        # Compute the variance under optimal weights
        # Here we assume C(x_0, x_0) = C(x_k, x_k) for all k
        variances[k] = C[0, 0] - np.sum(weights * rhs[:, k])

    return variances


def _simple_kriging_variances_loo_naive(C):
    """DO NOT USE. Naive computation of LOO variance. Used for tests."""

    num_datapoints = C.shape[0]

    variances = np.empty(num_datapoints)
    for i in range(num_datapoints):

        # The i'th column in the right hand side
        rhs = C[:, i]

        # Delete i'th row and column
        C_minus_i = np.delete(C, i, axis=0)
        C_minus_i = np.delete(C_minus_i, i, axis=1)
        rhs = np.delete(rhs, i, axis=0)

        # Solve for weights that minimize variance, then for the variance
        weights = np.linalg.solve(C_minus_i, rhs)
        variances[i] = C[i, i] - np.sum(weights * rhs)

    return variances


def simple_kriging_variances_loo(C):
    """Compute leave-one-out variance at each data point.

    This is equation (8) in the 1983 paper by Olivier Dubrule titled
    "Cross Validation of Kriging in a Unique Neighborhood".


    Parameters
    ----------
    C : np.ndarray
        A symmetric, positive definite covariance matrix of shape (n, n).

    Returns
    -------
    variances : np.ndarray
        The i'th entry is the kriging variance for point i.

    """
    return 1 / np.diag(inv(C, check_finite=False))


class SpatialSplit:
    def __init__(self, covmat_dataset, covmat_realworld):
        """Spatial data set splitter. Attempts to create test data sets with
        the same kriging variance as the planned real-world usage.


        Parameters
        ----------
        covmat_dataset : np.ndarray
            A symmetric, positive definite covariance matrix. Entry (i, j) is
            given by C_ij = cov(x_i, x_j) and there are n data points, so the
            matrix has shape (n, n).
        covmat_realworld : np.ndarray
            A matrix of covariances between the observed dataset (n points) and
            the planned real world usage dataset (k points). Must have shape
            (n, k).

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.spatial.distance import cdist
        >>> np.random.seed(42)
        >>> training_vectors = np.random.randn(100, 2)
        >>> real_world_vectors = np.random.randn(50, 2)
        >>> cov_matrix = np.exp(-cdist(training_vectors, training_vectors))
        >>> cov_realworld_matrix = np.exp(-cdist(training_vectors, real_world_vectors))
        >>> splitter = SpatialSplit(cov_matrix, cov_realworld_matrix)
        >>> splitter.draw_test_indices(test_size=10, random_state=42)
        array([34, 96, 71, 56, 14, 16,  4, 87, 57, 68])
        """
        self.covmat_dataset = covmat_dataset
        self.covmat_realworld = covmat_realworld

        assert self.covmat_dataset.shape[0] == self.covmat_dataset.shape[0]
        assert self.covmat_dataset.shape[1] == self.covmat_realworld.shape[0]

        # Compute simple kriging variance for the planned real world use
        # of the model, conditioned on the data points in the data set
        self.rw_variances = simple_kriging_variances(covmat_dataset, covmat_realworld)
        assert len(self.rw_variances) == self.covmat_realworld.shape[1]

    def variance_dataset_loo(self):
        """Get the leave-one-out simple kriging dataset variance.

        Index i contains the simple kriging variance at data point i,
        based on a model trained on all other data points.


        Returns
        -------
        array
            Array of variances, one for each point in the dataset.

        """
        return simple_kriging_variances_loo(self.covmat_dataset)

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

    def draw_test_indices(
        self, test_size=0.25, random_state=None, return_variance=False
    ):
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
         return_variance : bool
           If True, will return a tuple of arrays (indices, variances), where
           variances[i] is the variance of data point i evaluated on the
           iteration it was included in the test set.

        Returns
        -------
        list
            List of indices that are selected as test data.

        """
        num_datapoints = self.covmat_dataset.shape[0]

        # If the train size is a fraction, scale it
        if 0 < test_size < 1:
            n_samples = int(num_datapoints * test_size)
        else:
            n_samples = test_size

        indices = np.arange(num_datapoints)

        def eval_kriging_var(indices):
            # Around 98% of the time in the entire algorithm is spent
            # inverting matrices inside 'simple_kriging_variances_loo'
            covariance_mat = self.covmat_dataset[indices, :][:, indices]
            return simple_kriging_variances_loo(covariance_mat)

        samples = list(
            sample_to_match(
                indices,
                eval_kriging_var,
                self.rw_variances,
                n_samples,
                random_state=random_state,
            )
        )

        # The indices chosen by sampling, and the variances at each iteration
        inds = [index for (index, var) in samples]
        variances = [var for (index, var) in samples]

        if return_variance:
            return np.array(inds, dtype=int), np.array(variances)
        return np.array(inds, dtype=int)


# =============================================================================
# ============================== TESTS ========================================
# =============================================================================
import pytest

@pytest.mark.parametrize(
    "seed",
   list(range(100))
)
def test_leave_one_out(seed):
    """Test fast leave one out cross validation."""

    np.random.seed(seed)
    n = np.random.randint(3, 10)
    # Create PSD matrix
    F = np.random.randn(n, n)
    C = F.T @ F
    variances_fast = simple_kriging_variances_loo(C)
    variances_naive = _simple_kriging_variances_loo_naive(C)
    assert np.allclose(variances_fast, variances_naive)

@pytest.mark.parametrize(
    "seed",
   list(range(100))
)
def test_indices_in_range(seed):

    import numpy as np
    from scipy.spatial.distance import cdist

    np.random.seed(seed)

    train_size = np.random.randint(10, 100)
    test_size = np.random.randint(1, 10)

    training_vectors = np.random.randn(train_size, 2)
    real_world_vectors = np.random.randn(50, 2)
    cov_matrix = np.exp(-cdist(training_vectors, training_vectors))
    cov_realworld_matrix = np.exp(-cdist(training_vectors, real_world_vectors))
    splitter = SpatialSplit(cov_matrix, cov_realworld_matrix)
    indices = splitter.draw_test_indices(test_size=test_size)
    assert all( 0 <= i < train_size for i in indices)
    assert len(indices) == test_size
        
        
if __name__ == "__main__":


    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
