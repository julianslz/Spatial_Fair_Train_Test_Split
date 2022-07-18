# Spatial fair train-test split

This repository contains an implemention of the algorithm sketched in the paper [Fair train-test split in machine learning: Mitigating spatial autocorrelation for improved prediction accuracy](https://www.sciencedirect.com/science/article/pii/S0920410521015023).
A reference implementation is found in the GitHub repo [julianslz/Spatial_Fair_Train_Test_Split](https://github.com/julianslz/Spatial_Fair_Train_Test_Split/tree/b457b9953212f707cd6ac712d79ab91444d61276).

## Comments

- The reference implementation re-created the covariance matrix each time the kriging equations were solved. This implementation requires the user to supply covariance matrices (once and for all). This is in my mind (1) more flexible as the covariance function is not defined in the Cython level and (2) faster since the covariance matrix is created only once.
- The reference implementation solves the kriging equations with a generic `np.linalg.solve` call. This implementation uses the Cholesky factorization to solve for kriging variances, which means that we only factor the covariance matrix once whens solving for variances on the "real world" dataset.
- The reference implementation does not exploit the leave-one-out structure when solving the kriging equations over the test/train dataset. This implementation uses the results from the paper "Cross Validation of Kriging in a Unique Neighborhood" by Olivier Dubrule to solve the leave-one-out system in O(n^3) time instead of O(n^4).
- The reference implementation uses rejection sampling. This implementation uses weighted sampling. This means no samples are ever rejected.
- The reference implementation uses Cython. This implementation does not, but around 95% of the time is spent in fast `np.linalg` calls.


## Speed

On dataset 1, the speed difference is :

```
# This implementation
%timeit splitter.draw_test_indices(test_size=0.05)
163 ms ± 7.71 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

# Reference implementation
%timeit splitter.draw_test_indices(test_size=0.05)
163 ms ± 7.71 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```
