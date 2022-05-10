# Spatial fair train-test split

<p align="center">
    <img src="https://github.com/GeostatsGuy/GeostatsPy/blob/master/TCG_color_logo.png?raw=true" width="200" height="200" />
</p>

This repository contains the synthetic dataset and the Python and Cython codes to reproduce the results from the scientific publication **"Fair Train-Test Split in Machine Learning: Mitigating Spatial Autocorrelation and Ensuring Prediction Fairness."**

**Table of Contents**

# Executive summary
Neglecting the spatial correlation for subsurface applications in machine learning models could yield over-optimistic and unrealistic results. Furthermore, although some techniques have been suggested, the training from those techniques fails to replicate the difficulty of the final use of the model (the training may be excessively complicated or too easy).

For instance, take a look at Figure 2 from the paper::

<p>
    <img src="https://github.com/aerisjd/Spatial_Fair_Train_Test_Split/blob/main/Files/Figures/dataset_kmap.png?raw=true" width="800" height="500" />
</p>

On the left, we have the data available for training and the planned real-world use of the model (i.e., where we will apply our model to make predictions or inferences). We have the kriging variance on the right, and the larger the kriging variance, the more complex the prediction is.

The samples have spatial autocorrelation, and using the validation set approach ([James et al.](https://web.stanford.edu/~hastie/ISLRv2_website.pdf)) would result in training and tuning of the model that is too easy (Center of the following figure). On the other hand, using spatial cross-validation, extrapolation occurs, and we will train the model in a more complicated setup than the final use of the model (Right part of the following figure).

On the other hand, we propose a method to account for the spatial context and training fairness. This fairness mimics the difficulty in training the real application of the machine learning model, providing spatial-aware datasets for most problems. Moreover, our method imitates the prediction difficulty of the final use of the model in the training and tuning steps (this is confirmed on the left side of the following figure):

<p>
<img src="https://github.com/aerisjd/Spatial_Fair_Train_Test_Split/blob/main/Files/Figures/final_set.png?raw=true" width=“1000” height=“600” />
</p>

We use the simple kriging variance as a proxy of estimation difficulty. For example, in the following figure, we see that all the 100 realizations for our spatial fair train-test split replicate the estimation difficulty (red curve) of the final use of the model.

<p>
<img src="https://github.com/aerisjd/Spatial_Fair_Train_Test_Split/blob/main/Files/Figures/pdf_comparison.png?raw=true" width=“1000” height=“600” />
</p>

# Installation
Feel free to clone this repository in your local machine under the [MIT license](https://choosealicense.com/licenses/mit/).

For a better experience, create a new environment using the requirements.txt file.

Make sure you include the Code and Datasets folders as sources root.

## Cython compilation
After installing the [Cython](https://cython.org/) package, the code should run without problem because I have compiled the code for you. Otherwise, try to compile the cython_kriging_c.pyx file as:
1. Open the setup.py file
2. Open the terminal (make sure the terminal is linked to your environment)
3. Type cd[directory where your cython_kriging_c.pyx file is]
4. Run: python setup.py build_ext --inplace

The steps compile the cython code.

# Support
In case you require assistance, I am more than happy to help you at my [LinkedIn profile](https://www.linkedin.com/in/jsalazarneira/).

# Publication
You can find the publication [Fair train-test split in machine learning: Mitigating spatial autocorrelation for improved prediction accuracy](https://www.sciencedirect.com/science/article/pii/S0920410521015023) online.

# Authors and acknowledgment
I would like to thank my coauthors Lean Garland, Jesus Ochoa, and [Dr. Michael Pyrcz](https://github.com/GeostatsGuy) for their help on this project.

# License
[MIT](https://choosealicense.com/licenses/mit/)