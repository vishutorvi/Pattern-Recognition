# 1) Bayesian Classifier Based on Maximum Likelihood Estimation

Bayesian decision theory provides the optimal decision rule for classification when the
probabilities are known. In practice, however, these required probability models are rarely
available and are typically estimated based on the labeled training samples.
Since once we know the parametric forms, we apply Maximum likelihood estimation to
estimate the probability distribution.
For the below datasets we assume that it is gaussian distribution, so we use maximum
likelihood estimation to find the probability distribution.
We performed following task in our algorithm:.
1) We find the covariance and mean for each category.
2) Since the covariance matrix is not guaranteed to be invertible, we applied the
following formula:
Σ(β) = (1 − β)Σ + βI, and we chose β = 0.1 and obtained a better result.
3) Applied Bayesian classifier based on maximum likelihood estimation.
4) If we have any ties, we break it and assign the sample to the first in the list of
categories.
Parametric Form:
For all the dataset is Gaussian Distribution

We classify the sample to the category for which gi(x)> gj(x) for all j != i, where gi(x) is discriminant functions


# 2) Bayesian Classifier based on Parzen Window Estimation
We performed following task in our algorithm:.
1) We normalize our training and test data set to ensure centered means and unit
covariance
2) We perform leave one out on the training data, to find the best bandwidth parameter
(window width)
3) Applied the chosen window function and perform classification using a bayesian
classifier
4) Ties are broken by favoring earlier classes
Window function:
We assumed and applied Gaussian Kernel Window, which gave us high accuracy.

We classify the sample to the category for which gi(x)> gj(x) for all j != i, where gi(x) is discriminant functions


# 3) K Nearest Neighbor

We perform following task in our algorithm:
1) We normalize our training and test data set to ensure centered means and unit
covariance
2) We perform leave one out comparison to choose k value which gives minimum error
on the training data
3) We use manhattan distance metric
4) By finding k Nearest Neighbors, and choosing the class that maximizes the number
of neighbors
5) We break ties by favoring earlier classes
6) We find the nearest neighbors using the brute force and kd tree algorithms