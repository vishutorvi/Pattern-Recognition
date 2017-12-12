# Classification Using Linear Discriminant Functions and Boosting Algorithms

# Purpose: To know how to learn two-class/multi-class linear discriminant functions through perceptronlike algorithms and how to use boosting algorithms to create accurate classifiers using linear discriminant functions.

Implementing following Algorithms

1) Basic two-class classification using perceptron algorithms
	Abstractly, the problem is as follows. Given n labeled training samples, D={(x1,L1), (x2, L2), …,
	(xn, Ln)}, when Li is +1 / -1, implement Algorithm 4 (Fixed-Increment Single-Sample
	Perceptron Algorithm) and Algorithm 8 (Batch Relaxation with Margin) of Chapter 5 in the
	textbook.
2) Multi-class classification
	Use the basic two-class perceptron algorithms to solve multi-class classification problems by
	using the one-against-the-rest and one-against-the-other methods. Note that you need to handle
	ambiguous cases properly.
3) Adaboost to create strong classifiers
	Implement Algorithm 1 (AdaBoost) in Chapter 9 of the textbook to create a strong classifier
	using the above linear discriminant functions.