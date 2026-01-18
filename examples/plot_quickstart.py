"""
Quickstart example
====================================
Quick example of use of the library to optimize a decision tree classifier.
Firstly, we import the necessary libraries to get data and plot the results.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from mloptimizer.interfaces import HyperparameterSpaceBuilder, GeneticSearch

# %%
# Load the iris dataset to obtain a vector of features X and a vector of labels y.
# Another dataset or a custom one can be used
X, y = load_iris(return_X_y=True)

# %%
# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
# Define the HyperparameterSpace, you can use the default hyperparameters for the machine learning model
# that you want to optimize. In this case we use the default hyperparameters for a DecisionTreeClassifier.
# Another dataset or a custom one can be used
hyperparam_space = HyperparameterSpaceBuilder.get_default_space(estimator_class=DecisionTreeClassifier)

# %%
# The GeneticSearch class is the main wrapper for the optimization of a machine learning model.
# We configure the genetic algorithm parameters:
# - generations: Number of evolutionary iterations
# - population_size: Number of hyperparameter configurations per generation
# Note: These values are reduced for faster documentation builds.
# For production use, consider generations=30-50 and population_size=50-100.
genetic_params = {
    'generations': 10,
    'population_size': 20,
    'seed': 0
}

opt = GeneticSearch(
    estimator_class=DecisionTreeClassifier,
    hyperparam_space=hyperparam_space,
    cv=5,
    scoring='accuracy',
    **genetic_params
)

# %%
# To optimize the classifier we need to call the fit method.
# The method finds the best hyperparameters and stores them in ``best_estimator_``.
opt.fit(X, y)

print(opt.best_estimator_)

# %%
# Train the classifier with the best hyperparameters found
# Show the classification report and the confusion matrix
from sklearn.metrics import classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = opt.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=opt.best_estimator_.classes_,
    cmap=plt.cm.Blues
)
disp.plot()
plt.show()

del opt
