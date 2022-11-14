from sklearn import linear_model
from sklearn import metrics
from matplotlib import pyplot
import numpy as np

def coefficientOfDetermination(model: linear_model.LogisticRegression or linear_model.LinearRegression, input: list, label: list) -> float:
    """
    Returns the coefficient of determination for logistic regression model.
    Params:
        model (linear_model.LogisticRegression): Logistic regression model to evaluate.
        input(list): The list of input variables for the model.
        label(list): The list of classes for each input variable.
    
    Returns:
        score(float): The coefficient of determination.
    """
    return model.score(input, label)


def getConfusionMatrix(model: linear_model.LogisticRegression, inputs: list, labels: list) -> np.ndarray:
    """
    Returns the confusion matrix.
    Params:
        model (linear_model.LogisticRegression): The logistic regression model.
        inputs (list): The list of inputs.
        labels (list): The list of labels for the inputs.

    Returns:
        confusion_matrix (np.ndarray): The confusion matrix.
    """

    # predict the labels based on existing inputs
    predicted_labels = model.predict(inputs)
    # return the confusion matrix
    return metrics.confusion_matrix(labels, predicted_labels)


def showPlot(y_axis: list, x_axis: list, y_axis_label: str, x_axis_label: str, label: str):
    """
    Shows the plot.
    Params:
        y_axis(list): The list of values shown on y axis.
        x_axis(list): The list of values shown on x axis.
        y_axis_label(str): The label for the y axis.
        x_axis_label(str): The label for the x axis.
        label(str): Label for the line.
    """
    pyplot.plot(y_axis, x_axis, marker=".", label=label)
    pyplot.xlabel(x_axis_label)
    pyplot.ylabel(y_axis_label)
    pyplot.legend()
    pyplot.show()


def getCrossEntropyLoss(model: linear_model.LogisticRegression, inputs: list, labels: list) -> float:
    """
    Returns cross entropy loss (error).
    Params:
        model(linear_model.LogisticRegression): The model for which cross entropy loss is examined.
        inputs(list): The list of input data.
        labels(list): The list of labels/outputs for input data.

    Return:
        loss(float): The value for the cross entropy loss.
    """
    return metrics.log_loss(labels, model.predict_proba(inputs))