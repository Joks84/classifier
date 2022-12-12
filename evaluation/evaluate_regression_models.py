from sklearn import linear_model
from matplotlib import pyplot

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

