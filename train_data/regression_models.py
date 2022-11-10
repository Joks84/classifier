from sklearn import linear_model


def createLogicticRegressionModel(input: list, label: list) -> linear_model.LogisticRegression:
    """
    Creates and returns the logistic trained regression model.
    Params:
        input(list): The list of input variables to train on.
        label(list): The list of classes for each input variable.

    Returns:
        model(linear_model.LogisticRegression): The trained logistic regression model.
    """
    logistic_regression = linear_model.LogisticRegression(solver="liblinear", max_iter=1000, random_state=0).fit(input, label)
    return logistic_regression


def createLinearRegressionModel(observations: list, labels: list) -> linear_model.LinearRegression:
    """
    Creates and returns the trained linear regression model.
    Params:
        observations(list): The list of observed data - inputs.
        labels(list): The list of label data - outputs.

    Returns:
        modellinear_model.LinearRegression): The trained linear regression model.
    """
    linear_regression = linear_model.LinearRegression().fit(observations, labels)
    return linear_regression