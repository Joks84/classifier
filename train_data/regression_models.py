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