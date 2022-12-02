from process_data import preprocess_data
from train_data import regression_models
from evaluation import evaluate_regression_models

# LINEAR REGRESSION
images_array = preprocess_data.preprocessTrainImages("process_data/photos", 0.57)

X_train, X_test, y_train, y_test = preprocess_data.createTrainTestDataLinearRegression(images_array)

linear_model = regression_models.createLinearRegressionModel(X_train, y_train)
linear_score = evaluate_regression_models.coefficientOfDetermination(linear_model, X_test, y_test)
print("The linear regression score is: ", linear_score)