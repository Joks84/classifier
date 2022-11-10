from evaluation import evaluate_regression_models
from process_data import preprocess_data
from train_data import regression_models
from sklearn import metrics

# LOGISTIC REGRESSION
# get the images array
# images_array = preprocess_data.preprocessTrainImages("process_data/photos", 10)

# # get the inputs and the labels
# x = [item[:1001] for item in images_array]
# y = [item[-1] for item in images_array]

# # get the model
# model = regression_models.createLogicticRegressionModel(x, y)
# # coefficient of determination
# r_squared = evaluate_regression_models.coefficientOfDetermination(model, x, y)
# print("Logistic regression coefficient of determination is: ", r_squared)
# # confusion matrix
# confusion_matrix = evaluate_regression_models.getConfusionMatrix(model, x, y)
# print("The confusion matrix for logistic regression: ", confusion_matrix)
# # roc curve data
# false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y, model.predict(x))

# # plot the data
# evaluate_regression_models.showPlot(false_positive_rate, true_positive_rate, "False Positive Rate", "True Positive Rate", "Logistic")

# LINEAR REGRESSION
images_array = preprocess_data.preprocessTrainImages("process_data/photos", 0.57)

X_train, X_test, y_train, y_test = preprocess_data.createTrainTestData(images_array)

linear_model = regression_models.createLinearRegressionModel(X_train, y_train)
linear_score = evaluate_regression_models.coefficientOfDetermination(X_train, y_train)
print("The score is: ", linear_score)
