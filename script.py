from evaluation import evaluate_regression_models
from process_data import preprocess_data
from train_data import regression_models
from sklearn import metrics

# LOGISTIC REGRESSION
# get the images array
images_array = preprocess_data.preprocessTrainImages("process_data/photos", 10)

# get the inputs and the labels
X, y = preprocess_data.createTrainDataLogisticRegression(images_array)

# get the model
model = regression_models.createLogicticRegressionModel(X, y)
# coefficient of determination
r_squared = evaluate_regression_models.coefficientOfDetermination(model, X, y)
print("Logistic regression coefficient of determination is: ", r_squared)
# confusion matrix
confusion_matrix = evaluate_regression_models.getConfusionMatrix(model, X, y)
print("The confusion matrix for logistic regression: ", confusion_matrix)
# precision and recall
y_predicted = model.predict(X)
precision_score = metrics.precision_score(y, y_predicted)
print("Precision score is: ", precision_score)
recall_score = metrics.recall_score(y, y_predicted)
print("Recall score is: ", recall_score)
# roc curve data
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y, model.predict(X))
# # plot ROC curve
evaluate_regression_models.showPlot(false_positive_rate, true_positive_rate, "False Positive Rate", "True Positive Rate", "Logistic")

# LINEAR REGRESSION
# images_array = preprocess_data.preprocessTrainImages("process_data/photos", 0.57)

# X_train, X_test, y_train, y_test = preprocess_data.createTrainTestDataLinearRegression(images_array)

# linear_model = regression_models.createLinearRegressionModel(X_train, y_train)
# linear_score = evaluate_regression_models.coefficientOfDetermination(X_train, y_train)
# print("The linear regression score is: ", linear_score)
