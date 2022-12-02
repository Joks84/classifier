from evaluation import evaluate_regression_models
from process_data import preprocess_data
from train_data import regression_models
from sklearn import metrics, model_selection

# get the images array
images_array = preprocess_data.preprocessTrainImages("process_data/photos", 15)

# get the inputs and the labels
X, y = preprocess_data.createTrainDataLogisticRegression(images_array)
# split data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# train the model
model = regression_models.createLogicticRegressionModel(X_train, y_train)
# coefficient of determination
r_squared = evaluate_regression_models.coefficientOfDetermination(model, X_test, y_test)
print("Logistic regression coefficient of determination is: ", r_squared)
# confusion matrix
confusion_matrix = evaluate_regression_models.getConfusionMatrix(model, X_test, y_test)
print("The confusion matrix for logistic regression: ", confusion_matrix)
# precision and recall
y_predicted = model.predict(X_test)
print("Precision score is: ", metrics.precision_score(y_test, y_predicted))
print("Recall score is: ", metrics.recall_score(y_test, y_predicted))
print(metrics.classification_report(y_test, y_predicted))
# evaluate unseen image
# print("Cat: ", preprocess_data.evaluateUnseenImage("test_data/test_images/cat.jpg", model, (100, 100)))
print("Cross entropy loss for trained data: ", evaluate_regression_models.getCrossEntropyLoss(model, X_train, y_train))
# roc curve data
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, model.predict(X_test))
# plot ROC curve
# evaluate_regression_models.showPlot(false_positive_rate, true_positive_rate, "False Positive Rate", "True Positive Rate", "Logistic")

