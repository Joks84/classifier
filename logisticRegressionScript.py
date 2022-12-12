from process_data import preprocess_data
from train_data import regression_models
from evaluation import evaluate_regression_models
from sklearn import metrics, model_selection
import numpy as np
import matplotlib.pyplot as plt

# get the preprocessed images array
images_array = preprocess_data.preprocessTrainImages("process_data/photos/mix", 15)
# get the features and the labels
X, y = preprocess_data.createTrainDataLogisticRegression(images_array)
# split data to train and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# train the logistic regression model
model = regression_models.createLogicticRegressionModel(X_train, y_train)
# coefficient of determination
coef_determination = evaluate_regression_models.coefficientOfDetermination(model, X_test, y_test)
# get the predicted value
y_predict = model.predict(X_test)
# get the r2 score
r_squared = metrics.r2_score(y_test, y_predict)
# get the positive predicted probabilities
positive_probabilities = model.predict_proba(X_test)[:, 1]
# calculate area under the roc curve
roc_score = metrics.roc_auc_score(y_test, positive_probabilities)
fpr, tpr, _ = metrics.roc_curve(y_test, positive_probabilities)



# print results
print("The shape of data: ", np.asarray(X).shape)
print("Logistic regression coefficient of determination is: ", coef_determination)
print("R squared for logistic regression: ", r_squared)
print("Accuracy is: ", metrics.accuracy_score(y_test, y_predict))
print("Precision score is: ", metrics.precision_score(y_test, y_predict))
print("Recall score is: ", metrics.recall_score(y_test, y_predict))
print("Report: ", metrics.classification_report(y_test, y_predict))
print("Log loss: ", metrics.log_loss(y_test, model.predict_proba(X_test)))
print("Confusion matrix: ", metrics.confusion_matrix(y_test, y_predict))
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_predict)
# show roc curve
plt.plot(fpr, tpr, linestyle="--", label=roc_score)
plt.title("curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()