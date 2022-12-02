from process_data import preprocess_data
from evaluation import evaluate_regression_models
from train_data import regression_models
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
from sklearn import metrics


# get the images array
images_array = preprocess_data.preprocessTrainImages("process_data/photos", 15)
# get the inputs and the labels
X, y = preprocess_data.createTrainDataLogisticRegression(images_array)

pca = PCA(0.95)
# create X pca
X_pca = pca.fit_transform(X)
# split the data
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y, test_size=0.2)
# create the model
pca_model = regression_models.createLogicticRegressionModel(X_pca_train, y_pca_train)
print("R Squared with PCA: ", evaluate_regression_models.coefficientOfDetermination(pca_model, X_pca_test, y_pca_test))
# confusion matrix
confusion_matrix = evaluate_regression_models.getConfusionMatrix(pca_model, X_pca_test, y_pca_test)
print("The confusion matrix for PCA logistic regression: ", confusion_matrix)

y_pca_predict = pca_model.predict(X_pca_test)
print("Precision with PCA score is: ", metrics.precision_score(y_pca_test, y_pca_predict))
print("Recall with PCA score is: ", metrics.recall_score(y_pca_test, y_pca_predict))
print(metrics.classification_report(y_pca_test, y_pca_predict))

print("Cross entropy loss with PCA: ", evaluate_regression_models.getCrossEntropyLoss(pca_model, X_pca_train, y_pca_train))
# roc curve data
pca_false_positive_rate, pca_true_positive_rate, pca_thresholds = metrics.roc_curve(y_pca_test, y_pca_predict)
# plot ROC curve
# evaluate_regression_models.showPlot(pca_false_positive_rate, pca_true_positive_rate, "False Positive Rate", "True Positive Rate", "Logistic with PCA")




