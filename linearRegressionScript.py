from process_data import preprocess_data
from train_data import regression_models
from evaluation import evaluate_regression_models
from sklearn import model_selection
# LINEAR REGRESSION
# get the images array
images_array = preprocess_data.preprocessTrainImages("process_data/photos/plants", 0.57)
# get the features and the labels
X, y = preprocess_data.createTrainTestDataLinearRegression(images_array)
# split data to train and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

linear_model = regression_models.createLinearRegressionModel(X_train, y_train)
linear_score = evaluate_regression_models.coefficientOfDetermination(linear_model, X_test, y_test)
print("The linear regression score is: ", linear_score)