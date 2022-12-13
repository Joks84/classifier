from process_data import preprocess_data
from sklearn import model_selection
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from matplotlib import pyplot as plt

# images with multiple classes
# images_array = preprocess_data.createMulticlassData("process_data/photos/mix", 15)
# get images with 0 and 1 class
images_array = preprocess_data.preprocessTrainImages("process_data/photos/mix", 15)
# get the features and the labels
X, y = preprocess_data.createTrainDataLogisticRegression(images_array)
# split to train and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
# create K nearest neighbor model and fit the data
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)


y_predict = knn.predict(X_test)

print("Knn score: ", knn.score(X_test, y_test))

print("Confusion matrix: ", metrics.confusion_matrix(y_test, y_predict))

print("Classification report: ", metrics.classification_report(y_test, y_predict))

error_rate = []
accuracy = []
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    error_rate.append(np.mean(y_predict != y_test))
    accuracy.append(metrics.accuracy_score(y_test, y_predict))


plt.figure(figsize=(10, 6))
plt.plot(range(1, 30), error_rate, color="blue", linestyle="dashed", marker="o", markerfacecolor="red", markersize=10)
plt.title("Error rate vs K value")
plt.xlabel("K value")
plt.ylabel("Error rate")
print("Minimum error: ", min(error_rate), " at K = ", (error_rate.index(min(error_rate)) + 1))
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, 30), accuracy, color="blue", linestyle="dashed", marker="o", markerfacecolor="red", markersize=10)
plt.title("Accuracy vs K value")
plt.xlabel("K value")
plt.ylabel("Accuracy")
print("Minimum accuracy: ", max(accuracy), " at K = ", (accuracy.index(max(accuracy)) + 1))
plt.show()






