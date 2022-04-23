#Natural Computing - Assignment 5 - Question 5
#Sven Berberich, Thijs Schoppema en Gerhard van der Knijff

#Imports
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt


def runKNN():
    print("Running KNN.")
    #test K from 3 to 25
    accuracies = []
    for k in range(3,26):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    plt.plot(range(3,26), accuracies)
    plt.title("KNN on Wine dataset")
    plt.xlabel("Number of neighbours")
    plt.ylabel("Accuracy")
    plt.show()



def runAdaBoost():
    print("Running AdaBoost.")
    dtAccuracies = []
    lrAccuracies = []
    svcAccuracies = []
    for i in range(1,100):
        abcDT = AdaBoostClassifier(n_estimators=i)
        abcLR = AdaBoostClassifier(base_estimator=ExtraTreeClassifier(), n_estimators=i)
        abcSVC = AdaBoostClassifier(base_estimator=SVC(probability=True, kernel='linear'), n_estimators=i, algorithm='SAMME')
        modelDT = abcDT.fit(X_train, y_train)
        modelLR = abcLR.fit(X_train, y_train)
        modelSVC = abcSVC.fit(X_train, y_train)
        y_predDT = modelDT.predict(X_test)
        y_predLR = modelLR.predict(X_test)
        y_predSVC = modelSVC.predict(X_test)
        dtAccuracies.append(metrics.accuracy_score(y_test, y_predDT))
        lrAccuracies.append(metrics.accuracy_score(y_test, y_predLR))
        svcAccuracies.append(metrics.accuracy_score(y_test, y_predSVC))
    plt.plot(range(1,100), dtAccuracies, label="Decision Tree Classifier")
    plt.plot(range(1,100), lrAccuracies, label="Extra Tree Classifier")
    plt.plot(range(1,100), svcAccuracies, label="Support Vector Classifier")
    plt.title("AdaBoost on Wine dataset")
    plt.xlabel("Number of estimators")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

digits = load_wine()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=4)

runKNN()
runAdaBoost()
