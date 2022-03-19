import numpy as np
from sklearn import metrics
from matplotlib import pyplot

anomalyScores = []
classes = []

with open("output.txt") as english_file:
    englishScores = english_file.readlines()
    for score in englishScores:
        if not score.isspace():
            classes.append(0)
            anomalyScores.append(float(score))


with open("output2.txt") as anomaly_file:
    otherLanguageScores = anomaly_file.readlines()
    for score in otherLanguageScores:
        if not score.isspace():
            classes.append(1)
            anomalyScores.append(float(score))

fpr, tpr, thresholds = metrics.roc_curve(classes, anomalyScores, pos_label=1)

auc = metrics.roc_auc_score(classes, anomalyScores)

pyplot.plot(fpr, tpr, linestyle='--', label="Roc curve (AUC = " + str(auc) + ")")
pyplot.xlabel("Specificity")
pyplot.ylabel("Sensitivity")
pyplot.legend()
pyplot.show()
