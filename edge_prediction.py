
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import roc_auc_score

import sys

#This is the path to the feature-file where each row denotes the feature-vector generated for each edge
FEATURE_FILE = ""


def extract_feature_vector(feature_string):

    features = []
    [features.append(float(x)) for x in feature_string.split("|")]
    return (np.array(features))

def cross_validation(train_features,train_labels):
    """
    :param train_features: list of training features
    :param train_labels: labels assigned to each training record
    :return: the best regularizer
    """
    regs = [1E-3,1E-2,.1,1,10,100,1000]

    max_auc = -1
    max_reg = None
    for r in regs:
        indexes = range(0,len(train_features))
        kf = KFold(n_splits=10)
        auc = []
        for cv_train_index, cv_test_index in kf.split(indexes):
            cv_train_features = train_features[cv_train_index]
            cv_train_labels = train_labels[cv_train_index]

            cv_test_features = train_features[cv_test_index]
            cv_test_labels = train_labels[cv_test_index]

            logistic = linear_model.LogisticRegression(C=r)
            logistic.fit(cv_train_features, cv_train_labels)  # Train this classifier
            # calculate metrics now
            probabilities = logistic.predict_proba(cv_test_features)
            # area-under-curve
            y_probs = []
            [y_probs.append(x[1]) for x in probabilities]
            auc.append(roc_auc_score(cv_test_labels, y_probs))

        print("AUC for r="+str(r)+"="+str(np.mean(auc))+"...")
        if np.mean(auc)>max_auc:
            max_auc = np.mean(auc)
            max_reg = r

    return (max_reg)




if __name__=="__main__":
    #Load all the features and labels
    features = []
    labels = []
    with open(FEATURE_FILE,"r") as f:
        for line in f:
            data = line.strip().split("\t")
            labels.append(int(data[-1]))
            features.append(extract_feature_vector(data[2]))

    features = np.array(features)
    labels = np.array(labels)



    initial_indexes = range(0,len(features))
    kf = KFold(n_splits=10)

    mean_accuracy = []
    auc = []
    for train_index, test_index in kf.split(initial_indexes):
        train_features = features[train_index]
        train_labels = labels[train_index]

        #starting cross-validation
        print("Starting cross-validation...")
        best_reg = cross_validation(train_features,train_labels)
        #best_reg = 1

        test_features = features[test_index]
        test_labels = labels[test_index]

        #train the classifier
        #cross-validation for finding the best regulizer
        logistic = linear_model.LogisticRegression(C=best_reg)
        logistic.fit(train_features, train_labels)#Train this classifier
        #calculate metrics now
        probabilities = logistic.predict_proba(test_features)

        mean_accuracy.append(logistic.score(test_features,test_labels))
        #area-under-curve
        y_probs = []
        [y_probs.append(x[1]) for x in probabilities]
        auc.append(roc_auc_score(test_labels, y_probs))
    print("mean-accuracy is:"+str(np.mean(mean_accuracy)))
    print("mean-auc is:"+str(np.mean(auc)))
    random.shuffle(initial_indexes)




