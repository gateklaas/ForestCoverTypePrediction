"""
Library for plotting learning curves
Created by Klaas Schuijtemaker
"""

from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def test_error(model, train_X, train_y, test_X, test_y, score=metrics.accuracy_score, n_subsets=20):
    train_errs, test_errs = [],[]
    subset_sizes = np.exp(np.linspace(3, np.log(train_X.shape[0]), n_subsets)).astype(int)

    for m in subset_sizes:
        model.fit(train_X[:m], train_y[:m])
        train_err = 1 - score(train_y[:m], model.predict(train_X[:m]))
        test_err = 1 - score(test_y, model.predict(test_X))
        print "training error: %.3f test error: %.3f subset size: %d" % (train_err, test_err, m)
        train_errs.append(train_err)
        test_errs.append(test_err)

    return subset_sizes, train_errs, test_errs

def crossv_gamma_c(model, X, y, kf, score=metrics.accuracy_score, n_subsets=20):

    gammas = np.exp(np.linspace(np.log(0.00000001), np.log(10), n_subsets))
    Cs = np.exp(np.linspace(np.log(0.1), np.log(100000), n_subsets))
    
    train_errs, test_errs = np.zeros((n_subsets, n_subsets)), np.zeros((n_subsets, n_subsets))

    xx = 0
    for gamma in gammas:
        yy = 0
        model.gamma = gamma
        for C in Cs:
            model.C = C
    
            for train, test in kf:
                X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    
                model.fit(X_train, y_train)
                train_err = 1 - score(y_train, model.predict(X_train))
                crossv_err = 1 - score(y_test, model.predict(X_test))
    
                train_errs[xx][yy] += train_err
                test_errs[xx][yy] += crossv_err
    
            train_errs[xx][yy] /= kf.n_folds
            test_errs[xx][yy] /= kf.n_folds
            print "train_err: %.3f crossv_err: %.5f, gamma: %.5f, C: %.5f" % (train_errs[xx][yy], test_errs[xx][yy], gamma, C)
            yy += 1
        xx += 1

    return gammas, Cs, train_errs, test_errs

def crossv_gamma(model, X, y, kf, score=metrics.accuracy_score, n_subsets=20, start=0.001, stop=2):

    #gammas = np.linspace(start, stop, n_subsets)
    gammas = np.exp(np.linspace(np.log(start), np.log(stop), n_subsets))
    train_errs, test_errs = np.zeros(n_subsets), np.zeros(n_subsets)

    j = 0
    for gamma in gammas:
        model.gamma = gamma

        for train, test in kf:
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            model.fit(X_train, y_train)
            train_err = 1 - score(y_train, model.predict(X_train))
            crossv_err = 1 - score(y_test, model.predict(X_test))

            train_errs[j] += train_err
            test_errs[j] += crossv_err

        train_errs[j] /= kf.n_folds
        test_errs[j] /= kf.n_folds
        print "training error: %.3f crossv error: %.5f, gamma: %.5f" % (train_errs[j], test_errs[j], gamma)
        j += 1

    return gammas, train_errs, test_errs

def crossv_c(model, X, y, kf, score=metrics.accuracy_score, n_subsets=20, start=0.1, stop=100000):

    Cs = np.exp(np.linspace(np.log(start), np.log(stop), n_subsets))
    train_errs, test_errs = np.zeros(n_subsets), np.zeros(n_subsets)

    j = 0
    for C in Cs:
        model.C = C

        for train, test in kf:
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            model.fit(X_train, y_train)
            train_err = 1 - score(y_train, model.predict(X_train))
            crossv_err = 1 - score(y_test, model.predict(X_test))

            train_errs[j] += train_err
            test_errs[j] += crossv_err

        train_errs[j] /= kf.n_folds
        test_errs[j] /= kf.n_folds
        print "training error: %.3f crossv error: %.5f, C: %.5f" % (train_errs[j], test_errs[j], C)
        j += 1

    return Cs, train_errs, test_errs

def plot_test_error(subset_sizes, train_errs, test_errs, title="Model response to dataset size"):
    plt.plot(subset_sizes, train_errs, lw=2)
    plt.plot(subset_sizes, test_errs, lw=2)
    plt.legend(["Training Error", "Test Error"])
    plt.xscale("log")
    plt.xlabel("Dataset size")
    plt.ylim(0, 1)
    plt.ylabel("Error")
    plt.title(title)
    plt.show()

def plot_crossv_gamma_c(gammas, Cs, train_errs, crossv_errs, title="Model response to dataset size"):
    
    X, Y = np.meshgrid(Cs, gammas)
    
    plt.contourf(X, Y, crossv_errs, 100, cmap='jet')
    plt.colorbar()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("Gamma")
    plt.yscale("log")
    plt.title(title)
    plt.show()

    plt.contourf(X, Y, train_errs, 100, cmap='jet')
    plt.colorbar()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("Gamma")
    plt.yscale("log")    
    plt.title(title)
    plt.show()

    crossv_errs += train_errs
    plt.contourf(X, Y, crossv_errs, 100, cmap='jet')
    plt.colorbar()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("Gamma")
    plt.yscale("log")    
    plt.title(title)
    plt.show()

def plot_crossv_gamma(gammas, train_errs, crossv_errs, title="Model response to dataset size"):
    plt.plot(gammas, train_errs, lw=2)
    plt.plot(gammas, crossv_errs, lw=2)
    plt.legend(["Training Error", "Cross Validation Error"])
    plt.xscale("log")
    plt.xlabel("Gamma")
    plt.ylim(0, 1)
    plt.ylabel("Error")
    plt.title(title)
    plt.show()

def plot_crossv_c(Cs, train_errs, crossv_errs, title="Model response to dataset size"):
    plt.plot(Cs, train_errs, lw=2)
    plt.plot(Cs, crossv_errs, lw=2)
    plt.legend(["Training Error", "Cross Validation Error"])
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylim(0, 1)
    plt.ylabel("Error")
    plt.title(title)
    plt.show()

def get_precision_recall_table(y_true, y_pred, labels=None):
    p, r, f1, s = metrics.precision_recall_fscore_support(y_true, y_pred)
    cm = metrics.confusion_matrix(y_true, y_pred)
    
    if labels == None:
        labels = np.arange(1, len(r) + 1).astype(np.str)
    
    columns = ["true " + l for l in labels]
    index = ["pred. " + l for l in labels]
    
    df1 = pd.DataFrame(cm, columns=columns, index=index)
    df2 =  pd.DataFrame([r], index=["class recall"], columns=columns)
    
    df = pd.concat([df1, df2])
    df["class_precision"] = np.append(p, None)
    return df

def print_report(y_true, y_pred, labels=None):
    print "accuracy: %.5f" % metrics.accuracy_score(y_true, y_pred)
    print metrics.classification_report(y_true, y_pred, labels=labels)
    
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    print cm
    
    plt.matshow(cm)
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.show()
