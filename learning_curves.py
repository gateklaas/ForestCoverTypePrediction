"""
Library for plotting
"""

from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np

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
    
def crossv_gamma(model, train_X, train_y, crossv_X, crossv_y, score=metrics.accuracy_score, n_subsets=20, start=0.001, stop=2):
    train_errs, test_errs = [],[]
    gammas = np.linspace(start, stop, n_subsets)
    for gamma in gammas:
        model.gamma = gamma
        model.fit(train_X, train_y)
        train_err = 1 - score(train_y, model.predict(train_X))
        crossv_err = 1 - score(crossv_y, model.predict(crossv_X))
        print "training error: %.3f crossv error: %.5f, gamma: %.5f" % (train_err, crossv_err, gamma)
        train_errs.append(train_err)
        test_errs.append(crossv_err)
    
    return gammas, train_errs, test_errs
    
def crossv_c(model, train_X, train_y, crossv_X, crossv_y, score=metrics.accuracy_score, n_subsets=20):
    train_errs, test_errs = [],[]
    #Cs = np.linspace(0.1, 100, n_subsets)
    Cs = np.exp(np.linspace(np.log(0.1), np.log(100), n_subsets))

    for C in Cs:
        model.C = C
        model.fit(train_X, train_y)
        train_err = 1 - score(train_y, model.predict(train_X))
        crossv_err = 1 - score(crossv_y, model.predict(crossv_X))
        print "training error: %.3f crossv error: %.3f, C: %.3f" % (train_err, crossv_err, C)
        train_errs.append(train_err)
        test_errs.append(crossv_err)
    
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

def plot_crossv_gamma(gammas, train_errs, crossv_errs, title="Model response to dataset size"):
    plt.plot(gammas, train_errs, lw=2)
    plt.plot(gammas, crossv_errs, lw=2)
    plt.legend(["Training Error", "Cross Validation Error"])
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

def plot_confusion_matrix(true_y, pred_y, score=metrics.accuracy_score):
    plt.imshow(metrics.confusion_matrix(true_y, pred_y), cmap=plt.cm.binary, interpolation='nearest')
    plt.colorbar()
    plt.xlabel("true value")
    plt.ylabel("predicted value")
    plt.show()
    
def print_accuracy(true_y, pred_y):
    print "accuracy: %.5f" % metrics.accuracy_score(true_y, pred_y)
    #print "average_precision_score %.5f" % metrics.average_precision_score(true_y, pred_y)
    print "f1_score: %.5f" % metrics.f1_score(true_y, pred_y)
    print "precision_score: %.5f" % metrics.precision_score(true_y, pred_y)
    print "recall_score: %.5f" % metrics.recall_score(true_y, pred_y)
    #print "roc_auc_score: %.5f" % metrics.roc_auc_score(true_y, pred_y)
    print "adjusted_rand_score: %.5f" % metrics.adjusted_rand_score(true_y, pred_y)
    print "mean_absolute_error: %.5f" % metrics.mean_absolute_error(true_y, pred_y)
    print "mean_squared_error: %.5f" % metrics.mean_squared_error(true_y, pred_y)
    print "r2_score: %.5f" % metrics.r2_score(true_y, pred_y)