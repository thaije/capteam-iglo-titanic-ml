from sklearn import model_selection

# Wrapper methods for various cross validation methods of Scikit learn

# Finds the optimal Classifier Fit for the given train_test_split.
def optimal_fit(X, y, clf, train_test_split):
    optimal_score = 0
    optimal_clf = None
    scores = []
    for train, test in train_test_split:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        new_clf = clf.fit(X_train, y_train)
        new_score = new_clf.score(X_test, y_test)
        scores.append(new_score)
        if new_score > optimal_score:
            optimal_clf, optimal_score = new_clf, new_score

    return optimal_clf, optimal_score, scores


def KFold(X, y, clf, K=10):
    print ("KFold validation with K=%d" % K)
    kf = model_selection.KFold(n_splits=K)
    return optimal_fit(X, y, clf, kf.split(X))

def RepeatedKFold(X, y, clf, K=10, N=10):
    print ("RepeatedKFold validation with K=%d and N=%d" % (K, N))
    rkf = model_selection.RepeatedKFold(n_splits=K, n_repeats=N)
    return optimal_fit(X, y, clf, rkf.split(X))


def StratifiedKFold(X, y, clf, K=10):
    print ("StratifiedKFold validation with K=%d" % K)
    skf = model_selection.StratifiedKFold(K)
    return optimal_fit(X, y, clf, skf.split(X, y))

def LeaveOneOut(X, y, clf):
    print ("LeaveOneOut validation")
    loo = model_selection.LeaveOneOut()
    return optimal_fit(X, y, clf, loo.split(X))
