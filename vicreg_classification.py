import glob
import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA


classifiers = [
    KNeighborsClassifier(3),
    SVC(gamma="scale"),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    MLPClassifier(alpha=1, max_iter=500),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

N_PCs = 3

folder = "Extracted_Input_1"

list_files = glob.glob(f"{folder}/Extracted_Input_*.csv")

df = pd.read_csv(f"{folder}/Log.csv")
df.columns = [c.strip() for c in df.columns]
results_df = pd.DataFrame()

y = pd.read_csv(f"{folder}/Target.csv", header=None).values
y = np.vstack((y, y))
for file in list_files:
    i = int(file.split("_")[-1].split(".")[0])
    X = pd.read_csv(file, header=None).values
    L, W = X.shape
    W -= 1
    X1 = X[:, : W // 2]
    X2 = X[:, W // 2 : -1]
    X = np.vstack((X1, X2))

    del X1, X2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=12
    )
    y_train = np.squeeze(y_train.reshape(-1, 1))
    y_test = np.squeeze(y_test.reshape(-1, 1))

    pca = PCA(n_components=N_PCs)
    pca.fit(X_train)

    X_train_projected = pca.transform(X_train)
    X_test_projected = pca.transform(X_test)

    for clf, name in zip(classifiers, names):
        args = dict(df.iloc[i, :])
        print(i)
        print(file, "-", name)
        clf.fit(X_train_projected, y_train)
        y_pred = clf.predict(X_train_projected)
        train_acc = balanced_accuracy_score(y_train, y_pred)
        print("    Train: {:.6f}".format(train_acc * 100))

        y_pred = clf.predict(X_test_projected)
        test_acc = balanced_accuracy_score(y_test, y_pred)
        print("    Test: {:.6f}".format(test_acc * 100))
        print(" ")

        cm = confusion_matrix(y_test, y_pred)
        args["clf"] = name
        args["train_acc"] = train_acc
        args["test_acc"] = test_acc
        args["cm"] = list(cm)
        args["PCs"] = N_PCs

        results_df = results_df.append(args, ignore_index=True)
        results_df.to_csv(f"{folder}/results_L.csv", index=False)
