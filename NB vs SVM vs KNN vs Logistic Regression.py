NB vs SVM vs KNN vs Logistic Regression
#Comparing Gaussian naivebayes and SVM classifier
from sklearn import datasets
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model

iris = datasets.load_iris()
X, y = iris.data, iris.target
#NB
gnb = GaussianNB()
gnb.fit(X, y)
#SVM
clf = SVC()
clf.fit(X, y)
#KNN
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
logisticR = linear_model.LogisticRegression()
logisticR.fit(X, y)

case = clf.predict([[7.0, 3.2, 4.7, 1.4]])
if case == 0:
    print('The flower is I. setosa')
elif case == 1:
    print('The flower is I. versicolor')
elif case == 2:
    print('The flower is I. virginica')
print('The prediction score of SVM is: ', clf.score(X,y)*100, '%')

case1 = gnb.predict([[7.0, 3.2, 4.7, 1.4]])
if case1 == 0:
    print('The flower is I. setosa')
elif case1 == 1:
    print('The flower is I. versicolor')
elif case1 == 2:
    print('The flower is I. virginica')
print('The prediction score of Naive bayes is: ', gnb.score(X,y)*100, '%')

case2 = neigh.predict([[7.0, 3.2, 4.7, 1.4]])
if case2 == 0:
    print('The flower is I. setosa')
elif case2 == 1:
    print('The flower is I. versicolor')
elif case2 == 2:
    print('The flower is I. virginica')
print('The prediction score of KNN is: ', neigh.score(X,y)*100, '%')

case3 = logisticR.predict([[7.0, 3.2, 4.7, 1.4]])
if case3 == 0:
    print('The flower is I. setosa')
elif case3 == 1:
    print('The flower is I. versicolor')
elif case3 == 2:
    print('The flower is I. virginica')
print('The prediction score of KNN is: ', logisticR.score(X,y)*100, '%')