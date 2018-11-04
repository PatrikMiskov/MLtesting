from sklearn import datasets  # pylint: disable=E0401
from sklearn.cross_validation import train_test_split  # pylint: disable=E0401
from sklearn import tree  # pylint: disable=E0401
from sklearn.metrics import accuracy_score  # pylint: disable=E0401

iris = datasets.load_iris()

X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)

my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, Y_train)

predictions = my_classifier.predict(X_test)
print(accuracy_score(Y_test, predictions))
