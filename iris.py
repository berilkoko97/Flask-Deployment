import pandas as pd
import numpy as np
import pickle
import sklearn

df = pd.read_csv('Iris.csv')

#X = df.drop('Species', axis=1)
X = np.array(df.iloc[:, 0:4])
y = df.Species

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(X_train,y_train)


pickle.dump(sv, open('iris.pkl', 'wb'))


  