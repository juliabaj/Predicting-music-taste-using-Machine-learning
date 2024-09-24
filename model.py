#1.import data (csv file)
#2. clean data to prevent computer from learning wrong things
#3. split the data into Training/Test Sets
#4. Create a model
#5. Train the model
#6. Make predictions
#7. Evaluate and improve

#libraries for ML
#Numpy
#Pandas (data in rows and columns)
#MatPlotLib (2 dimension)
#Scikit-learn (decision algorithms)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])  #input data set
Y = music_data['genre']   #output data set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)  #20% of data is for testing
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

tree.export_graphviz(model, out_file='music-recommender.dot', feature_names=['age', 'gender'], class_names=sorted(Y.unique()), label='all', rounded=True, filled=True)
predictions = model.predict(X_test)    #predicting for 22 yr old men and woman
joblib.dump(model, 'music-recommender.joblib')

score = accuracy_score(Y_test, predictions)
print(score)
