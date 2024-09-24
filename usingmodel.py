import joblib

model = joblib.load('music-recommender.joblib')
predictions = model.predict([[21,1]])
print(predictions)