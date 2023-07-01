import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

dataframetrain = pd.read_csv("train.csv", encoding="Windows-1252")

#parse data from dataframes
X_Train = dataframetrain['text'].copy()
Y_Train = dataframetrain['sentiment'].copy()

dataframe = pd.read_csv("test.csv", encoding="Windows-1252")

X_Test = dataframe['text'].copy()
Y_Test = dataframe['sentiment'].copy()

#remove any null value in X_Train, Y_Train, X_Test, or Y_Test and its cooresponding value in
#Y_Train, X_Train, Y_Test, and X_Test, respectively
X_Train, Y_Train = X_Train.dropna(), Y_Train[X_Train.dropna().index].dropna()
X_Test, Y_Test = X_Test.dropna(), Y_Test[X_Test.dropna().index].dropna()

#values in X_Train and X_Test are tokenized
vectorizer = CountVectorizer(stop_words='english')
X_Train = vectorizer.fit_transform(X_Train)
X_Test = vectorizer.transform(X_Test)

le = LabelEncoder()

#Three types of sentiments: negative, neutral, and positive, these are normalized and preprocessed by LabelEncoder
Y_Train = le.fit_transform(Y_Train)

X_Test = X_Test.toarray()

model = LogisticRegression(max_iter=20000)

model.fit(X_Train, Y_Train)

predict = model.predict(X_Test)

#values in predict are tokenized, 0 = negative, 1 = neutral, 2 = positive
sentiments = []
for sentiment in predict:
    if sentiment == 0:
        sentiments.append("negative")
    elif sentiment == 1:
        sentiments.append("neutral")
    elif sentiment == 2:
        sentiments.append("positive")
    else:
        sentimnents.append("unknown")

contents = dataframe['text'].copy()
accurate = 0
num = 200

for x in range(num):
    print(f"Tweet:\n{contents[x]}")
    print(f"Predicted Sentiment: {sentiments[x]}\nActual Sentiment: {Y_Test[x]}\n")
    correct = (sentiments[x] == Y_Test[x])
    if correct: accurate += 1
    print("Guessed Correctly: %s\n" % correct)
    print("Overall Accuracy: %2f\n" % (accurate / (x+1)))

