 
## Importing the libraries
 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 ## Importing the dataset 

dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)

 ## Cleaning the texts 

import re,nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
  review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
  review=review.lower()
  review=review.split()
  ps=PorterStemmer()
  Total_stopwords=stopwords.words('english')
  Total_stopwords.remove('not')
  Total_stopwords.remove('no')
  Total_stopwords.remove('nor')
  Total_stopwords.remove('against')
  Total_stopwords.remove("wasn't")
  Total_stopwords.remove("weren't")
  review=[ps.stem(word) for word in review if not word in set(Total_stopwords)]
  review=' '.join(review)
  corpus.append(review)

 ## Creating the Bag of Words model 

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
Y=dataset.iloc[:,1].values

print(len(X[0]))

 ## Splitting the dataset into the Training set and Test set 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

 ## Training the Random Forest model on the Training set 

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)

  ## Applying Grid Search to find the best model and the best parameters

from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [10, 50, 100], 'criterion': ['gini']},
              {'n_estimators': [10, 50, 100], 'criterion': ['entropy']}
              ]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

 ## Predicting the Test set results 

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)

 ## Making the Confusion Matrix 

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

 #Predicting a single Review## 

Review=input("Enter a restaurant Review:")
Review=re.sub('[^a-zA-Z]',' ',Review)
Review=Review.lower()
Review=Review.split()
Review=[ps.stem(word) for word in Review if not word in set(Total_stopwords)]
Review=' '.join(Review)
Review=cv.transform([Review]).toarray()
result=classifier.predict(Review)
if result==1:
  print("Positive Review")
else:
  print("Negative Review")