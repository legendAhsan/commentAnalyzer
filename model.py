import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes  import MultinomialNB

data=pd.read_csv('AmazonMobileDataCleaned.csv')
data=data[:4000]
label = data['decision']
feedback=data['finaltext']

cv=CountVectorizer(max_features=5000,ngram_range=(1,3))

x=cv.fit(feedback)
x=cv.transform(feedback)


#pickle.dump(cv, open('cv.pkl', 'wb'))
X_train,X_test,y_train,y_test=train_test_split(x,label,test_size=0.2,random_state=0)

classifier=MultinomialNB()

classifier.fit(X_train,y_train)
features=cv.get_feature_names()
print(sorted(zip(classifier.coef_[0],features),reverse=True)[:50])
# # pickle.dump(classifier, open('model.pkl', 'wb'))
# # pred=classifier.predict(X_test)