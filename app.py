from flask import Flask, render_template, request, redirect
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import pickle

hell=Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
cv = pickle.load(open('cv.pkl','rb'))

data=[{'name':'Nokia 3310','price':'$50/-','review':0,'comments':[]},{'name':'samsung 3310','price':'$10/-','review':0,'comments':[]}]


@hell.route('/')
def helloWorld():
    return render_template('index.html',data=data)

@hell.route('/about/<mobilenumber>')
def about(mobilenumber):
    return render_template('about.html',mobileName=mobilenumber,data=data)

@hell.route('/feedback/<mobileName>', methods=['GET','POST'])
def comment(mobileName):
    user=[[request.form['post']]]

    myword=(user[0])[0]
    stop=set(stopwords.words('english'))
    stop.remove('no')
    stop.remove('not')
    stop.remove('nor')


    myword = ' '.join(e for e in myword.split() if e not in stop)
    myword.lower().strip()

    s = SnowballStemmer("english")
    p=[]

    for word in myword.split():
        p.append(s.stem(word))
    myword=' '.join(p)



    y= cv.transform([myword])
    prediction= model.predict(y)[0]

    global data
    for msg in data:
        if msg['name']==mobileName:
            msg['comments'].append((user[0])[0])
            if prediction==0:
                msg['review']-=1
            elif prediction==1:
                msg['review']+=1
            break
    
    data = sorted(data, key=lambda k: k['review'],reverse=True)
    return render_template('index.html', data=data)

if __name__ == "__main__":
    hell.run(debug=True)