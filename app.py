from flask import Flask, render_template, request, redirect
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pickle

hell=Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
cv = pickle.load(open('cv.pkl','rb'))

data=[{'name':'Nokia 3310','price':'$50/-','review':0},{'name':'samsung 3310','price':'$10/-','review':0}]


@hell.route('/')
def helloWorld():
    #return 'ahsan'
    return render_template('index.html',data=data)

@hell.route('/about/<mobilenumber>')
def about(mobilenumber):
    return render_template('about.html',mobileName=mobilenumber)

@hell.route('/feedback/<mobileName>', methods=['GET','POST'])
def comment(mobileName):
    user=[[request.form['post']]]
    userValue=pd.DataFrame(user)

    y= cv.transform(userValue[0]).toarray()
    prediction= model.predict(y)[0]

    global data
    for msg in data:
        if msg['name']==mobileName:
            if prediction==0:
                msg['review']-=1
            elif prediction==1:
                msg['review']+=1
            break
    
    data = sorted(data, key=lambda k: k['review'],reverse=True)
    return render_template('index.html', data=data)

if __name__ == "__main__":
    app.run(debug=True)