from flask import Flask,request,render_template
import pickle
import helper
import nltk


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

model=pickle.load(open('model.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def find():
    if request.method=='POST':
        q1=request.form['ques1']
        q2=request.form['ques2']

        query=helper.query_point_creator(q1,q2)

        result=model.predict(query)[0]

        if result:
            return render_template('index.html',prediction_text="This question's are Similar")

        else:
            return render_template('index.html',prediction_text="This question's are not Similar")

    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)
