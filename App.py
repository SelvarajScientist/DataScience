from flask import Flask,render_template,url_for,request
from flask_material import Material
import pandas as pd
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__,template_folder='templates')
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/services.html')
def service():
    return render_template("services.html")
@app.route('/about.html')
def about():
    return render_template("about.html")
@app.route('/success-stories.html')
def stories():
    return render_template("success-stories.html")

@app.route('/blog.html')
def blog():
    return render_template("blog.html")
@app.route('/contact.html')
def contact():
    return render_template("contact.html")
@app.route('/analyze',methods=['POST'])
def analyze():
    if request.method == 'POST':
        glucoselevel = request.form['glucose']
        bloodpressure = request.form['bp']
        insulinlevel = request.form['insulin']
        bmi          =request.form['BMI']
        Age          =request.form['AGE']




    #cleaning of data

    sample_data = [glucoselevel,bloodpressure,insulinlevel,bmi,Age]
    clean_data = [float(i) for i in sample_data]
    #reshaping of the data
    Exec = np.array(clean_data).reshape(1,-1)
    load_model = joblib.load('data/vector.pkl')
    result_prediction = load_model.predict(Exec)
    proba = load_model.predict_proba(Exec)
    percentage_true =(proba[0][1])*100
    percentage_false=(proba[0][0])*100

    if(result_prediction == [0]):
        result_prediction='Negative'
        content_for_Negative='Wow you are free from diabetes of nearly'+str(int(round(percentage_false)))+'%'
        content_for_positive='In future you might be affected by the diabetes of'+str(int(round(percentage_true)))+'%'
    else:
        result_prediction='Positive'
        content_for_Negative='Probality for futur you can be free from diabetes is of'+str(int(round(percentage_false)))+'%'
        content_for_positive='Ohh no you are affected by the diabetes is nearly'+str(int(round(percentage_true)))+'%'
    return render_template("services.html",glucose=glucoselevel,
                           bp=bloodpressure,
                           insulin=insulinlevel,
                           BMI=bmi,
                           AGE=Age,
                           result_prediction=result_prediction,
                           content_for_Negative=content_for_Negative,
                           content_for_positive=content_for_positive)
if __name__ =='__main__':
    app.run(debug=True)