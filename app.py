# save this as app.py
from flask import Flask, escape, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model_rf.pkl', 'rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        breath_prob = int(request.form['breath_prob'])
        fever = int(request.form['fever'])
        dry_cough = int(request.form['dry_cough'])
        sore_throat = int(request.form['sore_throat'])
        running_nose = int(request.form['running_nose'])
        asthma = int(request.form['asthma'])
        chronic_lung_disease = int(request.form['chronic_lung_disease'])
        headache = int(request.form['headache'])
        heart_disease = int(request.form['heart_disease'])
        diabetes = int(request.form['diabetes'])
        hyper_tension = int(request.form['hyper_tension'])
        fatigue = int(request.form['fatigue'])
        gastrointestinal = int(request.form['gastrointestinal'])
        abroad_travel = int(request.form['abroad_travel'])
        contact_covid = int(request.form['contact_covid'])
        attend_large_gathering = int(request.form['attend_large_gathering'])
        visit_public_place = int(request.form['visit_public_place'])
        family_work_in_public = int(request.form['family_work_in_public'])


        prediction = model.predict([[breath_prob, fever, dry_cough, sore_throat, running_nose, asthma, chronic_lung_disease, headache,
                                   heart_disease, diabetes, hyper_tension, fatigue, gastrointestinal, abroad_travel, contact_covid, 
                                   attend_large_gathering, visit_public_place, family_work_in_public]])

        # print(prediction)

        if (prediction == "No"):
            prediction = 0
        else:
            prediction = 1

        return render_template("predict.html", prediction=prediction)

    else:
        return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)
