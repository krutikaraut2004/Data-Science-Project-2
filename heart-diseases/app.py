# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    # Render the main HTML template
    return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Retrieving form data
        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        # Creating numpy array from form data
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        # Making prediction
        my_prediction = model.predict(data)
        
        # Render the result HTML template with prediction
        return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
