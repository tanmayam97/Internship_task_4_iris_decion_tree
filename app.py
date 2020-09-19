from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('dt_task4.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
     if request.method == 'POST':
            
        SepalLengthCm = request.form['SepalLengthCm']
        SepalWidthCm = request.form['SepalWidthCm']
        PetalLengthCm = request.form['PetalLengthCm']
        PetalWidthCm = request.form['PetalWidthCm']
        
        data = np.array([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
        my_prediction = model.predict(data)
        return render_template('result.html', prediction_text =  my_prediction)
        
            

if  __name__ == '__main__':
    app.run(debug = True)