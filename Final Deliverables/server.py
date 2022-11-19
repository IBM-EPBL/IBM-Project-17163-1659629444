import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle
from sklearn import preprocessing


app = Flask(__name__)
model = pickle.load(open(r'model3.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():

    import requests
    headers = {
        'X-Mboum-Secret': "BHbiTyBaWfYteexXDCS43kQB37lxuPbOMjNqcuhsSMeQSZ1gEemdmnjttKTo"
    }
    response = requests.get(url="https://mboum.com/api/v1/ne/news/?symbol=AAP", headers=headers)

    response_data = (response.json())
    val = (response_data['data']['item'][2]['description'])
    
    url = "https://api.apilayer.com/exchangerates_data/convert?to=INR&from=USD&amount=1"

    payload = {}
    headers = {
        "apikey": "H4ggfqM7nxp8nyyi8q1hJFv7HoVsWUUK"
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    result = response.json()
    rate = (result['result'])

    import finnhub
    finnhub_client = finnhub.Client(api_key="cdsgnbiad3icmfr07t20cdsgnbiad3icmfr07t2g")

    price = (finnhub_client.quote('BKSC')['c'])


    return render_template('register.html', news = val, rate= rate, price = price)


@app.route('/loan', methods=['GET', 'POST'])
def loan():

    return render_template('loan.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template("predict.html")


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    gender = request.form['query1']
    married = request.form['query2']
    dependents = request.form['query3']
    education = request.form['query4']
    self_employed = request.form['query5']
    applicant_income = request.form['query6']
    coapplicant_income = request.form['query7']
    loan_amount = request.form['query8']
    loan_amt_term = request.form['query9']
    credit_history = request.form['query10']
    property_area = request.form['query11']

    input_feature = [[gender, married, dependents, education, self_employed, applicant_income, coapplicant_income,
                      loan_amount, loan_amt_term, credit_history, property_area]]

    print(input_feature)
    names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
             'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    data = pd.DataFrame(input_feature, columns=names)

    new_df = data.copy()
    label_encoder = preprocessing.LabelEncoder()
    label_encoding_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for col in label_encoding_columns:
        new_df[col] = label_encoder.fit_transform(new_df[col])

    prediction = model.predict(new_df)
    print(prediction)
    prediction = int(prediction)

    if prediction == 0:
        return render_template("submit.html", result="Loan will not be approved")
    else:
        return render_template("submit.html", result="Loan will be approved")


if __name__ == "__main__":
    app.run(debug=True)
