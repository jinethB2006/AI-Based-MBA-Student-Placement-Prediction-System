from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model/student_placement_logistic_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/encoder.pkl")
feature_names = joblib.load("model/feature_names.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    ssc_percentage=float(request.form["ssc_percentage"])
    hsc_percentage=float(request.form["hsc_percentage"])
    degree_percentage=float(request.form["degree_percentage"])
    employability_test_percentage=float(request.form["employability_test_percentage"])
    mba_percentage=float(request.form["mba_percentage"])

    gender=request.form["gender"]
    ssc_board=request.form["ssc_board"]
    hsc_board=request.form["hsc_board"]
    hsc_stream=request.form["hsc_stream"]
    degree_type=request.form["degree_type"]
    work_experience=request.form["work_experience"]
    specialisation=request.form["specialisation"]

    numerical_data=pd.DataFrame([[
        ssc_percentage,
        hsc_percentage,
        degree_percentage,
        employability_test_percentage,
        mba_percentage
    ]],columns=[
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage",
        "employability_test_percentage",
        "mba_percentage"
    ])

    categorical_data=pd.DataFrame([[
        gender,
        ssc_board,
        hsc_board,
        hsc_stream,
        degree_type,
        work_experience,
        specialisation
    ]],columns=[
        "gender",
        "ssc_board",
        "hsc_board",
        "hsc_stream",
        "degree_type",
        "work_experience",
        "specialisation"
    ])

    encoded_data = encoder.transform(categorical_data)

    encoded_data = encoder.transform(categorical_data)

    encoded_columns = encoder.get_feature_names_out([
    "gender",
    "ssc_board",
    "hsc_board",
    "hsc_stream",
    "degree_type",
    "work_experience",
    "specialisation"
    ])

    encoded_data = pd.DataFrame(
    encoded_data,
    columns=encoded_columns,
    index=categorical_data.index
    )

    final_data = pd.concat([numerical_data, encoded_data], axis=1)

    final_data = final_data.reindex(columns=feature_names, fill_value=0)

    final_data = scaler.transform(final_data)

    prediction = model.predict(final_data)

    if prediction[0]==1:
        result="✅ Prediction: Student is likely to be Placed"
    else:
        result="❌ Prediction: Student is likely to be Not Placed"

    return render_template("index.html", prediction=result)

if __name__=="__main__":
    app.run(debug=True)