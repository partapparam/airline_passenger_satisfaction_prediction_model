import pickle

from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb



model_file = "xgb_model.bin"

with open(model_file, "rb") as f_in:
    dv, xgb_model = pickle.load(f_in)

app = Flask("satisfaction")


@app.route("/predict", methods=["POST"])
def predict():
    customer_survey = request.get_json()
    features = list(dv.get_feature_names_out())
    X_test = dv.transform([customer_survey])
    dtest = xgb.DMatrix(X_test, feature_names=features)

    xgb_pred = xgb_model.predict(dtest)
    xgb_satisfied = (xgb_pred >= 0.5)

    result = {"satisfaction": float(xgb_pred), "satisfied": bool(xgb_satisfied)}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)