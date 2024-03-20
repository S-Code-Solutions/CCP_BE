from flask import Blueprint, request, jsonify
from server.models import predict_arima, predict_lstm

api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/predict/arima', methods=['POST'])
def predict_with_arima():
    """
    Endpoint to predict future values using the ARIMA model.
    Expects a JSON payload with necessary parameters for prediction.
    """
    # Example: Expect JSON payload with {"series": [data_points], "steps": 5}
    data = request.get_json()
    series = data.get('series')
    steps = data.get('steps')

    if not series or steps is None:
        return jsonify({"error": "Missing 'series' or 'steps' in request"}), 400

    try:
        prediction = predict_arima(series, steps)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/predict/lstm', methods=['POST'])
def predict_with_lstm():
    """
    Endpoint to predict future values using the LSTM model.
    Expects a JSON payload with necessary parameters for prediction.
    """
    # Example: Expect JSON payload with {"series": [data_points], "steps": 5}
    data = request.get_json()
    series = data.get('series')
    steps = data.get('steps')

    if not series or steps is None:
        return jsonify({"error": "Missing 'series' or 'steps' in request"}), 400

    try:
        prediction = predict_lstm(series, steps)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
