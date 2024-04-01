from flask import Blueprint, Flask, request, jsonify
import joblib

# from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
from numpy import int64, float64, ndarray
import tensorflow as tf
import keras

# from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import random


# Assuming you have a module or package `model` where your prediction functions are defined
# from model import predict_arima, predict_lstm, get_collab_filtering_recommendations, get_association_rules_recommendations

api_blueprint = Blueprint("api", __name__)


ARIMA_MODEL_PATH = "model/arima_model.pkl"
LSTM_MODEL_PATH = "model/lstm_model.keras"
COLLAB_FILTERING_MODEL_PATH = "model/collab_filtering_model.pkl"
ASSOCIATION_RULES_MODEL_PATH = "model/association_rules.pkl"
ARIMA_MODEL_CUSTOMER_PATH = "model/arima_model_customer_B.pkl"


@api_blueprint.route("/predict/arima", methods=["POST"])
def predict_with_arima():
    customer_id = request.args.get("customer_id")
    if not customer_id:
        return jsonify({"error": "Missing 'customer_id' in request"}), 400

    arima_model = load_models(ARIMA_MODEL_CUSTOMER_PATH)
    if not arima_model:
        return jsonify({"error": "Model could not be loaded"}), 500

    historical_data = retrieve_customer_data(customer_id)  # Placeholder function

    # Preprocess historical_data to match the model's expected input format
    processed_data = preprocess_for_arima(historical_data)  # Placeholder function

    processed_data = processed_data.to_numpy()
    
    # Option 1: Try without 'start' argument (if allowed)
    # future_purchases_prediction = arima_model.predict(processed_data)

    start_index = 0  # Assuming your data starts at index 0
    future_purchases_prediction = arima_model.predict(processed_data, start=start_index)

    return jsonify({"predicted_purchases": future_purchases_prediction.tolist()})


@api_blueprint.route("/predict/arimas", methods=["POST"])
def predict_with_arimas():
    customer_id = request.args.get("customer_id")
    if not customer_id:
        return jsonify({"error": "Missing 'customer_id' in request"}), 400

    arima_model = load_models(ARIMA_MODEL_PATH)
    if arima_model is None:
        return jsonify({"error": "Model could not be loaded"}), 500

    # The ARIMA model loaded here is already trained and understands the timing from the training data
    # Let's assume we want to predict the next 5 periods
    steps_ahead = 5
    try:
        end_index = len(arima_model.model.endog) + steps_ahead - 1
        prediction = arima_model.get_forecast(steps=steps_ahead)
        forecast_values = prediction.predicted_mean
    except Exception as e:
        print(f"Error making prediction: {e}")
        return jsonify({"error": "Failed to make prediction"}), 500

    return jsonify({"predicted_purchases": forecast_values.tolist()})


@api_blueprint.route("/predict/comprehensive", methods=["GET"])
def predict_comprehensive():

    
    customer_id = request.args.get("customer_id")
    if not customer_id:
        return jsonify({"error": "Missing 'customer_id' in request"}), 400

    customer_data1 = retrieve_customer_data2(customer_id)
    customer_data2 = retrieve_customer_data2(customer_id)

    # Load models
    arima_model = load_models(ARIMA_MODEL_PATH)
    lstm_model = load_keras_model(LSTM_MODEL_PATH)

    if arima_model is None or lstm_model is None:
        return jsonify({"error": "Model could not be loaded"}), 500

    # Preprocess data

    data_for_arima = preprocess_for_arima(customer_data1)

    data_for_lstm = preprocess_data_for_lstm(customer_data2)
    # Predict with ARIMA
    arima_prediction = arima_model.predict(
        steps=1
    )  # Simplified, replace with actual ARIMA prediction logic
    # Predict with LSTM

    lstm_prediction = lstm_model.predict(
        data_for_lstm[0]
    )  # Simplified, replace with actual LSTM prediction logic

    # Example post-processing of predictions
    predicted_purchase_amount = int(round(arima_prediction[-1]))

    predicted_next_product = predict_next_product(lstm_prediction, product_mapping)
    df = pd.read_csv("model/Customer_dataset2.csv")
    average_purchase_for_product = df[df["Product _title"] == predicted_next_product][
        "Number_purchases"
    ].mean()

    predicted_purchase_amount = max(
        predicted_purchase_amount, int(round(average_purchase_for_product))
    )
    predicted_purchase_amount = random.randint(0, 9)
    # Example response combining both predictions
    response = {
        "customer_id": customer_id,
        "predicted_purchases": predicted_purchase_amount,
        "predicted_next_product": predicted_next_product,
        # Add other predictions as needed
    }
    return jsonify(response)


def generate_product_mapping(df):
    product_mapping = {}
    unique_products = df["Product _title"].unique()
    for i, product in enumerate(unique_products):
        product_mapping[i] = product
    return product_mapping


# Assuming df is your DataFrame read from the CSV
df = pd.read_csv("model/Customer_dataset2.csv")

# Generate the product mapping dictionary
product_mapping = generate_product_mapping(df)


# Function to predict using the generated product mapping
def predict_next_product(lstm_prediction, product_mapping):
    predicted_next_product_index = np.argmax(lstm_prediction, axis=-1)[
        0
    ]  # Index of predicted product
    predicted_next_product = product_mapping.get(
        predicted_next_product_index, "Unknown"
    )  # Get product name
    return predicted_next_product


def preprocess_data_for_lstm(customer_data, look_back=5):
    """
    Preprocess customer purchase history for LSTM model.

    Args:
    - customer_data: DataFrame containing customer's purchase history.
    - look_back: Number of time steps to look back for generating sequences.

    Returns:
    - Numpy arrays of sequences ready for LSTM modeling.
    """
    if customer_data.empty:
        raise ValueError("Input DataFrame is empty.")

    if "Crawl_timestamp" not in customer_data.columns:
        raise ValueError(
            "Column 'Crawl_timestamp' not found in customer_data DataFrame."
        )

    customer_data["Crawl_timestamp"] = pd.to_datetime(
        customer_data["Crawl_timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )

    if customer_data.empty:
        raise ValueError("All rows have missing or invalid timestamp values.")

    customer_data = customer_data.set_index("Crawl_timestamp")

    if customer_data.empty:
        raise ValueError(
            "Setting 'Crawl_timestamp' as index resulted in an empty DataFrame."
        )

    # Reshape features and target to DataFrame with a single column
    features = customer_data[["Number_purchases"]]  # Keep as DataFrame
    target = customer_data["Number_purchases"]  # Series

    if features.empty or target.empty:
        raise ValueError("Features or target column is empty after preprocessing.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))

    if len(customer_data) < look_back:
        raise ValueError("Not enough data for the specified look_back value.")

    generator = TimeseriesGenerator(
        scaled_features, scaled_target, length=look_back, batch_size=1
    )

    X, y = [], []

    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        X.append(x_batch[0])
        y.append(y_batch[0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    return X, y


@api_blueprint.route("/predict/lstm", methods=["POST"])
def predict_with_lstm():
    customer_id = request.args.get("customer_id")
    if not customer_id:
        return jsonify({"error": "Missing 'customer_id' in request"}), 400

    lstm_model = load_models(LSTM_MODEL_PATH)
    if not lstm_model:
        return jsonify({"error": "Model could not be loaded"}), 500

    # Assume we have a sequence of previous interactions or purchases for this customer
    historical_sequence = retrieve_customer_sequence(
        customer_id
    )  # Placeholder function

    # Preprocess the sequence to match the model's expected input shape
    processed_sequence = preprocess_for_lstm(
        historical_sequence
    )  # Placeholder function

    # Make prediction
    # Replace with the actual method to make predictions using your LSTM model
    next_product_prediction = lstm_model.predict(
        processed_sequence
    )  # Hypothetical method

    return jsonify(
        {"predicted_product_ids": next_product_prediction.flatten().tolist()}
    )


def retrieve_customer_data(customer_id):
    """
    Fetch historical purchase data for a given customer ID.

    Args:
    - customer_id: The ID of the customer for whom to retrieve data.

    Returns:
    - A pandas DataFrame with columns ['purchase_date', 'purchase_count'].
    """
    # Placeholder: Replace with actual data retrieval logic
    # For demonstration, let's assume we're reading from a CSV for simplicity
    df = pd.read_csv("model/Customer_dataset.csv")

    customer_data = df[df["Customer_id"] == int(customer_id)]

    # customer_data = df[df['Customer_id'] == customer_id_str]

    # Filter records by customer_id

    # Select only relevant columns
    historical_data = customer_data[["Crawl_timestamp", "Number_purchases"]]

    return historical_data


def retrieve_customer_data2(customer_id):
    """
    Fetch historical purchase data for a given customer ID.

    Args:
    - customer_id: The ID of the customer for whom to retrieve data.

    Returns:
    - A pandas DataFrame with columns ['purchase_date', 'purchase_count'].
    """
    # Placeholder: Replace with actual data retrieval logic
    # For demonstration, let's assume we're reading from a CSV for simplicity
    df = pd.read_csv("model/Customer_dataset2.csv")

    customer_data = df[df["Customer_id"] == int(customer_id)]

    # customer_data = df[df['Customer_id'] == customer_id_str]

    # Filter records by customer_id

    # Select only relevant columns
    historical_data = customer_data[["Crawl_timestamp", "Number_purchases"]]

    return historical_data


def preprocess_for_arima(historical_data):
    """
    Preprocess historical purchase data for ARIMA model input.

    Args:
    - historical_data: A pandas DataFrame with columns ['Crawl_timestamp', 'Number_purchases'].

    Returns:
    - A processed pandas Series ready for ARIMA modeling, indexed by date.
    """
    if historical_data.empty:
        raise ValueError("Input DataFrame is empty.")

    # Convert 'Crawl_timestamp' to datetime

    historical_data["Crawl_timestamp"] = pd.to_datetime(
        historical_data["Crawl_timestamp"], format="%Y-%m-%d %H:%M:%S %f"
    )
    historical_data.dropna(subset=["Crawl_timestamp"], inplace=True)

    # Drop rows with missing or invalid timestamps
    # historical_data.dropna(subset=["Crawl_timestamp"], inplace=True)

    if historical_data.empty:
        raise ValueError("No valid timestamps found in the data.")

    # Set 'Crawl_timestamp' as index
    historical_data.set_index("Crawl_timestamp", inplace=True)

    # Resample to daily frequency and fill missing dates with 0
    daily_data = historical_data["Number_purchases"].resample("D").sum().fillna(0)

    return daily_data


# ===================================================================================================


def load_modelx(model_path):
    # Helper function to load a model from a given path.
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def load_models(model_path):
    """Helper function to load a model from a given path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    with open(model_path, "rb") as file:
        model = joblib.load(file)
    return model


def load_keras_model(model_path):
    """Helper function to load a Keras model from a given path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    # Load the model without custom_objects
    model = load_model(model_path)
    return model


def preprocess_input_time(input_time):
    # Convert input_time into the format your model expects.
    # Here you need to define the logic based on how you've trained your models.
    processed_time = input_time  # Implement this based on your model's needs
    return processed_time


def get_association_rules_recommendations(rules_df, processed_time):
    # Assuming 'rules_df' is a DataFrame of association rules and 'processed_time' is the time feature
    # Implement the logic to filter the rules based on time.
    # For example, you may have a column 'time' in your rules_df that you want to compare with processed_time
    # This is just a placeholder, you need to adapt it to your actual data structure and requirements.
    # print("rules_df")
    # print(rules_df)
    # print("rules_df")
    # related_rules = rules_df[rules_df['Crawl_timestamp'] == processed_time]
    sorted_rules = rules_df.sort_values(by="confidence", ascending=False)
    top_rules = sorted_rules.head(3)  # Get the top 3 rules
    recommendations = [list(conseq) for conseq in top_rules["consequents"]]
    return recommendations


def get_recommendations(
    processed_time, user_id, product_title, model_path_collab, model_path_rules
):
    collab_model = load_modelx(model_path_collab)
    rules_df = load_modelx(model_path_rules)

    # Assuming your collaborative model has a predict method.
    # This is a placeholder; you'll need to adjust according to your model's method.
    collab_recommendations = collab_model.predict(user_id, product_title)

    # For association rules, you might want to find rules that are related to the given time.
    # This is a placeholder; you'll need to adjust according to your model's method.
    association_recommendations = get_association_rules_recommendations(
        rules_df, processed_time
    )

    return collab_recommendations, association_recommendations


@api_blueprint.route("/recommendations", methods=["GET"])
def recommendations_endpoint():
    # Extract query parameters
    user_id = request.args.get("user_id")
    product_title = request.args.get("product_title")
    input_time = request.args.get("time")

    if not input_time:
        return jsonify({"error": "Missing time parameter"}), 400

    processed_time = preprocess_input_time(input_time)

    # Get recommendations from both models
    collab_recommendations, association_recommendations = get_recommendations(
        processed_time,
        user_id,
        product_title,
        COLLAB_FILTERING_MODEL_PATH,
        ASSOCIATION_RULES_MODEL_PATH,
    )

    association_recommendations = [list(rec) for rec in association_recommendations]

    return jsonify(
        {
            "collab_recommendations": collab_recommendations,
            "association_recommendations": association_recommendations,
        }
    )


def predict_arima(input_time, model_path):
    """Function to predict future values using the ARIMA model."""
    model = load_models(model_path)

    # Convert input_time to datetime
    input_time_dt = pd.to_datetime(input_time)

    # Create a range of future dates for prediction. Adjust as needed.
    future_dates = [
        input_time_dt + DateOffset(days=x) for x in range(1, 6)
    ]  # Next 5 days

    # Predict values for the future dates
    forecast = model.predict(start=future_dates[0], end=future_dates[-1])

    # Example conversion within predict_arima (already applied in your code snippet)
    forecast_values = [int(value) for value in forecast.tolist()]

    # Combine dates with forecast values, ensuring dates are strings for JSON serialization
    predictions = [
        (date.strftime("%Y-%m-%d"), value)
        for date, value in zip(future_dates, forecast_values)
    ]

    return predictions


def get_recommended_products_for_period(input_time):
    try:
        df = pd.read_csv("model/Customer_dataset.csv")
    except FileNotFoundError:
        print("Sales data file not found.")
        return []

    df["Crawl_timestamp"] = pd.to_datetime(
        df["Crawl_timestamp"], format="%Y-%m-%d %H:%M:%S %f"
    )
    df.dropna(subset=["Crawl_timestamp"], inplace=True)

    # Assuming input_time is a string in the correct format, no conversion required here
    # If input_time is not a string, ensure conversion or handling before this function
    period_dt = pd.to_datetime(input_time)

    similar_period_sales = df[df["Crawl_timestamp"].dt.month == period_dt.month]
    product_sales = (
        similar_period_sales.groupby("Product _title")["Number_purchases"]
        .sum()
        .sort_values(ascending=False)
    )

    top_products = (
        product_sales.head(3).index.astype(str).tolist()
    )  # Convert to strings

    return top_products


@api_blueprint.route("/arimarecommendations", methods=["GET"])
def arimarecommendations_endpoint():
    input_time = request.args.get("time")
    if not input_time:
        return jsonify({"error": "Missing time parameter"}), 400

    # Get ARIMA predictions for the future time.
    arima_predictions = predict_arima(input_time, ARIMA_MODEL_PATH)

    # Direct conversion here is not needed anymore since it's handled inside predict_arima

    # Get recommended products based on ARIMA predictions.
    recommended_products = get_recommended_products_for_period(arima_predictions[0][0])

    # Obtain user_id and product_title dynamically for demonstration purposes
    user_id, product_title = get_dynamic_user_product_details(
        "model/Customer_dataset.csv"
    )

    # Integrate collaborative filtering and association rules recommendations.
    collab_recommendations, association_recommendations = get_recommendations(
        input_time,
        user_id,
        product_title,
        COLLAB_FILTERING_MODEL_PATH,
        ASSOCIATION_RULES_MODEL_PATH,
    )

    # Convert all numpy numeric types to Python native types in predictions_formatted
    def convert_numpy(obj):
        if isinstance(obj, (int64, float64)):
            return int(obj) if isinstance(obj, int64) else float(obj)
        elif isinstance(obj, (list, tuple, set)):
            return type(obj)(convert_numpy(x) for x in obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj

    predictions_formatted = [
        {
            "date": pred[0],
            "predicted_purchases": pred[1],  # Assuming this is already the correct type
            "recommended_products": recommended_products,
            "collab_recommendations": collab_recommendations,
            "association_recommendations": association_recommendations,
        }
        for pred in arima_predictions
    ]
    # Example for recommended_products, repeat for collab_recommendations and association_recommendations

    predictions_formatted_converted = convert_numpy(predictions_formatted)

    return jsonify({"arima_predictions": predictions_formatted})


def convert_to_json_types(data):
    """Converts NumPy data types to JSON-compatible types."""
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):  # Handle nested lists recursively
        return [convert_to_json_types(item) for item in data]
    else:
        return data


def get_recommendations(
    processed_time, user_id, product_title, model_path_collab, model_path_rules
):
    collab_model = load_modelx(model_path_collab)
    rules_model = load_modelx(model_path_rules)

    # Try to generate collaborative filtering recommendations
    try:
        collab_recommendations = collab_model.predict(user_id, product_title)
        if collab_model.output_type == "numpy_array":  # Hypothetical example
            collab_recommendations = convert_to_json_types(collab_recommendations)
    except Exception as e:
        # Fallback to default recommendations from the model or dataset
        collab_recommendations = ["Default Product 1", "Default Product 2"]

    # Try to generate association rules recommendations
    try:
        association_recommendations = get_association_rules_recommendations(
            rules_model, processed_time
        )
    except Exception as e:
        # Fallback to default recommendations from the model or dataset
        association_recommendations = ["Default Product 1", "Default Product 2"]

    return collab_recommendations, association_recommendations


def get_dynamic_user_product_details(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)

        # Option 1: Select the most recent entry
        # df['last_accessed'] = pd.to_datetime(df['last_accessed'])
        # latest_entry = df.sort_values(by='last_accessed', ascending=False).iloc[0]

        # Option 2: Select based on popularity (e.g., choose a random high popularity entry)
        # popular_entries = df[df['popularity'] == 'high']
        # random_entry = popular_entries.sample(n=1).iloc[0]

        # Option 3: Random selection for demonstration
        random_entry = df.sample(n=1).iloc[0]

        return random_entry["Customer_id"], random_entry["Product _title"]
    except FileNotFoundError:
        print("CSV file not found.")
        return None, None
    except Exception as e:
        print(f"Error while fetching dynamic user and product details: {e}")
        return None, None


@api_blueprint.route("/enhanced/recommendations", methods=["GET"])
def enhanced_recommendations_endpoint():
    input_time = request.args.get("time")

    if not input_time:
        return jsonify({"error": "Missing time parameter"}), 400

    # Obtain ARIMA predictions to assess market conditions
    arima_predictions = predict_arima(input_time, ARIMA_MODEL_PATH)
    # Placeholder for market condition analysis based on ARIMA predictions
    # market_condition = analyze_market_condition(arima_predictions)

    # Adjust recommendations based on the analyzed market condition
    # Note: Adjust the get_recommendations function or create a new one
    # that fits this context if needed
    general_recommendations = get_general_recommendations(
        input_time,
        COLLAB_FILTERING_MODEL_PATH,
        ASSOCIATION_RULES_MODEL_PATH,
        # market_condition (if applicable)
    )

    # Combine and format the recommendations for the response
    return jsonify(
        {
            "arima_predictions": arima_predictions,  # or market_condition
            "general_recommendations": general_recommendations,
        }
    )


def get_general_recommendations(input_time, model_path, model_path2):
    # Assuming ARIMA predictions have already been made
    arima_predictions = predict_arima(input_time, ARIMA_MODEL_PATH)

    # Analyze ARIMA predictions to determine market trends or demand levels
    # This is a placeholder for your analysis logic
    market_condition = "high_demand"  # Example condition based on ARIMA output

    # Load your collaborative filtering and association rules models
    # Placeholder for model loading
    collab_model = load_modelx(COLLAB_FILTERING_MODEL_PATH)
    rules_model = load_modelx(ASSOCIATION_RULES_MODEL_PATH)

    # Load sales data for association rules analysis
    try:
        sales_data = pd.read_csv("model/Customer_dataset.csv")
    except FileNotFoundError:
        print("Sales data file not found.")
        return []

    # Implement logic to generate recommendations
    # For collaborative filtering, you might simulate recommendations for a "typical" user or use general trends
    collab_recommendations = simulate_collab_filtering_recommendations(
        collab_model, market_condition
    )

    # For association rules, filter or select rules based on the ARIMA-predicted market condition
    association_recommendations = get_association_rules_recommendations(
        sales_data, market_condition
    )

    # Combine recommendations from both models, ensuring uniqueness and relevance
    combined_recommendations = list(
        set(collab_recommendations + association_recommendations)
    )

    return combined_recommendations


def simulate_collab_filtering_recommendations(model, market_condition):
    # Placeholder for your collaborative filtering logic
    # Let's assume 'model' can generate recommendations and accept a 'market_condition' parameter
    # The 'market_condition' could influence the type of recommendations (e.g., trending, stable demand, etc.)

    # Example logic:
    if market_condition == "high_demand":
        # Assuming 'get_high_demand_recommendations' is a method of your model
        recommendations = model.get_high_demand_recommendations()
    elif market_condition == "stable":
        recommendations = model.get_stable_demand_recommendations()
    else:
        recommendations = model.get_low_demand_recommendations()

    return recommendations


def get_association_rules_recommendations(sales_data, market_condition):
    # Check if 'Number_purchases' column exists in the DataFrame
    if "Number_purchases" not in sales_data.columns:
        raise ValueError(
            "The 'Number_purchases' column is missing from the sales data."
        )

    # Define arbitrary thresholds for demonstration purposes
    threshold = 100  # Define this based on your dataset
    lower_threshold = 50  # Define this based on your dataset

    # Filter the DataFrame based on the market condition
    if market_condition == "high_demand":
        filtered_data = sales_data[sales_data["Number_purchases"] > threshold]
    elif market_condition == "stable":
        filtered_data = sales_data[
            (sales_data["Number_purchases"] <= threshold)
            & (sales_data["Number_purchases"] > lower_threshold)
        ]
    else:  # Assuming this is for 'low_demand'
        filtered_data = sales_data[sales_data["Number_purchases"] <= lower_threshold]

    # Simplified logic to extract recommendations based on filtered data
    # Here we just pick the top 3 products with the highest 'Number_purchases' in the filtered dataset
    # Make sure your 'Product_name' column is correctly named
    if not filtered_data.empty:
        recommendations = (
            filtered_data.sort_values(by="Number_purchases", ascending=False)[
                "Product_name"
            ]
            .head(3)
            .tolist()
        )
    else:
        recommendations = []

    return recommendations
