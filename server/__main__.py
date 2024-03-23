from flask import Flask
from flask_cors import CORS
from server.config import Config

def create_app():
    # Initialize the Flask app
    app = Flask(__name__)
    print("app started")
    
    # Apply configuration settings
    app.config.from_object(Config)
    
    # Enable CORS for all domains on all routes
    CORS(app)
    
    # Import and register the API blueprints
    from server.routes import api_blueprint
    app.register_blueprint(api_blueprint, url_prefix="/api")
    
    return app

# Add this part to run the app
if __name__ == "__main__":
    app = create_app()
    # Optionally, specify the host and port
    app.run(debug=True, host='0.0.0.0', port=5000)
