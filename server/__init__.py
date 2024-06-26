from flask import Flask
from flask_cors import CORS
from server.config import Config

def create_app():
    # Initialize the Flask app
    app = Flask(__name__)
    
    # Apply configuration settings
    app.config.from_object(Config)
    
    # Enable CORS for all domains on all routes
    # For better security, you can restrict this to the origins you expect
    # e.g., CORS(app, resources={r"/api/*": {"origins": "http://localhost:4200"}})
    CORS(app)

    # Import and register the API blueprints
    from server.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app
