from flask import Flask
from server.config import Config
from flask_cors import CORS

def create_app():
    # Initialize the Flask app
    app = Flask(__name__, static_folder="../static", template_folder="../templates")
    
    # Apply configuration settings
    app.config.from_object(Config)
    
    # Enable CORS if your frontend is served from a different origin
    CORS(app)

    # Import and register the API blueprints
    from server.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app
