"""
Flask Application Factory

Creates and configures the Flask web application.
"""

from flask import Flask
from flask_cors import CORS

from utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def create_app(agent_system) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        agent_system: ContextAwareRetrievalAgent instance

    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    CORS(app)

    # Store agent in app config
    app.config['AGENT'] = agent_system

    # Register routes
    from api.routes import register_routes
    register_routes(app)

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Not found", "code": 404}, 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return {"error": "Internal server error", "code": 500}, 500

    @app.errorhandler(400)
    def bad_request(error):
        return {"error": "Bad request", "code": 400}, 400

    logger.info("Flask application created successfully")

    return app
