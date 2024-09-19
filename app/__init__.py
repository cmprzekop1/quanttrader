from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()

from .models.models import User

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hailtothevictors'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)

    # Set the login view for redirects
    login_manager.login_view = 'routes.login'

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Import and register blueprints
    from .routes import routes
    app.register_blueprint(routes)

    # Create the database and the User table
    with app.app_context():
        db.create_all()

    return app


