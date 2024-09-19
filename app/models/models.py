from .. import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from . import Backtester
from . import GetData

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    

    def __init__(self, username, password, id):
        self.username = username
        self.password = generate_password_hash(password, method='scrypt')

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def get_id(self):
        return self.id

