from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import jwt
import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hailtothevictors'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Create the database and the User table
with app.app_context():
    db.create_all()

# Route to serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling login
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        token = jwt.encode(
            {'username': username, 'exp': datetime.datetime.now() + datetime.timedelta(hours=1)},
            app.config['SECRET_KEY'],
            algorithm='HS256'
        )
        return jsonify({'token': token.decode('utf-8')})

    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/GUI')
def GUI():
    return render_template('.', 'GUI.html')

# Route for handling account creation
@app.route('/api/create-account', methods=['POST'])
def create_account():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if User.query.filter_by(username=username).first():
        return jsonify({'message': 'Username already exists'}), 409

    hashed_password = generate_password_hash(password, method='scrypt')
    new_user = User(username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'Account created successfully'}), 201

@app.route('/api/delete-account', methods=['DELETE'])
def delete_account():
    data = request.get_json()
    username = data.get('username')
    token = data.get('token')

    try: 
        decoded = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        if decoded['username'] == username:
            user = User.query.filter_by(username=username).first()
            if user:
                db.session.delete(user)
                db.session.commit()
                return jsonify({'message': 'Account deleted successfully'}), 200
            else:
                return jsonify({'message': 'User not found'}), 404
        else: return jsonify({'message': 'Token does not match username'}), 403
    except jwt.ExpiredSignatureError:
        return jsonify({'mesage': 'Token expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token'}), 401

if __name__ == '__main__':
    app.run(debug=True)
