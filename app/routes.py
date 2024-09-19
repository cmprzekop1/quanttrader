from flask import Blueprint, request, jsonify, render_template, redirect, url_for, current_app
from flask_login import login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from .models.models import User
import jwt
import datetime
from . import db

# Create a Blueprint
routes = Blueprint('routes', __name__)

@routes.route('/')
def index():
    return render_template('index.html')

@routes.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        token = jwt.encode(
            {'username': username, 'exp': datetime.datetime.now() + datetime.timedelta(hours=0.25)},
            current_app.config['SECRET_KEY'],
            algorithm='HS256'
        )
        return jsonify({'token': token})

    return jsonify({'message': 'Invalid credentials'}), 401

@routes.route('/GUI')
def GUI():
    return render_template('GUI.html')

@routes.route('/api/create-account', methods=['POST'])
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

@routes.route('/api/delete-account', methods=['DELETE'])
def delete_account():
    data = request.get_json()
    username = data.get('username')
    token = data.get('token')

    try:
        decoded = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        if decoded['username'] == username:
            user = User.query.filter_by(username=username).first()
            if user:
                db.session.delete(user)
                db.session.commit()
                return jsonify({'message': 'Account deleted successfully'}), 200
            else:
                return jsonify({'message': 'User not found'}), 404
        else:
            return jsonify({'message': 'Token does not match username'}), 403
    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token'}), 401

@routes.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('routes.index'))


@routes.route('/submit', methods = ['GET'])
def stock():
    data = request.get_json()
