from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import sqlite3
from flask import current_app, g

db = SQLAlchemy()

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    prediction_result = db.Column(db.JSON)
    model_used = db.Column(db.String(50))
    accuracy = db.Column(db.Float)

def get_db_connection():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db(app):
    db.init_app(app)
    app.teardown_appcontext(close_db)
    with app.app_context():
        db.create_all()
    return db