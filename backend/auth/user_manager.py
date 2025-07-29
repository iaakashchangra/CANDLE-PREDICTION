import hashlib
import secrets
from datetime import datetime, timedelta
import sqlite3
import json
from typing import Dict, Optional, List
from flask_login import UserMixin
import bcrypt

class User(UserMixin):
    """User model for Flask-Login"""
    
    def __init__(self, id, first_name, last_name, username, email, password_hash, created_at, last_login=None, is_active=True, is_admin=False):
        self.id = id
        self.first_name = first_name
        self.last_name = last_name
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.created_at = created_at
        self.last_login = last_login
        self.is_active = is_active
        self.is_admin = is_admin
    
    def check_password(self, password: str) -> bool:
        """Check if provided password matches user's password"""
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
    
    def to_dict(self) -> Dict:
        """Convert user object to dictionary"""
        return {
            'id': self.id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'last_login': self.last_login.isoformat() if isinstance(self.last_login, datetime) else self.last_login,
            'is_active': self.is_active,
            'is_admin': self.is_admin
        }

class UserManager:
    """Manages user authentication and configuration"""
    
    def __init__(self, db_path: str = 'candles.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                is_admin BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Create user_configs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                api_source TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                prediction_count INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create user_sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create model_performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                accuracy REAL,
                mae REAL,
                rmse REAL,
                mape REAL,
                sharpe_ratio REAL,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def create_user(self, username: str, email: str, password: str, first_name: str = "", last_name: str = "") -> Dict:
        """Create a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
            existing_user = cursor.fetchone()
            if existing_user:
                return {'success': False, 'error': 'Username or email already exists'}
            
            # Hash password
            password_hash = self.hash_password(password)
            
            # Insert new user
            cursor.execute('''
                INSERT INTO users (first_name, last_name, username, email, password_hash)
                VALUES (?, ?, ?, ?, ?)
            ''', (first_name, last_name, username, email, password_hash))
            
            user_id = cursor.lastrowid
            
            # Create default configuration
            cursor.execute('''
                INSERT INTO user_configs (user_id, api_source, symbol, timeframe, prediction_count, model_type)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, 'yahoo', 'AAPL', '1d', 5, 'lstm'))
            
            conn.commit()
            conn.close()
            
            return {'success': True, 'user_id': user_id}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def authenticate_user(self, username: str, password: str) -> Dict:
        """Authenticate user credentials"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, first_name, last_name, username, email, password_hash, created_at, last_login, is_active, is_admin
                FROM users WHERE username = ? AND is_active = TRUE
            ''', (username,))
            
            user_data = cursor.fetchone()
            conn.close()
            
            if not user_data:
                return {'success': False, 'error': 'User not found'}
            
            user = User(*user_data)
            
            if user.check_password(password):
                return {
                    'success': True,
                    'user': user.to_dict()
                }
            else:
                return {'success': False, 'error': 'Invalid password'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, first_name, last_name, username, email, password_hash, created_at, last_login, is_active, is_admin
                FROM users WHERE id = ? AND is_active = TRUE
            ''', (user_id,))
            
            user_data = cursor.fetchone()
            conn.close()
            
            if user_data:
                return User(*user_data)
            return None
            
        except Exception as e:
            print(f"Error getting user by ID: {e}")
            return None
    
    def get_user_config(self, user_id: int) -> Optional[Dict]:
        """Get user configuration"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT api_source, symbol, timeframe, prediction_count, model_type, updated_at
                FROM user_configs WHERE user_id = ?
                ORDER BY updated_at DESC LIMIT 1
            ''', (user_id,))
            
            config_data = cursor.fetchone()
            conn.close()
            
            if config_data:
                return {
                    'api_source': config_data[0],
                    'symbol': config_data[1],
                    'timeframe': config_data[2],
                    'prediction_count': config_data[3],
                    'model_type': config_data[4],
                    'updated_at': config_data[5]
                }
            return None
            
        except Exception as e:
            print(f"Error getting user config: {e}")
            return None
    
    def update_user_config(self, user_id: int, config: Dict) -> Dict:
        """Update user configuration"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Validate configuration
            required_fields = ['api_source', 'symbol', 'timeframe', 'prediction_count', 'model_type']
            for field in required_fields:
                if field not in config:
                    return {'success': False, 'error': f'Missing required field: {field}'}
            
            # Update or insert configuration
            cursor.execute('''
                INSERT OR REPLACE INTO user_configs 
                (user_id, api_source, symbol, timeframe, prediction_count, model_type, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                user_id,
                config['api_source'],
                config['symbol'],
                config['timeframe'],
                config['prediction_count'],
                config['model_type']
            ))
            
            conn.commit()
            conn.close()
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_session(self, user_id: int) -> str:
        """Create a new session token"""
        try:
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(hours=24)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, session_token, expires_at))
            
            conn.commit()
            conn.close()
            
            return session_token
            
        except Exception as e:
            print(f"Error creating session: {e}")
            return None
    
    def validate_session(self, session_token: str) -> Optional[int]:
        """Validate session token and return user ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id FROM user_sessions 
                WHERE session_token = ? AND expires_at > CURRENT_TIMESTAMP
            ''', (session_token,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            print(f"Error validating session: {e}")
            return None
    
    def delete_session(self, session_token: str) -> bool:
        """Delete a session token"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM user_sessions WHERE session_token = ?', (session_token,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM user_sessions WHERE expires_at <= CURRENT_TIMESTAMP')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error cleaning up sessions: {e}")
    
    def get_all_users(self) -> List[Dict]:
        """Get all active users"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, created_at
                FROM users WHERE is_active = TRUE
                ORDER BY created_at DESC
            ''')
            
            users = []
            for row in cursor.fetchall():
                users.append({
                    'id': row[0],
                    'username': row[1],
                    'email': row[2],
                    'created_at': row[3]
                })
            
            conn.close()
            return users
            
        except Exception as e:
            print(f"Error getting all users: {e}")
            return []
    
    def update_model_performance(self, user_id: int, model_type: str, symbol: str, 
                               timeframe: str, metrics: Dict) -> bool:
        """Update model performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO model_performance 
                (user_id, model_type, symbol, timeframe, accuracy, mae, rmse, mape, 
                 sharpe_ratio, total_predictions, correct_predictions, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                user_id, model_type, symbol, timeframe,
                metrics.get('accuracy', 0),
                metrics.get('mae', 0),
                metrics.get('rmse', 0),
                metrics.get('mape', 0),
                metrics.get('sharpe_ratio', 0),
                metrics.get('total_predictions', 0),
                metrics.get('correct_predictions', 0)
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"Error updating model performance: {e}")
            return False
    
    def get_model_performance(self, user_id: int) -> List[Dict]:
        """Get model performance for user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT model_type, symbol, timeframe, accuracy, mae, rmse, mape, 
                       sharpe_ratio, total_predictions, correct_predictions, updated_at
                FROM model_performance 
                WHERE user_id = ?
                ORDER BY updated_at DESC
            ''', (user_id,))
            
            performance = []
            for row in cursor.fetchall():
                performance.append({
                    'model_type': row[0],
                    'symbol': row[1],
                    'timeframe': row[2],
                    'accuracy': row[3],
                    'mae': row[4],
                    'rmse': row[5],
                    'mape': row[6],
                    'sharpe_ratio': row[7],
                    'total_predictions': row[8],
                    'correct_predictions': row[9],
                    'updated_at': row[10]
                })
            
            conn.close()
            return performance
            
        except Exception as e:
            print(f"Error getting model performance: {e}")
            return []
    
    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate a user account"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('UPDATE users SET is_active = FALSE WHERE id = ?', (user_id,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"Error deactivating user: {e}")
            return False
    
    def change_password(self, user_id: int, old_password: str, new_password: str) -> Dict:
        """Change user password"""
        try:
            # First verify old password
            user = self.get_user_by_id(user_id)
            if not user or not user.check_password(old_password):
                return {'success': False, 'error': 'Invalid current password'}
            
            # Hash new password
            new_password_hash = self.hash_password(new_password)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', 
                         (new_password_hash, user_id))
            
            conn.commit()
            conn.close()
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}