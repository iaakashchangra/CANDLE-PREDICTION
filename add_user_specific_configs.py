#!/usr/bin/env python3
"""
Database migration script to add user-specific configuration fields
to the user_selection models.

This script adds:
- default_prediction_count and risk_level to user_selections table
- prediction_count and risk_level to selected_users table
- selected_user_id foreign key to selected_models, selected_companies, and selected_timeframes tables
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
import sqlite3

# Simple database connection approach
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'candlestick_prediction.db')

def run_migration():
    """Run the database migration"""
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return
    
    try:
        print("Starting database migration...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if columns already exist before adding them
        def column_exists(table_name, column_name):
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [column[1] for column in cursor.fetchall()]
            return column_name in columns
        
        # Add new columns to user_selections table
        print("Adding columns to user_selections table...")
        if not column_exists('user_selections', 'default_prediction_count'):
            cursor.execute("ALTER TABLE user_selections ADD COLUMN default_prediction_count INTEGER DEFAULT 5")
        if not column_exists('user_selections', 'risk_level'):
            cursor.execute("ALTER TABLE user_selections ADD COLUMN risk_level VARCHAR(20) DEFAULT 'moderate'")
        
        # Add new columns to selected_users table
        print("Adding columns to selected_users table...")
        if not column_exists('selected_users', 'prediction_count'):
            cursor.execute("ALTER TABLE selected_users ADD COLUMN prediction_count INTEGER DEFAULT 5")
        if not column_exists('selected_users', 'risk_level'):
            cursor.execute("ALTER TABLE selected_users ADD COLUMN risk_level VARCHAR(20) DEFAULT 'moderate'")
        
        # Add selected_user_id foreign key to selected_models table
        print("Adding selected_user_id to selected_models table...")
        if not column_exists('selected_models', 'selected_user_id'):
            cursor.execute("ALTER TABLE selected_models ADD COLUMN selected_user_id INTEGER")
        
        # Add selected_user_id foreign key to selected_companies table
        print("Adding selected_user_id to selected_companies table...")
        if not column_exists('selected_companies', 'selected_user_id'):
            cursor.execute("ALTER TABLE selected_companies ADD COLUMN selected_user_id INTEGER")
        
        # Add selected_user_id foreign key to selected_timeframes table
        print("Adding selected_user_id to selected_timeframes table...")
        if not column_exists('selected_timeframes', 'selected_user_id'):
            cursor.execute("ALTER TABLE selected_timeframes ADD COLUMN selected_user_id INTEGER")
        
        # Commit all changes
        conn.commit()
        conn.close()
        print("Migration completed successfully!")
        
    except Exception as e:
        print(f"Migration failed: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        raise

if __name__ == '__main__':
    run_migration()