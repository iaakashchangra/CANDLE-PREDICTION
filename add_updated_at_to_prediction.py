#!/usr/bin/env python3
"""
Database migration script to add updated_at column to the prediction table.

This script adds:
- updated_at column to prediction table
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        
        # Check if column already exists before adding it
        def column_exists(table_name, column_name):
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [column[1] for column in cursor.fetchall()]
            return column_name in columns
        
        # Add updated_at column to prediction table
        print("Adding updated_at column to prediction table...")
        if not column_exists('prediction', 'updated_at'):
            cursor.execute("ALTER TABLE prediction ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            print("Added updated_at column to prediction table")
        else:
            print("updated_at column already exists in prediction table")
        
        # Commit changes and close connection
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