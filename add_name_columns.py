import sqlite3
import os

def add_missing_columns():
    """Add missing columns to the users table if they don't exist"""
    try:
        # Connect to the database
        conn = sqlite3.connect('candles.db')
        cursor = conn.cursor()
        
        # Check if columns exist
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add columns if they don't exist
        if 'first_name' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN first_name TEXT DEFAULT '' NOT NULL")
            print("Added first_name column to users table")
        
        if 'last_name' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN last_name TEXT DEFAULT '' NOT NULL")
            print("Added last_name column to users table")
        
        if 'last_login' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN last_login TIMESTAMP DEFAULT NULL")
            print("Added last_login column to users table")
        
        if 'is_admin' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE")
            print("Added is_admin column to users table")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        print("Database migration completed successfully")
        return True
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        return False

if __name__ == "__main__":
    add_missing_columns()