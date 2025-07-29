#!/usr/bin/env python3
"""
Script to create a default admin user for the application
"""

from backend.auth.user_manager import UserManager

def create_admin_user():
    """Create a default admin user using UserManager"""
    user_manager = UserManager()
    
    # Create admin user
    result = user_manager.create_user(
        username='admin',
        email='admin@example.com',
        password='admin123',  # Change this password!
        first_name='Admin',
        last_name='User'
    )
    
    if result['success']:
        print(f"Admin user created successfully!")
        print(f"User ID: {result['user_id']}")
        print(f"Username: admin")
        print(f"Password: admin123")
        print(f"WARNING: Please change the default password!")
    else:
        print(f"Failed to create admin user: {result['error']}")

if __name__ == '__main__':
    create_admin_user()