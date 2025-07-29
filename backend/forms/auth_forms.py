from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Length, Email, EqualTo

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=4, max=25)
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=6)
    ])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    first_name = StringField('First Name', validators=[
        DataRequired(),
        Length(min=2, message='First name must be at least 2 characters')
    ])
    last_name = StringField('Last Name', validators=[
        DataRequired(),
        Length(min=2, message='Last name must be at least 2 characters')
    ])
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=3, max=25, message='Username must be between 3 and 25 characters')
    ])
    email = StringField('Email', validators=[
        DataRequired(),
        Email(message='Please enter a valid email address')
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8, message='Password must be at least 8 characters')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='Passwords must match')
    ])
    terms_accepted = BooleanField('I accept the terms of service', validators=[
        DataRequired(message='You must accept the terms of service')
    ])
    submit = SubmitField('Register')