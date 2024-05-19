from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from joblib import load
import os
import numpy as np
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField
from wtforms.validators import DataRequired
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')  # Load secret key from .env file

csrf = CSRFProtect(app)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Load the model and vectorizer
model_name = 'nn_spam_classifier_BatchedEmails_0'
vectorizer_name = 'nn_vectorizer_BatchedEmails.joblib_0.joblib'

model = load_model(f'../Model_NN_batched_Ergebnisse/{model_name}.keras')
vectorizer = load(f'../Model_NN_batched_Ergebnisse/{vectorizer_name}')

class EmailForm(FlaskForm):
    subject = StringField('subject', validators=[DataRequired()])
    body = TextAreaField('body', validators=[DataRequired()])

def classify_email(subject, body):
    max_length = 2000
    subject = subject[:max_length]
    body = body[:max_length]
    email_text = f"Subject: {subject}\n\n{body}"
    email_vectorized = vectorizer.transform([email_text]).toarray()
    prediction = model.predict(email_vectorized)
    class_label = 'Ham' if prediction[0][0] > prediction[0][1] else 'Spam'
    return class_label

@app.route('/', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def index():
    form = EmailForm()
    result = None
    if form.validate_on_submit():
        subject = form.subject.data[:2000]
        body = form.body.data[:2000]
        result = classify_email(subject, body)
    return render_template('index.html', form=form, result=result)

@app.errorhandler(500)
def internal_error(error):
    return "Internal server error", 500

@app.errorhandler(404)
def not_found_error(error):
    return "Page not found", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
