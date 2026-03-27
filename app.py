from flask import Flask, render_template, request, redirect, session
import tensorflow as tf
import numpy as np
import os
import csv
from datetime import datetime
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
app.secret_key = "secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'blood_cell.h5')
CLASS_FILE = os.path.join(BASE_DIR, 'class_names.txt')
HISTORY_FILE = os.path.join(BASE_DIR, 'history.csv')
USERS_FILE = os.path.join(BASE_DIR, 'users.txt')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create folders/files if not exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Image", "Prediction", "Confidence"])

# Load Model
model = tf.keras.models.load_model(MODEL_PATH)

# Load Class Names
with open(CLASS_FILE) as f:
    class_names = [line.strip() for line in f.readlines()]

# Prediction Function
def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    confidence = float(np.max(prediction))
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]

    if confidence < 0.35:
        class_name = "UNKNOWN"

    return class_name, round(confidence * 100, 2)

# Save History
def save_history(filename, prediction, confidence):
    with open(HISTORY_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), filename, prediction, confidence])

# Login Page
@app.route('/')
def login():
    return render_template('login.html')

# Login Check
@app.route('/login', methods=['POST'])
def login_post():
    username = request.form['username']
    password = request.form['password']

    with open(USERS_FILE, 'r') as f:
        users = f.readlines()

    for user in users:
        parts = user.strip().split(',')
        if len(parts) == 2:
            u, p = parts
            if u == username and p == password:
                session['user'] = username
                return redirect('/home')

    return "Invalid Login"

# Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with open(USERS_FILE, 'a') as f:
            f.write(username + "," + password + "\n")

        return redirect('/')

    return render_template('register.html')

# Home Dashboard
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect('/')

    total = 0
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            total = len(f.readlines()) - 1

    return render_template('home.html', total=total)

# Predict
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect('/')

    file = request.files['image']
    filename = file.filename.replace(" ", "_")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    prediction, confidence = predict_image(filepath)
    save_history(filename, prediction, confidence)

    return render_template('result.html',
                           prediction=prediction,
                           confidence=confidence,
                           filename=filename,
                           username=session['user'])

# History (Admin Only)
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/')

    if session['user'] != 'admin':
        return render_template('not_admin.html')

    data = []
    with open(HISTORY_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append(row)

    return render_template('history.html', data=data)

# About
@app.route('/about')
def about():
    return render_template('about.html')

# Contact
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)