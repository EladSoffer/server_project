import threading
import tensorflow as tf
from globals import fruit_categories
from pymongo import MongoClient
import smtplib
import numpy as np
import cv2
import base64
import schedule
import time
from email.mime.text import MIMEText
import subprocess
from flask import Flask, request, send_file, jsonify
import os
from build_model import build_model

directory = "train"
fruit_categories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
fruit_categories = [word.replace("_", " ") for word in fruit_categories]


app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['All_emails']
emails_collection = db['mails']

# Load the trained model
input_shape = (128, 128, 3)
model = tf.keras.models.load_model('best_fruit_model_vgg16.keras', compile=False, custom_objects={'input_shape': input_shape})
# directory of the database
directory = "images"

def retrain_model():
    # Run model.py script
    subprocess.run(['python', 'model.py'])

# Function to send notification emails
def send_notification_emails(emails_collection):
    # Retrieve emails from MongoDB
    emails = emails_collection.find()
    for email in emails:
        # Send email to user
        send_email(email)

def send_email(email):
    subject = "Email Subject"
    body = "This is the body of the text message"
    sender = "fruitdetectionapp@gmail.com"
    recipients = [email]
    password = "dsgh doia zynk keof"

    def send_email(subject, body, sender, recipients, password):
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = ', '.join(recipients)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(sender, password)
            smtp_server.sendmail(sender, recipients, msg.as_string())
        print("Message sent!")

    send_email(subject, body, sender, recipients, password)

# Schedule the build_model function to run every week
schedule.every().week.do(build_model, emails_collection=emails_collection)

# Run the scheduled tasks continuously
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(30)  # Wait for 30 seconds before checking again

# Define a function to make predictions using the model
def predict(image):
    processed_image = cv2.resize(image, (128, 128))
    # Convert the image to the format expected by the model
    processed_image = processed_image.astype(np.float32) / 255.0  # Normalize pixel values
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

    # Make predictions using the model
    predictions = model.predict(processed_image)

    return predictions

@app.route('/upload', methods=['POST'])
def upload():
    image_data = request.get_data()
    image_data = base64.b64decode(image_data)
    # Convert image data to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Make predictions using the model
    predictions = predict(image)
    # Find the index of the maximum prediction
    max_index = np.argmax(predictions)
    # Get the corresponding fruit label
    predicted_fruit = fruit_categories[max_index]

    # Return the predicted fruit as JSON response
    return jsonify({'predicted_fruit': predicted_fruit})

# Define route for review submission
@app.route('/review', methods=['POST'])
def submit_review():
    data = request.json
    original_image = data.get('original')
    # Save review data to MongoDB
    if(data.get('email') != None):
        # Create a unique index on the email field
        emails_collection.create_index([('email', 1)], unique=True)
        # Insert a document
        try:
            emails_collection.insert_one({'email': data.get('email')})
            print("Document inserted successfully.")
        except Exception as e:
            pass
    right_label = data.get('rightRec')
    right_label = right_label.replace(" ", "_")
    right_label.removesuffix('es')
    right_label.removesuffix('s')
    directory = "train/" + right_label
    # If the directory doesn't exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Generate a unique filename using timestamp
    timestamp = int(time.time())  # Current timestamp
    filename = f"image_{timestamp}.jpg"
    # Decode base64 and save the image
    with open(os.path.join(directory, filename), 'wb') as f:
        f.write(base64.b64decode(original_image))
    return jsonify({'message': 'Data submitted successfully'}), 200

if __name__ == '__main__':
    # Start the scheduler in a separate thread
        # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)