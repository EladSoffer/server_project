import os
#from PIL import Image
#import io
#import cv2
#import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

from oneImage import get_oneImage, get_list_pressed_objects
from globals import output_layers,net, fruit_categories
from pymongo import MongoClient
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import numpy as np
import cv2
import base64
import schedule
import time
from email.mime.text import MIMEText
import subprocess
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
from build_model import build_model
from upload import get_upload
from yolo import detect_objects

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database_name']
#images_labels_collection = db['reviews']
emails_collection = db['mails']

# Load the trained model
model = tf.keras.models.load_model('fruit_classifi.h5')
directory = "images"

# Placeholder for your model building and training code

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


# def build_model(emails_collection):
#     # Replace this with your actual model building and training code
#     global fruit_categories
#     retrain_model()
#     send_notification_emails(emails_collection)
#     directory = "images"
#     fruit_categories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
#     fruit_categories = [word.replace("_", " ") for word in fruit_categories]
# Schedule the build_model function to run every Monday at 3:00 AM
schedule.every().second.do(build_model, emails_collection=emails_collection)

# Run the scheduled tasks continuously
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)  # Wait for 60 seconds before checking again
# Load YOLO model
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

# Ensure that net.getUnconnectedOutLayers() returns a list
unconnected_out_layers = net.getUnconnectedOutLayers()
if isinstance(unconnected_out_layers, int):  # If a scalar is returned, convert it to a list
    unconnected_out_layers = [unconnected_out_layers]


for i in unconnected_out_layers:
    output_layers.append(layer_names[i - 1])



# Preprocess the image





#
# # Overlay model detection results on the original image
# def overlay_detections(original_image, fruits_with_boxes):
#     for fruit_image, (x, y, w, h) in fruits_with_boxes:
#         # Draw square around the detected fruit
#         cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         # Classify the fruit using the custom classification model
#         fruit_label = classify_fruit(fruit_image)
#         # Annotate with class label
#         cv2.putText(original_image, fruit_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return original_image

    # dont know for what is it





    # image = cv2.imread("aa.jpg")
    #
    # # Process the image
    # processed_image = process_image(image)
    #
    # # Save the processed image
    # cv2.imwrite("processed_image.jpg", processed_image)
    #
    # # Display the processed image
    # cv2.imshow("Processed Image", processed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# Define a function to make predictions using the model
def predict(image):
    # Preprocess the input image (if needed)
    # Example: resize the image to match the input size expected by the model
    processed_image = cv2.resize(image, (100, 100))

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
#    img_base64 = get_upload(image_data)
#    return jsonify({'processed_image': img_base64}), 200

# @app.route('/oneImage', methods=['POST'])
# def detect_object():
#     # Receive image file
#     image_file = request.files['image']
#
#     # Receive coordinates data (JSON format)
#     coordinates = request.json['coordinates']
#
#     # Process coordinates data
#     for pair in coordinates:
#         x, y = pair
#         # Do something with x, y coordinates
#
#
#    #this image will be of one fruit
#    image_data = request.data
#    # Decode base64 encoded image
#    image_data = base64.b64decode(image_data)
#    # Convert image data to numpy array
#    nparr = np.frombuffer(image_data, np.uint8)
#    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Make predictions using the model
    predictions = predict(image)
    # Find the index of the maximum prediction
    max_index = np.argmax(predictions)

    # Get the corresponding fruit label
    predicted_fruit = fruit_categories[max_index]

    # Return the predicted fruit as JSON response
    return jsonify({'predicted_fruit': predicted_fruit})


# Assuming you have a function to perform object detection using YOLO

# def get_list_pressed_objects(data):
#     # Extract original image data and user choices
#     original_image = data.get('original')
#     user_choices = data.get('user_choices')
#     pressed_objects = get_list_pressed_objecta(original_image, user_choices)
#     return pressed_objects
@app.route('/oneImage', methods=['POST'])
def process_image():
    try:
        # Get JSON data from the request
        data = request.json
        original_image = data.get('original')
        user_choices = data.get('user_choices')
        original_image = get_oneImage(original_image, user_choices)
        # Extract images of pressed objects
        # pressed_object_images = []
        # for obj in pressed_objects:
        #     obj_x, obj_y, obj_width, obj_height = obj['bbox']
        #     pressed_object_image = original_image[obj_y:obj_y+obj_height, obj_x:obj_x+obj_width]
        #     pressed_object_images.append(pressed_object_image)
        return jsonify({"pressed_object_images": original_image}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400

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
    directory = "images/" + right_label
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
    # pressed_objects = get_list_pressed_objects(data)
    # pressed_object_images = []
    # for obj in pressed_objects:
    #     obj_x, obj_y, obj_width, obj_height = obj['bbox']
    #     pressed_object_image = original_image[obj_y:obj_y+obj_height, obj_x:obj_x+obj_width]
    #     pressed_object_images.append(pressed_object_image)
    # labels = data.get('labels')
    # #images_labels_collection.insert_one({'image': data['image'], 'label': data['label']})
    # # Specify the directory where you want to save the image
    #
    # for i in labels:
    #     i = i.replace(" ", "_")
    #     i.removesuffix('es')
    #     i.removesuffix('s')
    #     directory = "../../images/train/{i}"
    #     # If the directory doesn't exist, create it
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     # Generate a unique filename using timestamp
    #     timestamp = int(time.time())  # Current timestamp
    #     filename = f"image_{timestamp}.jpg"
    #     # Save the image to the specified directory
    #     pressed_object_images[i].save(os.path.join(directory, filename))
    # return jsonify({'message': 'Data submitted successfully'}), 200

if __name__ == '__main__':
    # Start the scheduler in a separate thread
    import threading
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)