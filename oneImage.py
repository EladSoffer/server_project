import base64

import io

import cv2
import numpy as np
from PIL import Image
from tensorboard.compat import tf
from globals import fruit_categories
from yolo import detect_objects

# Load the custom fruit classification model
model = tf.keras.models.load_model('fruit_clas.h5')

# Classify fruit using the custom classification model
def classify_fruit(fruit_image):
    # Preprocess the image for classification
    resized_image = cv2.resize(fruit_image, (100, 100))
    test_image = np.expand_dims(resized_image, axis=0)  # Add batch dimension
    test_image = test_image.astype(np.float32) / 255.0  # Normalize the image data

    # Predict the probabilities for each class
    predictions = model.predict(test_image)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions)
    predicted_class = fruit_categories[predicted_class_index]

    return predicted_class

def get_list_pressed_objects(original_image,decode_image, user_choices):
    detected_objects = detect_objects(decode_image)
    # Convert image data to PIL Image object
    # Decode the base64 encoded image data
    image_data = base64.b64decode(original_image)
    # Convert the binary data to a BytesIO object
    image_io = io.BytesIO(image_data)
    # Open the image using PIL
    pil_image = Image.open(image_io)
    #original_image = Image.open(io.BytesIO(original_image))

    # Get original image width and height
    original_width, original_height = pil_image.size
    # Map user choices to detected objects
    confidence_threshold = 0.5
    nms_threshold = 0.4
    pressed_objects = []
    fruits = []
    class_ids = []
    confidences = []
    boxes = []
    for user_x, user_y in user_choices:

        # Find the corresponding object based on user choice coordinates
        for obj in detected_objects:

            #obj_x, obj_y, obj_width, obj_height = obj['bbox']
                for detection in obj:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == 60 or class_id == 56 or class_id == 43 or class_id == 42 or class_id == 40:
                        continue
                    if class_id == 60 or class_id == 56 or class_id == 43 or class_id == 42 or class_id == 40:
                        continue
                    center_x = int(detection[0] * decode_image.shape[1])#check if x and y flip
                    center_y = int(detection[1] * decode_image.shape[0])
                    w = int(detection[2] * decode_image.shape[1])
                    h = int(detection[3] * decode_image.shape[0])
                    x = center_x - w // 2
                    y = center_y - h // 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        if len(indices) > 0:
            for i in indices.flatten():
                fruit = decode_image[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]]
                fruits.append((fruit, boxes[i]))  # Store fruit image and bounding box coordinates

            for fruit_image, (x, y, w, h) in fruits:
                print("w", w, " h", h, " x", x, " y", y)
                if x <= user_x*original_width <= (x + w) and y <= user_y*original_height <= (y + h):
                    # Add the detected object to the list of pressed objects
                    pressed_objects.append((fruit_image,(x, y, w, h)))
                    break  # Stop searching for other objects once found
    return pressed_objects

def get_oneImage(original_image, user_choices):

    image_data = base64.b64decode(original_image)
    # Convert image data to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    # Decode image
    decode_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pressed_objects = get_list_pressed_objects(original_image, decode_image, user_choices)
    for fruit_image, (x, y, w, h) in pressed_objects:
        # Draw square around the detected fruit
        cv2.rectangle(fruit_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        fruit_label = classify_fruit(fruit_image)
        # Annotate with class label
        cv2.putText(decode_image, fruit_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    _, img_encoded = cv2.imencode('.jpg', decode_image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return img_base64