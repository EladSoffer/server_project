import base64

import cv2
import numpy as np
from flask import jsonify
from yolo import detect_objects

# Extract fruits from the image with Non-Maximum Suppression
def extract_fruits(image, confidence_threshold=0.5, nms_threshold=0.4):
    outs = detect_objects(image)
    fruits = []
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 60 or class_id == 56 or class_id == 43 or class_id == 42 or class_id == 40:
                continue
            # Check if the class_id corresponds to a fruit class and confidence is above threshold
            #if confidence > confidence_threshold and classes[class_id] in ['apple', 'banana', 'orange', 'guava', 'lime', 'pomegranate']:
            # Extract bounding box coordinates
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])
            x = center_x - w // 2
            y = center_y - h // 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            fruit = image[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]]
            fruits.append((fruit, boxes[i]))  # Store fruit image and bounding box coordinates
    return fruits

def process_img(image):
    # Extract fruits from the image
    fruits_with_boxes = extract_fruits(image)
    for fruit_image, (x, y, w, h) in fruits_with_boxes:
        print("w", w, " h", h, " x", x, " y", y)

        # Draw square around the detected fruit
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Overlay model detection results on the original image
    #image_with_detections = overlay_detections(image.copy(), fruits_with_boxes) pay attention it function for detection
    cv2.imwrite("Processed Image.jpg", image)
    return image

def get_upload(image_data):
    try:

        # Get image data from request
        # Decode base64 encoded image
        image_data = base64.b64decode(image_data)

        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)

        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Process image using YOLO
        processed_image = process_img(image)

        # # Save the processed image
        # cv2.imwrite("processed_image.jpg", processed_image)
        #
        # # Display the processed image
        # cv2.imshow("Processed Image", processed_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Encode processed image to base64
        _, img_encoded = cv2.imencode('.jpg', processed_image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        return img_base64

    except Exception as e:
        print(e)