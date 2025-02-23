import sys
import os
import glob
import re
import numpy as np

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from flask import Flask , render_template , request , url_for
import pickle

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import img_to_array

import math
import matplotlib.pyplot as plt

from matplotlib.image import imread
import cv2
from PIL import Image

app = Flask('__name__')


###############################--- home page / Index page --#################################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sever_home')
def sever_home():
    return render_template('index.html')

@app.route('/emotion')
def emotion():
    return render_template('emotion.html')

@app.route('/eyes')
def eyes():
    return render_template('eyes.html')

@app.route('/integrated')
def integrated():
    return render_template('integrated.html')

###################################################################################



model_path1 = 'Model/model.h5'

emotion_model = load_model(model_path1)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route("/emotion_predict", methods=["GET", "POST"])
def emotion_predict():
    if request.method == 'POST':
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Normalize and preprocess the ROI for emotion prediction
        image = Image.open(f)
        image = image.resize((48, 48)) 
        img = np.array(image).astype('float') / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Predict the emotion from the ROI
        prediction = emotion_model.predict(img)[0]
        emotion_label = emotion_labels[prediction.argmax()]

        l1 = 100 * prediction[0]
        l2 = 100 * prediction[1]
        l3 = 100 * prediction[2]
        l4 = 100 * prediction[3]
        l5 = 100 * prediction[4]
        l6 = 100 * prediction[5]
        l7 = 100 * prediction[6]

        l1f = "{:.9f}%".format(l1)
        l2f = "{:.9f}%".format(l2)
        l3f = "{:.9f}%".format(l3)
        l4f = "{:.9f}%".format(l4)
        l5f = "{:.9f}%".format(l5)
        l6f = "{:.9f}%".format(l6)
        l7f = "{:.9f}%".format(l7)

        p1 = "\n  : " + str(l1f)
        p2 = "\n  : " + str(l2f)
        p3 = "\n  : " + str(l3f)
        p4 = "\n  : " + str(l4f)
        p5 = "\n  : " + str(l5f)
        p6 = "\n  : " + str(l6f)
        p7 = "\n  : " + str(l7f)

        result = 'The predicted emotion is \'' + emotion_label + '\''

        return render_template("result1.html", result=result, p1=p1, p2=p2, p3=p3, p4=p4, p5=p5, p6=p6, p7=p7)

###################################################################################



model_path2 = 'Model/drowiness_new7.h5'

eye_model = load_model(model_path2)

drowsiness_classes = ["", "", "Closed", "Open"]

@app.route("/eyes_predict", methods=["GET", "POST"])
def eyes_predict():
    if request.method == 'POST':
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Normalize and preprocess the ROI for emotion prediction
        image = Image.open(f)
        image = image.resize((145, 145)) 
        img = np.array(image).astype('float') / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Predict the emotion from the ROI
        prediction = eye_model.predict(img)[0]
        eye_label = drowsiness_classes[prediction.argmax()]



        l3 = 100 * prediction[2]
        l4 = 100 * prediction[3]

        l3f = "{:.9f}%".format(l3)
        l4f = "{:.9f}%".format(l4)

        p3 = "\n  : " + str(l3f)
        p4 = "\n  : " + str(l4f)

        result = 'The predicted emotion is \'' + eye_label + '\''

        return render_template("result2.html", result=result, p3=p3, p4=p4)


###################################################################################

from twilio.rest import Client

# Your Twilio Account SID and Auth Token
account_sid = 'AC2c0b63974ee82a8ad9f113f0799412b2'
auth_token = '5f4f0062b29a07addc4bf5e35156f076'

# Initialize Twilio client
client = Client(account_sid, auth_token)

def send_message():


    # Send a message using Twilio
    message = client.messages.create(
        body="You are falling Asleep, But be awake!.",
        from_='+12164522229',
        to='+916305201877'   
    )


    return "Send successfully"


@app.route("/image_recognition", methods=["GET", "POST"])
def image_recognition():
    if request.method == 'POST':

        import cv2
        import numpy as np
        from keras.models import load_model
        from tensorflow.keras.preprocessing.image import img_to_array
        from threading import Thread

        import pygame
        # Initialize Pygame mixer for sound playback
        pygame.mixer.init()

        # Load the drowsiness detection model
        drowsiness_model = load_model("Model/drowiness_new7.h5")

        # Load the computer vision pre-defined files
        face_cascade = cv2.CascadeClassifier("Model/data/haarcascade_frontalface_default.xml")
        left_eye_cascade = cv2.CascadeClassifier("Model/data/haarcascade_lefteye_2splits.xml")
        right_eye_cascade = cv2.CascadeClassifier("Model/data/haarcascade_righteye_2splits.xml")

        # Load the pre-trained face and emotion models
        face_classifier = cv2.CascadeClassifier(r'Model/data/haarcascade_frontalface_default.xml')
        emotion_classifier = load_model(r'Model/model.h5')

        # Define emotion labels
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

        # Create a window to display the video feed
        cv2.namedWindow('Combined Detector')

        # Define a function to play the alarm sound
        def start_alarm(sound):
            pygame.mixer.music.load(sound)
            pygame.mixer.music.play()

        # Open the default camera (usually the built-in webcam)
        cap = cv2.VideoCapture(0)

        # Initialize variables for drowsiness detection
        drowsiness_count = 0
        drowsiness_alarm_on = False
        drowsiness_alarm_sound = "Model/data/alarm.mp3"
        drowsiness_status1=0
        drowsiness_status2=0

        # Initialize variables for sad emotion detection
        sad_count = 0
        sad_alarm_on = False
        sad_alarm_sound = "Model/data/Happy.mp3"

        while True:
            # Read a frame from the camera
            _, frame = cap.read()
            height, width, _ = frame.shape

            # Convert the frame to grayscale for face and eye detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                # Detect left and right eyes within the detected face region
                left_eye = left_eye_cascade.detectMultiScale(roi_gray)
                right_eye = right_eye_cascade.detectMultiScale(roi_gray)

                for (x1, y1, w1, h1) in left_eye:
                    cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)

                    # Extract and preprocess the left eye image
                    eye1 = roi_color[y1:y1+h1, x1:x1+w1]
                    eye1 = cv2.resize(eye1, (145, 145))
                    eye1 = eye1.astype('float') / 255.0
                    eye1 = img_to_array(eye1)
                    eye1 = np.expand_dims(eye1, axis=0)

                    # Predict the drowsiness status for the left eye
                    pred1 = drowsiness_model.predict(eye1)
                    drowsiness_status1 = np.argmax(pred1)
                    break

                for (x2, y2, w2, h2) in right_eye:
                    cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)

                    # Extract and preprocess the right eye image
                    eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
                    eye2 = cv2.resize(eye2, (145, 145))
                    eye2 = eye2.astype('float') / 255.0
                    eye2 = img_to_array(eye2)
                    eye2 = np.expand_dims(eye2, axis=0)

                    # Predict the drowsiness status for the right eye
                    pred2 = drowsiness_model.predict(eye2)
                    drowsiness_status2 = np.argmax(pred2)
                    break

                # If both eyes are closed, start counting
                if drowsiness_status1 == 2 or drowsiness_status2 == 2:
                    drowsiness_count += 1
                    cv2.putText(frame,"Eyes Closed,count:"+str(drowsiness_count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

                    # If eyes are closed for 2 consecutive frames, start the alarm
                    if drowsiness_count >= 5 and not drowsiness_alarm_on:
                        drowsiness_alarm_on = True

                        # Play the drowsiness alarm sound in a new thread
                        t = Thread(target=start_alarm, args=(drowsiness_alarm_sound,))
                        t.daemon = True
                        t.start()


                        msg = send_message()
                        print(msg)

                else:
                    cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                    drowsiness_count = 0
                    drowsiness_alarm_on = False

            # Read a frame for emotion detection

            labels = []

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

                # Extract the region of interest (ROI) and resize it for emotion detection
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    # Normalize and preprocess the ROI for emotion prediction
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # Predict the emotion from the ROI
                    prediction = emotion_classifier.predict(roi)[0]
                    emotion_label = emotion_labels[prediction.argmax()]
                    label_position = (x, y)

                    # Display the predicted emotion label on the frame
                    cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    # Display a message if no faces are detected
                    cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # If the detected emotion is 'Sad', start counting
                if emotion_label == 'Sad':
                    sad_count += 1
                    cv2.putText(frame, "Emotion,count:"+str(sad_count), (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

                    # If 'Sad' emotion is detected for 3 consecutive frames, start the alarm
                    if sad_count >= 5 and not sad_alarm_on:
                        sad_alarm_on = True

                        # Play the sad emotion alarm sound in a new thread
                        t = Thread(target=start_alarm, args=(sad_alarm_sound,))
                        t.daemon = True
                        t.start()
                else:
                    sad_count = 0
                    sad_alarm_on = False

            # Display the frame with both drowsiness and emotion detection information
            cv2.imshow('Combined Detector', frame)

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close the OpenCV window
        cap.release()
        cv2.destroyAllWindows()

        return render_template("integrated.html")

###################################################################################

if __name__ == "__main__":
    app.run(debug = True)