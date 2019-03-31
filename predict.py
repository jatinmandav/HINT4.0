from training.model import CNNModel
from keras.models import load_model
import numpy as np
import cv2
from pyagender import PyAgender
import time
import os



def load_model_emotion(path):
    model = CNNModel(input_shape=(48, 48, 3), classes=7)
    model = model.build_model()
    model.load_weights('weights/model.h5')
    return model

def predict_emotion(gray, x, y, w, h, timestamp, gen, age):
    face = np.resize(gray[y:y+w, x:x+h], (48, 48))
    img = np.zeros((48, 48, 3))
    img[:,:,0] = face
    img[:,:,1] = face
    img[:,:,2] = face
    face = img[:]
    #face = np.expand_dims(np.expand_dims(np.resize(gray[y:y+w, x:x+h]/255.0, (48, 48)),-1), 0)
    face = np.reshape(face, (1, 48, 48, 3))
    prediction = model.predict([face])[0]
    index = int(np.argmax(prediction))

    with open('emotion_recorded.txt', 'a') as f:
        f.write(','.join([str(val) for val in prediction]))
        f.write(',{},{},{}\n'.format(timestamp, gen, age))

    '''prediction[index] = 0

    for i in range(len(prediction)):
        print(emotion_dict[i], ': ', prediction[i])
'''
    return(int(np.argmax(prediction)), round(max(prediction)*100, 2))

path = "Models/"
model = load_model_emotion(path)

fcc_path = "FaceDetector/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(fcc_path)
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Surprise", 6: "Sad"}
colour_cycle = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (230, 230, 250))
webcam = cv2.VideoCapture(0)
agender = PyAgender()

if os.path.exists('emotion_recorded.txt'):
    os.remove('emotion_recorded.txt')

start = time.time()
while True:
    ret, frame = webcam.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    faces = agender.detect_genders_ages(frame)

    from collections import deque
    queue = deque([0.5,0.5,0.5,0.5,0.5])
    if faces:
        gen=faces[0]['gender']
        age=faces[0]['age']
        queue.append(gen)
        queue.popleft()
        if (np.average(queue)<0.50):
            gen = 'Male'
        else:
            gen = 'Female'
        age = age - 5
    else:
        print("searching for face")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))

    for (count,(x, y, w, h)) in enumerate(faces):
        colour = colour_cycle[int(count%len(colour_cycle))]
        cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
        cv2.line(frame, (x+5, y+h+5),(x+100, y+h+5), colour, 20)
        cv2.putText(frame, "Face #"+str(count+1), (x+5, y+h+11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

        cv2.line(frame, (x+8, y),(x+150, y), colour, 20)
        emotion_id, confidence = predict_emotion(gray, x, y, w, h, (time.time() - start), gen, int(age))

        emotion = emotion_dict[emotion_id]

        cv2.putText(frame, emotion + ": " + str(confidence) + "%" , (x+20, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

    cv2.imshow('Emotion Recognition - Press q to exit.', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

webcam.release()
cv2.destroyAllWindows()
