import serial.tools.list_ports
import serial

import time
import cv2
import numpy as np
from keras.models import load_model

        
# import trained model
model = load_model('model_file_30epochs.h5')

# import Haar Cascade classifier
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# adding labels in data
labels_dict = {0: 'Angry', 1: 'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Unable to open camera")
    exit()

# Capture a photo
ret, frame = cap.read()
if not ret:
    print("Unable to capture photo")
    exit()

# If the frame was captured successfully, save it
if ret:
    cv2.imwrite('image.jpg', frame)

# Release the camera
cap.release()

frame = cv2.imread("image.jpg")
gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray, 1.3, 3)
for x,y,w,h in faces:
    sub_face_img=gray[y:y+h, x:x+w]
    resized=cv2.resize(sub_face_img,(48,48))
    normalize=resized/255.0
    reshaped = np.reshape(normalize, (1, 48, 48, 1))
    result = model.predict(reshaped) 
    label = np.argmax(result, axis=1)[0]
    print(label)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 1)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 2)
    cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255, 255, 255),2)

# Display the resulting image
cv2.imshow('Emotion Detection', frame)
time.sleep(2)

cv2.waitKey(0)

# pyduino code
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
portsList = []

for one in ports:
    portsList.append(str(one))
    print(str(one))

com = input("Select Com Port for Arduino #: ")

serialInst.port = 'COM6'  #port number
serialInst.baudrate = 9600
serialInst.open()
    

if labels_dict[label] == 'Neutral':
    string_to_send = "Neutral"
    
elif labels_dict[label] == 'Happy':
    string_to_send = "Happy"
    
elif labels_dict[label] == 'Surprise':
    string_to_send = "Surprise"

elif labels_dict[label] == 'Sad':
    string_to_send = "Sad"

else:
    string_to_send = "exit"
    
serialInst.write(string_to_send.encode())


cv2.destroyAllWindows()
