import requests

import serial.tools.list_ports
import serial

import time
import cv2
import numpy as np
from keras.models import load_model

# Replace with your ESP32-CAM IP address
esp32_ip = "192.168.210.195"

# Send HTTP GET request to turn on flashlight
# url = f"http://192.168.210.195/control?cmd=flash_on"
# response = requests.get(url)
# time.sleep(3)

# Send HTTP GET request to capture image
url = f"http://192.168.210.195/capture"
response = requests.get(url)

# Save image to file
with open("image.jpg", "wb") as file:
    file.write(response.content)

# Send HTTP GET request to turn off flashlight
# url = f"http://192.168.139.195/control?cmd=flash_off"
# response = requests.get(url)

print("Image saved successfully!")

# -----------------------------------------------------------------------------------------

# import trained model
model = load_model('model_file_30epochs.h5')

# import Haar Cascade classifier
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# adding labels in data
labels_dict = {0: 'Angry', 1: 'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

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

cv2.waitKey(0)
# -----------------------------------------------------------------------------------------

# Lit the LED
# pyduino code
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
portsList = []
serialInst.baudrate = 115200
serialInst.port = 'COM6'
serialInst.open()

if labels_dict[label] == 'Angry':
    string_to_send = "Angry"
    
elif labels_dict[label] == 'Happy':
    string_to_send = "Happy"
    
elif labels_dict[label] == 'Surprise':
    string_to_send = "Surprise"

elif labels_dict[label] == 'Sad':
    string_to_send = "Fear"

else:
    string_to_send = "exit"
    
serialInst.write(string_to_send.encode())



cv2.destroyAllWindows()



