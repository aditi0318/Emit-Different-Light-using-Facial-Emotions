import serial.tools.list_ports
import serial

import time
import cv2
import numpy as np
from keras.models import load_model

# pyduino code
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
portsList = []

for one in ports:
    portsList.append(str(one))
    print(str(one))

com = input("Select Com Port for Arduino #: ")

for i in range(len(portsList)):
    if portsList[i].startswith("COM" + str(com)):
        use = "COM" + str(com)
        print(use)

serialInst.baudrate = 9600
serialInst.port = use
serialInst.open()

        
# import trained model
model = load_model('model_file_30epochs.h5')

# import Haar Cascade classifier
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# adding labels in data
labels_dict = {0: 'Angry', 1: 'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

frame = cv2.imread("data00.jpg")
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
    
cv2.imshow("Frame",frame)
time.sleep(2)

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


cv2.waitKey(0)
cv2.destroyAllWindows()
