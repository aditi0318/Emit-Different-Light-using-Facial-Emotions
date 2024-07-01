import cv2
import numpy as np
from keras.models import load_model

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load pre-trained emotion detection model
model = load_model('model_file_30epochs.h5')

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
faces = face_cascade.detectMultiScale(gray, 1.3, 3)
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
    # cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), -1) #covering face
    cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255, 255, 255),2)
    
# Display the resulting image
cv2.imshow('Emotion Detection', frame)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()