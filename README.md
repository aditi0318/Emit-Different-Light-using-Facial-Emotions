# Emit-Different-Light-using-Facial-Emotions
Detect facial emotions using CNN and haar cascade and then emit different color lights with arduino and LEDs

Files in this project:
haarcascade_frontalface_default - haar cascade xml file for front face detection
model_file_30epochs.h5 - trained model file, created by running main.py
test_img.py - capture image from webcam and detect emotions in image
test_video.py - detecting emotions in video via webcam
testdata.py - giving pre-captured image and detecting emotion
esp32.py - capture image from esp32 cam and detect the emotion
joint_img.py - capture image from webcam, detect emotion and emit the lights
