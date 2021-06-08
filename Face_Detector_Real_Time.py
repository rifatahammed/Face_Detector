import cv2

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')


webcam = cv2.VideoCapture(0)
# 0 means the first/default Camera

# loops for get unlimited frames
while True:
    successful_frame_read, frame = webcam.read()
    # it will read the current frame

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draw rectengle
    for(x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('RIFFFFFs Live Face Detector', frame)
    key = cv2.waitKey(1)
    # waitkey 1 means the frame will change after 1 milisecond

    # segment for press q or Q to quit the app
    if key == 81 or key == 113:
        break

webcam.release()

print('THIS IS YOUR FACE')
