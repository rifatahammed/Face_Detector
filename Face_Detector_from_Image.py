import cv2

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# reading the image
img = cv2.imread('me.jpg')


grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect face
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# draw rectengle
for(x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# it will print the cordinates of the face
print(face_coordinates)

# this is the image detector frame
cv2.imshow('RIFFFFFs Face Detector', img)
cv2.waitKey()

print('THIS IS YOUR FACE')
