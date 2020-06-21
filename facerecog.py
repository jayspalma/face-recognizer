from cv2 import cv2
import face_recognition
import os

# Add the full file path for the following directory
# faceDir is for the training images
# unknownDirectory for the image you want to find the trained faces
# savedDir place where the processed images will be saved
####Directories####
faceDir = "/home/jtpalma/Desktop/face_recog/faces"
unknownDirectory = "/home/jtpalma/Desktop/face_recog/unknown"
savedDir = "/home/jtpalma/Desktop/face_recog/saved"

# Declare variables
font = cv2.FONT_HERSHEY_SIMPLEX
Encodings = []
Names = []

for face in os.listdir(faceDir):
    faceTest = os.path.join(faceDir, face)
    # print(faceTest)
    faceFilename = face.split(".")
    faceFilename[0]
    print("Training model with", faceFilename[0])
    Names.append(faceFilename[0])

    loadFace = face_recognition.load_image_file(faceTest)
    faceFilename[0] = face_recognition.face_encodings(loadFace)[0]
    Encodings.append(faceFilename[0])


for image_filename in os.listdir(unknownDirectory):
    print("Processing ", image_filename)
    full_image_dir = os.path.join(unknownDirectory, image_filename)

    testImage = face_recognition.load_image_file(full_image_dir)
    facePositions = face_recognition.face_locations(testImage)
    allEncodings = face_recognition.face_encodings(testImage, facePositions)

    testImage = cv2.cvtColor(testImage, cv2.COLOR_RGB2BGR)

    for (top, right, bottom, left), face_encoding in zip(facePositions, allEncodings):
        name = "Unknown Person"
        matches = face_recognition.compare_faces(Encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            name = Names[first_match_index]

        cv2.rectangle(testImage, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(testImage, name, (left, top - 8), font, 0.75, (0, 255, 255), 1)

    fullSavedDir = os.path.join(savedDir, image_filename)
    cv2.imwrite(fullSavedDir, testImage)


print("Finished Processing Images. Processed images are saved on", savedDir)

