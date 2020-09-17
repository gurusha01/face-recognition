import face_recognition
import os
import cv2

KF="known_faces"
UKF="unknown_faces"
TOL=0.6
frame_thickness=3 #pixel value
font_thickness=2
MODEL="cnn" #or hog

print("loading known faces")

known_faces=[]
known_names=[]

def name_to_color(name):
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color

for name in os.listdir(KF):
    for filename in os.listdir(f"{KF}/{name}"):
        image=face_recognition.load_image_file(f"{KF}/{name}/{filename}")
        encoding=face_recognition.face_encodings(image)[0]
        known_faces.append(encoding) 
        known_names.append(name)

print("processing unknown faces")
for filename in os.listdir(UKF):
    image = face_recognition.load_image_file(f'{UKF}/{filename}')

    locations=face_recognition.face_locations(image,model=MODEL)

    encodings=face_recognition.face_encodings(image,locations)

    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings,locations):

        results=face_recognition.compare_faces(known_faces,face_encoding,TOL)

        match=None
        print(results)
        if True in results:

            match=known_names[results.index(True)]
            print(f"match found")

            #cv2 part begins
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = name_to_color(match)
            cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), font_thickness)

    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)

