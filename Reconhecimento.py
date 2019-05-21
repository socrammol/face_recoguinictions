
import face_recognition
import cv2

# This is a demo of running face recognition on a video file and saving the results to a new video file.


# Open the input movie file
input_video = cv2.VideoCapture("arquivos/jp1.mp4")
length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

size = (
  int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
  int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')

output_video = cv2.VideoWriter('output.avi', fourcc, 25.07, size)

# Load some sample pictures and learn how to recognize them.
elli_image = face_recognition.load_image_file("arquivos/elli.jpeg")
elli_face_encoding = face_recognition.face_encodings(elli_image)[0]

allan_image = face_recognition.load_image_file("arquivos/allan_grant.jpeg")
alan_face_encoding = face_recognition.face_encodings(allan_image)[0]

ian_image = face_recognition.load_image_file("arquivos/iam_malcon.jpeg")
ian_face_encoding = face_recognition.face_encodings(ian_image)[0]

jhon_image = face_recognition.load_image_file("arquivos/jhon_hammond.jpg")
jhon_face_encoding = face_recognition.face_encodings(jhon_image)[0]

known_faces = [
    elli_face_encoding,
    alan_face_encoding,
    ian_face_encoding,
    jhon_face_encoding

]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_video.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        if match[0]:
            name = "Elli"
        elif match[1]:
            name = "Allan"
        elif match[2]:
            name = "Iam"
        elif match[3]:
            name = "Jhon"

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_video.write(frame)
    cv2.imshow('camera', frame)
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# All done!
input_video.release()
cv2.destroyAllWindows()

