import face_recognition as fr
import cv2
import numpy as np

face_names = []
detected_faces = []

image_path = "replace with path to image"

image = fr.load_image_file(image_path)
Pranav_face_encoding = fr.face_encodings(image)[0]

known_face_encodings = []
known_face_names = []


def face_recog():
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    face_locations = []
    face_encodings = []
    global face_names
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]

        if process_this_frame:
            face_locations = fr.face_locations(rgb_frame)
            face_encodings = fr.face_encodings(rgb_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = fr.compare_faces(known_face_encodings, face_encoding)
                fname = "unknown"
                face_distances = fr.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    fname = known_face_names[int(best_match_index)]
                face_names.append(fname)
        process_this_frame = not process_this_frame
        for (top, right, bottom, left), fname in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, fname, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    return face_names
