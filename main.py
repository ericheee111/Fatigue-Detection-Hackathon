from typing import List, Any
import array as arr
import numpy as np
import cv2
import dlib
import time
from scipy.spatial import distance
from playsound import playsound


def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio


def calculate_MAR(mouth):
    A = distance.euclidean(mouth[1], mouth[5])
    B = distance.euclidean(mouth[2], mouth[4])
    C = distance.euclidean(mouth[0], mouth[3])
    mar_aspect_ratio = (A + B) / (2.0 * C)
    return mar_aspect_ratio


cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

eye_thres = 0.23
eye_frame_thres = 3
blink_frame_count = 0
blink_numCount = 0
eyeClose_timer = -1
EAR = 0
init_EAR = np.zeros(10)
doze_counter = 0

mouth_thres = 0.93
mouth_frame_thres = 6
opmo_frame_counter = 0
opmo_counter = 0
mouthOpen_time = -1
MAR = 0
init_MAR = np.zeros(10)
yawn_counter = 0

time.sleep(1)
for i in range(10):
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
        mouth = []

        mp = 48
        while mp < 59:
            x = face_landmarks.part(mp).x
            y = face_landmarks.part(mp).y
            mouth.append((x, y))
            next_point = mp + 2
            if mp == 58:
                next_point = 48
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
            mp += 2

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)
        mouth_mar = calculate_MAR(mouth)

        init_EAR[i] = (left_ear + right_ear) / 2
        init_EAR[i] = round(init_EAR[i], 2)
        init_MAR[i] = round(mouth_mar, 2)

tolE = 0;
tolM = 0;
for i in range(10):
    tolE += init_EAR[i]
    tolM += init_MAR[i]

eye_thres = (tolE / 10) * 0.8
mouth_thres = (tolM / 10) * 1.8

time.sleep(1)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.putText(frame, "Press 'q' to quit.", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 250, 250), 2)

    # draw the total number of blinks on the frame along with
    # the computed eye aspect ratio for the frame
    cv2.putText(frame, "Blinks: {}".format(blink_numCount), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 250, 250), 2)
    cv2.putText(frame, "EAR: {:.2f}".format(EAR), (520, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 250, 250), 2)
    # draw the total number of yawns on the frame along with
    # the computed mouth aspect ratio for the frame
    cv2.putText(frame, "Mouth Opens: {}".format(opmo_counter), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 250, 250), 2)
    cv2.putText(frame, "MAR: {:.2f}".format(MAR), (520, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 250, 250), 2)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
        mouth = []

        mp = 48
        while mp < 59:
            x = face_landmarks.part(mp).x
            y = face_landmarks.part(mp).y
            mouth.append((x, y))
            next_point = mp + 2
            if mp == 58:
                next_point = 48
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
            mp += 2

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)
        mouth_mar = calculate_MAR(mouth)

        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 2)
        MAR = round(mouth_mar, 2)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if EAR < eye_thres:
            blink_frame_count += 1
            if eyeClose_timer == -1:
                eyeClose_timer = time.time()
                print("eye closed")

            if abs(time.time() - eyeClose_timer) >= 2 and eyeClose_timer != -1:
                print("Closed eye for 2 sec.")
                playsound('C:/Users/erich/Desktop/study_things/Hack/Hack/2022111302271.wav')
                eyeClose_timer = -1
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:


            eyeClose_timer = -1
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if blink_frame_count >= eye_frame_thres:
                blink_numCount += 1
            # reset the eye frame counter
            blink_frame_count = 0

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if MAR < mouth_thres:
            opmo_frame_counter += 1
            if mouthOpen_time == -1:
                mouthOpen_time = time.time()
                print("mouth opened")
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            if abs(time.time() - mouthOpen_time) >= 3 and mouthOpen_time != -1:
                print("Yawning.")

            mouthOpen_time = -1
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if opmo_frame_counter >= mouth_frame_thres:
                opmo_counter += 1
            # reset the eye frame counter
            opmo_frame_counter = 0

    cv2.imshow("Are you Sleepy", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
