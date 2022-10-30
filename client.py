from pickle import FALSE
import cv2
import numpy as np
import socket
import struct
from io import BytesIO
from bs4 import BeautifulSoup as bs
import os
import re
import pickle
import mediapipe as mp
import warnings
import pandas as pd
import json

from Const import const
import Dictionary
import Draw

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

MODEL = 'model/ActionV7.pkl'
with open(MODEL, 'rb') as f:
    model = pickle.load(f)

def IncreaseNum(increaseNum):
    increaseNum += 1
    return increaseNum

def ActionPerformed(prev,cur):
    if (prev == const.SQUAT_STRING and cur == const.STAND_STRING):
        return const.SQUAT_STRING
    
    elif (prev == const.LUNGE_STRING and cur == const.STAND_STRING):
        return const.LUNGE_STRING

    elif (prev == const.PUSHUP_STRING and cur == const.LYINGE_STRING):
        return const.PUSHUP_STRING

    else:
        return const.NONE_STRING

# Capture frame
cap = cv2.VideoCapture("full_test.avi")

# http://127.0.0.1:8000
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 8000))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    NumSquat,NumLunge,NumPushup = 0,0,0
    dict = Dictionary.initDict()

    with open("data.json", "r+") as jsonFile:
        data = json.load(jsonFile)

        data["Squat"] = str(NumSquat)
        data["Lunge"] = str(NumLunge)
        data["Pushup"] = str(NumPushup)

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5) as pose:
        cur = "None"
        # while을 돌며 계속 Webcam feed를 받음
        while cap.isOpened():
            _, frame = cap.read()

            results = pose.process(frame)

            frame.flags.writeable = True   
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = pose.process(frame)

            try:
                frame = Draw.DrawSkeleton(frame,results.pose_landmarks.landmark,(203, 192, 255))
            except:
                pass

            if results.pose_world_landmarks:
                # coordinate inference
                row = list(np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten())
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                cur = body_language_class

                # Eval count
                doAction = ""
                if (cur == const.STAND_STRING or cur == const.LYINGE_STRING):
                    doAction = Dictionary.EvaluateDictAction(dict,cur)
                    dict = Dictionary.initDict()
                else:
                    Dictionary.IncreaseDict(dict,cur)
                
                if (doAction == const.SQUAT_STRING):
                    NumSquat = IncreaseNum(NumSquat)

                elif (doAction == const.LUNGE_STRING):
                    NumLunge = IncreaseNum(NumLunge)

                elif (doAction == const.PUSHUP_STRING):
                    NumPushup = IncreaseNum(NumPushup)
                # 현재 행동이 무슨 행동인지에 따라 local에 있는 json file을 바꿈
                with open("data.json", "r+") as jsonFile:
                    data = json.load(jsonFile)

                    data["Action"] = cur
                    data["Squat"] = str(NumSquat)
                    data["Lunge"] = str(NumLunge)
                    data["Pushup"] = str(NumPushup)

                    jsonFile.seek(0)  # rewind
                    json.dump(data, jsonFile)
                    jsonFile.truncate()
                    
            # webcam을 socket에 담아 보냄
            memfile = BytesIO()
            np.save(memfile, frame)
            memfile.seek(0)
            data = memfile.read()

            # Send form byte array: frame size + frame content
            client_socket.sendall(struct.pack("L", len(data)) + data)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()