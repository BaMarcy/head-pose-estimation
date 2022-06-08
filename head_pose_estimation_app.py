# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 03:09:33 2022

@author: Deepworker
"""

import cv2
import numpy as np
import av
import time
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def process(image):
    face_3d = []
    face_2d = []
    
    image.flags.writeable = False
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    start = time.time()
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = image.shape
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Looking Forward"
                
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
            cv2.line(image, p1, p2, (255, 255, 255), 1)
                
            cv2.putText(image, text, (10, img_h-20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.putText(image, "x: " + str(np.round(x,2)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.putText(image, "y: " + str(np.round(y,2)), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.putText(image, "z: " + str(np.round(z,2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.putText(image, "by Marcell Balogh", (img_w-100, img_h-20), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
            
            end = time.time()
            totalTime = end - start
            try:
                fps = 1 / totalTime
            except:
                pass

            cv2.putText(image, f'FPS: {int(fps)}', (img_w-80, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            
            mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
    return image


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process(img)  
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    
webrtc_ctx = webrtc_streamer(
    key="head pose",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)