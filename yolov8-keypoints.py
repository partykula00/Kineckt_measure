import pandas as pd
from ultralytics import YOLO
from freenect import sync_camera_to_world
from utils import kineckt
import cv2
import torch
import math

import os

keypoints = {
    0: 'NOS',
    1: "LEWE OKO",
    2: "PRAWE OKO",
    3: "LEWE UCHO",
    4: "PRAWE UCHO",
    5: "LEWE RAMIE",
    6: "PRAWE RAMIE",
    7: "LEWY ŁOKIEĆ",
    8: "PRAWY ŁOKIEĆ",
    9: "LEWY NADGARSTEK",
    10: "PRAWY NADGARSTEK",
    11: "LEWE BIODRO",
    12: "PRAWE BIODRO",
    13: "LEWE KOLANO",
    14: "PRAWE KOLANO",
    15: "LEWA PIĘTA",
    16: "PRAWA PIĘTA"
}
sizes_ids = {(5, 6): 'BARKI',
         (11, 12): 'TUŁÓW',
         (5, 9): 'RĘKAW LEWY',
         (6, 10): 'RĘKAW PRAWY',
         (11, 15): 'NOGA LEWA',
         (12, 16): 'NOGA PRAWA'
}

sizes_mms = {(5, 6): 0,
             (11, 12): 0,
             (5, 9): 0,
             (6, 10): 0,
             (11, 15): 0,
             (12, 16): 0}


def calculate_distance(real_xyz1, real_xyz2):

    #CALCULATING DISTANCE IN 3D SPACE

    dx = abs(real_xyz1[0][0] - real_xyz2[0][0])
    dy = abs(real_xyz1[0][1] - real_xyz2[0][1])
    dz = abs(real_xyz1[1] - real_xyz2[1])

    distance_mm = math.sqrt(dx ** 2 + dy ** 2 + dz * 2)

    return distance_mm


df = pd.DataFrame(columns=list(sizes_ids.values()))
model = YOLO('yolov8s-pose.pt')
classes = [0]

while 1:
    DEPTH = kineckt.get_depth()
 
    VIDEO = kineckt.get_video()

    results = model.predict(VIDEO, conf=0.7, classes=classes, verbose=False)
    result_pic = results[0].plot()

    keypoints = results[0].keypoints.xy
    keypoints = keypoints.squeeze().cpu()

    if keypoints.numel() > 0:
        idx_list = torch.nonzero(keypoints[:, 0]).squeeze().tolist()

        if  isinstance(idx_list, list) and len(idx_list) >= 2:
            keys = sizes_ids.keys()
            for key in keys:
                if all(i in idx_list for i in key):
                    coordinates1 = (int(round(keypoints[key[0], 0].item())), int(round(keypoints[key[0], 1].item())))
                    coordinates2 = (int(round(keypoints[key[1], 0].item())), int(round(keypoints[key[1], 1].item())))
                    real_coordinates1 = sync_camera_to_world(coordinates1[0], coordinates1[1], DEPTH[coordinates1[1], coordinates1[0]])
                    real_coordinates2 = sync_camera_to_world(coordinates2[0], coordinates2[1], DEPTH[coordinates2[1], coordinates2[0]])

                    real_coordinates1 = (real_coordinates1, DEPTH[coordinates1[1], coordinates1[0]])
                    real_coordinates2 = (real_coordinates2, DEPTH[coordinates2[1], coordinates2[0]])

                    distance = calculate_distance(real_coordinates1, real_coordinates2)
                    sizes_mms[key] = distance


    cv2.imshow('VIDEO',result_pic)
    if cv2.waitKey(10) == ord('q'):
        break

print(sizes_mms)
measurement(sizes_mms)
