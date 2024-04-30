import cv2
import numpy as np
from ultralytics import YOLO
from freenect import sync_get_video, sync_get_depth, sync_camera_to_world

def change_class(model, i: int):
    class_id = model.names[i]

def predict(model, classes: list, frame):
    results = model.predict(frame, conf=0.7, classes=classes, verbose=False)
    results = results[0].cpu()

    mask = results.masks
    mask_array = np.array([])

    if mask:

        mask_coordinates = mask.xy[0]
        for coord in mask_coordinates:

            mask_array = np.append(mask_array, coord)


        #WSPÓŁRZĘDNE GRANICY MASKI:
        mask_array = mask_array.reshape(-1,2).astype(np.int32)

        mask_x = mask_array[:, 0]
        mask_y = mask_array[:, 1]

        height = 480
        width = 640

        mask = np.zeros((height, width), dtype=np.uint8)

        image = results.plot()

        cv2.fillPoly(mask, [mask_array.reshape((-1, 1, 2))], 255)


        masked_region_rgb = cv2.bitwise_and(frame, frame, mask=mask)
    else:
        image = results.plot()
        height = 480
        width = 640
        mask = np.zeros((height, width), dtype=np.uint8)
        masked_region_rgb = np.zeros((height, width), dtype=np.uint8)
        mask_x = None
        mask_y = None

    return image, mask, masked_region_rgb, mask_array, mask_x, mask_y

def pointcloud(depth, mask):
    if mask_x is None:
        masked_region_depth = np.zeros((480, 640), dtype=np.uint8)
    else:
        masked_region_depth = cv2.bitwise_and(depth, depth, mask=mask)

    return masked_region_depth

def pixels_to_world(depth, mask_array):
    
    for coord in mask_array:
        z = depth[coord[1], coord[0]]
        wx,wy = sync_camera_to_world(coord[0], coord[1], z)


model = YOLO('yolov8n-seg.pt')
print(model.names)
classes = [41]

while 1:
    frame = sync_get_video()[0][:,:,::-1]
    depth = sync_get_depth(format=4)[0]

    image_result, mask, masked_region, mask_array, mask_x, mask_y = predict(model,classes, frame)

    pixels_to_world(depth, mask_array)

    cv2.imshow('VIDEO', image_result)

    cv2.imshow('Masked region', masked_region)

    if cv2.waitKey(10) == ord('q'):
        break