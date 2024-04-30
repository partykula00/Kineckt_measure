import freenect
import numpy as np


def get_video():
    # FREENECT_VIDEO_RGB = 0, / ** < Decompressed RGB mode(demosaicing done by libfreenect) * /
    # FREENECT_VIDEO_BAYER = 1, / ** < Bayer compressed mode (raw information from camera) * /
    # FREENECT_VIDEO_IR_8BIT = 2, / ** < 8 - bit IR mode * /
    # FREENECT_VIDEO_IR_10BIT = 3, / ** < 10 - bit IR mode * /
    # FREENECT_VIDEO_IR_10BIT_PACKED = 4, / ** < 10 - bit packed IR mode * /
    # FREENECT_VIDEO_YUV_RGB = 5, / ** < YUV RGB mode * /
    # FREENECT_VIDEO_YUV_RAW = 6, / ** < YUV Raw mode * /
    # FREENECT_VIDEO_DUMMY = 2147483647, / ** < Dummy value to force enum to be 32 bits wide * /

    VIDEO = freenect.sync_get_video()[0][:,:,::-1]

    return VIDEO

def get_depth():
    # FREENECT_DEPTH_11BIT = 0 -> 11 bit depth information in one uint16_t/pixel
    # FREENECT_DEPTH_11BIT = 1 -> 10 bit depth information in one uint16_t pixel
    # FREENECT_DEPTH_11BIT = 2 -> 11 bit packed depth information
    # FREENECT_DEPTH_11BIT = 3 -> 10 bit packed depth infromation
    # FREENECT_DEPTH_11BIT = 4 -> processed depth data in mm, aligned to 640x480 RGB
    # FREENECT_DEPTH_11BIT = 5 -> depth to each pixel in mm, but left unaligned to RGB image
    # FREENECT_DEPTH_11BIT = 2147483647

    DEPTH = freenect.sync_get_depth(format=4)[0]
    return DEPTH

def show_both(pretty = False):
    import cv2
    def mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            value = depth[y, x]  # Wartość piksela na pozycji (x, y)
            cv2.displayStatusBar('Depth Image', f'x: {x}, y: {y}, value: {value}')
            cv2.displayStatusBar('RGB Image', f'x: {x}, y: {y}, value: {value}')

    # Utwórz puste okno
    cv2.namedWindow('Depth Image')
    cv2.setMouseCallback('Depth Image', mouse_event)

    while 1:
        if pretty is True:
            depth = get_depth()
            rgb = get_video()
            np.clip(depth, 0, 2**10 - 1, depth)
            depth >>= 2
            depth = depth.astype(np.uint8)
        else:
            depth = get_depth()
            rgb = get_video()

        cv2.imshow('Depth Image', depth)
        cv2.imshow('RGB Image', rgb)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break


def real_coordinates(coordinates):
    #coordinates is a tuple of (x,y,z), where x,y are coordinates in pixels, z is depth for (y,x)

    coord_real = freenect.sync_camera_to_world(coordinates[1], coordinates[0], coordinates[2])

    return coord_real


