import cv2, os
import math, random
import argparse
from utils_display import DisplayHand
from utils_mediapipe import MediaPipeHand
from utils_joint_angle import WristArmRom, GestureRecognition


def camera_a():
    global lxd
    global camera
    if lxd == 1:
        camera = cv2.VideoCapture(0)
        lxd = 2
        return camera
    else:
        return camera

def get_frame(camera):
    ret, frame = camera.read()
    #frame = frame[60:480,60:580]
    return frame


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--side', default='right')
parser.add_argument('-m', '--mode', default=2, help='Select mode: 0: Wrist flex/ext, 1:Wrist radial/ulnar dev, 2:Forearm pronation/supination')
args = parser.parse_args()
mode = int(args.mode)
side = args.side
pipe = MediaPipeHand(static_image_mode=False, max_num_hands=2)
disp = DisplayHand(max_num_hands=2)
rom = WristArmRom(mode, side)
gest = GestureRecognition(mode='eval')
counter = 0
while True:
    img_NEW = get_frame(camera_a())
    img = cv2.flip(img_NEW, 1)
    img.flags.writeable = False
    param = pipe.forward(img)
    for p in param:
        if p['class'] is not None:
            p['gesture'] = gest.eval(p['angle'])
    img.flags.writeable = True
    img_NEW = disp.draw_game_rps(img.copy(), param)
    cv2.imshow('Result', img_NEW)
    key = cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break