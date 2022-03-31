import numpy as np
import cv2
import time
import statistics

def open_video():
    # import video
    vid = cv2.VideoCapture('video.mp4')
    if (vid.isOpened() == False):
        print("error")
        exit(0)
    return vid

def get_background_img(vid):

    width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)  # float `height`

    video = np.zeros((frames,height,width,3)) # Array filled with zeros
    background = np.zeros((height,width,3)) # Array filled with zeros

    cnt = 0
    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret == False:
            break
        video[cnt] = frame
        cnt+=1

    # for each channel of each pixel, calculate the median across all frames
    for i in range(height):
        for j in range(width):
            for channel in range(3):
                values = []
                for frame in range(frames):
                    values += [video[frame][i][j][channel]]
                median = statistics.median(values)
                background[i][j][channel] = median

    # save image
    cv2.imwrite('back.png', background)
    vid.release()

def subtract_background():
    vid = open_video()
    width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # import background image
    background = cv2.imread('back.png')
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (width,height))

    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret == False:
            break
        subtracted_image = cv2.subtract(frame, background)
        out.write(subtracted_image)

    vid.release()
    out.release()

vid = open_video()
get_background_img(vid)
subtract_background()

