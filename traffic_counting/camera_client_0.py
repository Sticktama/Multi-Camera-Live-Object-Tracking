from imutils.video import VideoStream
import imagezmq
import time
import cv2

path = "http://192.168.1.11:81/stream"  # change to your IP stream address
cap = VideoStream(path)

sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')  # change to IP address and port of server thread
cam_id = 'Camera 0'  # this name will be displayed on the corresponding camera stream

stream = cap.start()

while True:
    frame = stream.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    sender.send_image(cam_id, frame)
