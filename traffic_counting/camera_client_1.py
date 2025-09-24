from imutils.video import VideoStream
import imagezmq
import time

# Use 0 for default webcam (change to 1, 2... if you have multiple cameras)
cap = VideoStream(src=0)

sender = imagezmq.ImageSender(connect_to='tcp://localhost:5566')  # change to server IP:port
cam_id = 'Webcam '  # identifier for this camera

stream = cap.start()
time.sleep(2.0)  # let the camera warm up

while True:
    frame = stream.read()
    sender.send_image(cam_id, frame)
