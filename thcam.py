import threading
from imutils.video import WebcamVideoStream
import numpy as np
import imutils
import cv2
import Queue
from camthread import camthread
import camprofile


qin = Queue.Queue()
qout= Queue.Queue()

width=camprofile.width
height=camprofile.height
hrange=np.array(range(height/2-camprofile.croph,height/2+camprofile.croph))
wrange=np.array(range(width/2-camprofile.cropw,width/2+camprofile.cropw))

vs = WebcamVideoStream(src=0).start()
myThread = camthread(qin,qout)
myThread.start()

outf = np.zeros((height,width,3),np.uint8)
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=width)
    frame = cv2.flip(frame, camprofile.flip)
    if qin.empty():
        qin.put(frame)

    if not qout.empty():
        outf = qout.get()

    key=cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        vs.stop()
        qin.put("q")
        break
    frame=cv2.add(frame,outf)
    cv2.imshow("q: quit",frame)
