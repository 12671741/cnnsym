import threading
from imutils.video import WebcamVideoStream
import numpy as np
import imutils
import cv2
import Queue
from camthread import camthread
import camprofile
#import classifier



qin = Queue.Queue()
qout= Queue.Queue()

width=camprofile.width
height=camprofile.height
hrange=np.array(range(height/2-camprofile.croph,height/2+camprofile.croph))
wrange=np.array(range(width/2-camprofile.cropw,width/2+camprofile.cropw))


kind=0
vs = WebcamVideoStream(src=0).start()
myThread = camthread(qin,qout)
myThread.start()
if 0:
    f=open('data/d.txt','w')
d=np.zeros(10,np.uint8)
k=0
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=width)
    if qin.empty():
        frame = cv2.flip(frame, camprofile.flip)
        qin.put(frame)

    if not qout.empty():
        outf = qout.get()
        key=cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            vs.stop()
            qin.put("q")
            break
        elif key & 0xFF == ord('c'):
            cframe=outf[hrange,:][:,wrange]
            fname='data/face'+str(k)+'.jpg'
            cv2.imwrite(fname,cframe)
            print "captured to: ",fname
            k=k+1
        elif key & 0xFF == ord('r'):
            votes=np.zeros((10))
            for i in range(-1,1):
                for j in range(-1,1):
                    cframe=outf[hrange+i,:][:,wrange+j]
                    f1d=np.reshape(cframe,(1024))/255
                    f1d = np.hstack((f1d,1))
                    l3c=cnnclassifyer.classify(f1d)
                    votes[np.argmax(l3c)]+=1

            print votes
            kind = np.argmax(votes)
            kk=''
            for each in l3c:
                kk+=" %.2f "%each
            print kk
            print "it is: "+str(kind)
            cv2.imshow("compimg",cframe)
        cv2.imshow("q: quit c: capture",outf)
