import numpy as np
import cv2
import camprofile
import cnnclassifyer

kernel = camprofile.kernal
width=camprofile.width
height=camprofile.height
hrange=np.array(range(height/2-camprofile.croph,height/2+camprofile.croph))
wrange=np.array(range(width/2-camprofile.cropw,width/2+camprofile.cropw))

str=["ground","DC votage","DC votage","current source","AC source","resistor","resistor","inductor","capacitor","diode"]
acu=[]

def imgproc(frame):
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.medianBlur(frame,5)
    frame=cv2.bilateralFilter(frame,4,30,30)
    frame=cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    #frame=cv2.dilate(frame, kernel,iterations=1)
    #frame=cv2.erode(frame, kernel,iterations=1)

    #frame, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #frame=cv2.rectangle(frame, (width/2-camprofile.cropw-1,height/2-camprofile.croph-2), (width/2+camprofile.cropw+1,height/2+camprofile.croph+2),(0,255,0))
    #frame=cv2.drawContours(frame, cont,-1, (0,255,0), 1)
    out = np.zeros((height,width,3),np.uint8)
    cv2.line(out, (width/2,height/2-10), (width/2,height/2+10),(0,255,0),2)
    cv2.line(out, (width/2-10,height/2), (width/2+10,height/2),(0,255,0),2)
    for i in range(-2,2):
        for j in range(-2,2):
            l3c=cnnclassifyer.classify(frame[hrange+2*i,:][:,wrange+2*j])
            acu.append(l3c)
            if len(acu)>100:
                acu.pop(0)
    out=cv2.putText(out,str[np.argmax(np.sum(np.array(acu),axis=0))], (width/3,height*3/8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,255))
    #print np.sum(np.array(acu),axis=0)
    return out
