import threading
import Queue
import camprofile
from imgproc import imgproc

class camthread(threading.Thread):
    def __init__ (self,qi,qo):
        self.qi=qi
        self.qo=qo
        threading.Thread.__init__ (self)

    def run(self):
        while True:
            if not self.qi.empty():
                framein=self.qi.get()
                if framein is not "q":
                    out=imgproc(framein)
                    if self.qo.empty():
                        self.qo.put(out)
                else:
                    break
