import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import cv2


if 0:
    Y=np.empty((totallen,10))
    X=np.empty((totallen,32,32,1))
    for i in xrange(totallen):
        fname='../data/face'+str(i)+'.jpg'
        img=cv2.imread(fname,0)
        img=img.astype(float)/255
        img=np.reshape(img,(1,32,32,1))
        Y[i]=eval(f[i].replace(" ",","))
        X[i]=img
    X,Y=tflearn.data_utils.shuffle(X,Y)
    np.savez_compressed("data32/X",X=X)
    np.save("data32/Y",Y)
else:
    z=np.load("data32/X.npz")
    namelist = z.zip.namelist()
    z.zip.extract(namelist[0])
    X = np.load(namelist[0], mmap_mode='r+')
    Y=np.load("data32/Y.npy")

totallen=len(Y)

trainsize=int(0.8*totallen)

convnet = input_data(shape=[None, 32, 32, 1], name='input')
#layer1
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
#layer2
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
#layer3
convnet = fully_connected(convnet, 128, activation='relu')
convnet = dropout(convnet, 0.8)
#layer4
convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet,tensorboard_dir='log')

model.load('model/symb.model')
model.fit({'input': X[0:trainsize]}, {'targets': Y[0:trainsize]}, n_epoch=5, validation_set=({'input': X[trainsize:totallen]}, {'targets': Y[trainsize:totallen]}),
    snapshot_step=500, show_metric=True, run_id='symb')

model.save('model/symb.model')
