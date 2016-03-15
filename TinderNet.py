import os
os.environ['GLOG_minloglevel'] = '2' 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import caffe
import glob
import pynder
import sys
import time

fb_id = "1438695358"
fb_auth = "CAAGm0PX4ZCpsBAHKzF9Wi8IpfsWHJfAnwLloLKJH5nRNU8wMbX4ZAeLg5R18Vd5RfWUY5ZAli7DPEIZChadV5ELZBfigucF3HFPnZBRqPlD153p6tdDot3CKGIjJ5haiCc1dSuXEFmyYRsIBJMvyNKXT4nFB5syzEWA2LVS85x3k35ZAyCP7b3B2M39fF7ZBB7tUZBlzveedSf8LC8YY653ZCIUhJTXEegxVEZD"

def initNet(iter, batchSize=1):
    """
    Instalizes the network by reading in the setup
    and the weights and input data size
    :param iter: the iteration of weights to users
    :param batchSize: the number of images run through at a time
    """
    nets = ["deploy.prototxt", "net_iter_{}.caffemodel".format(iter)]
    net = caffe.Net(nets[0], nets[1], caffe.TRAIN) #TEST means no dropout
    net.blobs['data'].reshape(batchSize, 3, 227, 227)
    return net

def initTransform(mean=None):
    """
    Initializes the data transformer to feed input images
    :param mean: the mean of each color channel to be subtracted
    """
    if mean is None:
        mean = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # Bands go first into caffe
    transformer.set_mean('data', mean)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))  # Reverse BGR to RGB
    return transformer

def processImg(img, net, tform):
    """
    Runs an image through the network
    :param img: the input image
    :param net: the network to use
    :param tfor: the image transformer
    """
    img = img[:, :, ::-1]
    timg = tform.preprocess('data', img)
    net.blobs['data'].data[...] = timg
    output = net.forward()
    prob = output['prob'][0]
    return prob

def softmax(prob):
    """
    Determines if we like an image:
    :param prob: the top layer data from the network
    """
    like = False
    max = np.argmax(prob)
    if max == prop.shape[0]-1:
        like = True
    return like

if __name__ == '__main__':
    # Input set up
    if len(sys.argv) not in [3, 5]:
        print "Useage: python TinderNet.py [Iteration Number] [latitude] [longitude] [send likes (true/false)]"
        print "\t Or: python TinderNet.py [Iteration Number] [send likes (true/false)]"
        raise AttributeError()
    else:
        if len(sys.argv) == 3:
            location = False
            if sys.argv[2].lower() == 'true':
                send = True
            else:
                send = False
        elif len(sys.argv) == 5:
            location = True
            lat = sys.argv[2]
            long = sys.argv[3]
            if sys.argv[4].lower() == 'true':
                send = True
            else:
                send = False
        iter = int(sys.argv[1])
    cont = True
    # Number of output classes
    classes = 3
    # Initializes some caffe parameters
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = initNet(iter, 1)
    # Initialize the transform with the same mean from the prototxt
    tform = initTransform(np.array([108, 114, 130], dtype=np.float64))
    session = pynder.Session(fb_id, fb_auth)
    # Update the location several times to make sure it takes
    if location:
        print "Please wait: updating location"
        for _ in range(8):
            session.update_location(lat, long)
            time.sleep(0.1)
    while cont:
        # Iterate over found users
        users = session.nearby_users()
        for i in range(len(users)):
            imgArr = np.zeros((len(users[i].photos), 256, 256, 3), dtype=np.uint8)
            score = np.zeros((classes), dtype=np.float64)
            # Read the users photos from the web
            for j in range(len(users[i].photos)):
                cap = cv2.VideoCapture(users[i].photos[j])
                if cap.isOpened():
                    ret, img = cap.read()
                    if ret:
                        img = cv2.resize(img, (256, 256))
                        imgArr[j, :, :, :] = img
            # Pass the images through the network
            scores = np.zeros((len(users[i].photos), 3),dtype=np.float64)
            for j in range(imgArr.shape[0]):
                prob = processImg(imgArr[j, :, :, :], net, tform)
                scores[j, :] = np.copy(prob)
            best = np.argmax(scores[:, -1])
            # Print some stats
            try:
                print "{}: {}".format(str(users[i].name), np.sum(scores[:, -1]))
            except Exception:
                print "Shit, thats a unicode name"
            # Determine swipe direction
            if np.sum(scores[:, -1]) >= 0.3:
                img = cv2.copyMakeBorder(imgArr[best,:,:,:],6,6,6,6,cv2.BORDER_CONSTANT,value=(0, 255, 0))
                if send:
                    users[i].like()
            else:
                img = cv2.copyMakeBorder(imgArr[best,:,:,:],6,6,6,6,cv2.BORDER_CONSTANT,value=(0, 0, 255))
                if send:
                    users[i].dislike()
            # Display Results
            cv2.imshow("Profile", cv2.resize(img, None, fx=2, fy=2))
            k = cv2.waitKey(60)
            if k == 27:
                cont = False
            if not cont:
                break
    cv2.destroyAllWindows()
