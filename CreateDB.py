import pynder
import cv2
import time
import glob
import os
from random import shuffle, randint
import thread
import copy
import sys
import numpy as np

fb_id = "1438695358"
fb_auth = "CAAGm0PX4ZCpsBAA7D7ZBTZB3ZCfhnIFWyiZBZBWWEASSDyIMc3vVb9FcnZBqGyOufZBBuAbbzhe5HztPigyODEKEtJZAlTRZCY9R61ZAV64d3EzkxpQLBWIeR6r6oOGyIEIK5RmovZBZCtNbqdZB3KZCsZAvQZAFwCuWs7iusY4NooCrM8TeMugOsAbED2x5KZAmORNASTK8ZCfuDHFujFjzfW7KrdsWfRZCTaJ3nwSUlQVQZD"

# Some locations for the tinder bot to prowl
Locations = [[42.3132882, -71.1972391],   # BC
             [41.8077414, -72.2561692],   # UCONN
             [42.3911569, -72.5289008],   # UMASS
             [40.7295134, -73.9986496],   # NYU
             [30.2849185, -97.7362454],   # UTA
             [38.8997145, -77.0507879],   # GWU
             [37.2283843, -80.4256054],   # VTech
             [40.0349011, -75.3395391],   # Villanova
             [34.0223519, -118.2873057]]  # USC

def getData(location, maxImgs=10000):
    """
    Logs into the tinder api and download a training dataset to
    the current directory /data
    If the fb_auth code is not accurate, then the code will error out
    :param location: [lat, long] list containing location from google maps
    :param maxImgs: number of images to stop at
        will still capture the rest of the current set of users
    :return: nothing, images are written to disk
    """
    session = pynder.Session(fb_id, fb_auth)
    trainDir = os.getcwd() + "/data/"
    if not os.path.exists(trainDir):
        os.mkdir(trainDir)
    itt = 0
    # Update location several times so it sticks
    if location is not None:
        print "Please wait: updating location"
        for _ in range(8):
            session.update_location(str(location[0]), str(location[1]))
            time.sleep(1)
    while len(glob.glob(trainDir + "*.jpg")) < maxImgs:
        itt += 1
        users = session.nearby_users()
        print "{}: processing {} users".format(itt, len(users))
        for i in range(len(users)):
            for imNum in range(0, len(users[i].photos)):
                cap = cv2.VideoCapture(users[i].photos[imNum])
                if cap.isOpened():
                    ret, img = cap.read()
                    if ret:  # Check if the image has been read correctly
                        img = cv2.resize(img, (256, 256))  # Resize for neural net
                        cv2.imwrite("{}{}.{}.jpg".format(trainDir, users[i].id, imNum), img)
                cap.release()

def removeUser(uid):
    """
    Prevents a user from showing up again by disliking them
    Useful for removing people you know from the data set
    :param uid: the user id to be remove. (first part of image filename)
    """
    session = pynder.Session(fb_id, fb_auth)
    session._api.dislike(uid)

class BufferWriter():
    __slots__ = ('num', 'fsize', 'flist')
    def __init__(self, fsize):
        """
        Creates a writer for the label buffer
        :param fsize: when to trigger the buffer write
        """
        self.num = 0
        self.fsize = fsize
        self.flist = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Makes sure data is written even when an error is encountered
        """
        self.write()

    def push(self, fname, label):
        """
        Adds a image and label to the stack to be written out
        Spawns an I/O thread when there are fsize images in the queue
        :param fname: the image filename and path
        :param label: the label (0 or 1)
        """
        self.flist.append([copy.deepcopy(fname), copy.deepcopy(label)])
        if len(self.flist) % (self.fsize + 1) == self.fsize:
            thread.start_new_thread(self.write, ())

    def write(self):
        """
        Thread to handle output I/O so main isnt blocked
        Writes the ratings to a data labels file
        Moves the image from it's current location to the created set folder
        """
        print "Writing {} image labels".format(len(self.flist))
        section = self.num % self.fsize
        if len(self.flist) > 0:
            # Only -1 when no user input
            if self.flist[0][1] != -1:
                # Go to new section to prevent overwrites
                while os.path.exists("data/set{}".format(section)):
                    section += 1
                os.mkdir("data/set{}".format(section))
                out = ""
                for i in range(len(self.flist)):
                    # Only write if the label is valid
                    if self.flist[i][1] in [0, 1]:
                        out += os.getcwd().replace("\\", "/")
                        out += "/data/set{}/".format(section)
                        out += "{} {}\n".format(self.flist[i][0].split("/")[-1], self.flist[i][1])
                        os.system("mv {} {}/data/set{}/".format(self.flist[i][0], os.getcwd(), section))
                with open("{}/data/set{}/Labels.txt".format(os.getcwd().replace("\\", "/"), section), 'w') as outFile:
                    outFile.write(out)
        self.flist = []

class BufferAssesser():
    __slots__ = ('buffer', 'done', 'quitOut')

    def __init__(self, bufferSize=16):
        """
        Creates a buffer assesser object
        :param bufferSize: size of buffer to make
        """
        self.buffer = [None] * bufferSize
        self.done = list([False])
        self.quitOut = False
        for i in range(bufferSize):
            self.buffer[i] = copy.deepcopy([None, 'fname', False])
        thread.start_new_thread(self.loadBuffer, (self.done, self.quitOut))
        time.sleep(1) 

    def loadBuffer(self, done, quitOut):
        """
        Thread that loads the buffer so no I/O limiting
        Goes through the data folder and loads all *.jpg images
        """
        files = glob.glob("{}/data/*.jpg".format(os.getcwd().replace('\\', '/')))
        #shuffle(files)
        for i in range(0, len(files)):
            sys.stdout.write("Files left to load: {}\n".format(len(files)-i))
            if quitOut:
                print "Early Exit"
                thread.exit()
            img = cv2.imread(files[i], 1)
            j = 0
            # Continually check for a free spot in buffer
            while True:
                j = (j + 1) % len(self.buffer)
                if not self.buffer[j][2]:
                    self.buffer[j][0] = img
                    self.buffer[j][1] = files[i]
                    self.buffer[j][2] = True
                    break
        done[0] = True

    def bufferUsed(self):
        """
        Checks if the entire buffer has been looked at
        :return: True only if all images have been viewed
        """
        out = False
        for i in range(0,len(self.buffer)):
            out = out or self.buffer[i][2]
        return not out

    def doubleSize(self, img):
        """
        Quickly makes a 512x512 image out of a 256x256 image
        with no interpolation at all
        :param img: the numpy array to be augmented 256x256
        :return: the output image as 512x512
        """
        out = np.zeros((512, 512, 3), dtype=img.dtype)
        out[0::2, 0::2, :] = img
        out[1::2, 0::2, :] = img
        out[0::2, 1::2, :] = img
        out[1::2, 1::2, :] = img
        return out

    def go(self):
        """
        Loop for manually labeling the images using left
        and right arrow keys. Calls write thread every 100
        images so you can quit with the 'q' key
        """
        cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
        with BufferWriter(100) as writer:
            while not (self.bufferUsed() and self.done[0]):
                # Repeatedly iterate over the buffer
                for i in range(0, len(self.buffer)):
                    if self.buffer[i][2]:  # If the image is ready to be read
                        label = -1  # Initializes the type
                        key = -1  # Initializes the key pressed
                        # Doubles the size for display
                        temp = self.doubleSize(self.buffer[i][0])
                        cv2.imshow("Image", temp)
                        while key not in [65361, 65363, 27, 81, 113]:
                            key = cv2.waitKey(0)
                            if key == 65361:
                                # OpenCV's left arrow
                                label = 0
                            elif key == 65363:
                                # OpenCV's right arrow
                                label = 1
                            elif key in [27, ord('q'), ord('Q')]:
                                # Exit the loading thread
                                self.quitOut = True
                        writer.push(self.buffer[i][1], label)
                        # Mark the image as used
                        self.buffer[i][2] = False
                    if self.quitOut:
                        break
                if self.quitOut:
                    break
        cv2.destroyAllWindows()

def resetFolders():
    """
    Resets the data folder to the get state
    rather than the post assess state
    """
    i = 0
    dir = os.getcwd().replace('\\', '/')
    while os.path.exists("{}/data/set{}".format(dir, i)):
        os.system("mv {}/data/set{}/*.jpg {}/data/".format(dir, i, dir))
        os.system("rm -rf {}/data/set{}".format(dir, i))
        i += 1

def trainTestSplit(split, passClasses=1):
    """
    Creates the test and train directories for the network
    Splits based on a given ratio, images randomized first
    Splits the no class into several to prevent over fitting
    :param split: the fraction of data for training
    :param passClasses: the number of classes to separate pass into
    """
    files = []
    i = 0
    dir = os.getcwd().replace('\\', '/')
    while os.path.exists("{}/data/set{}".format(dir, i)):
        with open("{}/data/set{}/Labels.txt".format(dir, i), 'r') as inDat:
            for line in inDat:
                files.append(line.strip())
        i += 1
    shuffle(files)
    with open("{}/train.txt".format(dir), 'w') as train:
        for i in range(int(len(files)*float(split))):
            files[i] = files[i].split(' ')
            if files[i][1] == '0':
                files[i][1] = str(randint(0, passClasses-1))
            else:
                files[i][1] = str(passClasses)
            train.write(" ".join(files[i]))
            train.write("\n")
    with open("{}/test.txt".format(dir), 'w') as test:
        for i in range(int(len(files)*float(split)), len(files)):
            files[i] = files[i].split(' ')
            if files[i][1] == '0':
                files[i][1] = str(randint(0, passClasses-1))
            else:
                files[i][1] = str(passClasses)
            test.write(" ".join(files[i]))
            test.write("\n")

if __name__ == '__main__':
    if sys.argv[1] == 'get':
        if len(sys.argv) == 5:
            loc = [sys.argv[3], sys.argv[4]]
        elif len(sys.argv) == 4:
            loc = Locations[int(sys.argv[3])]
        else:
            loc = None
        getData(loc, int(sys.argv[2]))
    if sys.argv[1] == 'remove':
        removeUser(sys.argv[2])
    if sys.argv[1] == 'assess':
        BufferAssesser(16).go()
    if sys.argv[1] == 'reset':
        resetFolders()
    if sys.argv[1] == 'split':
        trainTestSplit(float(sys.argv[2]), 2)