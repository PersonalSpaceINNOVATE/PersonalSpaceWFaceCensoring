from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import math
import requests

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

nnPathDefault = str((Path(__file__).parent / Path('models/mobilenet-ssd_openvino_2021.2_6shave.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-ff', '--full_frame', action="store_true", help="Perform tracking on full RGB frame", default=False)

args = parser.parse_args()

fullFrameTracking = args.full_frame

# Create pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

# Define sources and outputs
camRgb = pipeline.createColorCamera()
spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
objectTracker = pipeline.createObjectTracker()

xoutRgb = pipeline.createXLinkOut()
trackerOut = pipeline.createXLinkOut()

xoutRgb.setStreamName("preview")
trackerOut.setStreamName("tracklets")

# Properties
camRgb.setPreviewSize(300, 300)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# setting node configs
stereo.setConfidenceThreshold(255)

spatialDetectionNetwork.setBlobPath(args.nnPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

objectTracker.setDetectionLabelsToTrack([15,16])  # track only desired Object (from Label Map) 
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.SMALLEST_ID)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
objectTracker.out.link(trackerOut.input)

#Face Detection
image_path = "<../Path/To/Image.Extension>"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#function to calculate distance
def calculateDistance (x,y,z):
    disBet = math.sqrt(x ** 2 + y ** 2 + z **2)
    return disBet

if fullFrameTracking:
    camRgb.setPreviewKeepAspectRatio(False)
    camRgb.video.link(objectTracker.inputTrackerFrame)
    objectTracker.inputTrackerFrame.setBlocking(False)
    # do not block the pipeline if it's too slow on full frame
    objectTracker.inputTrackerFrame.setQueueSize(2)
else:
    spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
spatialDetectionNetwork.out.link(objectTracker.inputDetections)
stereo.depth.link(spatialDetectionNetwork.inputDepth)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    preview = device.getOutputQueue("preview", 4, False)
    tracklets = device.getOutputQueue("tracklets", 4, False)
    color = (255, 255, 255)
    
    #initializes libraries/thresholds
    minDis = 0.5
    disBetween = []
    atRisk = []
    percAtRisk = [0]
    
    #initializes Variables
    numObjects = 0
    
    #initializes counters
    masterCounter = 0
    guidelineBrokenCounter = 0
    atRiskCounter = 0
    pubCounter = 0
    percAtRiskCounter = 0
    prevNumAtRisk = 0
    difAtRisk = 0
    timesPublished = 0
    
    #Initializes Publish Stuff
    dataToPublish = {}
    dataToPublish["masterCounter"] = masterCounter
    dataToPublish["TimeAtRisk"] = percAtRiskCounter
    dataToPublish["NumberOfViolations"] = guidelineBrokenCounter
    dataToPublish["Room Occupancy"] = numObjects
    dataToPublish["Time Published"] = time.time()
    
    #Must Run initially to connect to Integromat
    #while(True):
     #   r = requests.post("https://hook.integromat.com/0xyum95wy0b26fj8wennlbg98ei57ml7", json= dataToPublish)
     #   print(r.status_code)
     #   if r.status_code == 200:
     #       break
    
    while(True):
        masterCounter+=1
        imgFrame = preview.get()
        track = tracklets.get()

        frame = imgFrame.getCvFrame()
        trackletsData = track.tracklets
        
        #counts number of objects
        numObjects = len(trackletsData)
        print("# of Objects Detected:",numObjects)
        print()
        
        #face detection code
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (10,10))
        face_detected = format(len(faces)) + "Face detected!"
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,0), -1)


        for t in trackletsData:
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)
            
            try:
                label = labelMap[t.label]
            except:
                label = t.label

            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)    
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            
         #calculate distances from camera           
            xposition = t.spatialCoordinates.x/1000
            yposition = t.spatialCoordinates.y/1000
            zposition = t.spatialCoordinates.z/1000
            distance = calculateDistance(xposition,yposition,zposition)
           
            print("Position (x,y,z) (m):    ",xposition,"X m",yposition,"Y m",zposition,"Z m")
            print("Distance From Camera (m):", distance,"m")
            print()   
        
        #calculating distances between each object
        if numObjects >= 2:
            for i in range(numObjects):
                for k in range(numObjects):
                    difx = (trackletsData[k].spatialCoordinates.x - trackletsData[i].spatialCoordinates.x)/1000
                    dify = (trackletsData[k].spatialCoordinates.y - trackletsData[i].spatialCoordinates.y)/1000
                    difz = (trackletsData[k].spatialCoordinates.z - trackletsData[i].spatialCoordinates.z)/1000
                    disBet = calculateDistance(difx,dify,difz)
                    if disBet != 0:
                        if disBet <= minDis:
                            atRisk.append(disBet)
                            atRisk = list(dict.fromkeys(atRisk))
                        disBetween.append(disBet)
                        disBetween = list(dict.fromkeys(disBetween))
            numDis = len(disBetween)
            numAtRisk = len(atRisk)
            
            if numAtRisk != prevNumAtRisk and numAtRisk > prevNumAtRisk:
                atRiskCounter=0
                difAtRisk = numAtRisk - prevNumAtRisk
            prevNumAtRisk = numAtRisk
            
            if numAtRisk != 0:
                atRiskCounter+=1
            else:
                atRiskCounter=0
            
            percAtRisk = (percAtRiskCounter/masterCounter) 
            
            print("# of Distances Calculated =  ", numDis )
            print("# of Distances at Risk =       ", numAtRisk)
            #print("# Prev at Risk = ", prevNumAtRisk)
            #print("#Dif at Risk = ", difAtRisk)
            print("Distance Between Objects (m):", disBetween, "m")
            disBetween = []
            atRisk = []
            
        if atRiskCounter == 16:
            guidelineBrokenCounter+=difAtRisk
        if atRiskCounter >= 16:        
            percAtRiskCounter+=1
            
        #publishes data every 1min    
        if pubCounter == 60:
            dataToPublish["masterCounter"] = masterCounter
            dataToPublish["TimeAtRisk"] = percAtRiskCounter
            dataToPublish["NumberOfViolations"] = guidelineBrokenCounter
            dataToPublish["Room Occupancy"] = numObjects
            dataToPublish["Time Published"] = time.time()
            r = requests.post("https://hook.integromat.com/0xyum95wy0b26fj8wennlbg98ei57ml7", json= dataToPublish)
            pubCounter = 0
            timesPublished+=1
        pubCounter +=1
       
        print()
        print("Master Clock            =", masterCounter, "s")
        print("Total Time At Risk      =", percAtRiskCounter, "s")
        print("Threshold Clock         =", atRiskCounter, "s")
        print("Publish Clock           =", pubCounter, "s")
        print("# Times Published       =", timesPublished)
        print("# of Violations         =", guidelineBrokenCounter)
        print("Percent of Time at Risk =", percAtRisk,"%")
        percAtRisk = []
        print()
        print()
        print()
        
        #displays livefeed 
        cv2.imshow("LiveFeed", frame)
        
        #refreshes every 1 seconds
        time.sleep(1)
        
        if cv2.waitKey(1) == ord('q'):
            break