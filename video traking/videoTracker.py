import cv2
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import gc

# Constants
LEFT_COLOR = (0, 0, 255)    # Red for left pair
RIGHT_COLOR = (0, 255, 0)    # Green for right pair
INITIAL_SIZE = 23
MIN_SIZE = 5
SIZE_STEP = 2
TEMPLATE_SCALE = 1.5
MATCH_THRESHOLD = 0.7
RECOVERY_ATTEMPTS = 5
VELOCITY_WEIGHT = 0.5  # Weight for previous motion vector
KNOWN_DIST = 26000 # um, window frame width
PIX_TO_DIST = KNOWN_DIST / np.mean([894, 895, 896]) # None if unknown
MAX_THREADS = 10

class VidTracker(object):

    vidFolder = "./converted"
    csvFolder = "./data"
    pltFolder = "./plots"
    tagFolder = "./tags"
    trackedFolder = "./tracked"
    ovlFolder = "./overlaid"
    oriFolder = "./original"
    inVidFormat = ".mp4"
    outVidFormat = ".mp4"
    
    def __init__(self, name):

        self.name = name

        self.vidPath = VidTracker.vidFolder + '/' + name + VidTracker.inVidFormat
        self.csvPath = VidTracker.csvFolder + '/' + name + ".csv"
        self.pltPath = VidTracker.pltFolder + '/' + name + ".png"
        self.tagPath = VidTracker.tagFolder + '/' + name + ".json"
        self.trackedPath = VidTracker.trackedFolder + '/' + name + VidTracker.outVidFormat
        self.ovlPath = VidTracker.ovlFolder + '/' + name + VidTracker.outVidFormat

        self.pixToDistRatio = None # float
        self.curSize = INITIAL_SIZE
        self.lines = []
        self.sqs = []
        
        self.templates = None # list
        self.p0 = []
        self.lineOffsets = []


        # video
        self.cap = None
        self.frame = None
        self.vidWriter = None
        
        # video
        self.frameWidth = None
        self.frameHeight = None
        self.frameCount = None
        self.fps = None
        self.vidLen = None

        self.dragging = False
        self.selectedElem = None

    def loadCap(self):

        # Load the video
        cap = cv2.VideoCapture(self.vidPath)
        
        # Check if the video was opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None

        self.cap = cap

        self.frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.vidLen = self.frameCount / self.fps # in seconds

    def readFrame(self):

        # Read the first frame from the video
        ret, frame = self.cap.read()
        
        if not ret:
            print(f"Error: Could not read frame from {self.name}")
            self.frame = None
            return None

        self.frame = frame
        return frame

    def cleanup(self):

        # Release the video capture object and close all OpenCV windows
        self.cap.release()
        self.cap = None
        self.frame = None
        cv2.destroyAllWindows()

    def getConvRatio(self, knownDist, cleanup=True):
        """
        Compute the conversion ratio between pixel and physical distance in a video.
        
        Parameters:
        video_path (str): Path to the video file.
        known_distance_um (float): Known physical distance in micrometers (um).
        
        Returns:
        float: Conversion ratio (um per pixel).
        """

        if(PIX_TO_DIST is not None):
            self.pixToDistRatio = PIX_TO_DIST
            return

        if(self.cap is None): self.loadCap()
        
        frame = self.readFrame() if self.frame is None else self.frame
        if(self.frame is None): return
        
        # Display the frame and let the user specify the horizontal line
        winTitle = "Please specify the horizontal line with known distance of {} um. Press enter when done".format(knownDist)
        dispFrame = frame.copy()
        cv2.namedWindow(winTitle)
        cv2.imshow(winTitle, dispFrame)
        
        # Wait for user input
        points = cv2.selectROI(winTitle, dispFrame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        
        # Calculate conversion ratio
        pixel_distance = points[2]  # Width of the selected ROI
        ratio = knownDist / pixel_distance
        self.pixToDistRatio = ratio

        if(cleanup): self.cleanup()

    def setupInitPos(self, cleanUp=True):
    
        if(self.cap is None): self.loadCap()
        
        frame = self.readFrame() if self.frame is None else self.frame
        if(frame is None): return
        
        winTitle = "Arrange ROIs | W/S: Resize all | D: Done | Left=Red, Right=Green"
        cv2.namedWindow(winTitle)
        cv2.setMouseCallback(winTitle, self._mouseCallBackFunc())

        self._setSqs()
        self._setLines()
        
        while True:
            dispFrame = self._drawFrame()

            cv2.putText(dispFrame, f"Size: {self.curSize}px | D to start", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(winTitle, dispFrame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('w'):
                self._setSqs(self.curSize + SIZE_STEP)
            elif key == ord('s'):
                self._setSqs(self.curSize - SIZE_STEP)
            elif key == ord('d'):
                break
        
        cv2.destroyAllWindows()

        self._getTemplates()
        self._getTrackingPts()

        if(cleanUp): self.cleanup()
        
    def tag(self, save=True):

        self.getConvRatio(KNOWN_DIST, False)
        self.setupInitPos()

        if(save): self.save()

    def track(self, showFrame=False):

        # initial frame
        if(self.frame is None):
            if(self.cap is None): self.loadCap()
            frame = self.readFrame()
            if(frame is None): return
        
        self._setWriter()
        initFrame = self._drawFrame()
        self.vidWriter.write(initFrame) # write initial frame

        if(showFrame):
            winTitle = f"Tracking {self.name}"
            cv2.namedWindow(winTitle)
            cv2.imshow(winTitle, initFrame)

        # tracking
        self._initTrackingPrms()

        for _ in tqdm(range(int(self.frameCount)), desc=f"Tracking {self.name}"):
            frame = self.readFrame()
            self.frameIndex += 1

            if(frame is None): break

            newGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # tracking and recovery
            if(self.isTrackingActive):
                self._attemptTracking(newGray)
            
            if(not self.isTrackingActive and self.recoveryCounter > 0):
                self._attemptRecovery(newGray)

            # update tracked positions and write frame
            self._logData()
            dispFrame = self._drawFrame(fromP0=True)
            self.vidWriter.write(dispFrame)

            if(showFrame): 
                cv2.imshow(winTitle, dispFrame)
                cv2.waitKey(1)
        
        # clean up
        self._saveData()
        self.cleanup()
        self.vidWriter.release()

    def track_single_point(self, initial_pos, showFrame=False):
        """
        Track a single point in the video using optical flow and template recovery.
        The tracked position is saved as CSV and a video with the point overlay.
        Args:
            initial_pos (tuple): (x, y) initial position of the point.
            showFrame (bool): If True, show tracking in a window.
        """
        # Prepare output paths
        out_csv = self.csvPath.replace(".csv", "_single_point.csv")
        out_vid = self.trackedPath.replace(".mp4", "_single_point.mp4")

        # Load video if not loaded
        if self.cap is None:
            self.loadCap()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            print(f"Error: Could not read first frame from {self.name}")
            return

        # Tracking parameters
        lkPrms = dict(winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        fps = self.fps
        frameCount = int(self.frameCount)
        frameWidth = self.frameWidth
        frameHeight = self.frameHeight

        # Prepare video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vidWriter = cv2.VideoWriter(out_vid, fourcc, fps, (frameWidth, frameHeight))

        # Prepare template for recovery
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x0, y0 = int(initial_pos[0]), int(initial_pos[1])
        tsize = 30
        template = gray[max(0, y0-tsize):y0+tsize, max(0, x0-tsize):x0+tsize].copy()

        # Tracking state
        p0 = np.array([[[x0, y0]]], dtype=np.float32)
        origin = np.array([x0, y0], dtype=np.float32)
        oldGray = gray.copy()
        velocities = np.zeros((1, 2), dtype=np.float32)
        lostPos = None
        recoveryCounter = 0
        isTrackingActive = True

        # DataFrame for output
        df = pd.DataFrame(columns=["Frame", "Second", "dx", "dy"])

        # Write first frame with overlay
        dispFrame = frame.copy()
        cv2.circle(dispFrame, (x0, y0), 6, (0, 255, 255), 2)
        vidWriter.write(dispFrame)
        if showFrame:
            winTitle = f"Tracking single point: {self.name}"
            cv2.namedWindow(winTitle)
            cv2.imshow(winTitle, dispFrame)
            cv2.waitKey(1)

        # Tracking loop
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        for frame_idx in tqdm(range(1, frameCount), desc=f"Tracking single point: {self.name}"):
            ret, frame = self.cap.read()
            if not ret:
                break
            newGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if isTrackingActive:
                # Optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(oldGray, newGray, p0, None, **lkPrms)
                if p1 is not None and st[0][0] == 1 and err.mean() < 10:
                    velocities = (p1[0] - p0[0]) * VELOCITY_WEIGHT
                    p0 = p1
                    oldGray = newGray.copy()
                    recoveryCounter = 0
                else:
                    lostPos = p0.copy()
                    isTrackingActive = False
                    recoveryCounter = RECOVERY_ATTEMPTS
            else:
                # Template recovery
                x, y = int(p0[0,0,0]), int(p0[0,0,1])
                expected_pos = lostPos[0] + velocities * VELOCITY_WEIGHT
                x, y = int(expected_pos[0,0]), int(expected_pos[0,1])
                search_size = int(tsize * TEMPLATE_SCALE)
                x1 = max(0, x - search_size)
                y1 = max(0, y - search_size)
                x2 = min(newGray.shape[1], x + search_size)
                y2 = min(newGray.shape[0], y + search_size)
                if x2 > x1 and y2 > y1:
                    search_area = newGray[y1:y2, x1:x2]
                    res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
                    _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
                    if maxVal > MATCH_THRESHOLD:
                        newX = x1 + maxLoc[0] + template.shape[1] // 2
                        newY = y1 + maxLoc[1] + template.shape[0] // 2
                        p0 = np.array([[[newX, newY]]], dtype=np.float32)
                        oldGray = newGray.copy()
                        isTrackingActive = True
                    else:
                        recoveryCounter -= 1
                        if recoveryCounter <= 0:
                            p0 = lostPos
                            isTrackingActive = True  # Give up recovery

            dx, dy = p0[0,0] - origin
            df.loc[len(df)] = [frame_idx, frame_idx / fps, dx, dy]

            # Draw and write frame
            dispFrame = frame.copy()
            cv2.circle(dispFrame, (int(p0[0,0,0]), int(p0[0,0,1])), 6, (0, 255, 255), 2)
            vidWriter.write(dispFrame)
            if showFrame:
                cv2.imshow(winTitle, dispFrame)
                cv2.waitKey(1)

        # Save CSV and release resources
        df.to_csv(out_csv, index=False)
        vidWriter.release()
        if showFrame:
            cv2.destroyAllWindows()
        self.cleanup()

    def save(self, path=None):

        if(path is None): path = self.tagPath

        tag = self._toJSON()

        # Write the dictionary to a JSON file
        with open(path, 'w') as jsonFile:
            json.dump(tag, jsonFile, indent=4)

    def load(self, path=None, cleanup=True):

        if(path is None): path = self.tagPath
        
        data = None
        with open(path, 'r') as file:
            data = json.load(file)
        
        self.pixToDistRatio = data["pixToDistRatio"]
        self.curSize = data["curSize"]
        self.sqs = data["sqs"]
        self.lines = data["lines"]

        self.loadCap()
        self.readFrame()

        self._getTemplates()
        self._getTrackingPts()

        if(cleanup): self.cleanup()

    def overlayPlot(self):

        cap = cv2.VideoCapture(self.trackedPath)
        data = pd.read_csv(self.csvPath)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vidWriter = cv2.VideoWriter(self.ovlPath, 
                                    fourcc, 
                                    cap.get(cv2.CAP_PROP_FPS), 
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                                   )

        for index in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc=f"Overlaying {self.name}"):
            # Read the first frame from the video
            ret, frame = cap.read()
            if(not ret): break

            frame = frame.copy()
            ovlFrame = self._overlayPltOnFrame(frame, data, index)
            vidWriter.write(ovlFrame)

            del frame
            del ovlFrame
        
        cap.release()
        vidWriter.release()
        gc.collect()  # Force garbage collection

    def process(self, loadTags=True):

        if(loadTags): self.load()
        else: self.tag()

        self.track()
        self.overlayPlot()

    def _getTemplates(self):

        # Capture templates for recovery
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        templates = []
        for sq in self.sqs:
            cx = int(sq['x'] + sq['w'] * .5)
            cy = int(sq['y'] + sq['h'] * .5)
            size = int(max(sq['w'], sq['h']) * 2)
            y1 = max(0, cy - size)
            x1 = max(0, cx - size)
            y2 = min(gray.shape[0], cy + size)
            x2 = min(gray.shape[1], cx + size)
            templates.append(gray[y1:y2, x1:x2].copy())
        
        self.templates = templates
    
    def _getTrackingPts(self):

        # Prepare tracking points
        p0 = []
        for sq in self.sqs:
            cx = sq['x'] + sq['w'] * .5
            cy = sq['y'] + sq['h'] * .5
            p0.append([cx, cy])
        p0 = np.array(p0, dtype=np.float32).reshape(-1, 1, 2)
        
        # Calculate line offsets
        pair1Avg = (p0[0][0][0] + p0[1][0][0])/2
        pair2Avg = (p0[2][0][0] + p0[3][0][0])/2
        lineOffsets = [self.lines[0]['x'] - pair1Avg, self.lines[1]['x'] - pair2Avg]

        self.p0 = p0
        self.lineOffsets = lineOffsets

    def _toJSON(self):

        data = {}
        data["pixToDistRatio"] = self.pixToDistRatio
        data["curSize"] = self.curSize
        data["sqs"] = self.sqs
        data["lines"] = self.lines
        data["time"] = time.asctime()

        return data
    
    def _setSqs(self, newSize=None):

        if(newSize is None):
            self.curSize = INITIAL_SIZE
            x0, x1 = 100, int(100 + INITIAL_SIZE * 1.5)
            y0, y1 = x0, x1
            self.sqs = [{'x': x0, 'y': y0, 'w': self.curSize, 'h': self.curSize, 'dragging': False},
                        {'x': x0, 'y': y1, 'w': self.curSize, 'h': self.curSize, 'dragging': False},
                        {'x': x1, 'y': y0, 'w': self.curSize, 'h': self.curSize, 'dragging': False},
                        {'x': x1, 'y': y1, 'w': self.curSize, 'h': self.curSize, 'dragging': False},
                       ]
        
        else:
            newSize = max(MIN_SIZE, newSize)
            self.curSize = newSize
            for sq in self.sqs:
                cx = sq['x'] + sq['w'] * .5
                cy = sq['y'] + sq['h'] * .5
                sq['w'] = newSize
                sq['h'] = newSize
                sq['x'] = int(cx - newSize * .5)
                sq['y'] = int(cy - newSize * .5)

    def _setLines(self):

        self.lines = [{'x': 125, 'dragging': False},
                      {'x': 225, 'dragging': False},
                     ]

    def _drawFrame(self, fromP0=False):

        dispFrame = self.frame.copy()
        self._drawSqs(dispFrame, fromP0=fromP0)
        self._drawLines(dispFrame, fromP0=fromP0)

        return dispFrame

    def _drawSqs(self, frame=None, fromP0=False):

        if(frame is None):
            frame = self.frame

        for i, sq in enumerate(self.sqs):
            color = LEFT_COLOR if i < 2 else RIGHT_COLOR

            if(fromP0):
                cx, cy = self.p0[i][0]
                x = int(cx - sq['w'] * .5)
                y = int(cy - sq['h'] * .5)
            else:
                x, y = sq['x'], sq['y']
            
            cv2.rectangle(frame, 
                         (x, y),
                         (x + sq['w'], y + sq['h']), 
                         color, 2)
    
    def _drawLines(self, frame=None, fromP0=False):

        if(frame is None): frame = self.frame

        if(fromP0):
            pair1Avg = (self.p0[0][0][0] + self.p0[1][0][0]) * .5
            pair2Avg = (self.p0[2][0][0] + self.p0[3][0][0]) * .5
            line1X = pair1Avg + self.lineOffsets[0]
            line2X = pair2Avg + self.lineOffsets[1]
            lineXs = [int(line1X), int(line2X)]
        else:
            lineXs = [self.lines[i]['x'] for i in range(len(self.lines))]
        
        for i, lineX in enumerate(lineXs):
            color = LEFT_COLOR if i < 1 else RIGHT_COLOR
            cv2.line(frame, (lineX, 0), (lineX, frame.shape[0]), color, 2)

    def _mouseCallBackFunc(self):

        def callBack(event, x, y, flags, param):

            if event == cv2.EVENT_LBUTTONDOWN:
                for i, sq in enumerate(self.sqs):
                    if (sq['x'] <= x <= sq['x'] + sq['w']) and (sq['y'] <= y <= sq['y'] + sq['h']):
                        sq['dragging'] = True
                        self.selectedElem = ('square', i)
                        self.dragging = True
                        break
                else:
                    for i, line in enumerate(self.lines):
                        if abs(x - line['x']) <= 5:
                            line['dragging'] = True
                            self.selectedElem = ('line', i)
                            self.dragging = True
                            break
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                if self.selectedElem:
                    elemType, index = self.selectedElem
                    if elemType == 'square':
                        sq = self.sqs[index]
                        sq['x'] = x - sq['w'] // 2
                        sq['y'] = y - sq['h'] // 2
                    elif elemType == 'line':
                        self.lines[index]['x'] = x
            elif event == cv2.EVENT_LBUTTONUP:
                if self.selectedElem:
                    elemType, index = self.selectedElem
                    if elemType == 'square':
                        self.sqs[index]['dragging'] = False
                    elif elemType == 'line':
                        self.lines[index]['dragging'] = False
                    self.selectedElem = None
                    self.dragging = False

        return callBack

    def _saveData(self, show=False):

        self.df.to_csv(self.csvPath, index=False)

        # Plot the distance over time
        time = self.df["Second"]
        dist = self.df['Line distance (um)']
        
        plt.title(f"Edge Distance Over Time")
        plt.figure(figsize=(5, 4))  # Larger plot size
        plt.plot(time, dist)

        plt.xlabel("Time (s)")
        plt.xlim(time.min(), int(math.ceil(time.max() / 10)) * 10)

        plt.ylabel("Distance (um)")
        plt.ylim(int(math.floor(dist.min() / 100)) * 100, int(math.ceil(dist.max() / 100)) * 100)
        plt.tight_layout()  # Ensure titles and labels fit
        
        # Save the plot
        plt.savefig(self.pltPath)
        if(show): plt.show()
        plt.close("all")  # Close all open figures

    def _attemptTracking(self, newGray):

        # Track points using optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.oldGray, newGray, self.p0, None, **self.lkPrms)
        
        if(p1 is not None and st.all() and err.mean() < 10):
            # Update velocities and pair offsets
            for i in range(4):
                self.velocities[i] = (p1[i] - self.p0[i]) * VELOCITY_WEIGHT
                if i < 2:
                    self.pairOffsets[i] = p1[i] - p1[i^1]
                else:
                    self.pairOffsets[i] = p1[i] - p1[i^1]
            
            self.p0 = p1.reshape(-1, 1, 2)
            self.oldGray = newGray.copy()
            self.recoveryCounter = 0
        else:
            print("Tracking lost - attempting recovery")
            self.lostPos = self.p0.copy()
            self.isTrackingActive = False
            self.recoveryCounter = RECOVERY_ATTEMPTS

    def _attemptRecovery(self, newGray):

        # Use pair relationships for recovery
        self.p0 = self._templateRecovery(newGray)
        
        # Validate recovered points
        validPts = sum(1 for pt in self.p0 if (0 <= pt[0,0] < newGray.shape[1] and 
                                               0 <= pt[0,1] < newGray.shape[0]))
        
        if(validPts >= 2):
            print(f"{self.name} Recovered {validPts} points")
            self.oldGray = newGray.copy()
            self.isTrackingActive = True
        else:
            self.recoveryCounter -= 1
            self.p0 = self.lostPos

    def _templateRecovery(self, newGray):
        
        newPts = self.lostPos.copy()
        
        for pair in [0, 2]:  # Process left pair (0-1) and right pair (2-3)
            p1, p2 = pair, pair + 1
            valid = [False, False]
            
            # Calculate expected positions based on pair relationships
            for i in [p1, p2]:
                if(self.templates[i].size == 0):
                    continue
                    
                # Calculate search center using partner position and historical offset
                partner = p2 if i == p1 else p1
                if(valid[partner % 2]):
                    expectedPos = newPts[partner] + self.pairOffsets[i]
                else:
                    # Use velocity prediction if no partner info
                    expectedPos = self.lostPos[i] + self.velocities[i] * VELOCITY_WEIGHT
                    
                x, y = expectedPos[0].astype(int)
                size = max(self.templates[i].shape)
                searchSize = int(size * TEMPLATE_SCALE)
                
                # Define search area
                x1 = max(0, x - searchSize)
                y1 = max(0, y - searchSize)
                x2 = min(newGray.shape[1], x + searchSize)
                y2 = min(newGray.shape[0], y + searchSize)
                
                if x2 > x1 and y2 > y1:
                    searchArea = newGray[y1:y2, x1:x2]
                    res = cv2.matchTemplate(searchArea, self.templates[i], cv2.TM_CCOEFF_NORMED)
                    _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
                    
                    if maxVal > MATCH_THRESHOLD:
                        newX = x1 + maxLoc[0] + self.templates[i].shape[1] // 2
                        newY = y1 + maxLoc[1] + self.templates[i].shape[0] // 2
                        newPts[i] = np.array([[newX, newY]], dtype=np.float32)
                        valid[i % 2] = True

        return newPts

    def _logData(self):
        
        # Calculate positions
        pair1Avg = (self.p0[0][0][0] + self.p0[1][0][0])/2
        pair2Avg = (self.p0[2][0][0] + self.p0[3][0][0])/2
        line1X = pair1Avg + self.lineOffsets[0]
        line2X = pair2Avg + self.lineOffsets[1]
        
        # Record data
        pairDist = abs(pair2Avg - pair1Avg) * self.pixToDistRatio
        lineDist = abs(line2X - line1X) * self.pixToDistRatio

        self.df.loc[len(self.df)] = [self.frameIndex, self.frameIndex / self.fps, pairDist, lineDist]
        
    def _initTrackingPrms(self):

        # tracking parameters
        self.lkPrms = dict(winSize=(15,15), maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.df = pd.DataFrame(columns=["Frame", "Second", "Pair distance (um)", "Line distance (um)"])
        self.oldGray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.frameIndex = 0
        self.isTrackingActive = True
        self.recoveryCounter = 0
        self.lostPos = None
        self.velocities = [np.zeros((1, 2), dtype=np.float32) for _ in range(4)]
        self.pairOffsets = [np.zeros((1, 2), dtype=np.float32) for _ in range(4)]

        # Calculate initial pair offsets
        self.pairOffsets[0] = self.p0[0] - self.p0[1]
        self.pairOffsets[1] = self.p0[1] - self.p0[0]
        self.pairOffsets[2] = self.p0[2] - self.p0[3]
        self.pairOffsets[3] = self.p0[3] - self.p0[2]

        self._logData()

    def _setWriter(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.vidWriter = cv2.VideoWriter(self.trackedPath, 
                                         fourcc, 
                                         self.fps, 
                                         (self.frameWidth, self.frameHeight))

    def _overlayPltOnFrame(self, frame, df, index):

        # Generate the plot
        time = df["Second"]
        dist = df["Line distance (um)"]
        
        plt.title(f"Electrical stimulation: {self.name}")
        plt.figure(figsize=(5, 4))  # Increase figure size to make plot 25% larger
        plt.plot(time[:index], dist[:index], linewidth=1)  # Thinner line (default is 1, you can reduce this further)

        plt.xlabel("Time (s)")  # Adjust labels for seconds
        plt.xlim(time.min(), int(math.ceil(time.max() / 10)) * 10)

        plt.ylabel("Distance (um)")
        plt.ylim(int(math.floor(dist.min() / 100)) * 100, int(math.ceil(dist.max() / 100)) * 100)

        curTime = time[index]
        curDistance = dist[index]
        plt.scatter(curTime, curDistance, color='red', zorder=5)  # Red dot for current value
        
        plt.tight_layout()  # Ensure titles and labels fit within the plot
        
        # Save the plot as an image in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)  # Go to the start of the buffer
        plotImage = np.frombuffer(buf.read(), dtype=np.uint8)
        plotImage = cv2.imdecode(plotImage, cv2.IMREAD_COLOR)
        plt.close("all")
        buf.close()

        # Resize the plot to fit on the video frame (bottom-right corner, 25% larger)
        plotImage = cv2.resize(plotImage, (int(frame.shape[1] * 0.3), int(frame.shape[0] * 0.5)))
        
        # Overlay the plot image onto the video frame (bottom-right corner)
        xOffset = frame.shape[1] - plotImage.shape[1] - 10  # 10px margin from the right
        yOffset = frame.shape[0] - plotImage.shape[0] - 10  # 10px margin from the bottom
        frame[yOffset:yOffset + plotImage.shape[0], xOffset:xOffset + plotImage.shape[1]] = plotImage

        del buf
        del plotImage

        return frame

    @staticmethod
    def checkFolders():

        os.makedirs(VidTracker.vidFolder, exist_ok=True)
        os.makedirs(VidTracker.csvFolder, exist_ok=True)
        os.makedirs(VidTracker.pltFolder, exist_ok=True)
        os.makedirs(VidTracker.tagFolder, exist_ok=True)
        os.makedirs(VidTracker.trackedFolder, exist_ok=True)
        os.makedirs(VidTracker.ovlFolder, exist_ok=True)
        os.makedirs(VidTracker.oriFolder, exist_ok=True)

    @staticmethod
    def measureImage(name):

        cap = cv2.VideoCapture(VidTracker.vidFolder + '/' + name + VidTracker.inVidFormat)
        ret, frame = cap.read()

        if(not ret):
            print("Unable to read frame")
            return
        
        print("Image size: ", frame.shape)
        
        winTitle = "Please select area to measure"
        cv2.namedWindow(winTitle)
        cv2.imshow(winTitle, frame)
        
        # Wait for user input
        points = cv2.selectROI(winTitle, frame, fromCenter=False, showCrosshair=True)
        width = points[2]
        height = points[3]
        print(f"Selected area pixel dimension: {width} x {height}")
        cv2.destroyAllWindows()
        cap.release()

    @staticmethod
    def batchTag(skipExisting=False):

        videos = os.listdir(VidTracker.vidFolder)
        vidNames = [v[:-4] for v in videos if v.endswith(VidTracker.inVidFormat)]

        if(skipExisting):
            vidNames = [name for name in vidNames if not os.path.exists(VidTracker.tagFolder + '/' + name + ".json")]

        for name in vidNames:
            tracker = VidTracker(name)
            tracker.tag()

    @staticmethod
    def batchTrack(skipExisting=False, parallel=False):

        videos = os.listdir(VidTracker.vidFolder)
        vidNames = [v[:-4] for v in videos if v.endswith(VidTracker.inVidFormat)]

        if(skipExisting):
            vidNames = [name for name in vidNames if not os.path.exists(VidTracker.trackedFolder + '/' + name + VidTracker.outVidFormat)]

        if(not parallel):
            for name in tqdm(vidNames, desc="Overall progress"):
                tracker = VidTracker(name)
                tracker.load()
                tracker.track()
        else:
            def runCmd(name):
                # Get the current working directory
                cwd = os.getcwd()
                escaped_cwd = cwd.replace('"', '\\"')  # Escape double quotes in the path
                command = f'osascript -e \'tell application "Terminal" to do script "cd \\"{escaped_cwd}\\" && python track.py {name}; exit"\''
                subprocess.Popen(command, shell=True)
        
            with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                executor.map(runCmd, vidNames)

    @staticmethod
    def batchOverlay(skipExisting=False, parallel=False):

        videos = os.listdir(VidTracker.trackedFolder)
        vidNames = [v[:-4] for v in videos if v.endswith(VidTracker.outVidFormat)]

        if(skipExisting):
            vidNames = [name for name in vidNames if not os.path.exists(VidTracker.ovlFolder + '/' + name + VidTracker.outVidFormat)]

        if(not parallel):
            for name in tqdm(vidNames, desc="Overall progress"):
                tracker = VidTracker(name)
                tracker.overlayPlot()
                del tracker
        else:
            def runCmd(name):
                # Get the current working directory
                cwd = os.getcwd()
                escaped_cwd = cwd.replace('"', '\\"')  # Escape double quotes in the path
                command = f'osascript -e \'tell application "Terminal" to do script "cd \\"{escaped_cwd}\\" && python overlay.py {name}; exit"\''
                subprocess.Popen(command, shell=True)
        
            with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                executor.map(runCmd, vidNames)

if(__name__ == "__main__"):
    #VidTracker.checkFolders()
    #VidTracker.measureImage("act 2x N2")
    #vid = VidTracker("act 2x N2")
    #vid.tag()
    #vid.track()
    #VidTracker.batchTag(skipExisting=True)
    #VidTracker.batchTrack(skipExisting=True, parallel=False)
    #VidTracker.batchOverlay(skipExisting=True)