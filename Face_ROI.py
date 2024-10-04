import cv2
import numpy as np
import random
from mtcnn.mtcnn import MTCNN
from GrabCut_v2 import grabCut
from scipy.signal import butter, lfilter

# Toggle these for different ROIs
REMOVE_EYES = False
FOREHEAD_ONLY = True 
USE_SEGMENTATION = True
USE_MY_GRABCUT = False
ADD_BOX_ERROR = False
USE_MTCNN = True  # Toggle between MTCNN and HaarCascade

CASCADE_PATH = "Haar_Cascade/haarcascade_frontalface_default.xml"
VIDEO_DIR = "videos/"
DEFAULT_VIDEO = "input.mp4"
RESULTS_SAVE_DIR = "results/segmentation/" if USE_SEGMENTATION else "results/no_segmentation/"
if REMOVE_EYES:
    RESULTS_SAVE_DIR += "no_eyes/"
if FOREHEAD_ONLY:
    RESULTS_SAVE_DIR += "forehead/"

MIN_FACE_SIZE = 100
WIDTH_FRACTION = 0.6
HEIGHT_FRACTION = 1.0

GRABCUT_ITERATIONS = 5
MY_GRABCUT_ITERATIONS = 2
SEGMENTATION_HEIGHT_FRACTION = 1.2
SEGMENTATION_WIDTH_FRACTION = 0.8
EYE_LOWER_FRAC = 0.25
EYE_UPPER_FRAC = 0.5
BOX_ERROR_MAX = 0.5

# Initialize face detection models
faceCascade = cv2.CascadeClassifier(CASCADE_PATH)
detector = MTCNN() if USE_MTCNN else None

def segment(image, faceBox):
    """ Perform segmentation using GrabCut_v2 or OpenCV's GrabCut. """
    if USE_MY_GRABCUT:
        foregrndMask, backgrndMask = grabCut(image, faceBox, MY_GRABCUT_ITERATIONS)
    else:
        mask = np.zeros(image.shape[:2], np.uint8)
        bgModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, faceBox, bgModel, fgModel, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)
        backgrndMask = (mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD)
    
    backgrndMask = np.broadcast_to(backgrndMask[:, :, np.newaxis], np.shape(image))
    return backgrndMask

def getROI(image, faceBox):
    """ Get the region of interest (ROI) from the detected face. """
    widthFrac = SEGMENTATION_WIDTH_FRACTION if USE_SEGMENTATION else WIDTH_FRACTION
    heightFrac = SEGMENTATION_HEIGHT_FRACTION if USE_SEGMENTATION else HEIGHT_FRACTION

    x, y, w, h = faceBox
    widthOffset = int((1 - widthFrac) * w / 2)
    heightOffset = int((1 - heightFrac) * h / 2)
    faceBoxAdjusted = (x + widthOffset, y + heightOffset, int(widthFrac * w), int(heightFrac * h))

    #My Grabcut or Open CVs Grabcut
    if USE_SEGMENTATION:
        backgrndMask = segment(image, faceBoxAdjusted)
    else:
        x, y, w, h = faceBoxAdjusted
        backgrndMask = np.full(image.shape, True, dtype=bool)
        backgrndMask[y:y+h, x:x+w, :] = False

    backgrndMask = backgrndMask.copy()

    # Either remove eyes or just keep forehead as the ROI
    if REMOVE_EYES:
        backgrndMask[y + int(h * EYE_LOWER_FRAC): y + int(h * EYE_UPPER_FRAC), :] = True
    if FOREHEAD_ONLY:
        backgrndMask[y + int(h * EYE_LOWER_FRAC):, :] = True  

    # Generate ROI while removing bg using Mask
    roi = image.copy()
    roi[backgrndMask == True] = 0  

    return roi


def distance(roi1, roi2):
    """ Calculate distance between ROIs. """
    return sum((roi1[i] - roi2[i])**2 for i in range(len(roi1)))

def detect_faces(frame):
    """ Detect faces using MTCNN or HaarCascade based on the toggle. """
    if USE_MTCNN:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb_frame)
        faces = [(d['box'][0], d['box'][1], d['box'][2], d['box'][3]) for d in detections]
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, 
                                             minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def getBestROI(frame, previousFaceBox):
    """ Get the best ROI by detecting the face and adjusting based on previous frame. """
    faces = detect_faces(frame)
    roi = None
    faceBox = None

    if len(faces) == 0:
        faceBox = previousFaceBox
    elif len(faces) > 1:
        if previousFaceBox is not None:
            minDist = float("inf")
            for face in faces:
                if distance(previousFaceBox, face) < minDist:
                    faceBox = face
        else:
            maxArea = 0
            for face in faces:
                if face[2] * face[3] > maxArea:
                    faceBox = face
    else:
        faceBox = faces[0]

    if faceBox is not None:
        if ADD_BOX_ERROR:
            noise = [random.uniform(-BOX_ERROR_MAX, BOX_ERROR_MAX) for _ in range(4)]
            x, y, w, h = faceBox
            x1 = x + int(noise[0] * w)
            y1 = y + int(noise[1] * h)
            x2 = x + w + int(noise[2] * w)
            y2 = y + h + int(noise[3] * h)
            faceBox = (x1, y1, x2 - x1, y2 - y1)

        #Draw Rect around ROI
        x, y, w, h = faceBox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        roi = getROI(frame, faceBox)

    return faceBox, roi

def butter_bandpass(lowcut, highcut, fs, order=3):
    """Butterworth bandpass filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    """Apply the bandpass filter to the data"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y

def amplify_video(frames, alpha=50, lowcut=0.83, highcut=1.0, fs=30):
    """Amplify the video signal"""
    filtered = np.zeros_like(frames)
    for channel in range(frames.shape[-1]):
        filtered[:, :, :, channel] = apply_butter_bandpass_filter(frames[:, :, :, channel], lowcut, highcut, fs)
    return frames + alpha * filtered  

def eulerian_magnification(frames, alpha=50, lowcut=0.83, highcut=1.0, fs=30):
    """Apply EVM to the entire video"""
    magnified_frames = amplify_video(frames, alpha=alpha, lowcut=lowcut, highcut=highcut, fs=fs)
    return magnified_frames


video = cv2.VideoCapture(VIDEO_DIR + DEFAULT_VIDEO)
capture_FPS = video.get(cv2.CAP_PROP_FPS)
WINDOW_TIME_SEC = 10
WINDOW_SIZE = int(np.ceil(WINDOW_TIME_SEC * capture_FPS))

frames = []
previousFaceBox = None

# Read all the frames from the video
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frames.append(frame)  

video.release()


frames_np = np.array(frames)

# Apply Eulerian Video Magnification to the input video and store it's frames 
magnified_video = eulerian_magnification(frames_np, alpha=10, lowcut=0.9, highcut=1.0, fs=capture_FPS)
print('EVM Done')
video_with_roi = []
#Now generate the ROI for each of the frames we got after EVM
for i, frame in enumerate(magnified_video):
    faceBox, roi = getBestROI(frame, previousFaceBox)
    previousFaceBox = faceBox  
    
    if roi is not None:
        video_with_roi.append(roi)
        #cv2.imshow('Region of Interest',roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Save the video of the magnified ROI 
height, width, _ = video_with_roi[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(RESULTS_SAVE_DIR+'output_magnified_roi_whitelight.mp4', fourcc, capture_FPS, (width, height))

for frame in video_with_roi:
    out.write(frame)

out.release()
cv2.destroyAllWindows()
