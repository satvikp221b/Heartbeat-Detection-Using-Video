import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA
import warnings
from GrabCut import grabCut
import random
import time
import tqdm

# Toggle these for different ROIs
REMOVE_EYES = True
FOREHEAD_ONLY = True
USE_SEGMENTATION = False
USE_MY_GRABCUT = True
ADD_BOX_ERROR = False

CASCADE_PATH = "haarcascade_frontalface_default.xml"
VIDEO_DIR = "../videos/"
# DEFAULT_VIDEO = "android-1.mp4"
# DEFAULT_VIDEO = "satvik_stationary14_80bpm.mp4"
# DEFAULT_VIDEO = "mahendra_22.mp4"
DEFAULT_VIDEO = "face.mp4"
RESULTS_SAVE_DIR = "results/" + ("segmentation/" if USE_SEGMENTATION else "no_segmentation/")
if REMOVE_EYES:
    RESULTS_SAVE_DIR += "no_eyes/"
if FOREHEAD_ONLY:
    RESULTS_SAVE_DIR += "forehead/"

MIN_FACE_SIZE = 100

WIDTH_FRACTION = 0.6 # Fraction of bounding box width to include in ROI
HEIGHT_FRACTION = 1

try:
    videoFile = sys.argv[1]
except:
    videoFile = DEFAULT_VIDEO  
video = cv2.VideoCapture(VIDEO_DIR + videoFile)
# video = cv2.VideoCapture(0)  # Change 0 to 1 or 2 if your camera is not the default camera
faceCascade = cv2.CascadeClassifier(CASCADE_PATH)
capture_FPS = video.get(cv2.CAP_PROP_FPS)
# capture_FPS = 702/31
FPS = capture_FPS;

print(f"capture_FPS: {capture_FPS}, FPS: {FPS}")

# FPS = 14.99
WINDOW_TIME_SEC = 10
WINDOW_SIZE = int(np.ceil(WINDOW_TIME_SEC * FPS))
MIN_HR_BPM = 61.0
MAX_HR_BPM = 120.0
# MAX_HR_CHANGE = 20.0
SEC_PER_MIN = 60

SEGMENTATION_HEIGHT_FRACTION = 1.2
SEGMENTATION_WIDTH_FRACTION = 0.8
GRABCUT_ITERATIONS = 5
MY_GRABCUT_ITERATIONS = 2

EYE_LOWER_FRAC = 0.25
EYE_UPPER_FRAC = 0.5

BOX_ERROR_MAX = 0.5

def segment(image, faceBox):
    if USE_MY_GRABCUT:
        foregrndMask, backgrndMask = grabCut(image, faceBox, MY_GRABCUT_ITERATIONS)
    
    else:
        mask = np.zeros(image.shape[:2],np.uint8)
        bgModel = np.zeros((1,65),np.float64)
        fgModel = np.zeros((1,65),np.float64)
        cv2.grabCut(image, mask, faceBox, bgModel, fgModel, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)
        backgrndMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),True,False).astype('uint8')
    
    backgrndMask = np.broadcast_to(backgrndMask[:,:,np.newaxis], np.shape(image))
    return backgrndMask

def getROI(image, faceBox): 
    if USE_SEGMENTATION:
        widthFrac = SEGMENTATION_WIDTH_FRACTION
        heigtFrac = SEGMENTATION_HEIGHT_FRACTION
    else:
        widthFrac = WIDTH_FRACTION
        heigtFrac = HEIGHT_FRACTION

    # Adjust bounding box
    (x, y, w, h) = faceBox
    widthOffset = int((1 - widthFrac) * w / 2)
    heightOffset = int((1 - heigtFrac) * h / 2)
    faceBoxAdjusted = (x + widthOffset, y + heightOffset,
        int(widthFrac * w), int(heigtFrac * h))

    # Segment
    if USE_SEGMENTATION:
        backgrndMask = segment(image, faceBoxAdjusted)

    else:
        (x, y, w, h) = faceBoxAdjusted
        backgrndMask = np.full(image.shape, True, dtype=bool)
        backgrndMask[y:y+h, x:x+w, :] = False 
    
    backgrndMask = backgrndMask.copy()
    (x, y, w, h) = faceBox
    if REMOVE_EYES:
        backgrndMask[y + int(h * EYE_LOWER_FRAC) : y + int(h * EYE_UPPER_FRAC), :] = True
    if FOREHEAD_ONLY:
        backgrndMask[y + int(h * EYE_LOWER_FRAC) :, :] = True

    roi = np.ma.array(image.copy(), mask=backgrndMask) # Masked array
    return roi

# Sum of square differences between x1, x2, y1, y2 points for each ROI
def distance(roi1, roi2):
    return sum((roi1[i] - roi2[i])**2 for i in range(len(roi1)))

def getBestROI(frame, faceCascade, previousFaceBox):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE), flags=cv2.CASCADE_SCALE_IMAGE)
    roi = None
    faceBox = None

    # If no face detected, use ROI from previous frame
    if len(faces) == 0:
        faceBox = previousFaceBox

    # if many faces detected, use one closest to that from previous frame
    elif len(faces) > 1:
        if previousFaceBox is not None:
            # Find closest
            minDist = float("inf")
            for face in faces:
                if distance(previousFaceBox, face) < minDist:
                    faceBox = face
        else:
            # Chooses largest box by area (most likely to be true face)
            maxArea = 0
            for face in faces:
                if (face[2] * face[3]) > maxArea:
                    faceBox = face

    # If only one face dectected, use it!
    else:
        faceBox = faces[0]

    if faceBox is not None:
        if ADD_BOX_ERROR:
            noise = []
            for i in range(4):
                noise.append(random.uniform(-BOX_ERROR_MAX, BOX_ERROR_MAX))
            (x, y, w, h) = faceBox
            x1 = x + int(noise[0] * w)
            y1 = y + int(noise[1] * h)
            x2 = x + w + int(noise[2] * w)
            y2 = y + h + int(noise[3] * h)
            faceBox = (x1, y1, x2-x1, y2-y1)

        # Show rectangle
        (x, y, w, h) = faceBox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        roi = getROI(frame, faceBox)

    return faceBox, roi

def plotSignals(signals, label):
    seconds = np.arange(0, WINDOW_TIME_SEC, 1.0 / FPS)
    colors = ["r", "g", "b"]
    
    fig, axes = plt.subplots(3, 1, figsize=(4, 6))  # Create a 3x1 grid of subplots
    
    for i in range(3):
        axes[i].plot(seconds, signals[:, i], colors[i])
        # axes[i].set_xlabel('Time (sec)')
        # axes[i].set_ylabel(f'{label} {i+1}', fontsize=17)
        axes[i].set_ylabel(f'transformed signal {i+1}')
        # axes[i].tick_params(axis='x')
        # axes[i].tick_params(axis='y')

    axes[0].set_title('ica components')
    axes[2].set_xlabel('Time (sec)')
    plt.tight_layout()  # Ensure proper spacing between subplots
    plt.show()
def plotSignalsAndSpectrum(signals, label, freqs, powerSpec):
    seconds = np.arange(0, WINDOW_TIME_SEC, 1.0 / FPS)
    colors = ["r", "g", "b"]
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))  # Create a 3x2 grid of subplots
    
    # Determine the y-limits for the signals and spectra
    signal_min = np.min(signals)
    signal_max = np.max(signals)
    spectrum_min = np.min(powerSpec)
    spectrum_max = np.max(powerSpec)
    
    for i in range(3):
        # Plot the signal on the left side
        axes[i, 0].plot(seconds, signals[:, i], colors[i])
        axes[i, 0].set_ylabel(f'transformed signal {i+1}')
        axes[i, 0].set_xlabel('Time (sec)')
        axes[i, 0].set_ylim(signal_min, signal_max)  # Set y-limits for signals
        
        # Plot the corresponding spectrum on the right side
        idx = np.argsort(freqs)
        axes[i, 1].plot(freqs[idx]*60, powerSpec[idx, i], colors[i])
        axes[i, 1].set_xlabel("Beats Per Minute (BPM)")
        axes[i, 1].set_ylabel("Power")
        axes[i, 1].set_xlim([0.5*60, 4*60])
        axes[i, 1].set_ylim(spectrum_min, spectrum_max)  # Set y-limits for spectra
    
    axes[0, 0].set_title('ICA Components')
    axes[0, 1].set_title('FFT Power Spectrum')
    axes[2, 0].set_xlabel('Time (sec)')
    
    plt.tight_layout()  # Ensure proper spacing between subplots
    plt.show()

def plotSpectrum(freqs, powerSpec):
    idx = np.argsort(freqs)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    colors = ["r", "g", "b"]
    for i in range(3):
        plt.plot(freqs[idx]*60, powerSpec[idx, i], colors[i])

    temp = powerSpec[idx, :]
    plt.xlabel("Beats Per Minute (BPM)")
    plt.ylabel("Power")
    plt.legend(['ica component #1', 'ica component #2', 'ica component #3'])
    plt.xlim([0.5*60, 4*60])
    plt.title('FFT power spectrum of the ICA components')
    plt.show()

def getHeartRate(window, lastHR):
    # Normalize across the window to have zero-mean and unit variance
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    normalized = (window - mean) / std
    
    # # Separate into three source signals using ICA
    ica = FastICA()
    # srcSig = ica.fit_transform(reduced)
    srcSig = ica.fit_transform(normalized)
    temp = np.array(srcSig)
    np.save('src_sig_frames.npy', temp)
    # srcSig = reduced
    # srcSig = normalized

    # Find power spectrum
    powerSpec = np.abs(np.fft.fft(srcSig, axis=0))**2
    freqs = np.fft.fftfreq(WINDOW_SIZE, 1.0 / FPS)

    # Find heart rate
    maxPwrSrc = np.max(powerSpec, axis=1)
    # print(powerSpec.shape, maxPwrSrc.shape)
    validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BPM / SEC_PER_MIN))
    validPwr = maxPwrSrc[validIdx]
    # print(len(validIdx), validIdx[0], validPwr.shape)
    validFreqs = freqs[validIdx]
    
    maxPwrIdx = np.argmax(validPwr)
    hr = validFreqs[maxPwrIdx]
    print(np.round(hr*60*100)/100)

    # plotSignals(normalized, "Normalized color intensity")
    # plotSignals(srcSig, "Source signal strength")
    # plotSpectrum(freqs, powerSpec)
    plotSignalsAndSpectrum(srcSig, "Source signal strength", freqs, powerSpec)

    return hr

colorSig = [] # Will store the average RGB color values in each frame's ROI
heartRates = [] # Will store the heart rate calculated every 1 second
previousFaceBox = None
n_frames_processed = 0
print(f"window size: {WINDOW_SIZE}, fps: {FPS}")
start_time = time.time()
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
try: 
    # while True:
    # loaded_frames = np.load('../video_frames.npy')
    # print(loaded_frames.shape, n_frames)
    for i in tqdm.tqdm(range(n_frames-1), desc="Processing Video"):
        # Capture frame-by-frame
        ret, frame = video.read()
        # frame = loaded_frames[i+1,:,:,:]
        if not ret:
            break
        n_frames_processed += 1
        if n_frames_processed % (int(capture_FPS/FPS)) == 0:
            previousFaceBox, roi = getBestROI(frame, faceCascade, previousFaceBox)

            if (roi is not None) and (np.size(roi) > 0):
                colorChannels = roi.reshape(-1, roi.shape[-1])
                avgColor = colorChannels.mean(axis=0)
                # avgColor = colorChannels[:,2]
                # print(avgColor.shape)
                colorSig.append(avgColor)

            # Calculate heart rate every one second (once have 30-second of data)
            if (len(colorSig) >= WINDOW_SIZE) and (len(colorSig) % np.ceil(FPS) == 0):
                # print("calculating window start and stop")
                windowStart = len(colorSig) - WINDOW_SIZE
                window = colorSig[windowStart : windowStart + WINDOW_SIZE]
                lastHR = heartRates[-1] if len(heartRates) > 0 else None
                heartRates.append(getHeartRate(window, lastHR))

            if np.ma.is_masked(roi):
                roi = np.where(roi.mask == True, 0, roi)
            # cv2.imshow('ROI', roi)
            # cv2.waitKey(1)
            # if n_frames_processed%100==0:
            #     print(f"Frame: {n_frames_processed}, time: {time.time()-start_time}, len: {len(heartRates)}, {len(colorSig)} ")

except KeyboardInterrupt:  # Handle Ctrl+C interruption
    print("Received Ctrl+C. Closing the program...")

print(f"Frame: {n_frames_processed}, time: {time.time()-start_time}, len: {len(heartRates)}, {len(colorSig)} ")
print(videoFile)
print(np.round(np.array(heartRates)*60))
video.release()
cv2.destroyAllWindows()
