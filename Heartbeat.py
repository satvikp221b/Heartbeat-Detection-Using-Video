import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import time
from collections import deque

# Function to calculate heart rate using Auto-correlation instead of FFT
def getHeartRate_auto(window, lastHR, FPS, MIN_HR_BPM=50, MAX_HR_BPM=120, SEC_PER_MIN=60, smooth_factor=0.9):
    """Get Heartrate using Auto-correlation"""
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    normalized = (window - mean) / std
    
    #ICA
    ica = FastICA()
    srcSig = ica.fit_transform(normalized)

    # Perform auto-correlation on the first ICA component (usually green color channel)
    signal = srcSig[:, 0]
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  

    # Find the peaks 
    peak_lags = np.arange(1, len(autocorr))
    valid_lags = (peak_lags * SEC_PER_MIN / FPS >= MIN_HR_BPM) & (peak_lags * SEC_PER_MIN / FPS <= MAX_HR_BPM)
    
    valid_lags = valid_lags[:len(peak_lags)]
    
    #Shifting auto corr to align with valid lags
    if np.any(valid_lags):
        valid_autocorr = autocorr[1:][valid_lags]  
        max_lag_idx = np.argmax(valid_autocorr)
        peak_lag = peak_lags[valid_lags][max_lag_idx]
        hr = SEC_PER_MIN * FPS / peak_lag  # Convert lag to BPM
    else:
        return 0  # Return zero if no valid peaks found
    
    # Smooth heart rate to ensure gradual increase if doubling
    if lastHR:
        if hr > lastHR * 1.5 or hr < lastHR / 1.5:
            hr = smooth_factor * lastHR + (1 - smooth_factor) * hr

    return hr

# Function to calculate heart rate using ICA and FFT, not using because double peaking was frequent
def getHeartRate(window, lastHR, FPS, MIN_HR_BPM=50, MAX_HR_BPM=120, SEC_PER_MIN=60, smooth_factor=0.9):
    """Calculate heart rate using ICA and FFT"""
    # Normalize across the window to have zero-mean and unit variance
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    normalized = (window - mean) / std
    
    # ICA
    ica = FastICA()
    srcSig = ica.fit_transform(normalized)
    
    # Find power spectrum
    powerSpec = np.abs(np.fft.fft(srcSig, axis=0))**2
    freqs = np.fft.fftfreq(len(window), 1.0 / FPS)

    # Peak Detection
    validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BPM / SEC_PER_MIN))
    validFreqs = freqs[validIdx]
    validPwr = powerSpec[validIdx]

    # Ensure validFreqs and validPwr are not empty
    if len(validFreqs) == 0 or len(validPwr) == 0:
        # 
        return lastHR if lastHR is not None else 0

    
    maxPwrIdx = np.argmax(validPwr)
    if maxPwrIdx >= len(validFreqs):
        return lastHR if lastHR is not None else 0

    hr = validFreqs[maxPwrIdx] * 60  # Convert to beats per minute (BPM)

    # Harmonic suppression: if heart rate is a multiple of 60, smooth it
    if lastHR:
        if hr > lastHR * 1.5 or hr < lastHR / 1.5:
            hr = smooth_factor * lastHR + (1 - smooth_factor) * hr

    return hr
def plotSignalsAndAutocorr(signals, autocorr, FPS, WINDOW_TIME_SEC):
    seconds = np.arange(0, WINDOW_TIME_SEC, 1.0 / FPS)
    colors = ["r", "g", "b"]
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))  

    for i in range(3):
        # Plot the signal on the left
        axes[i, 0].plot(seconds, signals[:, i], colors[i])
        axes[i, 0].set_ylabel(f'transformed signal {i+1}')
        axes[i, 0].set_xlabel('Time (sec)')

    # Plot the auto-correlation result on the right
    axes[0, 1].plot(autocorr)
    axes[0, 1].set_title('Auto-correlation')
    axes[0, 1].set_xlabel("Lags")
    
    plt.tight_layout()  
    plt.show()

def process_video_and_calculate_hr(original_video_path, magnified_video_path, FPS, WINDOW_SIZE, results_dir, capture_FPS):
    """# Process the original input video and calculate heart rate using the magnified video"""
    original_video = cv2.VideoCapture(original_video_path)
    magnified_video = cv2.VideoCapture(magnified_video_path)
    
    colorSig = []  
    # Store last 30 heart rate values
    heartRates = deque(maxlen=30)  
    n_frames_processed = 0
    n_frames = int(original_video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(f"{results_dir}/output_with_hr_whitelight_forehead_fft.mp4", fourcc, capture_FPS, (width, height))

    try:
        for i in tqdm.tqdm(range(n_frames), desc="Processing Video"):
            ret_original, frame_original = original_video.read()
            ret_magnified, frame_magnified = magnified_video.read()
            if not (ret_original and ret_magnified):
                break

            n_frames_processed += 1

            # Get average color from the entire magnified frame (already focused on ROI)
            avgColor = frame_magnified.reshape(-1, frame_magnified.shape[-1]).mean(axis=0)
            colorSig.append(avgColor)

            # Every second, calculate heart rate from the last window
            if len(colorSig) >= WINDOW_SIZE:
                window = colorSig[-WINDOW_SIZE:]
                lastHR = np.mean(heartRates) if len(heartRates) > 0 else 0  # Use average of past heart rates
                hr = getHeartRate(window, lastHR, FPS)

    
                heartRates.append(hr)

                # If we don't have 30 frames of heart rate data yet, set HR to 0
                if len(heartRates) < 30:
                    hr = 0
                else:
                    hr = np.mean(heartRates)  # Use the average of the last 30 frames' heart rates

                # Overlay heart rate on the original video frame
                cv2.putText(frame_original, f"HR: {int(hr)} BPM", (width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                output_video.write(frame_original)

    except KeyboardInterrupt:
        print("Process interrupted. Exiting...")

    finally:
        original_video.release()
        magnified_video.release()
        output_video.release()
        cv2.destroyAllWindows()


FPS = 30  # Frames per second for the video
WINDOW_SIZE = FPS * 1  
original_video_path = "videos/input_whitelight.mp4"  # Path to the original input video
magnified_video_path = "results/segmentation/forehead/output_magnified_roi_whitelight.mp4"  # Path to the magnified video
results_dir = "results/"

process_video_and_calculate_hr(original_video_path, magnified_video_path, FPS, WINDOW_SIZE, results_dir, FPS)
