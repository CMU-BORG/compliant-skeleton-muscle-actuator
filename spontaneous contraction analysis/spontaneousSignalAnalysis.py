import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, periodogram
from typeTable import skeletonType

# ==== GLOBAL PARAMETERS ====
INPUT_FOLDER = "data"
OUTPUT_CSV = "summary_results.csv"
PLOT_FOLDER = "freqAnalysis"
SIGNAL_RANGE_THRESHOLD = 44.656253  # Skip files with range below this value
PEAK_PROMINENCE = SIGNAL_RANGE_THRESHOLD
PEAK_DISTANCE = 3  # in number of samples
FILE_SORT_KEY = skeletonType

# ==== MAIN SCRIPT ====

def makeFolders():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(PLOT_FOLDER, exist_ok=True)

def main():
    makeFolders()
    csvFiles = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    results = []

    for filePath in csvFiles:
        fileName = os.path.basename(filePath)
        df = pd.read_csv(filePath)

        if "Second" not in df.columns or "Line distance (um)" not in df.columns:
            continue

        time = df["Second"].values
        signal = df["Line distance (um)"].values
        signalRange = np.ptp(signal)

        if signalRange < SIGNAL_RANGE_THRESHOLD:
            continue

        duration = time[-1] - time[0]
        if duration <= 0:
            continue

        # Method 1: FFT
        freqs, amps, fftSummary = analyzeWithFFT(signal, time, duration)

        # Save FFT result to CSV
        fft_df = pd.DataFrame({
            "Frequency (Hz)": freqs,
            "Amplitude (um)": 2 * np.sqrt(amps)
        })
        fft_csv_name = os.path.splitext(fileName)[0] + "_fft.csv"
        fft_csv_path = os.path.join(PLOT_FOLDER, fft_csv_name)
        fft_df.to_csv(fft_csv_path, index=False)

        # Method 2: Peak Counting
        peakFreq, peakAmp, numPeaks, peakTimes = analyzeWithPeaks(signal, time, duration)

        # Save plot
        plotResults(fileName, time, signal, freqs, amps, fftSummary, peakTimes)

        # Store results
        p, s, d, psKey = extractFileTags(fileName)
        sortGroup = FILE_SORT_KEY.get(psKey, "Unknown")

        resultRow = {
            "File": fileName,
            "Group": sortGroup,
            "P": p,
            "S": s,
            "D": d,
            "Signal range (um)": signalRange,
            "Duration (s)": duration,
            "Peak count": numPeaks,
            "Peak count freq (Hz)": peakFreq,
            "Peak count amp (um)": peakAmp,
        }
        for i, (f, a) in enumerate(zip(fftSummary["frequencies"], fftSummary["amplitudes"])):
            resultRow[f"FFT freq {i+1} (Hz)"] = f
            resultRow[f"FFT amp {i+1} (um)"] = a
        results.append(resultRow)


    # Save summary CSV
    resultsDf = pd.DataFrame(results)
    resultsDf = resultsDf.sort_values(by=["Group", "D"], ascending=[True, True])
    resultsDf.to_csv(OUTPUT_CSV, index=False)
    print(f"Analysis complete. Results saved to {OUTPUT_CSV} and plots to {PLOT_FOLDER}")

# ==== FUNCTION DEFINITIONS ====

def analyzeWithFFT(signal, time, duration):
    n = len(time)
    fs = 1 / np.median(np.diff(time))  # Sampling frequency

    # Define frequency bounds
    fMin = 1 / (2 * duration)
    fMax = fs

    freqs, power = periodogram(signal, fs=fs)
    validIdx = np.where((freqs >= fMin) & (freqs <= fMax))
    freqs = freqs[validIdx]
    power = power[validIdx]

    # Find 3 most prominent peaks
    peakIdxs, _ = find_peaks(power)
    prominentIdxs = peakIdxs[np.argsort(power[peakIdxs])][-3:][::-1]
    fftFreqs = freqs[prominentIdxs]
    fftAmps = 2 * np.sqrt(power[prominentIdxs])  # Convert to amplitude

    return freqs, power, {
        "frequencies": fftFreqs.tolist(),
        "amplitudes": fftAmps.tolist()
    }

def analyzeWithPeaks(signal, time, duration):
    invertedSignal = -signal  # Invert the signal so valleys become peaks
    peakIdxs, _ = find_peaks(invertedSignal, prominence=PEAK_PROMINENCE, distance=PEAK_DISTANCE)
    peakTimes = time[peakIdxs]
    peakValues = signal[peakIdxs]  # Use original signal to report true amplitude
    peakAmp = np.max(np.max(signal) - np.min(signal))
    if len(peakValues) < 1:
        return 0.0, peakAmp, 0, []
    peakFreq = len(peakIdxs) / duration
    return peakFreq, peakAmp, len(peakIdxs), peakTimes

def plotResults(fileName, time, signal, freqs, amps, fftSummary, peakTimes):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)

    # Top: FFT Plot
    axs[0].plot(freqs, 2 * np.sqrt(amps), label="FFT")
    axs[0].set_title("FFT Analysis")
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Amplitude (um)")
    for f, a in zip(fftSummary["frequencies"], fftSummary["amplitudes"]):
        axs[0].plot(f, a, "ro")
        axs[0].annotate(f"{f:.2f} Hz", (f, a), textcoords="offset points", xytext=(0,5), ha='center')

    # Bottom: Peak Counting
    axs[1].plot(time, signal, label="Signal")
    axs[1].plot(peakTimes, [signal[np.abs(time - t).argmin()] for t in peakTimes], "rx", label="Peaks")
    axs[1].set_title("Peak Counting")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Line distance (um)")
    axs[1].legend()

    plotPath = os.path.join(PLOT_FOLDER, os.path.splitext(fileName)[0] + ".png")
    plt.savefig(plotPath)
    plotPath = os.path.join(PLOT_FOLDER, os.path.splitext(fileName)[0] + ".svg")
    plt.savefig(plotPath)
    plt.close()

def extractFileTags(fileName):
    """Extracts numeric values for P, S, and D from file names like P{d}_S{d}_D{d}."""
    match = re.match(r"P(\d+)_S(\d+)_D(\d+)", fileName)
    if match:
        p = int(match.group(1))
        s = int(match.group(2))
        d = int(match.group(3))
        return p, s, d, f"P{p}_S{s}"
    else:
        return None, None, None

if __name__ == "__main__":
    #makeFolders()
    main()