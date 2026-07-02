import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
import os

try:
    import noisereduce as nr
except ImportError:
    nr = None


def load_audio(file_path, target_sr=16000):
    """Loads an audio file and resamples it to target_sr."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    y, sr = librosa.load(file_path, sr=target_sr)
    return y, sr


def apply_bandpass_filter(y, sr, low_cut=50.0, high_cut=8000.0, order=5):
    """Applies a Butterworth bandpass filter using scipy.signal."""
    nyquist = 0.5 * sr
    low = low_cut / nyquist
    high = high_cut / nyquist
    # Ensure frequencies are in [0, 1] range
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    b, a = butter(order, [low, high], btype='band')
    y_filtered = filtfilt(b, a, y)
    return y_filtered


def remove_noise(y, sr):
    """Suppresses background noise using the noisereduce library."""
    if nr is not None:
        # Stationary noise suppression using spectral noise profiling
        y_denoised = nr.reduce_noise(y=y, sr=sr)
        return y_denoised
    else:
        print("Warning: noisereduce library not installed. Skipping noise reduction.")
        return y


def remove_speaker_transcript(y, sr, transcript_csv_path, keep_speaker="participant"):
    """
    Parses a DAIC-WOZ transcript CSV and removes non-participant speech segments.
    The DAIC-WOZ transcript format is:
    start_time\tstop_time\tspeaker\tvalue
    or comma-separated.
    """
    if not os.path.exists(transcript_csv_path):
        print(f"Warning: Transcript path {transcript_csv_path} does not exist. Skipping speaker isolation.")
        return y

    try:
        # Load CSV (handles tab or comma separation)
        try:
            df = pd.read_csv(transcript_csv_path, sep='\t')
            if 'speaker' not in df.columns:
                df = pd.read_csv(transcript_csv_path)
        except Exception:
            df = pd.read_csv(transcript_csv_path)

        # Clean column names
        df.columns = [col.strip().lower() for col in df.columns]

        required_cols = {'start_time', 'stop_time', 'speaker'}
        if not required_cols.issubset(set(df.columns)):
            print(f"Warning: Transcript CSV missing required columns: {required_cols}. Found: {df.columns}. Skipping speaker isolation.")
            return y

        # Create a mask for speaker sections to keep
        participant_mask = np.zeros_like(y, dtype=bool)

        for _, row in df.iterrows():
            speaker = str(row['speaker']).strip().lower()
            start_sec = float(row['start_time'])
            stop_sec = float(row['stop_time'])
            
            # Map time to sample indices
            start_idx = int(start_sec * sr)
            stop_idx = int(stop_sec * sr)
            
            # Bound check
            start_idx = max(0, min(start_idx, len(y)))
            stop_idx = max(0, min(stop_idx, len(y)))

            if keep_speaker in speaker:
                participant_mask[start_idx:stop_idx] = True

        # Extract only participant audio (concatenating active segments)
        participant_audio = y[participant_mask]
        
        # Fallback if transcript keeps nothing
        if len(participant_audio) == 0:
            print("Warning: Speaker filtering returned empty audio. Keeping original audio.")
            return y
            
        return participant_audio

    except Exception as e:
        print(f"Error parsing transcript CSV: {e}. Skipping speaker isolation.")
        return y


def normalize_volume(y, target_db=-20.0):
    """Normalizes the volume/RMS level of the audio signal to target_db."""
    rms = np.sqrt(np.mean(y**2))
    if rms == 0:
        return y
    target_rms = 10 ** (target_db / 20.0)
    gain = target_rms / rms
    return y * gain


def trim_silence(y, sr, top_db=25):
    """Splits audio on silence and returns non-silent parts concatenated."""
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y
    chunks = [y[start:end] for start, end in intervals]
    return np.concatenate(chunks)


def segment_audio(y, sr, frame_len_sec=2.0, hop_len_sec=1.0):
    """
    Slices a continuous audio signal into overlapping segments.
    Returns:
        List of NumPy arrays representing each frame.
    """
    frame_len = int(frame_len_sec * sr)
    hop_len = int(hop_len_sec * sr)
    
    if len(y) < frame_len:
        # Pad with zeros if the signal is shorter than one frame
        pad_width = frame_len - len(y)
        return [np.pad(y, (0, pad_width), 'constant')]
        
    frames = []
    for start in range(0, len(y) - frame_len + 1, hop_len):
        frames.append(y[start : start + frame_len])
    return frames


def preprocess_pipeline(audio_path, transcript_path=None, target_sr=16000, frame_len_sec=2.0, hop_len_sec=1.0):
    """
    Performs the full preprocessing workflow:
    1. Resampling and loading
    2. Band-pass filtering
    3. Noise reduction
    4. Speaker isolation (if transcript provided)
    5. Volume normalization
    6. Silence trimming
    7. Segmentation into micro-frames
    """
    # 1. Load Audio
    y, sr = load_audio(audio_path, target_sr=target_sr)
    
    # 2. Bandpass Filter
    y = apply_bandpass_filter(y, sr)
    
    # 3. Noise Reduction
    y = remove_noise(y, sr)
    
    # 4. Speaker Isolation
    if transcript_path:
        y = remove_speaker_transcript(y, sr, transcript_path)
        
    # 5. Volume Normalization
    y = normalize_volume(y)
    
    # 6. Silence Trimming
    y = trim_silence(y, sr)
    
    # 7. Segmentation
    frames = segment_audio(y, sr, frame_len_sec, hop_len_sec)
    
    return frames, sr
