import librosa
import numpy as np

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def wav_to_image(file_name):
    # Load the data
    import librosa
    y, sr = librosa.load(file_name)

    # Trim silent edges
    sound, _ = librosa.effects.trim(y)

    # Construct the mel-log spectrogram
    spectrogram = librosa.feature.melspectrogram(
        sound,
        sr=sr,
        n_fft=2048, 
        hop_length=512, 
        n_mels=128
    )
    
    spectrogram = np.log(spectrogram + 1e-9)  # add small number to avoid log(0)

    # Reshape to have 3 channels
    if len(spectrogram.shape) == 2:
        spectrogram = np.stack((spectrogram,) * 3, axis=-1)
    elif len(spectrogram.shape) == 3:
        spectrogram = np.concatenate((spectrogram,) * 3, axis=-1)
    else:
        raise ValueError("Spectogram has strange shape", spectrogram.shape)

    return spectrogram

def wav_to_pyplot_image(file_name):
    image = wav_to_image(file_name)
    
    image = scale_minmax(image, 0, 255).astype(np.uint8)
    image = np.flip(image, axis=0)  # put low frequencies at the bottom in image
    
    return image
