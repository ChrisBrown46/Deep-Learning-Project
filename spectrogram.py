import librosa
import numpy as np
import pickle

def scale_minmax(X, min=0.0, max=1.0):
    """ Returns a normalized version of the input numpy array """
    
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def load_wav(file_name):
    """ Returns a tuple of (sound, sr) """
    
    y, sr = librosa.load(file_name)
    sound, _ = librosa.effects.trim(y)
    return sound, sr

def load_image(file_name):
    """ Returns a tuple of (sound, sr) """
    
    with open(file_name, "rb") as fin:
        return pickle.load(fin)

def save_wav(file_name, sound, sr):
    """ Librosa is deprecating in favor of PySoundFile  """
    
    librosa.output.write_wav(file_name, sound, sr)
    
def save_image(file_name, image, sr):
    """ Pickles the data as a tuple of (image, sr) """
    
    with open(file_name, "wb+") as fout:
        pickle.dump((image, sr), fout)

def wav_to_image(sound, sr):
    """ Converts (sound, sr) into a 3-D mel-log spectrogram """
    
    # Construct the mel spectrogram
    spectrogram = librosa.feature.melspectrogram(
        sound,
        sr=sr,
        n_fft=2048, 
        hop_length=512, 
        n_mels=128
    )
    
    # Log the magnitude - add a small number to avoid log(0)
    spectrogram = np.log(spectrogram + 1e-9)

    # Reshape to have 3 channels
    if len(spectrogram.shape) == 2:
        spectrogram = np.stack((spectrogram,) * 3, axis=-1)
    elif len(spectrogram.shape) == 3:
        spectrogram = np.concatenate((spectrogram,) * 3, axis=-1)
    else:
        raise ValueError("Spectogram has strange shape", spectrogram.shape)

    return spectrogram

def image_to_wav(spectrogram, sr):
    """ Converts (image, sr) into a 2-D audio numpy array """
    
    # Remove extra channels
    spectrogram = spectrogram[:, :, 0]
    
    # Undo the log operation and remove the small number
    spectrogram = np.exp(spectrogram) - 1e-9
    
    # Invert a mel power spectrogram to audio using Griffin-Lim
    sound = librosa.feature.inverse.mel_to_audio(
        spectrogram,
        sr=sr,
        n_fft=2048,
        hop_length=512,
    )
    
    return sound

def wav_to_pyplot_image(sound, sr):
    """ Converts (sound, sr) into a 3-D image for pyplot to render """
    
    image = wav_to_image(sound, sr)
    
    image = scale_minmax(image, 0, 255).astype(np.uint8)
    image = np.flip(image, axis=0)  # put low frequencies at the bottom in image
    
    return image
