import wave
import audioop
import glob

def amplify_wav(filepath, factor=3.5):
    print(f"Processing {filepath}...")
    with wave.open(filepath, 'rb') as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    
    # Multiply each sample by the given factor
    new_frames = audioop.mul(frames, params.sampwidth, factor)
    
    with wave.open(filepath, 'wb') as w:
        w.setparams(params)
        w.writeframes(new_frames)
    print(f"Done amplifying {filepath}.")

if __name__ == "__main__":
    files = glob.glob("dataset/amicorpus/*/audio/*.wav")
    for f in files:
        amplify_wav(f, factor=3.5)
