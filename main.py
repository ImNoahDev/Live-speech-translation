import os
import pyaudio
import wave
import numpy as np
import pvporcupine
import struct
from faster_whisper import WhisperModel
import secrets

# Initialize Whisper model
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# Parameters for recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
THRESHOLD = 2500  # Adjust this threshold based on your environment and microphone sensitivity
SILENT_CHUNKS_TO_STOP = int(1 * RATE / CHUNK)  # Number of silent chunks to stop recording

# Access key for Porcupine
access_key = os.getenv('porcupine')
keyword_paths = ['./translator.ppn']


def is_silent(chunk):
    # Check if the amplitude of the audio chunk is below the threshold
    return np.max(np.abs(np.frombuffer(chunk, dtype=np.int16))) < THRESHOLD

def record_until_silence():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []
    silent_chunks = 0

    print("Recording...")

    while True:
        try:
            data = stream.read(CHUNK)
        except IOError as ex:
            if ex.errno != pyaudio.paInputOverflowed:
                raise
            data = b'\x00' * CHUNK

        frames.append(data)

        if is_silent(data):
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks > SILENT_CHUNKS_TO_STOP:
            break

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()
    # Convert binary data to numpy array and return
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    return audio_data, RATE

def save_audio(filename, audio_data, sample_rate):
    audio = pyaudio.PyAudio()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

def transcribe_audio(filename):
    segments, info = model.transcribe(filename)
    transcription = ' '.join([segment.text for segment in segments])
    return transcription

def wakeword():
    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=keyword_paths)
    p = pyaudio.PyAudio()
    stream = p.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print("Listening for wake word...")

    try:
        while True:
            pcm = stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Wake word detected!")
                return True

    except KeyboardInterrupt:
        print("Stopping wake word detection")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        porcupine.delete()

while True:
    try:
        if wakeword():
            audio_data, sample_rate = record_until_silence()
            audio_filename = "recorded_audio.wav"
            save_audio(audio_filename, audio_data, sample_rate)

            print("Audio saved to:", audio_filename)

            # Perform transcription
            transcription = transcribe_audio(audio_filename)
            print("Transcription:", transcription)
            

    except Exception as e:
        print(f"Error during execution: {e}")


