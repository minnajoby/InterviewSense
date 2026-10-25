
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import numpy as np
import re


NO_SPEECH_MESSAGE = (
    "No speech detected. Please check your microphone and speak clearly."
)
SPEECH_NOT_UNDERSTOOD_MESSAGE = "Speech could not be understood."
TRANSCRIPTION_SERVICE_UNAVAILABLE_MESSAGE = "Transcription service unavailable."
TRANSCRIPTION_FAILED_MESSAGE = "Transcription failed."
INSUFFICIENT_SPEECH_MESSAGE = (
    "Not enough clear speech was detected to score this answer."
)
MINIMUM_TRANSCRIPT_WORDS = 3


def _contains_speech_signal(filename):
    """Return whether the recording contains enough audio energy to transcribe."""
    audio, _ = sf.read(filename, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if len(audio) == 0:
        return False

    audio = np.nan_to_num(audio)
    return float(np.sqrt(np.mean(audio ** 2))) >= 0.005


# Record the student's answer
def record_audio(duration, filename):
    sample_rate = 16000

    print("Audio recording started...")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    sf.write(filename, audio, sample_rate)

    print("Audio recording completed.")

    return filename


# Convert speech into text
def transcribe_audio(filename):
    recognizer = sr.Recognizer()

    try:
        if not _contains_speech_signal(filename):
            return NO_SPEECH_MESSAGE

        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)

        transcript = recognizer.recognize_google(audio)
        return transcript

    except sr.UnknownValueError:
        return SPEECH_NOT_UNDERSTOOD_MESSAGE

    except sr.RequestError:
        return TRANSCRIPTION_SERVICE_UNAVAILABLE_MESSAGE

    except Exception as error:
        print("Audio error:", error)
        return TRANSCRIPTION_FAILED_MESSAGE


# Analyze the student's speaking
def analyze_speech(transcript, duration, audio_file=None):
    transcription_status = "ok"

    if transcript == NO_SPEECH_MESSAGE:
        transcription_status = "no_speech"
    elif transcript == SPEECH_NOT_UNDERSTOOD_MESSAGE:
        transcription_status = "not_understood"
    elif transcript == TRANSCRIPTION_SERVICE_UNAVAILABLE_MESSAGE:
        transcription_status = "service_unavailable"
    elif transcript == TRANSCRIPTION_FAILED_MESSAGE:
        transcription_status = "failed"

    transcription_available = transcription_status == "ok"
    words = transcript.split() if transcription_available else []

    word_count = len(words)

    if (
        transcription_available
        and word_count < MINIMUM_TRANSCRIPT_WORDS
    ):
        transcription_status = "insufficient_speech"
        transcription_available = False

    if duration > 0:
        speaking_rate = word_count / (duration / 60)
    else:
        speaking_rate = 0

    filler_words = [
        "um", "uh", "erm", "actually",
        "basically", "like"
    ]

    transcript_words = []
    if transcription_available:
        transcript_words = re.findall(
            r"\b[\w']+\b",
            transcript.lower()
        )

    filler_count = 0

    for word in transcript_words:
        if word in filler_words:
            filler_count += 1

    long_pause_count = 0
    long_pause_seconds = 0.0
    voice_activity_ratio = 0.0
    volume_stability = 0.0

    if audio_file:
        audio, sample_rate = sf.read(audio_file, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        frame_size = max(1, int(sample_rate * 0.1))
        frame_rms = []
        for start in range(0, len(audio), frame_size):
            frame = audio[start:start + frame_size]
            if len(frame):
                frame_rms.append(float(np.sqrt(np.mean(frame ** 2))))

        if frame_rms:
            rms_values = np.array(frame_rms)
            noise_floor = np.percentile(rms_values, 20)
            speech_peak = np.percentile(rms_values, 90)
            threshold = max(0.01, noise_floor + (speech_peak - noise_floor) * 0.2)
            voiced = rms_values >= threshold
            voice_activity_ratio = float(voiced.mean())

            pause_start = None
            for index, is_voiced in enumerate(voiced):
                if not is_voiced and pause_start is None:
                    pause_start = index
                elif is_voiced and pause_start is not None:
                    pause_duration = (index - pause_start) * 0.1
                    if pause_duration >= 0.8:
                        long_pause_count += 1
                        long_pause_seconds += pause_duration
                    pause_start = None

            if pause_start is not None:
                pause_duration = (len(voiced) - pause_start) * 0.1
                if pause_duration >= 0.8:
                    long_pause_count += 1
                    long_pause_seconds += pause_duration

            voiced_rms = rms_values[voiced]
            if len(voiced_rms) > 1 and voiced_rms.mean() > 0:
                variation = voiced_rms.std() / voiced_rms.mean()
                volume_stability = max(0.0, 100.0 - min(variation * 100, 100))

    return {
        "transcription_status": transcription_status,
        "transcription_available": transcription_available,
        "word_count": word_count,
        "speaking_rate": round(speaking_rate, 2),
        "filler_count": filler_count,
        "long_pause_count": long_pause_count,
        "long_pause_seconds": round(long_pause_seconds, 2),
        "voice_activity_ratio": round(voice_activity_ratio, 2),
        "volume_stability": round(volume_stability, 2),
    }
