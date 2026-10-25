
import numpy as np


# Score posture
def analyze_posture(df):
    if df.empty or "shoulder_difference" not in df.columns:
        return 0

    values = df["shoulder_difference"].dropna()

    if len(values) == 0:
        return 0

    alignment_score = (values < 0.03).mean() * 100

    if "shoulder_movement" not in df.columns:
        return round(alignment_score, 2)

    movement_values = df["shoulder_movement"].dropna()

    if len(movement_values) == 0:
        return round(alignment_score, 2)

    average_movement = movement_values.mean()

    if average_movement < 0.01:
        stability_score = 100
    elif average_movement < 0.03:
        stability_score = 80
    elif average_movement < 0.06:
        stability_score = 60
    else:
        stability_score = 40

    return round((alignment_score * 0.7) + (stability_score * 0.3), 2)


# Score head movement
def analyze_head_movement(df):
    if df.empty or "head_movement" not in df.columns:
        return 0

    values = df["head_movement"].dropna()

    if len(values) == 0:
        return 0

    average_movement = values.mean()

    if average_movement < 0.01:
        score = 100
    elif average_movement < 0.03:
        score = 80
    elif average_movement < 0.06:
        score = 60
    else:
        score = 40

    return round(score, 2)


# Score approximate gaze
def analyze_head_alignment(df):
    required_columns = {"head_yaw", "head_pitch", "head_roll"}

    if df.empty or not required_columns.issubset(df.columns):
        return 0

    values = df[list(required_columns)].dropna()

    if len(values) == 0:
        return 0

    roll_distance = values["head_roll"].abs()
    roll_distance = np.minimum(
        roll_distance,
        np.minimum(
            (values["head_roll"] - 180).abs(),
            (values["head_roll"] + 180).abs(),
        ),
    )

    aligned = (
        (values["head_yaw"].abs() <= 20)
        & (values["head_pitch"].abs() <= 15)
        & (roll_distance <= 15)
    )

    return round(aligned.mean() * 100, 2)


def analyze_gaze(df):
    required_columns = {
        "gaze_ratio",
        "head_yaw",
        "head_pitch",
        "head_roll",
    }

    if df.empty or not required_columns.issubset(df.columns):
        return 0

    values = df[
        ["gaze_ratio", "head_yaw", "head_pitch", "head_roll"]
    ].dropna()

    if len(values) == 0:
        return 0

    roll_distance = values["head_roll"].abs()
    roll_distance = np.minimum(
        roll_distance,
        np.minimum(
            (values["head_roll"] - 180).abs(),
            (values["head_roll"] + 180).abs(),
        ),
    )

    centered = (
        (values["gaze_ratio"] >= 0.35)
        & (values["gaze_ratio"] <= 0.65)
        & (values["head_yaw"].abs() <= 25)
        & (values["head_pitch"].abs() <= 20)
        & (roll_distance <= 20)
    ).sum()

    return round((centered / len(values)) * 100, 2)


# Score speaking
def analyze_speaking(audio_results):
    if not audio_results.get("transcription_available", True):
        return None

    words = audio_results["word_count"]
    rate = audio_results["speaking_rate"]
    fillers = audio_results["filler_count"]
    long_pauses = audio_results.get("long_pause_count", 0)

    score = 100

    if words < 10:
        score -= 30

    if rate < 60 or rate > 180:
        score -= 20

    score -= min(fillers * 5, 30)
    score -= min(long_pauses * 10, 30)

    return max(0, round(score, 2))


def analyze_confidence(audio_results):
    if not audio_results.get("transcription_available", False):
        return None

    activity = audio_results.get("voice_activity_ratio", 0)
    stability = audio_results.get("volume_stability", 0)
    long_pauses = audio_results.get("long_pause_count", 0)

    if activity == 0 and stability == 0:
        return 0

    activity_score = min(activity * 100 / 0.7, 100)
    score = (activity_score * 0.35) + (stability * 0.65)
    score -= min(long_pauses * 10, 30)

    return max(0, round(score, 2))


# Calculate overall score
def calculate_overall_score(
    posture,
    head_alignment,
    movement,
    gaze,
    speaking,
    confidence,
):
    weighted_scores = [
        (posture, 0.20),
        (head_alignment, 0.20),
        (movement, 0.15),
        (gaze, 0.15),
        (speaking, 0.20),
        (confidence, 0.10),
    ]

    available_scores = [
        (score, weight)
        for score, weight in weighted_scores
        if score is not None
    ]

    if not available_scores:
        return 0

    total_weight = sum(weight for _, weight in available_scores)
    overall = sum(
        score * weight
        for score, weight in available_scores
    ) / total_weight

    return round(overall, 2)


def generate_improvement_tips(scores, audio_results=None):
    """Return the most useful practice tips for the current interview."""
    tips = []

    score_tips = [
        (
            "Gaze",
            "Try to look toward the camera more consistently instead of looking down or to the side.",
        ),
        (
            "Head Alignment",
            "Keep your head level and face the camera while answering.",
        ),
        (
            "Posture",
            "Sit upright with both shoulders relaxed and approximately aligned.",
        ),
        (
            "Head Movement",
            "Keep head movements controlled and avoid frequent unnecessary movement.",
        ),
        (
            "Speaking",
            "Give a complete answer with a steady pace and clear pronunciation.",
        ),
        (
            "Confidence (audio proxy)",
            "Use a steady speaking volume and reduce long pauses.",
        ),
    ]

    for metric, tip in score_tips:
        score = scores.get(metric)
        if score is not None and score < 60:
            tips.append((score, tip))

    if audio_results:
        total_fillers = sum(
            audio.get("filler_count", 0)
            for audio in audio_results
        )
        total_pauses = sum(
            audio.get("long_pause_count", 0)
            for audio in audio_results
        )

        if total_fillers > 0:
            tips.append((
                55,
                "Replace filler words such as 'um', 'uh', and 'like' with a short pause.",
            ))

        if total_pauses > 0:
            tips.append((
                55,
                "Plan your main points before answering so you can reduce long pauses.",
            ))

    tips.sort(key=lambda item: item[0])
    return [tip for _, tip in tips[:3]]
