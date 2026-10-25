import os
import threading

import pandas as pd

from audio_analysis import record_audio, transcribe_audio, analyze_speech
from question_bank import get_categories, get_questions
from reporting import generate_pdf_report
from scoring import (
    analyze_gaze,
    analyze_confidence,
    analyze_head_alignment,
    analyze_head_movement,
    analyze_posture,
    analyze_speaking,
    calculate_overall_score,
    generate_improvement_tips,
)
from vision import analyze_interview_video


def choose_category():
    categories = get_categories()

    print("\nInterview categories:")
    for number, category in enumerate(categories, start=1):
        print(str(number) + ". " + category)

    while True:
        try:
            choice = int(input("Choose a category: "))
            if 1 <= choice <= len(categories):
                return categories[choice - 1]
        except ValueError:
            pass

        print("Please enter a valid category number.")


def choose_question_count(maximum):
    while True:
        try:
            count = int(input("Number of questions (1-" + str(maximum) + "): "))
            if 1 <= count <= maximum:
                return count
        except ValueError:
            pass

        print("Please enter a valid question count.")


def choose_duration():
    while True:
        try:
            duration = int(input("Answer time per question in seconds: "))
            if duration > 0:
                return duration
        except ValueError:
            pass

        print("Please enter a positive number of seconds.")


def record_question_answer(question, question_number, duration):
    audio_file = "answer_" + str(question_number) + ".wav"
    vision_file = "vision_" + str(question_number) + ".csv"
    stop_event = threading.Event()

    print("\nQuestion " + str(question_number) + ": " + question)
    input("Press Enter when you are ready to answer...")

    camera_thread = threading.Thread(
        target=analyze_interview_video,
        args=(stop_event, vision_file),
    )
    camera_thread.start()

    recording_successful = False
    try:
        print("Recording for " + str(duration) + " seconds...")
        record_audio(duration, audio_file)
        recording_successful = os.path.exists(audio_file)
    except Exception as error:
        print("Microphone recording failed:", error)
    finally:
        stop_event.set()
        camera_thread.join()

    if not recording_successful:
        raise RuntimeError(
            "Audio recording failed for question " + str(question_number)
        )

    transcript = transcribe_audio(audio_file)
    audio_data = analyze_speech(transcript, duration, audio_file)

    visual_data = pd.DataFrame()
    if os.path.exists(vision_file):
        visual_data = pd.read_csv(vision_file)
    else:
        print("No webcam data was saved for this question.")

    print("Transcript:", transcript)

    return {
        "answer": {
            "question": question,
            "answer": transcript,
        },
        "audio_data": audio_data,
        "visual_data": visual_data,
    }


def run_interview():
    print("\nWelcome to InterviewSense Pro!")
    print("Answer each question aloud while facing the camera.")

    category = choose_category()
    available_questions = get_questions(category)
    question_count = choose_question_count(len(available_questions))
    duration = choose_duration()
    questions = available_questions[:question_count]

    answers = []
    audio_results = []
    vision_results = []

    for question_number, question in enumerate(questions, start=1):
        result = record_question_answer(
            question,
            question_number,
            duration,
        )
        answers.append(result["answer"])
        audio_results.append(result["audio_data"])

        if not result["visual_data"].empty:
            vision_results.append(result["visual_data"])

    if vision_results:
        vision_data = pd.concat(vision_results, ignore_index=True)
    else:
        vision_data = pd.DataFrame()

    posture_score = analyze_posture(vision_data)
    head_alignment_score = analyze_head_alignment(vision_data)
    movement_score = analyze_head_movement(vision_data)
    gaze_score = analyze_gaze(vision_data)

    speaking_scores = []
    for audio_data in audio_results:
        speaking_score_item = analyze_speaking(audio_data)
        if speaking_score_item is not None:
            speaking_scores.append(speaking_score_item)
    speaking_score = round(
        sum(speaking_scores) / len(speaking_scores),
        2,
    ) if speaking_scores else None
    confidence_scores = [
        analyze_confidence(audio_data)
        for audio_data in audio_results
    ]
    confidence_scores = [
        score for score in confidence_scores
        if score is not None
    ]
    confidence_score = round(
        sum(confidence_scores) / len(confidence_scores),
        2,
    ) if confidence_scores else None

    overall_score = calculate_overall_score(
        posture_score,
        head_alignment_score,
        movement_score,
        gaze_score,
        speaking_score,
        confidence_score,
    )

    scores = {
        "Posture": posture_score,
        "Head Alignment": head_alignment_score,
        "Head Movement": movement_score,
        "Gaze": gaze_score,
        "Speaking": speaking_score,
        "Confidence (audio proxy)": confidence_score,
        "Overall": overall_score,
    }
    improvement_tips = generate_improvement_tips(
        scores,
        audio_results
    )

    print("\n==============================")
    print(" INTERVIEWSENSE PRO REPORT")
    print("==============================")
    for metric, score in scores.items():
        if score is None:
            print(metric + ": N/A")
        else:
            print(metric + ": " + str(score) + " / 100")

    print("\nHow to improve:")
    if improvement_tips:
        for tip in improvement_tips:
            print("- " + tip)
    else:
        print("No major improvement areas were detected.")

    report_file = generate_pdf_report(
        scores,
        answers,
        output_file="interview_report.pdf",
        tips=improvement_tips,
    )
    print("\nPDF report created:", os.path.abspath(report_file))
    print("Interview completed!")


if __name__ == "__main__":
    run_interview()
