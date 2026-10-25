
import streamlit as st
import pandas as pd
import threading
import os

from question_bank import get_categories, get_questions
from vision import analyze_interview_video
from audio_analysis import record_audio, transcribe_audio, analyze_speech
from scoring import (
    analyze_posture,
    analyze_head_alignment,
    analyze_head_movement,
    analyze_gaze,
    analyze_confidence,
    analyze_speaking,
    calculate_overall_score,
    generate_improvement_tips
)
from reporting import generate_pdf_report


# Page setup
st.set_page_config(
    page_title="InterviewSense Pro",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 InterviewSense Pro")
st.write("Practice your mock interview and improve your body language.")


# Store interview progress
if "started" not in st.session_state:
    st.session_state.started = False

if "questions" not in st.session_state:
    st.session_state.questions = []

if "question_index" not in st.session_state:
    st.session_state.question_index = 0

if "answers" not in st.session_state:
    st.session_state.answers = []

if "audio_results" not in st.session_state:
    st.session_state.audio_results = []

if "vision_results" not in st.session_state:
    st.session_state.vision_results = []


# Start screen
if not st.session_state.started:

    category = st.selectbox(
        "Interview category",
        get_categories()
    )

    question_count = st.selectbox(
        "Number of questions",
        [3, 5]
    )

    duration = st.selectbox(
        "Answer time per question",
        [15, 30, 60],
        index=1,
        format_func=lambda seconds: str(seconds) + " seconds"
    )

    if st.button("Start Mock Interview", type="primary"):

        st.session_state.questions = get_questions(category)[:question_count]
        st.session_state.question_index = 0
        st.session_state.answers = []
        st.session_state.audio_results = []
        st.session_state.vision_results = []
        st.session_state.duration = duration
        st.session_state.started = True

        st.rerun()


# Interview screen
if st.session_state.started:

    index = st.session_state.question_index
    questions = st.session_state.questions

    if index < len(questions):

        question = questions[index]

        st.progress(
            (index + 1) / len(questions),
            text="Question " + str(index + 1) + " of " + str(len(questions))
        )

        st.subheader(question)

        # Record this question only once
        if len(st.session_state.answers) == index:

            st.info(
                "Your webcam and microphone will start. "
                "Answer aloud when you are ready."
            )

            audio_file = "answer_" + str(index + 1) + ".wav"
            vision_file = "vision_" + str(index + 1) + ".csv"

            stop_event = threading.Event()

            # Start webcam analysis in the background
            camera_thread = threading.Thread(
                target=analyze_interview_video,
                args=(stop_event, vision_file)
            )

            camera_thread.start()

            # Record audio while the webcam is running
            
            recording_successful = False

            try:
                record_audio(st.session_state.duration, audio_file)
                recording_successful = os.path.exists(audio_file)

            except Exception as error:
                st.error("Microphone recording failed: " + str(error))

            finally:
                stop_event.set()
                camera_thread.join()

            if not recording_successful:
                st.error("Recording failed. Please check your microphone and try again.")
                st.stop()

            # Analyze the recorded answer
            transcript = transcribe_audio(audio_file)

            audio_data = analyze_speech(
                transcript,
                st.session_state.duration,
                audio_file
            )

            st.session_state.answers.append({
                "question": question,
                "answer": transcript,
                "transcription_status": audio_data.get("transcription_status", "ok")
            })

            st.session_state.audio_results.append(audio_data)

            # Save this question's webcam results
            if os.path.exists(vision_file):
                visual_data = pd.read_csv(vision_file)
                st.session_state.vision_results.append(visual_data)

            st.rerun()

        else:

            st.write("**Your answer:**")
            current_answer = st.session_state.answers[index]
            answer_text = current_answer["answer"]
            transcription_status = current_answer.get("transcription_status", "ok")

            if transcription_status == "no_speech":
                st.warning(answer_text)
            elif transcription_status in {
                "not_understood",
                "service_unavailable",
                "failed",
                "insufficient_speech",
            }:
                st.warning(
                    answer_text
                    + " This answer is excluded from speaking-content scoring."
                )
            else:
                st.write(answer_text)

            if index + 1 < len(questions):

                if st.button("Next Question", type="primary"):
                    st.session_state.question_index += 1
                    st.rerun()

            else:

                if st.button("Finish Interview", type="primary"):
                    st.session_state.question_index += 1
                    st.rerun()


    # Final report
    else:

        st.success("Mock interview completed!")

        st.subheader("Your Performance")

        # Combine webcam measurements
        if st.session_state.vision_results:
            vision_data = pd.concat(
                st.session_state.vision_results,
                ignore_index=True
            )
        else:
            vision_data = pd.DataFrame()

        posture_score = analyze_posture(vision_data)
        head_alignment_score = analyze_head_alignment(vision_data)
        movement_score = analyze_head_movement(vision_data)
        gaze_score = analyze_gaze(vision_data)

        # Average speaking score across questions
        speaking_scores = []

        for audio_data in st.session_state.audio_results:
            speaking_score_item = analyze_speaking(audio_data)
            if speaking_score_item is not None:
                speaking_scores.append(speaking_score_item)

        if speaking_scores:
            speaking_score = round(
                sum(speaking_scores) / len(speaking_scores), 2
            )
        else:
            speaking_score = None

        confidence_scores = []

        for audio_data in st.session_state.audio_results:
            confidence_score_item = analyze_confidence(audio_data)
            if confidence_score_item is not None:
                confidence_scores.append(confidence_score_item)

        if confidence_scores:
            confidence_score = round(
                sum(confidence_scores) / len(confidence_scores), 2
            )
        else:
            confidence_score = None

        overall_score = calculate_overall_score(
            posture_score,
            head_alignment_score,
            movement_score,
            gaze_score,
            speaking_score,
            confidence_score
        )

        scores = {
            "Posture": posture_score,
            "Head Alignment": head_alignment_score,
            "Head Movement": movement_score,
            "Gaze": gaze_score,
            "Speaking": speaking_score,
            "Confidence (audio proxy)": confidence_score,
            "Overall": overall_score
        }
        improvement_tips = generate_improvement_tips(
            scores,
            st.session_state.audio_results
        )

        st.metric("Overall Practice Score", str(overall_score) + " / 100")

        first_column, second_column = st.columns(2)

        with first_column:
            st.metric("Posture", str(posture_score) + " / 100")
            st.metric("Head Alignment", str(head_alignment_score) + " / 100")
            st.metric("Head Movement", str(movement_score) + " / 100")
            st.metric("Gaze", str(gaze_score) + " / 100")

        with second_column:
            if speaking_score is None:
                st.metric("Speaking", "N/A")
            else:
                st.metric("Speaking", str(speaking_score) + " / 100")
            if confidence_score is None:
                st.metric("Confidence (audio proxy)", "N/A")
            else:
                st.metric(
                    "Confidence (audio proxy)",
                    str(confidence_score) + " / 100"
                )

        if speaking_score is None:
            st.info(
                "Speaking score is unavailable because no answer had a usable "
                "transcript."
            )
        elif confidence_score is None:
            st.info(
                "Audio confidence is unavailable because no answer had a "
                "usable transcript."
            )

        st.caption(
            "Scores are approximate practice feedback, not a validated "
            "measure of interview ability."
        )

        st.subheader("How to Improve")
        if improvement_tips:
            for tip in improvement_tips:
                st.write("- " + tip)
        else:
            st.success(
                "No major improvement areas were detected. Keep practicing "
                "for more consistent results."
            )

        st.subheader("Answer Summary")

        for answer in st.session_state.answers:
            with st.expander(answer["question"]):
                status = answer.get("transcription_status", "ok")
                if status == "ok":
                    st.write(answer["answer"])
                else:
                    st.warning(answer["answer"])

        # Create PDF report
        if st.button("Generate PDF Report", type="primary"):

            pdf_file = generate_pdf_report(
                scores,
                st.session_state.answers,
                tips=improvement_tips
            )

            with open(pdf_file, "rb") as file:
                st.download_button(
                    "Download PDF",
                    data=file,
                    file_name="InterviewSense_Report.pdf",
                    mime="application/pdf"
                )

        if st.button("Start New Interview"):
            st.session_state.started = False
            st.session_state.questions = []
            st.session_state.question_index = 0
            st.session_state.answers = []
            st.session_state.audio_results = []
            st.session_state.vision_results = []
            st.rerun()
