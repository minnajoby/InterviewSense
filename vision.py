import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time


mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils


# Estimate head orientation
def estimate_head_pose(face, width, height):
    model_points = np.array([
        (0, 0, 0),
        (0, -330, -65),
        (-225, 170, -135),
        (225, 170, -135),
        (-150, -150, -125),
        (150, -150, -125)
    ], dtype=np.float64)

    ids = [1, 152, 263, 33, 287, 57]

    image_points = []

    for idx in ids:
        landmark = face.landmark[idx]

        image_points.append([
            landmark.x * width,
            landmark.y * height
        ])

    image_points = np.array(image_points, dtype=np.float64)

    focal_length = width

    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(
            rotation_matrix
        )

        return float(angles[1]), float(angles[0]), float(angles[2])

    return np.nan, np.nan, np.nan


# Estimate whether the eyes are approximately centered
def estimate_eye_gaze(face):
    landmarks = face.landmark

    left_iris = [468, 469, 470, 471, 472]
    right_iris = [473, 474, 475, 476, 477]

    left_eye = [33, 133]
    right_eye = [362, 263]

    left_iris_x = sum(landmarks[i].x for i in left_iris) / 5
    right_iris_x = sum(landmarks[i].x for i in right_iris) / 5

    left_min = min(landmarks[i].x for i in left_eye)
    left_max = max(landmarks[i].x for i in left_eye)

    right_min = min(landmarks[i].x for i in right_eye)
    right_max = max(landmarks[i].x for i in right_eye)

    left_ratio = (left_iris_x - left_min) / (
        left_max - left_min + 0.000001
    )

    right_ratio = (right_iris_x - right_min) / (
        right_max - right_min + 0.000001
    )

    gaze_ratio = (left_ratio + right_ratio) / 2

    if 0.35 <= gaze_ratio <= 0.65:
        status = "Eyes approximately centered"
    else:
        status = "Gaze may be away from camera"

    return gaze_ratio, status


# Run webcam analysis until the audio recording ends
def analyze_interview_video(stop_event=None,
                            output_file="vision_data.csv"):

    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("Could not open webcam.")
        return

    session_data = []
    previous_shoulder = None
    previous_head = None

    start_time = time.time()

    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            while True:

                if stop_event is not None and stop_event.is_set():
                    break

                success, frame = video.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                pose_result = pose.process(rgb)
                face_result = face_mesh.process(rgb)

                shoulder_difference = np.nan
                shoulder_movement = np.nan
                head_movement = np.nan

                yaw = np.nan
                pitch = np.nan
                roll = np.nan
                gaze_ratio = np.nan

                posture_status = "Not detected"
                gaze_status = "Not detected"
                face_detected = 0

                # Analyze shoulders
                if pose_result.pose_landmarks:

                    landmarks = pose_result.pose_landmarks.landmark

                    left = landmarks[
                        mp_pose.PoseLandmark.LEFT_SHOULDER.value
                    ]

                    right = landmarks[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                    ]

                    mp_draw.draw_landmarks(
                        frame,
                        pose_result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                    if left.visibility > 0.5 and right.visibility > 0.5:

                        shoulder_difference = abs(left.y - right.y)

                        shoulder_position = np.array([
                            (left.x + right.x) / 2,
                            (left.y + right.y) / 2
                        ])

                        if previous_shoulder is not None:
                            shoulder_movement = np.linalg.norm(
                                shoulder_position - previous_shoulder
                            )

                        previous_shoulder = shoulder_position

                        if shoulder_difference < 0.03:
                            posture_status = "Shoulders approximately aligned"
                        else:
                            posture_status = "Shoulder height differs"

                # Analyze face and head movement
                if face_result.multi_face_landmarks:

                    face_detected = 1
                    face = face_result.multi_face_landmarks[0]

                    yaw, pitch, roll = estimate_head_pose(
                        face, frame.shape[1], frame.shape[0]
                    )

                    gaze_ratio, gaze_status = estimate_eye_gaze(face)

                    nose = face.landmark[1]

                    head_position = np.array([
                        nose.x, nose.y
                    ])

                    if previous_head is not None:
                        head_movement = np.linalg.norm(
                            head_position - previous_head
                        )

                    previous_head = head_position

                    cv2.circle(
                        frame,
                        (
                            int(nose.x * frame.shape[1]),
                            int(nose.y * frame.shape[0])
                        ),
                        5,
                        (255, 0, 0),
                        -1
                    )

                cv2.putText(
                    frame,
                    posture_status,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Face detected: " + str(face_detected),
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.imshow(
                    "InterviewSense Pro - Visual Analysis",
                    frame
                )

                session_data.append({
                    "time_seconds": round(time.time() - start_time, 2),
                    "shoulder_difference": shoulder_difference,
                    "shoulder_movement": shoulder_movement,
                    "head_movement": head_movement,
                    "head_yaw": yaw,
                    "head_pitch": pitch,
                    "head_roll": roll,
                    "gaze_ratio": gaze_ratio,
                    "gaze_status": gaze_status,
                    "posture_status": posture_status,
                    "face_detected": face_detected
                })

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

    video.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(session_data)
    df.to_csv(output_file, index=False)

    print("Visual analysis completed.")
