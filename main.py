import cv2
import mediapipe as mp
import numpy as np
import time

###############################################################################
# HELPER CLASSES / FUNCTIONS
###############################################################################

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class PointerSmoother:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.prev_point = None

    def smooth(self, current_point):
        if self.prev_point is None:
            self.prev_point = current_point
            return current_point

        smoothed_point = (
            int(self.alpha * self.prev_point[0] + (1 - self.alpha) * current_point[0]),
            int(self.alpha * self.prev_point[1] + (1 - self.alpha) * current_point[1])
        )
        self.prev_point = smoothed_point
        return smoothed_point

def average_landmark(landmarks, indices, img_w, img_h):
    """
    Given a set of face mesh landmarks and a list of landmark indices,
    return the average (x, y) in image coordinates.
    """
    xs = []
    ys = []
    for i in indices:
        lx = landmarks[i].x * img_w
        ly = landmarks[i].y * img_h
        xs.append(lx)
        ys.append(ly)
    return (int(np.mean(xs)), int(np.mean(ys)))

def compute_transformation_matrix(eye_points, screen_points):
    """
    Compute a 2D affine transformation M (2x3) that maps
    (eye_x, eye_y, 1) -> (screen_x, screen_y).

    eye_points: list of (ex, ey) from calibration
    screen_points: list of (sx, sy) from calibration
    """
    # We will solve for M in the equation:
    # [sx_i]   [m00 m01 m02] [ex_i]
    # [sy_i] = [m10 m11 m12] [ey_i]
    #                       [ 1  ]

    # Build A and b for least squares
    # For each calibration point i:
    #   [ex_i  ey_i  1  0    0    0  ]   [m00]   [sx_i]
    #   [0     0     0  ex_i ey_i 1  ] * [m01] = [sy_i]
    #                                     [m02]
    #                                     [m10]
    #                                     [m11]
    #                                     [m12]

    A = []
    b = []
    for (ex, ey), (sx, sy) in zip(eye_points, screen_points):
        A.append([ex, ey, 1, 0, 0, 0])
        A.append([0, 0, 0, ex, ey, 1])
        b.append(sx)
        b.append(sy)

    A = np.array(A, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    # Solve for M in least squares sense: A * M_vec = b
    # M_vec has shape (6,) => [m00, m01, m02, m10, m11, m12]
    M_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M = M_vec.reshape(2, 3)
    return M

def apply_transformation_matrix(M, ex, ey):
    """
    Apply 2D affine transformation M (2x3) to point (ex, ey).
    Return (sx, sy).
    """
    # [sx]   [m00 m01 m02] [ex]
    # [sy] = [m10 m11 m12] [ey]
    #                      [1 ]
    point = np.array([ex, ey, 1], dtype=np.float32)
    screen_xy = M @ point
    return int(screen_xy[0]), int(screen_xy[1])
    
def compute_dynamic_sensitivity(base_sensitivity, distance, min_sensitivity=0.5, max_distance=200):
    """
    Reduce sensitivity as the distance decreases.
    :param base_sensitivity: The base sensitivity value.
    :param distance: The current distance between the red and green cursors.
    :param min_sensitivity: The minimum sensitivity value.
    :param max_distance: The distance beyond which sensitivity remains unchanged.
    :return: Adjusted sensitivity value.
    """
    if distance > max_distance:
        return base_sensitivity
    return max(min_sensitivity, base_sensitivity * (distance / max_distance))

###############################################################################
# MAIN SCRIPT
###############################################################################

def countdown(cap, message):
    """Display a countdown before calibration."""
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame,
            f"{message} in {i}...",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0, (0, 255, 255), 3
        )
        cv2.imshow("Countdown", frame)
        cv2.waitKey(1000)  # Wait for 1 second
    cv2.destroyWindow("Countdown")


def main():
    # Initialize FaceMesh and Hands
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Read one frame to get size
    success, frame = cap.read()
    if not success:
        print("Error: Cannot read from webcam.")
        return
    h, w, _ = frame.shape

    # ----------------------------
    # 1) DEFINE CALIBRATION POINTS
    # ----------------------------
    calibration_screen_points = [
        (int(w*0.1), int(h*0.1)),   # top-left
        (int(w*0.9), int(h*0.1)),   # top-right
        (int(w*0.9), int(h*0.9)),   # bottom-right
        (int(w*0.1), int(h*0.9)),   # bottom-left
        (int(w*0.5), int(h*0.5)),   # center
    ]

    calibration_eye_points = []
    calibration_hand_points = []

    right_eye_indices = [33, 133]
    left_eye_indices  = [362, 263]

    sensitivity_reduction_enabled = False  # Toggle for sensitivity reduction function
    base_sensitivity = 2.0  # Sensitivity for hand movement

    # ----------------------------
    # 2) RUN EYE CALIBRATION
    # ----------------------------
    countdown(cap, "Eye calibration starts")

    for idx, cal_pt in enumerate(calibration_screen_points):
        print(f"[CALIBRATION] Look at point {idx+1}/{len(calibration_screen_points)}: {cal_pt}")
        start_time = time.time()
        collected_eye_samples = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            cv2.circle(frame, cal_pt, 20, (0, 255, 255), -1)
            cv2.putText(
                frame,
                f"Calibration {idx+1}/{len(calibration_screen_points)}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2
            )
            cv2.putText(
                frame,
                "Keep looking at the circle...",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2
            )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb_frame)

            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark

                right_eye_center = average_landmark(landmarks, right_eye_indices, w, h)
                left_eye_center  = average_landmark(landmarks, left_eye_indices, w, h)

                ex = (right_eye_center[0] + left_eye_center[0]) // 2
                ey = (right_eye_center[1] + left_eye_center[1]) // 2

                cv2.circle(frame, right_eye_center, 5, (255, 0, 0), -1)
                cv2.circle(frame, left_eye_center, 5, (255, 0, 0), -1)
                cv2.circle(frame, (ex, ey), 5, (0, 0, 255), -1)

                if time.time() - start_time > 1.0:
                    collected_eye_samples.append((ex, ey))

            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            if time.time() - start_time > 2.0:
                break

        if len(collected_eye_samples) > 0:
            avg_ex = int(np.mean([p[0] for p in collected_eye_samples]))
            avg_ey = int(np.mean([p[1] for p in collected_eye_samples]))
        else:
            avg_ex, avg_ey = 0, 0

        calibration_eye_points.append((avg_ex, avg_ey))
        print(f"   -> Eye calibration point = ({avg_ex}, {avg_ey})")

    # ----------------------------
    # 3) RUN HAND CALIBRATION
    # ----------------------------
    countdown(cap, "Hand calibration starts")

    for idx, cal_pt in enumerate(calibration_screen_points):
        print(f"[CALIBRATION] Move your hand to point {idx+1}/{len(calibration_screen_points)}: {cal_pt}")
        start_time = time.time()
        collected_hand_samples = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            cv2.circle(frame, cal_pt, 20, (0, 255, 255), -1)
            cv2.putText(
                frame,
                f"Calibration {idx+1}/{len(calibration_screen_points)}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2
            )
            cv2.putText(
                frame,
                "Move your hand to the circle...",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2
            )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_results = hands.process(rgb_frame)

            if hands_results.multi_hand_landmarks:
                hand_landmarks = hands_results.multi_hand_landmarks[0]
                index_tip = hand_landmarks.landmark[8]
                hx = int(index_tip.x * w)
                hy = int(index_tip.y * h)

                cv2.circle(frame, (hx, hy), 5, (0, 255, 0), -1)

                if time.time() - start_time > 1.0:
                    collected_hand_samples.append((hx, hy))

            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            if time.time() - start_time > 2.0:
                break

        if len(collected_hand_samples) > 0:
            avg_hx = int(np.mean([p[0] for p in collected_hand_samples]))
            avg_hy = int(np.mean([p[1] for p in collected_hand_samples]))
        else:
            avg_hx, avg_hy = 0, 0

        calibration_hand_points.append((avg_hx, avg_hy))
        print(f"   -> Hand calibration point = ({avg_hx}, {avg_hy})")

    cv2.destroyWindow("Calibration")

    M_eye = compute_transformation_matrix(calibration_eye_points, calibration_screen_points)
    M_hand = compute_transformation_matrix(calibration_hand_points, calibration_screen_points)

    print("[CALIBRATION] Eye transformation matrix:\n", M_eye)
    print("[CALIBRATION] Hand transformation matrix:\n", M_hand)

    # ----------------------------
    # 4) MAIN TRACKING LOOP
    # ----------------------------
    gaze_smoother = PointerSmoother(alpha=0.7)
    hand_pointer = [w // 2, h // 2]  # Start at screen center

    print("[INFO] Entering main tracking loop. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(rgb_frame)
        hands_results = hands.process(rgb_frame)

        gaze_pointer = None
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            right_eye_center = average_landmark(landmarks, right_eye_indices, w, h)
            left_eye_center  = average_landmark(landmarks, left_eye_indices, w, h)
            ex = (right_eye_center[0] + left_eye_center[0]) // 2
            ey = (right_eye_center[1] + left_eye_center[1]) // 2

            gx, gy = apply_transformation_matrix(M_eye, ex, ey)
            gx = max(0, min(gx, w-1))
            gy = max(0, min(gy, h-1))

            gaze_pointer = gaze_smoother.smooth((gx, gy))

        if hands_results.multi_hand_landmarks:
            hand_landmarks = hands_results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[8]
            hx = int(index_tip.x * w)
            hy = int(index_tip.y * h)

            mx, my = apply_transformation_matrix(M_hand, hx, hy)
            hand_pointer[0] = max(0, min(mx, w-1))
            hand_pointer[1] = max(0, min(my, h-1))

            # if gaze_pointer is not None:
            #     distance = np.linalg.norm(np.array([mx, my]) - np.array(gaze_pointer))
            #     sensitivity = base_sensitivity
            #     if sensitivity_reduction_enabled:
            #         sensitivity = compute_dynamic_sensitivity(base_sensitivity, distance)
            #     hand_pointer[0] += int(sensitivity * (mx - hand_pointer[0]))
            #     hand_pointer[1] += int(sensitivity * (my - hand_pointer[1]))
            
            # hand_pointer[0] = max(0, min(hand_pointer[0], w-1))
            # hand_pointer[1] = max(0, min(hand_pointer[1], h-1))

        if gaze_pointer is not None:
            cv2.circle(frame, gaze_pointer, 10, (0, 0, 255), -1)

        cv2.circle(frame, tuple(hand_pointer), 10, (0, 255, 0), -1)

        cv2.imshow("Eye and Hand Tracking (Calibrated)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    hands.close()

if __name__ == "__main__":
    main()
