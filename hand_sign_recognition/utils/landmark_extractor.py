import numpy as np
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class HandLandmarkExtractor:
    def __init__(self, max_hands=1, detection_confidence=0.6, tracking_confidence=0.6):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1
        )

    def close(self):
        self.hands.close()

    def process(self, frame_bgr, draw=False):
        """
        Returns:
          landmarks: np.ndarray shape (42,) for a single hand, or None if no hand.
          annotated: frame with drawings if draw=True else original frame.
        """
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        annotated = frame_bgr.copy()
        all_points = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            w = max(max_x - min_x, 1e-6)
            h = max(max_y - min_y, 1e-6)
            norm_xs = [(x - min_x) / w for x in xs]
            norm_ys = [(y - min_y) / h for y in ys]

            points = []
            for x, y in zip(norm_xs, norm_ys):
                points.extend([x, y])
            all_points = np.array(points, dtype=np.float32)  # (42,)

            if draw:
                mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        return all_points, annotated
