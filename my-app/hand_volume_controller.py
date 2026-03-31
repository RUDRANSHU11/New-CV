import cv2
import mediapipe as mp
import numpy as np
import math

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ── Volume range ───────────────────────────────────────────────────────────────
VOL_MIN    = 0          # 0%
VOL_MAX    = 100        # 100%
DIST_MIN   = 20         # pixels — thumb and index fully pinched
DIST_MAX   = 220        # pixels — thumb and index fully spread

# ── Camera ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

volume      = 50        # start at 50%
smoothed    = 50.0      # for smoothing jitter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark

        # Landmark 4 = thumb tip, Landmark 8 = index fingertip
        x4 = int(lm[4].x * w);  y4 = int(lm[4].y * h)
        x8 = int(lm[8].x * w);  y8 = int(lm[8].y * h)
        cx = (x4 + x8) // 2;    cy = (y4 + y8) // 2

        # Euclidean distance between thumb tip and index tip
        dist = math.hypot(x8 - x4, y8 - y4)

        # Map distance → volume
        raw_vol   = np.interp(dist, [DIST_MIN, DIST_MAX], [VOL_MIN, VOL_MAX])
        smoothed  = 0.8 * smoothed + 0.2 * raw_vol       # exponential smoothing
        volume    = int(smoothed)

        # ── Draw hand landmarks ────────────────────────────────────────────────
        mp_draw.draw_landmarks(
            frame,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
            mp_draw.DrawingSpec(color=(0, 180, 180), thickness=2)
        )

        # ── Draw fingertip circles and connecting line ─────────────────────────
        cv2.circle(frame, (x4, y4), 10, (255, 80,  80),  cv2.FILLED)
        cv2.circle(frame, (x8, y8), 10, (80,  80,  255), cv2.FILLED)
        cv2.circle(frame, (cx, cy),  7, (255, 255, 255), cv2.FILLED)
        line_color = (0, 255, 0) if dist < 40 else (255, 255, 0)
        cv2.line(frame, (x4, y4), (x8, y8), line_color, 3)

        # ── Distance label at midpoint ─────────────────────────────────────────
        cv2.putText(frame, f"{int(dist)}px", (cx + 12, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # ── Volume bar (right side) ────────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = w - 80, 80, 35, 400
    filled_h = int(np.interp(volume, [VOL_MIN, VOL_MAX], [0, bar_h]))

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), cv2.FILLED)                                  # background
    cv2.rectangle(frame, (bar_x, bar_y + bar_h - filled_h),
                  (bar_x + bar_w, bar_y + bar_h),
                  (0, 255, 120), cv2.FILLED)                                  # fill
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (180, 180, 180), 2)                                         # border

    cv2.putText(frame, f"{volume}%",
                (bar_x - 5, bar_y + bar_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2)
    cv2.putText(frame, "VOL",
                (bar_x + 2, bar_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # ── HUD ───────────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (420, 36), (20, 20, 20), cv2.FILLED)
    cv2.putText(frame,
                "Pinch fingers to control volume  |  Q to quit",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Hand Gesture Volume Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()