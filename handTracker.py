import cv2
import mediapipe as mp
import time

frame = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

previous_time = 0
current_time = 0

while True:
    (success, img) = frame.read()

    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(im_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, 'FPS: ', (1000, 68), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.putText(img, str(int(fps)), (1130, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('Hand Tracking', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
