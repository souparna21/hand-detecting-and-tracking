import cv2
import mediapipe as mp
# import time library
import time

frame = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

# create two variable previous time and current time
previous_time = 0
current_time = 0

while True:
    (success, img) = frame.read()

    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(im_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    # Assign the value of current time in cTime variable
    current_time = time.time()
    # Write the formula of FPS
    fps = 1 / (current_time - previous_time)
    # Now assign the current time to pTime variable
    previous_time = current_time

    # Show the text 'FPS: ' on the image frame (*Extra info below*)
    cv2.putText(img, 'FPS: ', (1000, 68), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    # Show the value of FPS on the image frame
    cv2.putText(img, str(int(fps)), (1130, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('Hand Tracking', img)  # You have to write this in last otherwise fps will not show

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
