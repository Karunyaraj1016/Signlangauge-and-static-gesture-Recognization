import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: '0', 1: '0', 2: '1', 3: '1', 4: '2', 5: '2', 6: '3', 7: '3', 8: '4', 9: '4',10: '5', 11: '5', 12: '6', 13: '6', 14: '7', 15: '7', 16: '8', 17: '8', 18: '9', 19: '9',20: 'A', 21: 'A', 22: 'B', 23: 'B', 24: 'C', 25: 'C', 26: 'D', 27: 'D', 28: 'E', 29: 'E',30: 'F', 31: 'F', 32: 'G', 33: 'G', 34: 'H', 35: 'H', 36: 'I', 37: 'I', 38: 'J', 39: 'J',40: 'K', 41: 'K', 42: 'L', 43: 'L', 44: 'M', 45: 'M', 46: 'N', 47: 'N', 48: 'O', 49: 'O',50: 'P', 51: 'P', 52: 'P', 53: 'P', 54: 'Q', 55: 'Q', 56: 'R', 57: 'R', 58: 'S', 59: 'S',60: 'T', 61: 'T', 62: 'U', 63: 'U', 64: 'V', 65: 'V', 66: 'W', 67: 'W', 68: 'X', 69: 'X',70: 'Y', 71: 'Y', 72: 'Z', 73: 'Z'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Process only the first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        mp_drawing.draw_landmarks(
            frame,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Calculate landmark differences for the first hand
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predict using the model
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
