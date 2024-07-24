import cv2
import mediapipe as mp
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Yeni pencere için boyutlar
new_width = width
new_height = height

persistent_canvas = np.zeros((new_height, new_width, 3), np.uint8)

last_point = None
draw = False
cursor_radius = 15

colors = {
    'Yellow': (0, 255, 255),
    'Red': (0, 0, 255),
    'Blue': (255, 0, 0),
    'Green': (0, 255, 0),
    'Cyan': (255, 255, 0),  # Cyan rengi eklendi
    'Purple': (128, 0, 128)  # Mor (Purple) rengi eklendi
}

button_height = 50
button_width = width // len(colors)

def draw_color_buttons(frame):
    for i, (color_name, color) in enumerate(colors.items()):
        x_start = i * button_width
        x_end = (i + 1) * button_width
        cv2.rectangle(frame, (x_start, 0), (x_end, button_height), color, -1)
        cv2.putText(frame, color_name, (x_start + 10, button_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def draw_clear_button(frame):
    cv2.rectangle(frame, (new_width - button_width, 0), (new_width, button_height), (128, 128, 128), -1)
    cv2.putText(frame, 'Temizle', (new_width - button_width + 10, button_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def clear_canvas():
    global persistent_canvas
    persistent_canvas = np.zeros((new_height, new_width, 3), np.uint8)       


selected_color = (0, 0, 0, 0)  # Seçilen rengi saklamak için değişken
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    gesture = ' '
    num_fingers = 0
    
    draw_color_buttons(frame)
    draw_clear_button(frame)
    
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            landmarks = []
            for lm in handslms.landmark:
                lmx = int(lm.x * width)
                lmy = int(lm.y * height)

                landmarks.append([lmx, lmy])
                
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                
            if len(landmarks) > 0:
                if landmarks[4][0] < landmarks[3][0]:
                    num_fingers += 1
                if landmarks[8][1] < landmarks[6][1]:
                    num_fingers += 1
                if landmarks[12][1] < landmarks[10][1]:
                    num_fingers += 1
                if landmarks[16][1] < landmarks[14][1]:
                    num_fingers += 1
                if landmarks[20][1] < landmarks[18][1]:
                    num_fingers += 1
                
                if num_fingers == 1:
                    for i, (color_name, color) in enumerate(colors.items()):
                        x_start = i * button_width
                        x_end = (i + 1) * button_width
                        if landmarks[8][0] >= x_start and landmarks[8][0] <= x_end and landmarks[8][1] <= button_height:
                            selected_color = color  # Renk seçildiğinde 'selected_color' değişkenini tanımla
                            break

                if num_fingers == 2:
                    gesture = 'Şeklini Çiz!'
                    if selected_color and draw:
                        current_point = landmarks[8]  # Index finger tip
                        cv2.line(persistent_canvas, last_point, current_point, selected_color, 15)
                        cv2.line(frame, last_point, current_point, selected_color, 15)
                        last_point = current_point
                    else:
                        last_point = landmarks[8]  # Index finger tip
                        draw = True
                    cursor_pos = last_point
                    cv2.circle(frame, cursor_pos, cursor_radius, (0, 255, 0), cv2.FILLED)

                elif num_fingers in [1, 3]:
                    gesture = 'Pointer'
                    cursor_pos = landmarks[8]  # Index finger tip position
                    cv2.circle(frame, cursor_pos, cursor_radius, (0, 255, 0), cv2.FILLED)
                    draw = False
                    
                elif num_fingers == 5:
                    gesture = 'Silgi Modu'
                    if draw:
                        current_point = landmarks[8]  # Index finger tip
                        cv2.line(persistent_canvas, last_point, current_point, (0, 0, 0), 40)
                        cv2.line(frame, last_point, current_point, (0, 0, 0), 40)
                        last_point = current_point
                        draw = True
                    else:
                        draw = False

            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Parmak Sayacı: {num_fingers}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    combined_frame = cv2.addWeighted(frame, 1, persistent_canvas, 0.5, 0)

    # Yeni pencerede videoyu göster
    cv2.namedWindow("KBM", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("KBM", new_width, new_height)
    cv2.imshow("KBM", combined_frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('t'):
        clear_canvas()

    if num_fingers == 5 and not result.multi_hand_landmarks and draw:
        persistent_canvas = np.zeros((new_height, new_width, 3), np.uint8)

    if num_fingers == 1:
        if landmarks[8][0] >= width - button_width and landmarks[8][1] <= button_height:
            persistent_canvas = np.zeros((new_height, new_width, 3), np.uint8)

cap.release()
cv2.destroyAllWindows()
