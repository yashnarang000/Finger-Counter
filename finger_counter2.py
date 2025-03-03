import cv2
from handtracking_module import HandTrace
import pyttsx3

camera = cv2.VideoCapture(1)

hand = HandTrace(max_num_hands=1)

# ftips: ids of tips of finger
ftips = [8, 12, 16, 20]

# ttip: id of tip of the thumb
ttip = 4

finger_sum = 0

tts = pyttsx3.init()
tts.setProperty('rate', 95)

while True:
    success, frame = camera.read()
    frame = cv2.flip(frame, 1)

    finger_values = {}

    try:
        # ftip: id of tip of finger
        # ftip-2: id of mid of finger

        # hand.id2axes(...)[0] gives x coordinates
        # hand.id2axes(...)[1] gives y coordinates

        for ftip in ftips:
            if hand.id2axes(frame, ftip)[1] < hand.id2axes(frame, ftip-2)[1]:
                finger_values.update({ftip:1}) # finger open
            else:
                finger_values.update({ftip:0}) # finger closed

        if hand.id2axes(frame, ttip)[0] < hand.id2axes(frame, ttip-2)[0]:
            finger_values.update({ttip:1}) # thumb open
        else:
            finger_values.update({ttip:0}) # thumb closed

        prev_finger_sum = finger_sum    
        finger_sum = 0

        for value in finger_values.values():
            finger_sum = finger_sum + value

        if finger_sum != prev_finger_sum:
            print(finger_sum)
            tts.say(f"{finger_sum}")
            tts.runAndWait()
    
    except TypeError:
        pass
    
    cv2.waitKey(1)