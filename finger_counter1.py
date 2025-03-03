import cv2
from handtracking_module import HandTrace
from handtracking_module import distance

camera = cv2.VideoCapture(0)

hand = HandTrace(max_num_hands=1)

# 0: center
# 4: thumb tip
# 8: index tip
# 12: middle tip
# 16: ring tip
# 20: pinky tip

# sum of all the values of fingers (basically, how many fingers are up)
finger_sum = 0


def distanceFromCenter(point):
    return distance(point, center)

def dfc_visualize(frame, finger):

    # aboveFinger: where text has to be placed
    aboveFinger = (finger[0], finger[1]-30) 

    dfc_finger = dfcDict[finger]
    cv2.putText(frame, str(dfc_finger), aboveFinger, 0, 0.5, (255, 0, 255))

while True:
    success, frame = camera.read()
    frame = cv2.flip(frame, 1)
    
    center = hand.id2axes(frame, 0)
    thumb = hand.id2axes(frame, 4)
    index = hand.id2axes(frame, 8)
    middle = hand.id2axes(frame, 12)
    ring = hand.id2axes(frame, 16)
    pinky = hand.id2axes(frame, 20)

    try:
        # distance from center dictionary
        dfcDict = {
            thumb: distanceFromCenter(thumb),
            index: distanceFromCenter(index),
            middle: distanceFromCenter(middle),
            ring: distanceFromCenter(ring),
            pinky: distanceFromCenter(pinky)
        }

        distance_thresholds = {
            thumb: 130,
            index: 170,
            middle: 190,
            ring: 160,
            pinky: 140
        }

        finger_values = {}

        # writing the distance between each finger and center above the tip of the finger
        for finger in dfcDict:
            # getting the 'distance from center' and distance threshold of the finger
            dfc_finger = dfcDict[finger]
            threshold = distance_thresholds[finger]

            if dfc_finger >= threshold:
                finger_values.update({finger: 1})
            else:
                finger_values.update({finger: 0})

        prev_finger_sum = finger_sum
        finger_sum = 0

        for value in finger_values.values():
            finger_sum = finger_sum + value

        if finger_sum != prev_finger_sum:
            print(finger_sum)
        
    except TypeError:
        pass

    cv2.imshow("Live", frame)
    cv2.waitKey(1)