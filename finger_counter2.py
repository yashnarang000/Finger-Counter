import cv2
from handtracking_module import HandTrace
import pyttsx3
import threading # Required to stop the lag

# --- SETUP ---
camera = cv2.VideoCapture(1)
# camera.set(cv2.CAP_PROP_FPS, 60) # Optional: limit FPS for stability

hand = HandTrace(max_num_hands=1)

ftips = [8, 12, 16, 20]
ttip = 4
finger_sum = 0
prev_finger_sum = -1 # Start with -1 so it speaks 0 on start if you want

# --- SPEECH FUNCTION (THREADED) ---
def speak_number(num):
    def _speak():
        # We create a new engine instance per thread to avoid conflicts
        engine = pyttsx3.init()
        engine.setProperty('rate', 95)
        engine.say(str(num))
        engine.runAndWait()
    
    # Run this in a separate "lane" so it doesn't block the video
    threading.Thread(target=_speak).start()

# --- MAIN LOOP ---
while True:
    success, frame = camera.read()
    
    # 1. SAFETY CHECK (Crucial!)
    if not success:
        print("Camera not found or busy. Retrying...")
        continue

    frame = cv2.flip(frame, 1)

    finger_values = {}

    # Check if a hand is actually detected before asking for coordinates
    # (Assuming your HandTrace module has a method to check this, 
    # usually by checking if results are not None. If not, try/except is okay)
    try:
        # 2. Logic to check fingers
        for ftip in ftips:
            # Check Y-axis (height)
            if hand.id2axes(frame, ftip)[1] < hand.id2axes(frame, ftip-2)[1]:
                finger_values[ftip] = 1
            else:
                finger_values[ftip] = 0

        # Check Thumb (X-axis) - Note: This logic depends on left/right hand
        if hand.id2axes(frame, ttip)[0] < hand.id2axes(frame, ttip-2)[0]:
             finger_values[ttip] = 1
        else:
             finger_values[ttip] = 0

        # 3. Calculate Sum
        new_finger_sum = sum(finger_values.values())

        # 4. Speak only if the number CHANGED
        if new_finger_sum != finger_sum:
            print(f"Count: {new_finger_sum}")
            speak_number(new_finger_sum)
            finger_sum = new_finger_sum
    
    except Exception as e:
        # This catches errors when no hand is in the frame
        pass
    
    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
