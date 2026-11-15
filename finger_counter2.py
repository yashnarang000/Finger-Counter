import cv2
from handtracking_module import HandTrace
import pyttsx3
import threading

# --- SETUP ---
camera = cv2.VideoCapture(0) # Change to 1 if your external webcam is at index 1
hand = HandTrace(max_num_hands=1)

ftips = [8, 12, 16, 20]
ttip = 4
finger_sum = -1

# Initialize TTS engine once globally to avoid thread/memory leaks
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 110)
except Exception:
    tts_engine = None

def speak_number(num):
    if tts_engine is None:
        return
    def _speak():
        try:
            tts_engine.say(str(num))
            tts_engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()

# --- MAIN LOOP ---
while True:
    success, frame = camera.read()
    if not success:
        print("Camera not found or busy. Retrying...")
        continue

    frame = cv2.flip(frame, 1)

    # 1. PROCESS ONCE PER FRAME (Fixes the performance killer)
    axes = hand.axesDict(frame)
    
    # 2. DRAW VISUAL FEEDBACK ON SCREEN
    # hand.drawHandConnections(frame)

    if axes:
        try:
            finger_values = {}
            
            # Check 4 fingers (Y-axis: lower Y value means higher on screen)
            for ftip in ftips:
                if ftip in axes and (ftip - 2) in axes:
                    if axes[ftip][1] < axes[ftip - 2][1]:
                        finger_values[ftip] = 1
                    else:
                        finger_values[ftip] = 0

            # Check Thumb (X-axis)
            if ttip in axes and (ttip - 2) in axes:
                if axes[ttip][0] < axes[ttip - 2][0]:
                    finger_values[ttip] = 1
                else:
                    finger_values[ttip] = 0

            if finger_values:
                new_finger_sum = sum(finger_values.values())

                # Speak only if the number CHANGED
                if new_finger_sum != finger_sum:
                    print(f"Count: {new_finger_sum}")
                    speak_number(new_finger_sum)
                    finger_sum = new_finger_sum

        except Exception as e:
            print(f"Logic error: {e}") # Log it instead of hiding it silently

    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()