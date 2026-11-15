import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# deprecated
# mpHands = mp.solutions.hands
# mpDraw = mp.solutions.drawing_utils

HAND_CONNECTIONS = {
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)
}

class ResultWrapper:
    """Bridges new MediaPipe HandLandmarkerResult attributes to match old legacy names."""
    def __init__(self, original_result):
        self._result = original_result
        
        # Map new names to old legacy attribute names
        self.multi_hand_landmarks = getattr(original_result, 'hand_landmarks', None)
        self.multi_handedness = getattr(original_result, 'handedness', None)
        self.multi_hand_world_landmarks = getattr(original_result, 'hand_world_landmarks', None)
        
        # Keep original attributes accessible too
        self.hand_landmarks = self.multi_hand_landmarks
        self.handedness = self.multi_handedness
        self.hand_world_landmarks = self.multi_hand_world_landmarks

class DetectorWrapper:
    """Wrapper to map legacy .process() calls to the new Tasks API."""
    def __init__(self, detector, parent_instance):
        self.detector = detector
        self.parent = parent_instance
    
    def process(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        if self.parent.static_image_mode:
            res = self.detector.detect(mp_image)
        else:
            self.parent.timestamp_ms += 33
            res = self.detector.detect_for_video(mp_image, self.parent.timestamp_ms)
        
        return ResultWrapper(res)

class HandTrace:
    def __init__(self, static_image_mode=False,
               max_num_hands=2,
               model_complexity=1, # although deprecated, kept for argument compatibility
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        # self.model_complexity = model_complexity // deprecated
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        running_mode = vision.RunningMode.IMAGE if self.static_image_mode else vision.RunningMode.VIDEO

        options = vision.HandLandmarkerOptions(
            base_options = base_options,
            running_mode = running_mode,
            num_hands = self.max_num_hands,
            min_hand_detection_confidence = self.min_detection_confidence,
            min_hand_presence_confidence = self.min_tracking_confidence
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0

        # deprecated
        # self.hands = mpHands.Hands(static_image_mode = self.static_image_mode, 
        #                       max_num_hands = self.max_num_hands, 
        #                       model_complexity = self.model_complexity,
        #                       min_detection_confidence = self.min_detection_confidence,
        #                       min_tracking_confidence = self.min_tracking_confidence)

        self.hands = DetectorWrapper(self.detector, self)

        parent_instance = self

        self.hands = DetectorWrapper(self.detector, self)

        
    
    def id2axes(self, frame, input_id):
        '''
        returns a tuple (x,y)
        '''
        axes = self.axesDict(frame)
        
        if axes == None:
            (x,y) = (None, None)
        else:
            (x,y) = axes[input_id]
        
        return (x,y)
        
    def axesDict(self, frame):
        '''
        returns a dictionary {id: (x,y)}
        '''
        palm_data = self.hands.process(frame)
        multi_hand_landmarks = palm_data.multi_hand_landmarks
        
        if multi_hand_landmarks != None:
            axes = {}
            for single_hand_landmarks in multi_hand_landmarks:
                for temp_id, landmark in enumerate(single_hand_landmarks):
                    height, width, channels = frame.shape
                    cx, cy = int(width*landmark.x), int(height*landmark.y)
                    axes.update({temp_id:(cx, cy)})

            return axes

# deprecated
    # def drawHandConnections(self, detection_frame, display_frame=0):
    #     '''
    #     detection_frame: frame in which the hand is to be tracked
    #     display_frame: frame in which the landmark connections are to be drawn (optional)
    #     '''
    #     palm_data = self.hands.process(detection_frame)
    #     multi_hand_landmarks = palm_data.multi_hand_landmarks
    #     if multi_hand_landmarks != None:
    #         for single_hand_landmarks in multi_hand_landmarks:
    #             if str(type(display_frame)) == "<class 'numpy.ndarray'>" :
    #                 mpDraw.draw_landmarks(display_frame, single_hand_landmarks, mpHands.HAND_CONNECTIONS)
    #             else:
    #                 mpDraw.draw_landmarks(detection_frame, single_hand_landmarks, mpHands.HAND_CONNECTIONS)

    def drawHandConnections(self, detection_frame, display_frame=0):
            '''
            detection_frame: frame in which the hand is to be tracked
            display_frame: frame in which the landmark connections are to be drawn (optional)
            '''
            palm_data = self.hands.process(detection_frame)
            multi_hand_landmarks = palm_data.multi_hand_landmarks
            
            target_frame = display_frame if isinstance(display_frame, type(detection_frame)) and display_frame.ndim == 3 else detection_frame
            
            if multi_hand_landmarks:
                for single_hand_landmarks in multi_hand_landmarks:
                    height, width, _ = target_frame.shape
                    
                    pixel_landmarks = []
                    for landmark in single_hand_landmarks:
                        px, py = int(landmark.x * width), int(landmark.y * height)
                        pixel_landmarks.append((px, py))
                        cv2.circle(target_frame, (px, py), 5, (0, 255, 0), cv2.FILLED)
                    
                    for connection in HAND_CONNECTIONS:
                                pt1_idx, pt2_idx = connection
                                pt1 = pixel_landmarks[pt1_idx]
                                pt2 = pixel_landmarks[pt2_idx]
                                cv2.line(target_frame, pt1, pt2, (0, 0, 255), 2)

def distance(pt1, pt2):
        (xa, ya) = pt1
        (xb, yb) = pt2

        xn = abs(xb - xa)
        yn = abs(yb - ya)

        xs = pow(xn, 2)
        ys = pow(yn, 2)

        h = int(pow(xs + ys, 0.5))
        return h