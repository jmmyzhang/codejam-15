"""
ASL Recognition Module
Uses MediaPipe for hand tracking and PyTorch model for sign recognition
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional
import torch


class ASLRecognizer:
    """Recognizes ASL signs from video frames"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ASL recognizer
        
        Args:
            model_path: Path to PyTorch model for ASL recognition. If None, uses placeholder.
        """
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize PyTorch model (placeholder - you'll need to load your actual model)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path:
            try:
                self.model = torch.jit.load(model_path, map_location=self.device)
                self.model.eval()
                print(f"Loaded ASL model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
                print("Using placeholder recognition")
        
        # State for accumulating signs
        self.sign_buffer = []
        self.buffer_size = 10
        self.last_text = ""
        
    def extract_hand_features(self, frame, hand_landmarks) -> np.ndarray:
        """
        Extract features from hand landmarks
        
        Args:
            frame: Video frame
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            Feature vector
        """
        features = []
        
        # Get image dimensions
        h, w = frame.shape[:2]
        
        # Extract normalized landmark coordinates
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        # Calculate relative distances between key points
        # (thumb tip, index tip, middle tip, ring tip, pinky tip)
        key_points = [4, 8, 12, 16, 20]  # Landmark indices for fingertips
        
        if len(hand_landmarks.landmark) >= 21:
            for i in range(len(key_points)):
                for j in range(i + 1, len(key_points)):
                    p1 = hand_landmarks.landmark[key_points[i]]
                    p2 = hand_landmarks.landmark[key_points[j]]
                    dist = np.sqrt(
                        (p1.x - p2.x)**2 + 
                        (p1.y - p2.y)**2 + 
                        (p1.z - p2.z)**2
                    )
                    features.append(dist)
        
        return np.array(features, dtype=np.float32)
    
    def recognize_sign(self, features: np.ndarray) -> Optional[str]:
        """
        Recognize sign from features using PyTorch model
        
        Args:
            features: Feature vector extracted from hand landmarks
            
        Returns:
            Recognized sign text or None
        """
        if self.model is None:
            # Placeholder: return None or simple gesture detection
            # In production, this would use your trained ASL model
            return None
        
        try:
            # Prepare input tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(features_tensor)
                # Assuming model outputs class probabilities or logits
                # You'll need to adapt this based on your model's output format
                predicted_class = torch.argmax(output, dim=1).item()
                
                # Map class index to sign text
                # You'll need to implement this mapping based on your model
                sign_text = self._class_to_text(predicted_class)
                
                return sign_text
        except Exception as e:
            print(f"Error in sign recognition: {e}")
            return None
    
    def _class_to_text(self, class_idx: int) -> str:
        """
        Map class index to sign text
        
        This is a placeholder - you'll need to implement based on your model's classes
        """
        # Placeholder mapping - replace with your actual class mapping
        class_mapping = {
            0: "hello",
            1: "thank you",
            2: "yes",
            3: "no",
            # Add more mappings based on your model
        }
        return class_mapping.get(class_idx, "")
    
    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        Process a video frame to recognize ASL signs
        
        Args:
            frame: Video frame (BGR format)
            
        Returns:
            Recognized text or None
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Process each detected hand
            all_features = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                features = self.extract_hand_features(frame, hand_landmarks)
                all_features.append(features)
            
            # Combine features from both hands if available
            if len(all_features) == 1:
                combined_features = all_features[0]
            elif len(all_features) == 2:
                # Concatenate features from both hands
                combined_features = np.concatenate(all_features)
            else:
                return None
            
            # Recognize sign
            text = self.recognize_sign(combined_features)
            
            # Accumulate signs in buffer for temporal smoothing
            if text:
                self.sign_buffer.append(text)
                if len(self.sign_buffer) > self.buffer_size:
                    self.sign_buffer.pop(0)
                
                # Return most common sign in buffer
                if len(self.sign_buffer) >= 3:
                    from collections import Counter
                    most_common = Counter(self.sign_buffer).most_common(1)[0]
                    if most_common[1] >= len(self.sign_buffer) // 2:
                        return most_common[0]
            
            return text
        
        return None

