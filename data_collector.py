import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime

class DataCollector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.exercises = [
            "smile",
            "blink_3_times", 
            "head_rotation",
            "head_nod",
            "show_tongue"
        ]
        
    def show_ethical_guidelines(self):
        print("\n" + "="*60)
        print("🤝 AI EMOTION - ETHICAL GUIDELINES")
        print("="*60)
        print("1. Your privacy is protected - no images are stored")
        print("2. Only facial landmarks coordinates are saved")
        print("3. You can stop at any time by pressing 'q'")
        print("4. Data will be used only for research purposes")
        print("5. You have the right to withdraw your data")
        print("="*60)
        input("Press Enter to continue...")
    
    def get_participant_info(self):
        print("\n📝 PARTICIPANT INFORMATION")
        participant_id = input("Enter your participant ID (or press Enter for auto-generate): ")
        if not participant_id:
            participant_id = f"participant_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        age = input("Enter your age: ")
        gender = input("Enter your gender (M/F/O): ")
        
        return {
            "participant_id": participant_id,
            "age": age,
            "gender": gender,
            "collection_date": datetime.now().isoformat()
        }
    
    def extract_features(self, landmarks):
        """Extract 10 important facial features"""
        if not landmarks:
            return None
            
        landmarks = landmarks[0]  # Take first face
        
        # Define important points (MediaPipe indices)
        LIP_CORNER_LEFT = 61
        LIP_CORNER_RIGHT = 291
        LIP_TOP = 13
        LIP_BOTTOM = 14
        EYE_LEFT_TOP = 159
        EYE_LEFT_BOTTOM = 145
        EYE_RIGHT_TOP = 386
        EYE_RIGHT_BOTTOM = 374
        NOSE_TIP = 1
        FOREHEAD = 10
        
        features = []
        
        try:
            # 1. Mouth openness (vertical distance)
            mouth_open = abs(landmarks[LIP_TOP].y - landmarks[LIP_BOTTOM].y)
            features.append(mouth_open)
            
            # 2. Smile width (horizontal distance)
            smile_width = abs(landmarks[LIP_CORNER_LEFT].x - landmarks[LIP_CORNER_RIGHT].x)
            features.append(smile_width)
            
            # 3-4. Eye openness (both eyes)
            left_eye_open = abs(landmarks[EYE_LEFT_TOP].y - landmarks[EYE_LEFT_BOTTOM].y)
            right_eye_open = abs(landmarks[EYE_RIGHT_TOP].y - landmarks[EYE_RIGHT_BOTTOM].y)
            features.extend([left_eye_open, right_eye_open])
            
            # 5-6. Lip corner movements
            lip_left_y = landmarks[LIP_CORNER_LEFT].y
            lip_right_y = landmarks[LIP_CORNER_RIGHT].y
            features.extend([lip_left_y, lip_right_y])
            
            # 7-8. Eye ratios (for blinking detection)
            left_eye_ratio = left_eye_open / smile_width if smile_width > 0 else 0
            right_eye_ratio = right_eye_open / smile_width if smile_width > 0 else 0
            features.extend([left_eye_ratio, right_eye_ratio])
            
            # 9. Nose position (for head movement)
            nose_position = landmarks[NOSE_TIP].x
            features.append(nose_position)
            
            # 10. Face vertical position
            face_center = landmarks[FOREHEAD].y
            features.append(face_center)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def collect_exercise_data(self, exercise_name, participant_id):
        """Collect data for one specific exercise"""
        print(f"\n🎬 Exercise: {exercise_name.replace('_', ' ').title()}")
        print("Get ready...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot access camera")
            return []
        
        data_samples = []
        frame_count = 0
        max_frames = 150  # ~5 seconds at 30fps
        
        print("Recording... Press 'q' to stop early")
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks
                features = self.extract_features(landmarks)
                
                if features:
                    sample = {
                        "timestamp": datetime.now().isoformat(),
                        "exercise": exercise_name,
                        "participant_id": participant_id,
                        "features": features,
                        "frame_count": frame_count
                    }
                    data_samples.append(sample)
                    frame_count += 1
                    
                    # Display frame with instruction
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"Exercise: {exercise_name}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Frames: {frame_count}/{max_frames}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('AI Emotion - Data Collection', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Collected {len(data_samples)} samples for {exercise_name}")
        return data_samples
    
    def run_collection_session(self):
        """Run complete data collection session"""
        print("🚀 AI EMOTION - DATA COLLECTION")
        print("="*50)
        
        # Show ethical guidelines
        self.show_ethical_guidelines()
        
        # Get participant info
        participant_info = self.get_participant_info()
        
        all_data = []
        
        # Collect data for each exercise
        for exercise in self.exercises:
            input(f"\nPress Enter to start {exercise.replace('_', ' ')}...")
            
            exercise_data = self.collect_exercise_data(exercise, participant_info["participant_id"])
            all_data.extend(exercise_data)
            
            print(f"✅ Completed {exercise}")
        
        # Save data
        self.save_data(all_data, participant_info)
        
        print(f"\n🎉 DATA COLLECTION COMPLETE!")
        print(f"📊 Total samples collected: {len(all_data)}")
        print(f"👤 Participant: {participant_info['participant_id']}")
    
    def save_data(self, data, participant_info):
        """Save collected data to JSON file"""
        if not os.path.exists('collected_data'):
            os.makedirs('collected_data')
        
        filename = f"collected_data/{participant_info['participant_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output = {
            "participant_info": participant_info,
            "collection_metadata": {
                "exercises": self.exercises,
                "total_samples": len(data),
                "collection_date": datetime.now().isoformat()
            },
            "samples": data
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Data saved to: {filename}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run_collection_session()