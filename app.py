import streamlit as st
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Optional, Tuple
from datetime import datetime
from config import Config
from database import DatabaseManager
from face_recognition import FaceRecognition
from utils import ImageUtils
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceRecognitionApp:
    def __init__(self):
        Config.create_directories()
        self.db = DatabaseManager(Config.DB_PATH)
        self.face_rec = FaceRecognition(Config.DEVICE)
        self.image_utils = ImageUtils()
        st.set_page_config(page_title="Voter Face Recognition System", layout="wide")
    
    def validate_voter_id(self, voter_id: str) -> bool:
        """Validate voter ID format (example: ABC1234567)"""
        pattern = r'^[A-Z]{3}\d{7}$'
        return bool(re.match(pattern, voter_id.upper()))

    def check_duplicate_voter(self, voter_id: str, embedding: Optional[np.ndarray] = None) -> bool:
        """Check if voter ID or face already exists."""
        try:
            # Check voter ID
            cursor = self.db.conn.execute(
                "SELECT id FROM users WHERE voter_id = ? AND is_active = 1",
                (voter_id.upper(),)
            )
            if cursor.fetchone():
                return True
                
            # Check face if embedding provided
            if embedding is not None:
                users = self.db.get_all_users()
                match = self.face_rec.match_face(embedding, users, Config.SIMILARITY_THRESHOLD)
                if match and match[2] > Config.HIGH_CONFIDENCE_THRESHOLD:
                    return True
                    
            return False
        except Exception as e:
            logger.error(f"Error checking duplicate voter: {e}")
            return False

    def register_voter(self, name: str, dob: str, gender: str, voter_id: str, image: Image.Image):
        """Register a new voter."""
        try:
            # Validate inputs
            if not all([name, dob, gender, voter_id]):
                st.error("All fields are required")
                return False
                
            if not self.validate_voter_id(voter_id):
                st.error("Invalid voter ID format. Use format: ABC1234567")
                return False
                
            # Preprocess image
            processed_image = self.image_utils.preprocess_image(image, Config.FACE_SIZE)
            
            # Get face embedding
            embedding, _ = self.face_rec.get_embedding(processed_image)
            if embedding is None:
                st.error("No face detected in the image")
                return False
                
            # Check for duplicates
            if self.check_duplicate_voter(voter_id.upper(), embedding):
                st.error("Voter ID or face already registered")
                return False
                
            # Create thumbnail
            thumbnail = self.image_utils.create_thumbnail(processed_image, Config.THUMBNAIL_SIZE)
            
            # Generate unique filenames
            image_filename = self.image_utils.generate_unique_filename(name)
            thumbnail_filename = self.image_utils.generate_unique_filename(name, '_thumb.jpg')
            
            # Save images
            image_path = f"{Config.IMAGE_DIR}/{image_filename}"
            thumbnail_path = f"{Config.THUMBNAIL_DIR}/{thumbnail_filename}"
            processed_image.save(image_path)
            thumbnail.save(thumbnail_path)
            
            # Add to database
            self.db.add_user(name, embedding, image_path, thumbnail_path, dob, gender, voter_id.upper())
            st.success(f"Voter {name} registered successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            st.error(f"Registration failed: {str(e)}")
            return False

    def recognize_voter(self, frame: np.ndarray) -> Optional[Tuple]:
        """Recognize voter from video frame and return bounding box."""
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            embedding, box = self.face_rec.get_embedding(image)
            if embedding is None:
                return None, None
                
            users = self.db.get_all_users()
            match = self.face_rec.match_face(embedding, users, Config.SIMILARITY_THRESHOLD)
            
            if match:
                user_id, name, similarity, image_path, thumbnail_path, created_at, dob, gender, voter_id = match
                self.db.log_recognition(user_id, similarity)
                return (name, similarity, dob, gender, voter_id), box
            return None, None
            
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return None, None

    def run(self):
        """Main application interface."""
        st.title("Voter Face Recognition System")
        
        # Navigation
        page = st.sidebar.selectbox("Select Page", ["Register", "Recognize", "Stats"])
        
        if page == "Register":
            st.header("Register New Voter")
            name = st.text_input("Name")
            dob = st.date_input("Date of Birth", min_value=datetime(1900, 1, 1))
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            voter_id = st.text_input("Voter ID (ABC1234567)")
            
            # File upload or webcam capture
            col1, col2 = st.columns(2)
            with col1:
                uploaded_file = st.file_uploader("Upload Image", type=Config.SUPPORTED_FORMATS)
            with col2:
                use_webcam = st.checkbox("Use Webcam for Registration")
            
            if use_webcam:
                frame = st.camera_input("Capture Face for Registration")
                if frame:
                    image = Image.open(frame)
                    if st.button("Register with Webcam Image"):
                        self.register_voter(name, str(dob), gender, voter_id, image)
            elif uploaded_file:
                image = Image.open(uploaded_file)
                if st.button("Register with Uploaded Image"):
                    self.register_voter(name, str(dob), gender, voter_id, image)
        
        elif page == "Recognize":
            st.header("Face Recognition")
            frame = st.camera_input("Capture Face for Recognition")
            
            if frame:
                image = Image.open(frame)
                frame_np = np.array(image)
                frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                
                if st.button("Verify Face"):
                    result, box = self.recognize_voter(frame_np)
                    if result:
                        name, similarity, dob, gender, voter_id = result
                        # Draw bounding box if available
                        if box is not None:
                            x1, y1, x2, y2 = box
                            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        st.image(frame_rgb, channels="RGB", caption="Recognized Face")
                        st.write(f"Recognized: {name}")
                        st.write(f"Similarity: {similarity:.2%}")
                        st.write(f"Date of Birth: {dob}")
                        st.write(f"Gender: {gender}")
                        st.write(f"Voter ID: {voter_id}")
                    else:
                        st.image(frame_rgb, channels="RGB", caption="Face Image")
                        st.write("No match found")
        
        elif page == "Stats":
            st.header("Recognition Statistics")
            days = st.slider("Select time range (days)", 1, 90, 30)
            stats = self.db.get_recognition_stats(days)
            
            st.subheader(f"Recognition Stats (Last {days} days)")
            for name, voter_id, count, avg_similarity in stats:
                st.write(f"Name: {name}, Voter ID: {voter_id}, Recognitions: {count}, Avg Similarity: {avg_similarity:.2%}")
                
            st.subheader("Total Active Voters")
            st.write(f"Total: {self.db.get_user_count()}")

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()