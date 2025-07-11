import torch
import numpy as np
from typing import Tuple, Optional, List
from PIL import Image
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
import logging

logger = logging.getLogger(__name__)

class FaceRecognition:
    """Enhanced face recognition with face detection and improved matching."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        try:
            # Initialize face detection model
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=device
            )
            
            # Initialize face recognition model
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            
            # Image transformation pipeline
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
            
            logger.info(f"Face recognition models loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Error initializing face recognition: {e}")
            raise
    
    def detect_faces(self, image: Image.Image) -> Optional[List[Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
        """Detect faces in the image and return face crops with bounding boxes."""
        try:
            # Convert PIL image to tensor
            img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
            
            # Detect faces with bounding boxes
            boxes, probs = self.mtcnn.detect(img_tensor, landmarks=False)
            
            if boxes is not None and len(boxes[0]) > 0:
                valid_faces = []
                for i, (box, prob) in enumerate(zip(boxes[0], probs[0])):
                    if prob > 0.9:  # High confidence threshold
                        face = self.mtcnn.extract(img_tensor, boxes, None)[0].cpu().numpy()
                        valid_faces.append((face, box.astype(int)))
                return valid_faces if valid_faces else None
            
            return None
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return None
    
    def get_embedding(self, image: Image.Image) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extract face embedding from image and return bounding box."""
        try:
            # First try to detect faces
            faces = self.detect_faces(image)
            
            if faces is not None and len(faces) > 0:
                # Use the first detected face
                face_tensor = torch.tensor(faces[0][0]).unsqueeze(0).to(self.device)
                box = faces[0][1]  # Bounding box coordinates (x1, y1, x2, y2)
            else:
                # Fallback to using the entire image
                logger.warning("No faces detected, using entire image")
                img_tensor = self.transform(image).unsqueeze(0).to(self.device)
                face_tensor = img_tensor
                box = None  # No bounding box if no face detected
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model(face_tensor).cpu().numpy()
            
            return embedding.flatten(), box
            
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return None, None
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            emb1_norm = emb1 / np.linalg.norm(emb1)
            emb2_norm = emb2 / np.linalg.norm(emb2)
            similarity = np.dot(emb1_norm, emb2_norm)
            return float(similarity)
        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return 0.0
    
    def match_face(self, embedding: np.ndarray, users: List[Tuple], 
                   threshold: float = 0.6) -> Optional[Tuple]:
        """Match face embedding against stored users."""
        try:
            if not users:
                return None
            
            best_match = None
            best_similarity = -1
            
            for user_data in users:
                uid, name, db_embedding = user_data[:3]
                
                similarity = self.cosine_similarity(embedding, db_embedding)
                
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (uid, name, similarity) + user_data[3:]
            
            return best_match
            
        except Exception as e:
            logger.error(f"Face matching error: {e}")
            return None
    
    def get_face_quality_score(self, image: Image.Image) -> float:
        """Calculate a quality score for the face image."""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate image sharpness (Laplacian variance)
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            laplacian_var = np.var(np.gradient(gray))
            
            # Normalize to 0-1 range
            quality_score = min(laplacian_var / 1000, 1.0)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Quality score calculation error: {e}")
            return 0.0