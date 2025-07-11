
import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    """Configuration settings for the face recognition system."""
    
    # Database settings
    DB_PATH: str = "faces.db"
    
    # Image settings
    IMAGE_DIR: str = "registered_images"
    THUMBNAIL_DIR: str = "thumbnails"
    MAX_IMAGE_SIZE: Tuple[int, int] = (1024, 1024)
    THUMBNAIL_SIZE: Tuple[int, int] = (150, 150)
    
    # Face recognition settings
    FACE_SIZE: Tuple[int, int] = (160, 160)
    SIMILARITY_THRESHOLD: float = 0.6
    HIGH_CONFIDENCE_THRESHOLD: float = 0.8
    
    # Device settings
    DEVICE: str = "cpu"  # Change to "cuda" if GPU available
    
    # UI settings
    MAX_DISPLAY_USERS: int = 50
    
    # File formats
    SUPPORTED_FORMATS: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.IMAGE_DIR, exist_ok=True)
        os.makedirs(cls.THUMBNAIL_DIR, exist_ok=True)