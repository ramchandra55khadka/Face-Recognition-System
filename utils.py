

# utils.py
import os
import hashlib
import logging
from datetime import datetime
from typing import Optional, Tuple
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageUtils:
    """Utility functions for image processing."""
    
    @staticmethod
    def validate_image(image: Image.Image) -> bool:
        """Validate if image is suitable for face recognition."""
        try:
            if image.mode not in ['RGB', 'L']:
                return False
            if image.size[0] < 50 or image.size[1] < 50:
                return False
            return True
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False
    
    @staticmethod
    def preprocess_image(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Preprocess image for face recognition with enhanced normalization."""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Auto-orient based on EXIF data
            image = ImageOps.exif_transpose(image)
            
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)
            
            # Apply histogram equalization to improve lighting
            img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img_array = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            
            # Convert back to PIL Image
            image = Image.fromarray(img_array)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Resize while maintaining aspect ratio
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create a new image with the exact target size and paste the resized image
            new_image = Image.new('RGB', target_size, (255, 255, 255))
            paste_x = (target_size[0] - image.size[0]) // 2
            paste_y = (target_size[1] - image.size[1]) // 2
            new_image.paste(image, (paste_x, paste_y))
            
            return new_image
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            raise
    
    @staticmethod
    def create_thumbnail(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Create a thumbnail of the image."""
        try:
            thumbnail = image.copy()
            thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
            return thumbnail
        except Exception as e:
            logger.error(f"Thumbnail creation error: {e}")
            raise
    
    @staticmethod
    def generate_unique_filename(name: str, extension: str = '.jpg') -> str:
        """Generate a unique filename based on name and timestamp."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name_clean = "".join(c for c in name if c.isalnum() or c in (' ', '_')).strip()
        name_clean = name_clean.replace(' ', '_')
        return f"{name_clean}_{timestamp}{extension}"