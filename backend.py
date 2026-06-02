"""
Backend module for AI-powered virtual try-on using Gemini API
"""
import os
from google import genai
from google.genai import types
from PIL import Image
import io
import base64
from typing import Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GeminiTryOnBackend:
    """Backend handler for Gemini API virtual try-on generation"""
    
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key or api_key == 'your_api_key_here':
            raise ValueError(
                "GEMINI_API_KEY not found or not set. "
                "Please set your API key in the .env file"
            )
        
        self.client = genai.Client(api_key=api_key)
        # Use Gemini model with image generation capabilities
        self.model_id = 'gemini-3-pro-image'
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def generate_tryon_image(
        self, 
        person_image: Image.Image, 
        garment_image: Image.Image
    ) -> tuple[Optional[Image.Image], Optional[str]]:
        """
        Generate virtual try-on image using Gemini API
        
        Args:
            person_image: PIL Image of the person
            garment_image: PIL Image of the garment/dress
            
        Returns:
            tuple: (generated_image, error_message)
                - generated_image: PIL Image of the generated try-on result
                - error_message: Error message if any, None otherwise
        """
        try:
            # Default prompt for virtual try-on
            prompt = """Create a realistic virtual try-on image where the model is wearing the garment. 
Preserve the original color, texture, and features of both the model and the garment. 
Ensure the garment fits naturally on the model's body with proper proportions and lighting. 
The result should look photorealistic and professionally edited."""
            
            # Convert PIL images to bytes for upload
            person_bytes = io.BytesIO()
            person_image.save(person_bytes, format='PNG')
            person_bytes.seek(0)
            
            garment_bytes = io.BytesIO()
            garment_image.save(garment_bytes, format='PNG')
            garment_bytes.seek(0)
            
            # Upload images and generate content
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    types.Part.from_bytes(
                        data=person_bytes.read(),
                        mime_type='image/png'
                    ),
                    types.Part.from_bytes(
                        data=garment_bytes.read(),
                        mime_type='image/png'
                    ),
                    prompt,  # Just pass the text directly
                ]
            )
            
            # Check if response has image data
            if response.candidates and len(response.candidates) > 0:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # Extract the image data
                        image_data = part.inline_data.data
                        # Convert bytes to PIL Image
                        generated_image = Image.open(io.BytesIO(image_data))
                        return generated_image, None
            
            # If no image found, return error
            return None, "No image generated from Gemini API"
                
        except Exception as e:
            error_msg = f"Error generating try-on: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def validate_images(
        self, 
        person_image: Optional[Image.Image], 
        garment_image: Optional[Image.Image]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate uploaded images
        
        Args:
            person_image: PIL Image of person
            garment_image: PIL Image of garment
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if person_image is None:
            return False, "Please upload a person image"
        
        if garment_image is None:
            return False, "Please upload a garment image"
        
        # Check image formats
        try:
            person_image.verify()
        except Exception:
            return False, "Invalid person image format"
        
        try:
            garment_image.verify()
        except Exception:
            return False, "Invalid garment image format"
        
        return True, None


def test_backend():
    """Test function for backend module"""
    try:
        backend = GeminiTryOnBackend()
        print("✓ Backend initialized successfully")
        print(f"✓ Using model: {backend.model_id}")
        return True
    except Exception as e:
        print(f"✗ Backend initialization failed: {e}")
        return False


if __name__ == "__main__":
    test_backend()
