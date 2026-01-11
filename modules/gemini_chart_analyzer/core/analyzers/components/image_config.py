
from dataclasses import dataclass

"""Image validation configuration for Gemini analyzer."""



@dataclass
class ImageValidationConfig:
    """Configuration for validating chart images."""

    max_file_size_mb: float = 20.0
    max_width: int = 4096
    max_height: int = 4096
    min_width: int = 100
    min_height: int = 100
    supported_formats: tuple = ("PNG", "JPEG", "JPG", "WEBP", "GIF")
