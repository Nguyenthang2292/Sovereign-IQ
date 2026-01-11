
from typing import List, Optional
import os

from .image_config import ImageValidationConfig
from .model_config import GeminiModelType
import PIL.Image
import PIL.Image

"""Helper utilities used by the Gemini chart analyzer."""





def select_best_model(available_models: Optional[List[str]] = None) -> str:
    """Choose the highest priority Gemini model from the available list."""
    if available_models:
        available_model_types: List[GeminiModelType] = []
        for model_name in available_models:
            model_type = GeminiModelType.from_name(model_name)
            if model_type:
                available_model_types.append(model_type)
        if available_model_types:
            available_model_types.sort(key=lambda m: m.priority)
            return available_model_types[0].name
        return available_models[0] if available_models else GeminiModelType.FLASH_3_PREVIEW.name
    return GeminiModelType.FLASH_3_PREVIEW.name


def validate_image(image_path: str, config: Optional[ImageValidationConfig] = None) -> tuple[bool, Optional[str]]:
    """Validate an image file against the configured limits."""
    if config is None:
        config = ImageValidationConfig()

    if not os.path.exists(image_path):
        return False, f"Image file not found: {image_path}"

    file_ext = os.path.splitext(image_path)[1].upper().lstrip(".")
    if file_ext not in config.supported_formats:
        return False, f"Unsupported image format: {file_ext}. Supported: {config.supported_formats}"

    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if file_size_mb > config.max_file_size_mb:
        return False, f"Image file too large: {file_size_mb:.2f}MB (max: {config.max_file_size_mb}MB)"

    try:
        with PIL.Image.open(image_path) as img:
            width, height = img.size
            if width < config.min_width or height < config.min_height:
                return False, f"Image too small: {width}x{height} (min: {config.min_width}x{config.min_height})"
            if width > config.max_width or height > config.max_height:
                return False, f"Image too large: {width}x{height} (max: {config.max_width}x{config.max_height})"
    except Exception as exc:
        return False, f"Failed to validate image: {exc}"

    return True, None
