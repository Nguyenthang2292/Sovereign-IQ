"""Analyzer configuration classes for Gemini chart analyzer."""

from dataclasses import dataclass
from enum import Enum
from typing import List


@dataclass
class ImageValidationConfig:
    """Configuration for validating chart images."""

    max_file_size_mb: float = 20.0
    max_width: int = 4096
    max_height: int = 4096
    min_width: int = 100
    min_height: int = 100
    supported_formats: tuple = ("PNG", "JPEG", "JPG", "WEBP", "GIF")


class GeminiModelType(Enum):
    """Enum for Gemini model types with priority."""

    FLASH_3_PREVIEW = ("models/gemini-3-flash-preview", 0)
    PRO_3_PREVIEW = ("models/gemini-3-pro-preview", 1)

    FLASH_25_LITE = ("models/gemini-2.5-flash-lite", 2)

    FLASH_3 = ("models/gemini-3-flash", 3)
    FLASH_25 = ("models/gemini-2.5-flash", 4)
    FLASH_15 = ("models/gemini-1.5-flash", 5)

    PRO_3 = ("models/gemini-3-pro", 6)
    PRO_25 = ("models/gemini-2.5-pro", 7)
    PRO_15 = ("models/gemini-1.5-pro", 8)

    @property
    def name(self) -> str:
        """Get model name."""
        return self.value[0]

    @property
    def priority(self) -> int:
        """Priority order (lower = higher priority)."""
        return self.value[1]

    @property
    def is_preview(self) -> bool:
        return "preview" in self.name.lower()

    @property
    def is_flash(self) -> bool:
        return "flash" in self.name.lower()

    @property
    def is_lite(self) -> bool:
        return "lite" in self.name.lower()

    @property
    def is_pro(self) -> bool:
        return "pro" in self.name.lower() and "flash" not in self.name.lower()

    @classmethod
    def from_name(cls, model_name: str):
        """Retrieve model enum from its name."""
        if not model_name:
            return None

        normalized = model_name.strip().lower()
        for model_type in cls:
            if model_type.name.lower() == normalized:
                return model_type

        return None

    @classmethod
    def get_fallback_models(cls, primary_model):
        """Get fallback models ordered by priority."""
        fallbacks = []
        for model_type in cls:
            if model_type == primary_model:
                continue
            fallbacks.append(model_type)
        return fallbacks
