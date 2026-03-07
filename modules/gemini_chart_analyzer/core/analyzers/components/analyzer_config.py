"""Analyzer configuration classes for Gemini chart analyzer."""

from dataclasses import dataclass
from enum import Enum


@dataclass
class ImageValidationConfig:
    """Configuration for validating chart images."""

    max_file_size_mb: float = 20.0
    max_width: int = 4096
    max_height: int = 4096
    min_width: int = 100
    min_height: int = 100
    supported_formats: tuple = ("PNG", "JPEG", "JPG", "WEBP", "GIF")


# Stable models used as emergency fallback when preview quota is exhausted.
# These models have independent quotas and reliably support image input.
_RATE_LIMIT_FALLBACK_NAMES: list[str] = [
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
]


class GeminiModelType(Enum):
    """Enum for Gemini model types with priority."""

    # --- Gemini 3.1 (highest priority) ---
    PRO_31_PREVIEW = ("models/gemini-3.1-pro-preview", 0)
    PRO_31_PREVIEW_CUSTOMTOOLS = ("models/gemini-3.1-pro-preview-customtools", 1)

    # --- Gemini 3.x ---
    FLASH_3_PREVIEW = ("models/gemini-3-flash-preview", 2)
    PRO_3_PREVIEW = ("models/gemini-3-pro-preview", 3)
    FLASH_3 = ("models/gemini-3-flash", 4)
    PRO_3 = ("models/gemini-3-pro", 5)

    # --- Gemini 2.5 ---
    FLASH_25 = ("models/gemini-2.5-flash", 6)
    FLASH_25_LITE = ("models/gemini-2.5-flash-lite", 7)
    PRO_25 = ("models/gemini-2.5-pro", 8)

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
        """Get fallback models ordered by priority (excludes primary)."""
        fallbacks = []
        for model_type in cls:
            if model_type == primary_model:
                continue
            fallbacks.append(model_type)
        return fallbacks

    @classmethod
    def get_rate_limit_fallbacks(cls) -> list[str]:
        """Return stable model names for emergency fallback on 429/quota errors.

        These models have independent quotas from preview/experimental models
        and reliably support multimodal (image) input.
        """
        return list(_RATE_LIMIT_FALLBACK_NAMES)
