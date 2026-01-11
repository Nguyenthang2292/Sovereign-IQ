
from unittest.mock import Mock, patch
import io
import sys

from config.iching import FONT_SIZE
from modules.iching.utils.helpers import (
from modules.iching.utils.helpers import (

"""
Tests for utility functions.
"""


    clean_images_folder,
    ensure_utf8_stdout,
    get_font,
)


class TestEnsureUtf8Stdout:
    """Test ensure_utf8_stdout function."""

    def test_ensure_utf8_stdout_already_utf8(self):
        """Test when stdout is already UTF-8."""
        original_stdout = sys.stdout
        try:
            # Create a mock stdout with UTF-8 encoding
            mock_stdout = Mock()
            mock_stdout.encoding = "utf-8"
            sys.stdout = mock_stdout

            ensure_utf8_stdout()

            # Should not replace stdout if already UTF-8
            assert sys.stdout is mock_stdout
        finally:
            sys.stdout = original_stdout

    def test_ensure_utf8_stdout_not_utf8(self):
        """Test when stdout is not UTF-8."""
        original_stdout = sys.stdout
        try:
            # Create a mock stdout without UTF-8 encoding
            mock_stdout = Mock()
            mock_stdout.encoding = "ascii"
            mock_stdout.buffer = Mock()
            sys.stdout = mock_stdout

            ensure_utf8_stdout()

            # Should replace stdout with TextIOWrapper
            assert sys.stdout is not mock_stdout
            assert isinstance(sys.stdout, io.TextIOWrapper)
        finally:
            sys.stdout = original_stdout


class TestGetFont:
    """Test get_font function."""

    @patch("modules.iching.utils.helpers.ImageFont.truetype")
    @patch("modules.iching.utils.helpers.ImageFont.load_default")
    @patch("modules.iching.utils.helpers.platform.system")
    @patch("modules.iching.utils.helpers.os.path.exists")
    def test_get_font_default_success(self, mock_exists, mock_system, mock_load_default, mock_truetype):
        """Test get_font with default font success."""
        expected_font_path = "arial.ttf"
        expected_size = FONT_SIZE
        mock_font = Mock()
        mock_truetype.return_value = mock_font
        mock_system.return_value = "Windows"
        mock_exists.return_value = False

        font = get_font()

        # Verify truetype was called once with correct path and size
        # Note: truetype is called with positional arguments: (path, size)
        mock_truetype.assert_called_once_with(expected_font_path, expected_size)
        # Verify the returned font is the mock
        assert font is mock_font

    @patch("modules.iching.utils.helpers.ImageFont.truetype")
    @patch("modules.iching.utils.helpers.ImageFont.load_default")
    @patch("modules.iching.utils.helpers.platform.system")
    @patch("modules.iching.utils.helpers.os.path.exists")
    def test_get_font_fallback_to_default(self, mock_exists, mock_system, mock_load_default, mock_truetype):
        """Test get_font falls back to default font."""
        mock_truetype.side_effect = OSError("Font not found")
        mock_load_default.return_value = Mock()
        mock_system.return_value = "Windows"
        mock_exists.return_value = False

        font = get_font()

        # Should fall back to load_default
        mock_load_default.assert_called()
        assert font is not None


class TestCleanImagesFolder:
    """Test clean_images_folder function."""

    def test_clean_images_folder_empty(self, tmp_path, monkeypatch):
        """Test cleaning empty folder."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Monkeypatch IMAGES_DIR directly instead of mocking Path
        monkeypatch.setattr("modules.iching.utils.helpers.IMAGES_DIR", images_dir)

        count = clean_images_folder()
        assert count == 0

    def test_clean_images_folder_with_files(self, tmp_path, monkeypatch):
        """Test cleaning folder with files."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        # Create some dummy files
        filenames = ["a.png", "b.jpg", "not_image.txt"]
        for fn in filenames:
            (images_dir / fn).write_text("dummy")

        # Monkeypatch IMAGES_DIR directly
        monkeypatch.setattr("modules.iching.utils.helpers.IMAGES_DIR", images_dir)

        count = clean_images_folder()
        # Only .png and .jpg files should be removed (assuming known impl.)
        assert count == 2
        # Confirm only non-image file remains
        assert set(f.name for f in images_dir.iterdir()) == {"not_image.txt"}
