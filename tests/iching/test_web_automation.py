
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from modules.iching.core.web_automation import (

from modules.iching.core.web_automation import (

"""
Tests for I Ching web automation.
"""



    clear_all_haodong,
    click_with_fallback,
    ensure_checkbox_checked,
    fill_web_form,
    handle_single_line,
    save_result_screenshot,
    scroll_into_view,
    wait_for_page_load,
)


class TestWaitForPageLoad:
    """Test wait_for_page_load function."""

    @patch("modules.iching.core.web_automation.time.sleep")
    @patch("modules.iching.core.web_automation.WebDriverWait")
    def test_wait_for_page_load_success(self, mock_wait_class, mock_sleep):
        """Test successful page load."""
        mock_driver = Mock()
        mock_driver.execute_script.return_value = 1000  # scrollHeight

        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait
        mock_wait.until.return_value = True

        wait_for_page_load(mock_driver, wait_timeout=30)

        # Verify WebDriverWait was created
        mock_wait_class.assert_called_once_with(mock_driver, 30)
        # Verify execute_script was called for scrolling
        assert mock_driver.execute_script.call_count >= 1

    @patch("modules.iching.core.web_automation.time.sleep")
    @patch("modules.iching.core.web_automation.WebDriverWait")
    def test_wait_for_page_load_exception(self, mock_wait_class, mock_sleep):
        """Test page load with exception (should not raise)."""
        mock_driver = Mock()
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait
        mock_wait.until.side_effect = Exception("Timeout")

        # Should not raise exception, just log warning
        wait_for_page_load(mock_driver, wait_timeout=30)


class TestScrollIntoView:
    """Test scroll_into_view function."""

    @patch("modules.iching.core.web_automation.time.sleep")
    def test_scroll_into_view_success(self, mock_sleep):
        """Test successful scroll into view."""
        mock_driver = Mock()
        mock_element = Mock()

        scroll_into_view(mock_driver, mock_element, delay=0.5)

        mock_driver.execute_script.assert_called_once()
        mock_sleep.assert_called_once_with(0.5)

    @patch("modules.iching.core.web_automation.time.sleep")
    def test_scroll_into_view_exception(self, mock_sleep):
        """Test scroll with exception (should not raise)."""
        mock_driver = Mock()
        mock_driver.execute_script.side_effect = Exception("Error")
        mock_element = Mock()

        # Should not raise exception
        scroll_into_view(mock_driver, mock_element)


class TestClickWithFallback:
    """Test click_with_fallback function."""

    def test_click_with_fallback_normal_click(self):
        """Test normal click success."""
        mock_driver = Mock()
        mock_element = Mock()

        click_with_fallback(mock_driver, mock_element)

        mock_element.click.assert_called_once()
        mock_driver.execute_script.assert_not_called()

    def test_click_with_fallback_js_fallback(self):
        """Test JavaScript fallback when normal click fails."""
        mock_driver = Mock()
        mock_element = Mock()
        mock_element.click.side_effect = Exception("Click failed")

        click_with_fallback(mock_driver, mock_element)

        mock_driver.execute_script.assert_called_once()

    def test_click_with_fallback_both_fail(self):
        """Test when both normal and JS click fail."""
        mock_driver = Mock()
        mock_driver.execute_script.side_effect = Exception("JS failed")
        mock_element = Mock()
        mock_element.click.side_effect = Exception("Click failed")

        with pytest.raises(RuntimeError):
            click_with_fallback(mock_driver, mock_element)


class TestEnsureCheckboxChecked:
    """Test ensure_checkbox_checked function."""

    @patch("modules.iching.core.web_automation.scroll_into_view")
    @patch("modules.iching.core.web_automation.click_with_fallback")
    @patch("modules.iching.core.web_automation.WebDriverWait")
    def test_checkbox_already_checked(self, mock_wait_class, mock_click, mock_scroll):
        """Test when checkbox is already checked."""
        mock_driver = Mock()
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait

        mock_checkbox = Mock()
        mock_checkbox.is_selected.return_value = True
        mock_wait.until.return_value = mock_checkbox

        ensure_checkbox_checked(mock_driver, mock_wait, "test_checkbox")

        # Should not click if already selected
        mock_click.assert_not_called()

    @patch("modules.iching.core.web_automation.scroll_into_view")
    @patch("modules.iching.core.web_automation.click_with_fallback")
    @patch("modules.iching.core.web_automation.WebDriverWait")
    def test_checkbox_not_checked(self, mock_wait_class, mock_click, mock_scroll):
        """Test when checkbox needs to be checked."""
        mock_driver = Mock()
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait

        mock_checkbox = Mock()
        mock_checkbox.is_selected.return_value = False
        mock_wait.until.return_value = mock_checkbox

        ensure_checkbox_checked(mock_driver, mock_wait, "test_checkbox")

        # Should click to check
        mock_click.assert_called_once()


class TestHandleSingleLine:
    """Test handle_single_line function."""

    @patch("modules.iching.core.web_automation.time.sleep")
    @patch("modules.iching.core.web_automation.scroll_into_view")
    @patch("modules.iching.core.web_automation.click_with_fallback")
    @patch("modules.iching.core.web_automation.Select")
    @patch("modules.iching.core.web_automation.WebDriverWait")
    def test_handle_solid_line_not_red(self, mock_wait_class, mock_select_class, mock_click, mock_scroll, mock_sleep):
        """Test handling solid line that is not red."""
        mock_driver = Mock()
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait

        mock_select_element = Mock()
        mock_wait.until.return_value = mock_select_element

        mock_select = Mock()
        mock_select_class.return_value = mock_select

        line_info = {"is_solid": True, "is_red": False}
        handle_single_line(mock_driver, mock_wait, line_info, 1)

        mock_select.select_by_value.assert_called_once_with("1")
        mock_click.assert_not_called()  # Should not click checkbox

    @patch("modules.iching.core.web_automation.time.sleep")
    @patch("modules.iching.core.web_automation.scroll_into_view")
    @patch("modules.iching.core.web_automation.click_with_fallback")
    @patch("modules.iching.core.web_automation.Select")
    @patch("modules.iching.core.web_automation.WebDriverWait")
    def test_handle_solid_line_red(self, mock_wait_class, mock_select_class, mock_click, mock_scroll, mock_sleep):
        """Test handling solid line that is red."""
        mock_driver = Mock()
        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait

        mock_select_element = Mock()
        mock_checkbox = Mock()
        mock_checkbox.is_selected.return_value = False

        def wait_side_effect(condition):
            if "Hao1" in str(condition):
                return mock_select_element
            return mock_checkbox

        mock_wait.until.side_effect = wait_side_effect

        mock_select = Mock()
        mock_select_class.return_value = mock_select

        line_info = {"is_solid": True, "is_red": True}
        handle_single_line(mock_driver, mock_wait, line_info, 1)

        mock_select.select_by_value.assert_called_once_with("1")
        mock_click.assert_called_once()  # Should click checkbox

    def test_handle_single_line_invalid_line_num(self):
        """Test handling invalid line number."""
        mock_driver = Mock()
        mock_wait = Mock()

        line_info = {"is_solid": True, "is_red": False}

        with pytest.raises(ValueError, match="line_num phải trong khoảng"):
            handle_single_line(mock_driver, mock_wait, line_info, 0)  # Invalid


class TestClearAllHaodong:
    """Test clear_all_haodong function."""

    @patch("modules.iching.core.web_automation.scroll_into_view")
    @patch("modules.iching.core.web_automation.click_with_fallback")
    def test_clear_all_haodong(self, mock_click, mock_scroll):
        """Test clearing all haodong checkboxes."""
        mock_driver = Mock()

        # Mock find_element to return checkboxes
        # Checkboxes at even positions (2, 4, 6) will be selected
        mock_checkboxes = []
        selected_indices = []
        for i in range(1, 7):
            mock_cb = Mock()
            is_selected = i % 2 == 0  # Some selected
            mock_cb.is_selected.return_value = is_selected
            if is_selected:
                selected_indices.append(i)
            mock_checkboxes.append(mock_cb)

        mock_driver.find_element.side_effect = mock_checkboxes

        clear_all_haodong(mock_driver)

        # Should click on selected checkboxes only
        expected_clicks = len(selected_indices)  # Should be 3 (checkboxes 2, 4, 6)
        assert mock_click.call_count == expected_clicks


class TestSaveResultScreenshot:
    """Test save_result_screenshot function."""

    @patch("modules.iching.core.web_automation.wait_for_page_load")
    @patch("modules.iching.core.web_automation.IChingResultExtractor")
    def test_save_result_screenshot_success(self, mock_extractor_class, mock_wait_load, tmp_path):
        """Test successful screenshot save."""
        # Setup: create images directory
        images_dir = tmp_path / "images"
        images_dir.mkdir(exist_ok=True)

        mock_driver = Mock()
        mock_driver.execute_script.return_value = 1920  # width/height

        # Make save_screenshot create a real file in tmp_path
        def save_screenshot_side_effect(filepath):
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            Path(filepath).touch()

        mock_driver.save_screenshot.side_effect = save_screenshot_side_effect

        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.extract_from_image.return_value = None
        mock_extractor_class.return_value = mock_extractor

        # Patch IMAGES_DIR (the Path used by save_result_screenshot) to point to tmp_path
        with patch("modules.iching.core.web_automation.IMAGES_DIR", images_dir):
            # Execute
            screenshot_path, result = save_result_screenshot(mock_driver, wait_timeout=30)

        # Assertions
        assert mock_driver.save_screenshot.call_count == 1
        assert screenshot_path is not None
        screenshot_file = Path(screenshot_path)
        assert screenshot_file.exists()
        assert screenshot_file.parent == images_dir
        assert mock_extractor.extract_from_image.called
        assert mock_extractor.extract_from_image.call_args[0][0] == screenshot_path


class TestFillWebForm:
    """Test fill_web_form function."""

    def test_fill_web_form_insufficient_lines(self):
        """Test with insufficient line_info."""
        line_info = [{"is_solid": True, "is_red": False}] * 3  # Only 3 lines

        with pytest.raises(ValueError, match="phải có ít nhất 6 phần tử"):
            fill_web_form(line_info)

    @patch("modules.iching.core.web_automation.webdriver")
    @patch("modules.iching.core.web_automation.save_result_screenshot")
    @patch("modules.iching.core.web_automation.click_submit_button")
    @patch("modules.iching.core.web_automation.handle_single_line")
    @patch("modules.iching.core.web_automation.clear_all_haodong")
    @patch("modules.iching.core.web_automation.ensure_checkbox_checked")
    @patch("modules.iching.core.web_automation.WebDriverWait")
    def test_fill_web_form_success(
        self, mock_wait_class, mock_ensure, mock_clear, mock_handle, mock_submit, mock_screenshot, mock_webdriver
    ):
        """Test successful form filling."""
        # Setup mocks
        mock_driver = Mock()
        mock_webdriver.Chrome.return_value = mock_driver

        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait

        mock_screenshot.return_value = ("path/to/screenshot.png", None)

        line_info = [{"is_solid": True, "is_red": False}] * 6

        fill_web_form(line_info, auto_close=True)

        # Verify driver was created and used
        mock_webdriver.Chrome.assert_called()
        mock_driver.get.assert_called_once()
        mock_driver.quit.assert_called_once()

        # Verify form was filled
        assert mock_handle.call_count == 6  # One for each line
        mock_submit.assert_called_once()

    @patch("modules.iching.core.web_automation.webdriver")
    @patch("modules.iching.core.web_automation.save_result_screenshot")
    @patch("modules.iching.core.web_automation.click_submit_button")
    @patch("modules.iching.core.web_automation.handle_single_line")
    @patch("modules.iching.core.web_automation.clear_all_haodong")
    @patch("modules.iching.core.web_automation.ensure_checkbox_checked")
    @patch("modules.iching.core.web_automation.WebDriverWait")
    @patch("modules.iching.core.web_automation.log_error")
    def test_fill_web_form_continue_on_error(
        self,
        mock_log_error,
        mock_wait_class,
        mock_ensure,
        mock_clear,
        mock_handle,
        mock_submit,
        mock_screenshot,
        mock_webdriver,
    ):
        """Test that form continues processing when stop_on_error=False (default)."""
        # Setup mocks
        mock_driver = Mock()
        mock_webdriver.Chrome.return_value = mock_driver

        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait

        mock_screenshot.return_value = ("path/to/screenshot.png", None)

        # Make handle_single_line fail on line 3, but succeed on others
        def handle_side_effect(driver, wait, info, line_num):
            if line_num == 3:
                raise RuntimeError("Test error on line 3")

        mock_handle.side_effect = handle_side_effect

        line_info = [{"is_solid": True, "is_red": False}] * 6

        # Should not raise exception, should continue processing
        fill_web_form(line_info, auto_close=True, stop_on_error=False)

        # Verify driver was created and used
        mock_webdriver.Chrome.assert_called()
        mock_driver.get.assert_called_once()
        mock_driver.quit.assert_called_once()

        # Verify all 6 lines were attempted (even though line 3 failed)
        assert mock_handle.call_count == 6
        # Verify error was logged
        assert mock_log_error.called
        # Verify submit was still called (form submission may be incomplete)
        mock_submit.assert_called_once()

    @patch("modules.iching.core.web_automation.webdriver")
    @patch("modules.iching.core.web_automation.save_result_screenshot")
    @patch("modules.iching.core.web_automation.click_submit_button")
    @patch("modules.iching.core.web_automation.handle_single_line")
    @patch("modules.iching.core.web_automation.clear_all_haodong")
    @patch("modules.iching.core.web_automation.ensure_checkbox_checked")
    @patch("modules.iching.core.web_automation.WebDriverWait")
    @patch("modules.iching.core.web_automation.log_error")
    def test_fill_web_form_stop_on_error(
        self,
        mock_log_error,
        mock_wait_class,
        mock_ensure,
        mock_clear,
        mock_handle,
        mock_submit,
        mock_screenshot,
        mock_webdriver,
    ):
        """Test that form stops and re-raises when stop_on_error=True."""
        # Setup mocks
        mock_driver = Mock()
        mock_webdriver.Chrome.return_value = mock_driver

        mock_wait = Mock()
        mock_wait_class.return_value = mock_wait

        mock_screenshot.return_value = ("path/to/screenshot.png", None)

        # Make handle_single_line fail on line 3
        def handle_side_effect(driver, wait, info, line_num):
            if line_num == 3:
                raise RuntimeError("Test error on line 3")

        mock_handle.side_effect = handle_side_effect

        line_info = [{"is_solid": True, "is_red": False}] * 6

        # Should raise exception when stop_on_error=True
        with pytest.raises(RuntimeError, match="Test error on line 3"):
            fill_web_form(line_info, auto_close=True, stop_on_error=True)

        # Verify driver was created
        mock_webdriver.Chrome.assert_called()
        mock_driver.get.assert_called_once()
        # Verify driver was cleaned up in finally block
        mock_driver.quit.assert_called_once()

        # Verify only 3 lines were attempted (stopped at line 3)
        assert mock_handle.call_count == 3
        # Verify error was logged before re-raising
        assert mock_log_error.called
        # Verify submit was NOT called (stopped before reaching it)
        mock_submit.assert_not_called()
