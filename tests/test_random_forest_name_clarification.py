"""Tests to verify Random Forest name clarification docstrings.

This test module ensures that both Random Forest implementations
have clear docstrings to prevent confusion.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

import pytest


class TestRandomForestNameClarification:
    """Test that both Random Forest modules have clarifying docstrings."""

    def test_random_forest_module_docstring(self):
        """Test that modules.random_forest has clarifying docstring."""
        import modules.random_forest

        docstring = modules.random_forest.__doc__

        assert docstring is not None
        assert "sklearn" in docstring or "scikit-learn" in docstring
        assert "RandomForestClassifier" in docstring
        assert "decision_matrix" in docstring or "RandomForestCore" in docstring
        assert "NOT to be confused" in docstring or "IMPORTANT" in docstring

    def test_decision_matrix_random_forest_core_docstring(self):
        """Test that RandomForestCore has clarifying docstring."""
        from modules.decision_matrix.core.random_forest_core import RandomForestCore

        docstring = RandomForestCore.__doc__

        assert docstring is not None
        assert "Pine Script" in docstring
        assert "NOT" in docstring or "NOT sklearn" in docstring
        assert "modules.random_forest" in docstring

    def test_decision_matrix_module_docstring(self):
        """Test that modules.decision_matrix has clarifying docstring."""
        import modules.decision_matrix

        docstring = modules.decision_matrix.__doc__

        assert docstring is not None
        assert "RandomForestCore" in docstring
        assert "NOT sklearn" in docstring or "IMPORTANT" in docstring

    def test_random_forest_model_module_docstring(self):
        """Test that random_forest.core.model has clarifying docstring."""
        from modules.random_forest.core import model

        docstring = model.__doc__

        assert docstring is not None
        assert "sklearn" in docstring or "RandomForestClassifier" in docstring
        assert "decision_matrix" in docstring or "Pine Script" in docstring

    def test_imports_are_distinct(self):
        """Test that both modules can be imported without confusion."""
        # Import sklearn-based RF
        from modules.random_forest.core.model import train_random_forest_model
        from modules.random_forest.core.model import RandomForestClassifier

        # Import Pine Script-based RF
        from modules.decision_matrix.core.random_forest_core import RandomForestCore

        # Verify they are different types
        assert train_random_forest_model is not None
        assert RandomForestCore is not None

        # Verify RandomForestCore is not sklearn's RandomForestClassifier
        from sklearn.ensemble import RandomForestClassifier as SklearnRF

        assert RandomForestCore is not SklearnRF
        assert not issubclass(RandomForestCore, SklearnRF)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
