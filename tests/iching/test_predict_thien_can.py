import pytest

from modules.iching.predict.constants import HOA, KIM, MOC, THO, THUY, YANG, YIN

"""
Tests for I Ching Thiên Can (Heavenly Stems).
"""

from modules.iching.predict.thien_can import (
    AT,
    BINH,
    CANH,
    DINH,
    GIAP,
    KY,
    MAU,
    NHAM,
    QUY,
    TAN,
    THIEN_CAN_DICT,
    THIEN_CAN_LIST,
    ThienCan,
)


class TestThienCanInstances:
    """Test Thiên Can instances."""

    def test_thien_can_list_length(self):
        """Test that THIEN_CAN_LIST has 10 elements."""
        assert len(THIEN_CAN_LIST) == 10

    def test_thien_can_dict_length(self):
        """Test that THIEN_CAN_DICT has 10 elements."""
        assert len(THIEN_CAN_DICT) == 10

    def test_all_instances_in_list(self):
        """Test that all instances are in THIEN_CAN_LIST."""
        instances = [GIAP, AT, BINH, DINH, MAU, KY, CANH, TAN, NHAM, QUY]
        assert all(tc in THIEN_CAN_LIST for tc in instances)

    def test_all_instances_in_dict(self):
        """Test that all instances are in THIEN_CAN_DICT."""
        assert THIEN_CAN_DICT["Giáp"] == GIAP
        assert THIEN_CAN_DICT["Ất"] == AT
        assert THIEN_CAN_DICT["Bính"] == BINH
        assert THIEN_CAN_DICT["Đinh"] == DINH
        assert THIEN_CAN_DICT["Mậu"] == MAU
        assert THIEN_CAN_DICT["Kỷ"] == KY
        assert THIEN_CAN_DICT["Canh"] == CANH
        assert THIEN_CAN_DICT["Tân"] == TAN
        assert THIEN_CAN_DICT["Nhâm"] == NHAM
        assert THIEN_CAN_DICT["Quý"] == QUY

    def test_yin_yang_alternation(self):
        """Test that YIN and YANG alternate correctly."""
        # Giáp (1) - Dương, Ất (2) - Âm, Bính (3) - Dương, etc.
        assert GIAP.yin_yang == YANG
        assert AT.yin_yang == YIN
        assert BINH.yin_yang == YANG
        assert DINH.yin_yang == YIN
        assert MAU.yin_yang == YANG
        assert KY.yin_yang == YIN
        assert CANH.yin_yang == YANG
        assert TAN.yin_yang == YIN
        assert NHAM.yin_yang == YANG
        assert QUY.yin_yang == YIN

    def test_wu_hang_assignments(self):
        """Test Ngũ hành assignments."""
        # Giáp, Ất - Mộc
        assert GIAP.wu_hang == MOC
        assert AT.wu_hang == MOC
        # Bính, Đinh - Hoả
        assert BINH.wu_hang == HOA
        assert DINH.wu_hang == HOA
        # Mậu, Kỷ - Thổ
        assert MAU.wu_hang == THO
        assert KY.wu_hang == THO
        # Canh, Tân - Kim
        assert CANH.wu_hang == KIM
        assert TAN.wu_hang == KIM
        # Nhâm, Quý - Thuỷ
        assert NHAM.wu_hang == THUY
        assert QUY.wu_hang == THUY


class TestThienCanClass:
    """Test ThienCan class."""

    def test_creation_valid(self):
        """Test creating ThienCan with valid values."""
        tc = ThienCan(name="Test", yin_yang=YANG, wu_hang=MOC)
        assert tc.name == "Test"
        assert tc.yin_yang == YANG
        assert tc.wu_hang == MOC

    def test_creation_invalid_yin_yang(self):
        """Test creating ThienCan with invalid yin_yang."""
        with pytest.raises(ValueError, match="yin_yang must be"):
            ThienCan(name="Test", yin_yang="Invalid", wu_hang=MOC)

    def test_creation_invalid_wu_hang(self):
        """Test creating ThienCan with invalid wu_hang."""
        with pytest.raises(ValueError, match="wu_hang must be one of"):
            ThienCan(name="Test", yin_yang=YANG, wu_hang="Invalid")

    def test_repr(self):
        """Test string representation."""
        tc = ThienCan(name="Giáp", yin_yang=YANG, wu_hang=MOC)
        repr_str = repr(tc)
        assert "ThienCan" in repr_str
        assert "Giáp" in repr_str
        assert YANG in repr_str
        assert MOC in repr_str

    def test_eq_same_name(self):
        """Test equality with same name."""
        tc1 = ThienCan(name="Giáp", yin_yang=YANG, wu_hang=MOC)
        # Same name, but intentionally choose different yin_yang and wu_hang to confirm
        # that only 'name' affects equality
        tc2 = ThienCan(name="Giáp", yin_yang=YIN, wu_hang=HOA)
        # Should be equal because they have same name
        assert tc1 == tc2

    def test_eq_different_name(self):
        """Test equality with different name."""
        tc1 = ThienCan(name="Giáp", yin_yang=YANG, wu_hang=MOC)
        tc2 = ThienCan(name="Ất", yin_yang=YIN, wu_hang=MOC)
        assert tc1 != tc2

    def test_eq_with_instance(self):
        """Test equality with module instance."""
        tc = ThienCan(name="Giáp", yin_yang=YANG, wu_hang=MOC)
        assert tc == GIAP

    def test_eq_with_non_thien_can(self):
        """Test equality with non-ThienCan object."""
        tc = ThienCan(name="Giáp", yin_yang=YANG, wu_hang=MOC)
        assert tc != "Giáp"
        assert tc != 123

    def test_hash(self):
        """Test hashing."""
        tc1 = ThienCan(name="Giáp", yin_yang=YANG, wu_hang=MOC)
        tc2 = ThienCan(name="Giáp", yin_yang=YIN, wu_hang=HOA)
        # Same name should produce same hash
        assert hash(tc1) == hash(tc2)
        # Should be usable in set
        test_set = {tc1, tc2}
        assert len(test_set) == 1  # Same hash, so only one element


class TestThienCanInstanceMethods:
    """Test ThienCan instance methods."""

    def test_get_yin_yang(self):
        """Test get_yin_yang method."""
        assert GIAP.get_yin_yang() == YANG
        assert AT.get_yin_yang() == YIN

    def test_get_wu_hang(self):
        """Test get_wu_hang method."""
        assert GIAP.get_wu_hang() == MOC
        assert BINH.get_wu_hang() == HOA
        assert CANH.get_wu_hang() == KIM

    def test_is_yang(self):
        """Test is_yang method."""
        assert GIAP.is_yang() is True
        assert AT.is_yang() is False
        assert BINH.is_yang() is True
        assert DINH.is_yang() is False

    def test_is_yin(self):
        """Test is_yin method."""
        assert GIAP.is_yin() is False
        assert AT.is_yin() is True
        assert BINH.is_yin() is False
        assert DINH.is_yin() is True


class TestThienCanClassMethods:
    """Test ThienCan class methods."""

    def test_get_by_name_valid(self):
        """Test get_by_name with valid name."""
        assert ThienCan.get_by_name("Giáp") == GIAP
        assert ThienCan.get_by_name("Ất") == AT
        assert ThienCan.get_by_name("Quý") == QUY

    def test_get_by_name_invalid(self):
        """Test get_by_name with invalid name."""
        assert ThienCan.get_by_name("Invalid") is None
        assert ThienCan.get_by_name("") is None

    def test_get_yin_yang_by_name_valid(self):
        """Test get_yin_yang_by_name with valid name."""
        assert ThienCan.get_yin_yang_by_name("Giáp") == YANG
        assert ThienCan.get_yin_yang_by_name("Ất") == YIN
        assert ThienCan.get_yin_yang_by_name("Bính") == YANG

    def test_get_yin_yang_by_name_invalid(self):
        """Test get_yin_yang_by_name with invalid name."""
        assert ThienCan.get_yin_yang_by_name("Invalid") is None

    def test_get_wu_hang_by_name_valid(self):
        """Test get_wu_hang_by_name with valid name."""
        assert ThienCan.get_wu_hang_by_name("Giáp") == MOC
        assert ThienCan.get_wu_hang_by_name("Bính") == HOA
        assert ThienCan.get_wu_hang_by_name("Canh") == KIM
        assert ThienCan.get_wu_hang_by_name("Nhâm") == THUY

    def test_get_wu_hang_by_name_invalid(self):
        """Test get_wu_hang_by_name with invalid name."""
        assert ThienCan.get_wu_hang_by_name("Invalid") is None

    def test_is_yang_by_name_valid(self):
        """Test is_yang_by_name with valid names."""
        assert ThienCan.is_yang_by_name("Giáp") is True
        assert ThienCan.is_yang_by_name("Ất") is False
        assert ThienCan.is_yang_by_name("Bính") is True
        assert ThienCan.is_yang_by_name("Đinh") is False

    def test_is_yang_by_name_invalid(self):
        """Test is_yang_by_name with invalid name."""
        assert ThienCan.is_yang_by_name("Invalid") is False

    def test_is_yin_by_name_valid(self):
        """Test is_yin_by_name with valid names."""
        assert ThienCan.is_yin_by_name("Giáp") is False
        assert ThienCan.is_yin_by_name("Ất") is True
        assert ThienCan.is_yin_by_name("Bính") is False
        assert ThienCan.is_yin_by_name("Đinh") is True

    def test_is_yin_by_name_invalid(self):
        """Test is_yin_by_name with invalid name."""
        assert ThienCan.is_yin_by_name("Invalid") is False
