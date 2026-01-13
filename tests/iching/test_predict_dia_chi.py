"""
Tests for I Ching Địa Chi (Earthly Branches).
"""

from collections import Counter

import pytest

from modules.iching.predict.constants import HOA, KIM, MOC, THO, THUY, YANG, YIN
from modules.iching.predict.dia_chi import (
    DAN,
    DAU,
    DIA_CHI_DICT,
    DIA_CHI_LIST,
    HOI,
    LUC_HOP_DICT,
    LUC_HOP_PAIRS,
    LUC_XUNG_DICT,
    LUC_XUNG_PAIRS,
    MAO,
    MUI,
    NGO,
    SUU,
    THAN,
    THIN,
    TUAT,
    TY,
    TY_AM,
    DiaChi,
)


class TestDiaChiInstances:
    """Test Địa Chi instances."""

    def test_dia_chi_list_length(self):
        """Test that DIA_CHI_LIST has 12 elements."""
        assert len(DIA_CHI_LIST) == 12

    def test_dia_chi_dict_length(self):
        """Test that DIA_CHI_DICT has 12 elements."""
        assert len(DIA_CHI_DICT) == 12

    def test_all_instances_in_list(self):
        """Test that all instances are in DIA_CHI_LIST."""
        instances = [TY, SUU, DAN, MAO, THIN, TY_AM, NGO, MUI, THAN, DAU, TUAT, HOI]
        assert all(dc in DIA_CHI_LIST for dc in instances)

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("Tý", TY),
            ("Sửu", SUU),
            ("Dần", DAN),
            ("Mão", MAO),
            ("Thìn", THIN),
            ("Tỵ", TY_AM),
            ("Ngọ", NGO),
            ("Mùi", MUI),
            ("Thân", THAN),
            ("Dậu", DAU),
            ("Tuất", TUAT),
            ("Hợi", HOI),
        ],
    )
    def test_all_instances_in_dict(self, key, expected):
        """Test that all instances are in DIA_CHI_DICT."""
        assert DIA_CHI_DICT[key] == expected

    @pytest.mark.parametrize(
        "dia_chi_constant,expected_yin_yang",
        [
            (TY, YANG),
            (SUU, YIN),
            (DAN, YANG),
            (MAO, YIN),
            (THIN, YANG),
            (TY_AM, YIN),
            (NGO, YANG),
            (MUI, YIN),
            (THAN, YANG),
            (DAU, YIN),
            (TUAT, YANG),
            (HOI, YIN),
        ],
    )
    def test_yin_yang_alternation(self, dia_chi_constant, expected_yin_yang):
        """Test that YIN and YANG alternate correctly."""
        # Tý (1) - Dương, Sửu (2) - Âm, Dần (3) - Dương, etc.
        assert dia_chi_constant.yin_yang == expected_yin_yang

    def test_wu_hang_assignments(self):
        """Test Ngũ hành assignments."""
        # Tý, Hợi - Thuỷ
        assert TY.wu_hang == THUY
        assert HOI.wu_hang == THUY
        # Dần, Mão - Mộc
        assert DAN.wu_hang == MOC
        assert MAO.wu_hang == MOC
        # Tỵ, Ngọ - Hoả
        assert TY_AM.wu_hang == HOA
        assert NGO.wu_hang == HOA
        # Thân, Dậu - Kim
        assert THAN.wu_hang == KIM
        assert DAU.wu_hang == KIM
        # Sửu, Thìn, Mùi, Tuất - Thổ
        assert SUU.wu_hang == THO
        assert THIN.wu_hang == THO
        assert MUI.wu_hang == THO
        assert TUAT.wu_hang == THO


class TestDiaChiClass:
    """Test DiaChi class."""

    def test_creation_valid(self):
        """Test creating DiaChi with valid values."""
        dc = DiaChi(name="Test", yin_yang=YANG, wu_hang=MOC)
        assert dc.name == "Test"
        assert dc.yin_yang == YANG
        assert dc.wu_hang == MOC

    def test_creation_invalid_yin_yang(self):
        """Test creating DiaChi with invalid yin_yang."""
        with pytest.raises(ValueError, match="yin_yang must be"):
            DiaChi(name="Test", yin_yang="Invalid", wu_hang=MOC)

    def test_creation_invalid_wu_hang(self):
        """Test creating DiaChi with invalid wu_hang."""
        with pytest.raises(ValueError, match="wu_hang must be one of"):
            DiaChi(name="Test", yin_yang=YANG, wu_hang="Invalid")

    def test_repr(self):
        """Test string representation."""
        dc = DiaChi(name="Tý", yin_yang=YANG, wu_hang=THUY)
        repr_str = repr(dc)
        assert "DiaChi" in repr_str
        assert "Tý" in repr_str
        assert YANG in repr_str
        assert THUY in repr_str

    def test_eq_same_name(self):
        """Test equality with same name."""
        dc1 = DiaChi(name="Tý", yin_yang=YANG, wu_hang=THUY)
        dc2 = DiaChi(name="Tý", yin_yang=YIN, wu_hang=MOC)  # Different attributes
        # Should be equal because they have same name
        assert dc1 == dc2

    def test_eq_different_name(self):
        """Test equality with different name."""
        dc1 = DiaChi(name="Tý", yin_yang=YANG, wu_hang=THUY)
        dc2 = DiaChi(name="Sửu", yin_yang=YIN, wu_hang=THO)
        assert dc1 != dc2

    def test_eq_with_instance(self):
        """Test equality with module instance."""
        dc = DiaChi(name="Tý", yin_yang=YANG, wu_hang=THUY)
        assert dc == TY

    def test_eq_with_non_dia_chi(self):
        """Test equality with non-DiaChi object."""
        dc = DiaChi(name="Tý", yin_yang=YANG, wu_hang=THUY)
        assert dc != "Tý"
        assert dc != 123

    def test_hash(self):
        """Test hashing."""
        dc1 = DiaChi(name="Tý", yin_yang=YANG, wu_hang=THUY)
        dc2 = DiaChi(name="Tý", yin_yang=YIN, wu_hang=MOC)
        # Same name should produce same hash
        assert hash(dc1) == hash(dc2)
        # Should be usable in set
        test_set = {dc1, dc2}
        assert len(test_set) == 1  # Same hash, so only one element


class TestDiaChiInstanceMethods:
    """Test DiaChi instance methods."""

    def test_get_yin_yang(self):
        """Test get_yin_yang method."""
        assert TY.get_yin_yang() == YANG
        assert SUU.get_yin_yang() == YIN

    def test_get_wu_hang(self):
        """Test get_wu_hang method."""
        assert TY.get_wu_hang() == THUY
        assert DAN.get_wu_hang() == MOC
        assert THAN.get_wu_hang() == KIM

    def test_is_yang(self):
        """Test is_yang method."""
        assert TY.is_yang() is True
        assert SUU.is_yang() is False
        assert DAN.is_yang() is True
        assert MAO.is_yang() is False

    def test_is_yin(self):
        """Test is_yin method."""
        assert TY.is_yin() is False
        assert SUU.is_yin() is True
        assert DAN.is_yin() is False
        assert MAO.is_yin() is True


class TestLucHop:
    """Test Lục hợp (6 pairs of harmony)."""

    def test_luc_hop_pairs_count(self):
        """Test that LUC_HOP_PAIRS has 6 pairs."""
        assert len(LUC_HOP_PAIRS) == 6

    def test_luc_hop_pairs_content(self):
        """Test content of LUC_HOP_PAIRS."""
        expected_pairs = [
            ("Tý", "Sửu"),
            ("Dần", "Hợi"),
            ("Mão", "Tuất"),
            ("Thìn", "Dậu"),
            ("Tỵ", "Thân"),
            ("Ngọ", "Mùi"),
        ]
        for pair in expected_pairs:
            assert pair in LUC_HOP_PAIRS or (pair[1], pair[0]) in LUC_HOP_PAIRS

    def test_luc_hop_dict_bidirectional(self):
        """Test that LUC_HOP_DICT is bidirectional."""
        assert LUC_HOP_DICT["Tý"] == "Sửu"
        assert LUC_HOP_DICT["Sửu"] == "Tý"
        assert LUC_HOP_DICT["Dần"] == "Hợi"
        assert LUC_HOP_DICT["Hợi"] == "Dần"

    def test_get_luc_hop_by_name(self):
        """Test get_luc_hop_by_name class method."""
        assert DiaChi.get_luc_hop_by_name("Tý") == "Sửu"
        assert DiaChi.get_luc_hop_by_name("Sửu") == "Tý"
        assert DiaChi.get_luc_hop_by_name("Dần") == "Hợi"
        assert DiaChi.get_luc_hop_by_name("Invalid") is None

    def test_get_luc_hop_instance_method(self):
        """Test get_luc_hop instance method."""
        assert TY.get_luc_hop() == SUU
        assert SUU.get_luc_hop() == TY
        assert DAN.get_luc_hop() == HOI
        assert HOI.get_luc_hop() == DAN

    def test_is_luc_hop_class_method(self):
        """Test is_luc_hop class method."""
        assert DiaChi.is_luc_hop("Tý", "Sửu") is True
        assert DiaChi.is_luc_hop("Sửu", "Tý") is True
        assert DiaChi.is_luc_hop("Tý", "Dần") is False
        assert DiaChi.is_luc_hop("Dần", "Hợi") is True

    def test_is_luc_hop_with_instance_method(self):
        """Test is_luc_hop_with instance method."""
        assert TY.is_luc_hop_with(SUU) is True
        assert SUU.is_luc_hop_with(TY) is True
        assert TY.is_luc_hop_with(DAN) is False
        assert DAN.is_luc_hop_with(HOI) is True

    def test_get_all_luc_hop(self):
        """Test get_all_luc_hop class method."""
        all_pairs = DiaChi.get_all_luc_hop()
        assert len(all_pairs) == 6
        # Normalize pairs to canonical form (sorted tuple) for order-independent comparison
        normalized_all_pairs = Counter(tuple(sorted(pair)) for pair in all_pairs)
        normalized_expected = Counter(tuple(sorted(pair)) for pair in LUC_HOP_PAIRS)
        assert normalized_all_pairs == normalized_expected


class TestLucXung:
    """Test Lục xung (6 pairs of conflict)."""

    def test_luc_xung_pairs_count(self):
        """Test that LUC_XUNG_PAIRS has 6 pairs."""
        assert len(LUC_XUNG_PAIRS) == 6

    def test_luc_xung_pairs_content(self):
        """Test content of LUC_XUNG_PAIRS."""
        expected_pairs = [
            ("Tý", "Ngọ"),
            ("Sửu", "Mùi"),
            ("Dần", "Thân"),
            ("Mão", "Dậu"),
            ("Thìn", "Tuất"),
            ("Tỵ", "Hợi"),
        ]
        for pair in expected_pairs:
            assert pair in LUC_XUNG_PAIRS or (pair[1], pair[0]) in LUC_XUNG_PAIRS

    def test_luc_xung_dict_bidirectional(self):
        """Test that LUC_XUNG_DICT is bidirectional."""
        assert LUC_XUNG_DICT["Tý"] == "Ngọ"
        assert LUC_XUNG_DICT["Ngọ"] == "Tý"
        assert LUC_XUNG_DICT["Dần"] == "Thân"
        assert LUC_XUNG_DICT["Thân"] == "Dần"

    def test_get_luc_xung_by_name(self):
        """Test get_luc_xung_by_name class method."""
        assert DiaChi.get_luc_xung_by_name("Tý") == "Ngọ"
        assert DiaChi.get_luc_xung_by_name("Ngọ") == "Tý"
        assert DiaChi.get_luc_xung_by_name("Dần") == "Thân"
        assert DiaChi.get_luc_xung_by_name("Invalid") is None

    def test_get_luc_xung_instance_method(self):
        """Test get_luc_xung instance method."""
        assert TY.get_luc_xung() == NGO
        assert NGO.get_luc_xung() == TY
        assert DAN.get_luc_xung() == THAN
        assert THAN.get_luc_xung() == DAN

    def test_is_luc_xung_class_method(self):
        """Test is_luc_xung class method."""
        assert DiaChi.is_luc_xung("Tý", "Ngọ") is True
        assert DiaChi.is_luc_xung("Ngọ", "Tý") is True
        assert DiaChi.is_luc_xung("Tý", "Sửu") is False
        assert DiaChi.is_luc_xung("Dần", "Thân") is True

    def test_is_luc_xung_with_instance_method(self):
        """Test is_luc_xung_with instance method."""
        assert TY.is_luc_xung_with(NGO) is True
        assert NGO.is_luc_xung_with(TY) is True
        assert TY.is_luc_xung_with(SUU) is False
        assert DAN.is_luc_xung_with(THAN) is True

    def test_get_all_luc_xung(self):
        """Test get_all_luc_xung class method."""
        all_pairs = DiaChi.get_all_luc_xung()
        assert len(all_pairs) == 6
        # Normalize pairs to sets of frozensets for order-insensitive comparison
        normalized_all_pairs = {frozenset(pair) for pair in all_pairs}
        normalized_luc_xung_pairs = {frozenset(pair) for pair in LUC_XUNG_PAIRS}
        assert normalized_all_pairs == normalized_luc_xung_pairs


class TestDiaChiClassMethods:
    """Test DiaChi class methods."""

    def test_get_by_name_valid(self):
        """Test get_by_name with valid name."""
        assert DiaChi.get_by_name("Tý") == TY
        assert DiaChi.get_by_name("Sửu") == SUU
        assert DiaChi.get_by_name("Hợi") == HOI

    def test_get_by_name_invalid(self):
        """Test get_by_name with invalid name."""
        assert DiaChi.get_by_name("Invalid") is None
        assert DiaChi.get_by_name("") is None

    def test_get_yin_yang_by_name_valid(self):
        """Test get_yin_yang_by_name with valid name."""
        assert DiaChi.get_yin_yang_by_name("Tý") == YANG
        assert DiaChi.get_yin_yang_by_name("Sửu") == YIN
        assert DiaChi.get_yin_yang_by_name("Dần") == YANG

    def test_get_yin_yang_by_name_invalid(self):
        """Test get_yin_yang_by_name with invalid name."""
        assert DiaChi.get_yin_yang_by_name("Invalid") is None

    def test_get_wu_hang_by_name_valid(self):
        """Test get_wu_hang_by_name with valid name."""
        assert DiaChi.get_wu_hang_by_name("Tý") == THUY
        assert DiaChi.get_wu_hang_by_name("Dần") == MOC
        assert DiaChi.get_wu_hang_by_name("Thân") == KIM
        assert DiaChi.get_wu_hang_by_name("Sửu") == THO

    def test_get_wu_hang_by_name_invalid(self):
        """Test get_wu_hang_by_name with invalid name."""
        assert DiaChi.get_wu_hang_by_name("Invalid") is None

    def test_is_yang_by_name_valid(self):
        """Test is_yang_by_name with valid names."""
        assert DiaChi.is_yang_by_name("Tý") is True
        assert DiaChi.is_yang_by_name("Sửu") is False
        assert DiaChi.is_yang_by_name("Dần") is True
        assert DiaChi.is_yang_by_name("Mão") is False

    def test_is_yang_by_name_invalid(self):
        """Test is_yang_by_name with invalid name."""
        assert DiaChi.is_yang_by_name("Invalid") is False

    def test_is_yin_by_name_valid(self):
        """Test is_yin_by_name with valid names."""
        assert DiaChi.is_yin_by_name("Tý") is False
        assert DiaChi.is_yin_by_name("Sửu") is True
        assert DiaChi.is_yin_by_name("Dần") is False
        assert DiaChi.is_yin_by_name("Mão") is True

    def test_is_yin_by_name_invalid(self):
        """Test is_yin_by_name with invalid name."""
        assert DiaChi.is_yin_by_name("Invalid") is False


class TestCalculateScore:
    """Test calculate_score method."""

    def test_calculate_score_single_luc_hop(self):
        """Test calculate_score with one lục hợp pair."""
        # Tý hợp Sửu = +1
        score = DiaChi.calculate_score(["Tý", "Sửu"])
        assert score == 1

    def test_calculate_score_single_luc_xung(self):
        """Test calculate_score with one lục xung pair."""
        # Tý xung Ngọ = -1
        score = DiaChi.calculate_score(["Tý", "Ngọ"])
        assert score == -1

    def test_calculate_score_mixed(self):
        """Test calculate_score with both hợp and xung."""
        # Tý hợp Sửu (+1), Tý xung Ngọ (-1) = 0
        score = DiaChi.calculate_score(["Tý", "Sửu", "Ngọ"])
        assert score == 0

    def test_calculate_score_multiple_hops(self):
        """Test calculate_score with multiple hợp pairs."""
        # Tý hợp Sửu (+1), Dần hợp Hợi (+1) = +2
        score = DiaChi.calculate_score(["Tý", "Sửu", "Dần", "Hợi"])
        assert score == 2

    def test_calculate_score_multiple_xungs(self):
        """Test calculate_score with multiple xung pairs."""
        # Tý xung Ngọ (-1), Dần xung Thân (-1) = -2
        score = DiaChi.calculate_score(["Tý", "Ngọ", "Dần", "Thân"])
        assert score == -2

    def test_calculate_score_no_relationships(self):
        """Test calculate_score with no hợp or xung."""
        # Tý, Dần - no direct relationship = 0
        score = DiaChi.calculate_score(["Tý", "Dần"])
        assert score == 0

    def test_calculate_score_empty_list(self):
        """Test calculate_score with empty list."""
        score = DiaChi.calculate_score([])
        assert score == 0

    def test_calculate_score_single_element(self):
        """Test calculate_score with single element."""
        score = DiaChi.calculate_score(["Tý"])
        assert score == 0

    def test_calculate_score_complex(self):
        """Test calculate_score with complex scenario."""
        # Tý hợp Sửu (+1), Tý xung Ngọ (-1), Dần hợp Hợi (+1), Dần xung Thân (-1)
        # Total: +1 -1 +1 -1 = 0
        score = DiaChi.calculate_score(["Tý", "Sửu", "Ngọ", "Dần", "Hợi", "Thân"])
        assert score == 0

    def test_calculate_score_all_pairs(self):
        """Test calculate_score with all 12 chi."""
        # This will have 6 hợp pairs and 6 xung pairs
        all_chi = ["Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi"]
        score = DiaChi.calculate_score(all_chi)
        # Should be 0 (6 hợp + 6 xung = +6 -6 = 0)
        assert score == 0
