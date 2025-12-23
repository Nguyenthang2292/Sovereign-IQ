"""
Tests for I Ching data models.
"""

import pytest

from modules.iching.core.data_models import HaoInfo, IChingResult


class TestHaoInfo:
    """Test HaoInfo dataclass."""
    
    def test_hao_info_creation(self):
        """Test creating HaoInfo with required fields."""
        hao = HaoInfo(
            hao=1,
            luc_than="Thê Tài",
            can_chi="Tý-Thủy"
        )
        assert hao.hao == 1
        assert hao.luc_than == "Thê Tài"
        assert hao.can_chi == "Tý-Thủy"
        assert hao.luc_thu is None
        assert hao.is_dong is False
        assert hao.phuc_than is None
    
    def test_hao_info_with_optional_fields(self):
        """Test creating HaoInfo with all optional fields."""
        hao = HaoInfo(
            hao=2,
            luc_than="Quan Quỷ",
            can_chi="Dần-Mộc",
            luc_thu="Thanh Long",
            is_dong=True,
            phuc_than="Giáp Tý"
        )
        assert hao.hao == 2
        assert hao.luc_than == "Quan Quỷ"
        assert hao.can_chi == "Dần-Mộc"
        assert hao.luc_thu == "Thanh Long"
        assert hao.is_dong is True
        assert hao.phuc_than == "Giáp Tý"


class TestIChingResult:
    """Test IChingResult dataclass."""
    
    @pytest.fixture
    def sample_hao_info_list(self):
        """Create a sample list of HaoInfo for testing."""
        return [
            HaoInfo(hao=1, luc_than="Thê Tài", can_chi="Tý-Thủy"),
            HaoInfo(hao=2, luc_than="Quan Quỷ", can_chi="Dần-Mộc"),
            HaoInfo(hao=3, luc_than="Huynh Đệ", can_chi="Thìn-Thổ"),
            HaoInfo(hao=4, luc_than="Phụ Mẫu", can_chi="Ngọ-Hỏa"),
            HaoInfo(hao=5, luc_than="Tử Tôn", can_chi="Thân-Kim"),
            HaoInfo(hao=6, luc_than="Huynh Đệ", can_chi="Tuất-Thổ"),
        ]
    
    def test_iching_result_creation(self, sample_hao_info_list):
        """Test creating IChingResult."""
        result = IChingResult(
            nhat_than="Sửu-Thổ",
            nguyet_lenh="Tý-Thủy",
            que_trai=sample_hao_info_list.copy(),
            que_phai=sample_hao_info_list.copy(),
            the_vi_tri=4,
            ung_vi_tri=1
        )
        assert result.nhat_than == "Sửu-Thổ"
        assert result.nguyet_lenh == "Tý-Thủy"
        assert len(result.que_trai) == 6
        assert len(result.que_phai) == 6
        assert result.the_vi_tri == 4
        assert result.ung_vi_tri == 1
        assert result.tk_vi_tri is None
    
    def test_iching_result_with_tk(self, sample_hao_info_list):
        """Test creating IChingResult with TK positions."""
        result = IChingResult(
            nhat_than="Sửu-Thổ",
            nguyet_lenh="Tý-Thủy",
            que_trai=sample_hao_info_list.copy(),
            que_phai=sample_hao_info_list.copy(),
            the_vi_tri=4,
            ung_vi_tri=1,
            tk_vi_tri=[5, 6]
        )
        assert result.tk_vi_tri == [5, 6]
    
    def test_iching_result_to_dict(self, sample_hao_info_list):
        """Test converting IChingResult to dictionary."""
        result = IChingResult(
            nhat_than="Sửu-Thổ",
            nguyet_lenh="Tý-Thủy",
            que_trai=sample_hao_info_list.copy(),
            que_phai=sample_hao_info_list.copy(),
            the_vi_tri=4,
            ung_vi_tri=1,
            tk_vi_tri=[6]
        )
        
        data = result.to_dict()
        
        assert data["nhat_than"] == "Sửu-Thổ"
        assert data["nguyet_lenh"] == "Tý-Thủy"
        assert data["the_vi_tri"] == 4
        assert data["ung_vi_tri"] == 1
        assert data["tk_vi_tri"] == [6]
        assert len(data["que_trai"]) == 6
        assert len(data["que_phai"]) == 6
        assert data["que_trai"][0]["hao"] == 1
        assert data["que_trai"][0]["luc_than"] == "Thê Tài"
    
    def test_iching_result_from_dict(self):
        """Test creating IChingResult from dictionary."""
        data = {
            "nhat_than": "Sửu-Thổ",
            "nguyet_lenh": "Tý-Thủy",
            "que_trai": [
                {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "is_dong": False, "phuc_than": None},
                {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "is_dong": True, "phuc_than": "Giáp Tý"},
                {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "is_dong": False, "phuc_than": None},
                {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "is_dong": False, "phuc_than": None},
                {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "is_dong": False, "phuc_than": None},
                {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "is_dong": False, "phuc_than": None}
            ],
            "que_phai": [
                {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "luc_thu": "Thanh Long", "is_dong": False},
                {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "luc_thu": "Chu Tước", "is_dong": False},
                {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "luc_thu": "Câu Trần", "is_dong": False},
                {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "luc_thu": "Đằng Xà", "is_dong": False},
                {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "luc_thu": "Bạch Hổ", "is_dong": False},
                {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "luc_thu": "Huyền Vũ", "is_dong": False}
            ],
            "the_vi_tri": 4,
            "ung_vi_tri": 1,
            "tk_vi_tri": [6]
        }
        
        result = IChingResult.from_dict(data)
        
        assert result.nhat_than == "Sửu-Thổ"
        assert result.nguyet_lenh == "Tý-Thủy"
        assert len(result.que_trai) == 6
        assert len(result.que_phai) == 6
        assert result.the_vi_tri == 4
        assert result.ung_vi_tri == 1
        assert result.tk_vi_tri == [6]
        assert result.que_trai[0].hao == 1
        assert result.que_trai[1].is_dong is True
        assert result.que_trai[1].phuc_than == "Giáp Tý"
        assert result.que_phai[0].luc_thu == "Thanh Long"
    
    def test_iching_result_serialization_roundtrip(self, sample_hao_info_list):
        """Test serialization and deserialization roundtrip."""
        # Create que_trai with optional fields (phuc_than, is_dong)
        que_trai = []
        for i, hao_info in enumerate(sample_hao_info_list):
            que_trai.append(HaoInfo(
                hao=hao_info.hao,
                luc_than=hao_info.luc_than,
                can_chi=hao_info.can_chi,
                is_dong=(i == 1),  # Set is_dong for second item
                phuc_than="Giáp Tý" if i == 1 else None  # Set phuc_than for second item
            ))
        
        # Create que_phai with optional fields (luc_thu, is_dong)
        que_phai = []
        for i, hao_info in enumerate(sample_hao_info_list):
            que_phai.append(HaoInfo(
                hao=hao_info.hao,
                luc_than=hao_info.luc_than,
                can_chi=hao_info.can_chi,
                luc_thu="Thanh Long" if i == 0 else None,  # Set luc_thu for first item
                is_dong=(i == 1)  # Set is_dong for second item
            ))
        
        original = IChingResult(
            nhat_than="Sửu-Thổ",
            nguyet_lenh="Tý-Thủy",
            que_trai=que_trai,
            que_phai=que_phai,
            the_vi_tri=4,
            ung_vi_tri=1,
            tk_vi_tri=[5, 6]
        )
        
        # Convert to dict and back
        data = original.to_dict()
        restored = IChingResult.from_dict(data)
        
        assert restored.nhat_than == original.nhat_than
        assert restored.nguyet_lenh == original.nguyet_lenh
        assert restored.the_vi_tri == original.the_vi_tri
        assert restored.ung_vi_tri == original.ung_vi_tri
        assert restored.tk_vi_tri == original.tk_vi_tri
        assert len(restored.que_trai) == len(original.que_trai)
        assert len(restored.que_phai) == len(original.que_phai)
        
        for i in range(6):
            # Assert que_trai fields
            assert restored.que_trai[i].hao == original.que_trai[i].hao
            assert restored.que_trai[i].luc_than == original.que_trai[i].luc_than
            assert restored.que_trai[i].can_chi == original.que_trai[i].can_chi
            assert restored.que_trai[i].is_dong == original.que_trai[i].is_dong
            assert restored.que_trai[i].phuc_than == original.que_trai[i].phuc_than
            
            # Assert que_phai fields
            assert restored.que_phai[i].hao == original.que_phai[i].hao
            assert restored.que_phai[i].luc_than == original.que_phai[i].luc_than
            assert restored.que_phai[i].can_chi == original.que_phai[i].can_chi
            assert restored.que_phai[i].luc_thu == original.que_phai[i].luc_thu
            assert restored.que_phai[i].is_dong == original.que_phai[i].is_dong

