
from dataclasses import dataclass
from typing import List, Optional

"""
Data models for I Ching result extraction.
"""



@dataclass(frozen=True)
class HaoInfo:
    """Thông tin một hào trong quẻ."""

    hao: int  # Số hào (1-6, từ dưới lên)
    luc_than: str  # Lục thân (Thế, Ứng, Phụ Mẫu, Tử Tôn, Thê Tài, Quan Quỷ, Huynh Đệ)
    can_chi: str  # Can Chi (ví dụ: "Tý-Thủy", "Dần-Mộc")
    luc_thu: Optional[str] = (
        None  # Lục thú (chỉ có ở quẻ phải): Thanh Long, Chu Tước, Câu Trần, Đằng Xà, Bạch Hổ, Huyền Vũ
    )
    is_dong: bool = False  # Hào động (màu đỏ) - có thể ở cả hai quẻ
    phuc_than: Optional[str] = None  # P.Thần (Phục Thần) - chỉ có ở quẻ trái


@dataclass(frozen=True)
class IChingResult:
    """Kết quả trích xuất thông tin từ ảnh kết quả I Ching."""

    nhat_than: str  # Nhật thần (ví dụ: "Sửu-Thổ")
    nguyet_lenh: str  # Nguyệt lệnh (ví dụ: "Tý-Thủy")
    que_trai: List[HaoInfo]  # Quẻ bên trái: 6 hào từ dưới lên
    que_phai: List[HaoInfo]  # Quẻ bên phải: 6 hào từ dưới lên
    the_vi_tri: Optional[int]  # Vị trí Thế (hào số 1-6, hoặc None)
    ung_vi_tri: Optional[int]  # Vị trí Ứng (hào số 1-6, hoặc None)
    tk_vi_tri: Optional[List[int]] = None  # Vị trí TK (danh sách hào số có "K" trong cột T.K)

    def __post_init__(self):
        """Thực thi các ràng buộc miền cho IChingResult."""
        # Kiểm tra nhat_than nếu không phải None: phải là string không rỗng
        if self.nhat_than is not None:
            if not str(self.nhat_than).strip():
                raise ValueError("nhat_than must be a non-empty string")

        # Kiểm tra nguyet_lenh nếu không phải None: phải là string không rỗng
        if self.nguyet_lenh is not None:
            if not str(self.nguyet_lenh).strip():
                raise ValueError("nguyet_lenh must be a non-empty string")

        # Validate que_trai: may contain at most 6 HaoInfo entries (0..6 allowed) and must not contain duplicate hao numbers
        if len(self.que_trai) > 6:
            raise ValueError(f"que_trai must have at most 6 HaoInfo, got {len(self.que_trai)}")

        # Kiểm tra mỗi HaoInfo.hao trong que_trai là 1-6 (validate range before duplicate check)
        for hao_info in self.que_trai:
            if not isinstance(hao_info.hao, int) or hao_info.hao < 1 or hao_info.hao > 6:
                raise ValueError(f"que_trai: HaoInfo.hao must be an integer in range 1-6, got {hao_info.hao}")

        # Check for duplicate hao numbers
        hao_nums_trai = [h.hao for h in self.que_trai]

        if len(set(hao_nums_trai)) != len(hao_nums_trai):
            raise ValueError("que_trai must not contain duplicate hao numbers")

        # Validate que_phai: may be empty or contain 1-6 HaoInfo objects; all hao numbers must be unique
        if len(self.que_phai) > 6:
            raise ValueError(f"que_phai must have at most 6 HaoInfo, got {len(self.que_phai)}")

        # Kiểm tra mỗi HaoInfo.hao trong que_phai là 1-6 (validate range before duplicate check)
        for hao_info in self.que_phai:
            if not isinstance(hao_info.hao, int) or hao_info.hao < 1 or hao_info.hao > 6:
                raise ValueError(f"que_phai: HaoInfo.hao must be an integer in range 1-6, got {hao_info.hao}")

        # Check for duplicate hao numbers after range validation
        hao_nums_phai = [h.hao for h in self.que_phai]

        if len(set(hao_nums_phai)) != len(hao_nums_phai):
            raise ValueError("que_phai must not contain duplicate hao numbers")

        # Kiểm tra the_vi_tri là None hoặc 1-6
        if self.the_vi_tri is not None:
            if not isinstance(self.the_vi_tri, int) or self.the_vi_tri < 1 or self.the_vi_tri > 6:
                raise ValueError(f"the_vi_tri must be None or an integer in range 1-6, got {self.the_vi_tri}")

        # Kiểm tra ung_vi_tri là None hoặc 1-6
        if self.ung_vi_tri is not None:
            if not isinstance(self.ung_vi_tri, int) or self.ung_vi_tri < 1 or self.ung_vi_tri > 6:
                raise ValueError(f"ung_vi_tri must be None or an integer in range 1-6, got {self.ung_vi_tri}")

        # Kiểm tra tk_vi_tri nếu có: mỗi entry phải là 1-6
        if self.tk_vi_tri is not None:
            if not isinstance(self.tk_vi_tri, list):
                raise ValueError(f"tk_vi_tri must be None or a list of integers, got {type(self.tk_vi_tri).__name__}")
            for idx, vi_tri in enumerate(self.tk_vi_tri):
                if not isinstance(vi_tri, int) or vi_tri < 1 or vi_tri > 6:
                    raise ValueError(f"tk_vi_tri[{idx}] must be an integer in range 1-6, got {vi_tri}")

    def to_dict(self) -> dict:
        """Chuyển đổi sang dictionary."""
        return {
            "nhat_than": self.nhat_than,
            "nguyet_lenh": self.nguyet_lenh,
            "que_trai": [
                {
                    "hao": h.hao,
                    "luc_than": h.luc_than,
                    "can_chi": h.can_chi,
                    "is_dong": h.is_dong,
                    "phuc_than": h.phuc_than,
                }
                for h in self.que_trai
            ],
            "que_phai": [
                {"hao": h.hao, "luc_than": h.luc_than, "can_chi": h.can_chi, "luc_thu": h.luc_thu, "is_dong": h.is_dong}
                for h in self.que_phai
            ],
            "the_vi_tri": self.the_vi_tri,
            "ung_vi_tri": self.ung_vi_tri,
            "tk_vi_tri": self.tk_vi_tri,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IChingResult":
        """Tạo IChingResult từ dictionary."""
        # Coerce hao to int - let ValueError bubble if conversion fails
        que_trai = [
            HaoInfo(
                hao=int(h["hao"]),
                luc_than=h["luc_than"],
                can_chi=h["can_chi"],
                is_dong=h.get("is_dong", False),
                phuc_than=h.get("phuc_than"),
            )
            for h in data.get("que_trai", [])
        ]

        que_phai = [
            HaoInfo(
                hao=int(h["hao"]),
                luc_than=h["luc_than"],
                can_chi=h["can_chi"],
                luc_thu=h.get("luc_thu"),
                is_dong=h.get("is_dong", False),
            )
            for h in data.get("que_phai", [])
        ]

        # Required fields: raise KeyError if missing
        nhat_than = data["nhat_than"]
        nguyet_lenh = data["nguyet_lenh"]

        # Validate required fields are not empty
        if not nhat_than:
            raise ValueError("nhat_than is required and cannot be empty")
        if not nguyet_lenh:
            raise ValueError("nguyet_lenh is required and cannot be empty")

        # Position fields: coerce to int if not None - let ValueError bubble if conversion fails
        the_vi_tri = data.get("the_vi_tri")
        if the_vi_tri is not None:
            the_vi_tri = int(the_vi_tri)

        ung_vi_tri = data.get("ung_vi_tri")
        if ung_vi_tri is not None:
            ung_vi_tri = int(ung_vi_tri)

        # Coerce tk_vi_tri entries to int if present - let ValueError bubble if conversion fails
        tk_vi_tri = data.get("tk_vi_tri")
        if tk_vi_tri is not None:
            tk_vi_tri = [int(v) for v in tk_vi_tri]

        return cls(
            nhat_than=nhat_than,
            nguyet_lenh=nguyet_lenh,
            que_trai=que_trai,
            que_phai=que_phai,
            the_vi_tri=the_vi_tri,
            ung_vi_tri=ung_vi_tri,
            tk_vi_tri=tk_vi_tri,
        )
