# hst_agent.py - HST Agent with Text2SQL and ReAct mechanism
"""HST Agent for querying tender records database using Text2SQL with ReAct strategy"""
import logging
import os
import time
import redis
import json
import hashlib
import asyncio
from decimal import Decimal
from typing import List, Dict, Any, Generator, Optional, Tuple, AsyncGenerator
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError, OperationalError, TimeoutError as SQLTimeoutError
from openai import OpenAI, AsyncOpenAI, RateLimitError, APIConnectionError, AuthenticationError
from src.common import TokenUsage
from src.utils.utils_logging import log_openai_agent_response, log_openai_agent_error
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# ERROR CODE MAPPING
ERROR_CODES = {
    "rate_limit": "01",
    "authentication": "02",
    "not_found": "03",
    "connection": "04",
    "timeout": "05",
    "sql_error": "06",
    "general": "99"
}


# ============================================================================
# SCHEMA METADATA - Ngữ nghĩa cho từng cột
# ============================================================================
SCHEMA_METADATA = {
    "Năm_phê_duyệt_KQLCNT": {
        "description": "Năm phê duyệt kết quả lựa chọn nhà thầu (YYYY)",
        "type": "bigint",
        "unique values": "2024, 2025"
    },
    "Thời_gian_phê_duyệt_KQLCNT": {
        "description": "Tháng phê duyệt kết quả lựa chọn nhà thầu",
        "type": "text",
        "unique values": "Tháng 01, tháng 02, ..., Tháng 12",
    },
    "Số_thông_báo_mời_thầu": {
        "description": "Mã số thông báo mời thầu",
        "type": "text",
        "example": "IB2400101502-00"
    },
    "Tên_bên_trúng_thầu": {
        "description": "Tên công ty trúng thầu",
        "type": "text",
        "example": "CÔNG TY TNHH FPT IS, CÔNG TY TNHH HỆ THỐNG THÔNG TIN FPT"
    },
    "Tên_bên_mời_thầu": {
        "description": "Tên khách hàng/bên mời thầu",
        "type": "text",
        "example": "NGÂN HÀNG THƯƠNG MẠI CỔ PHẦN ĐẦU TƯ VÀ PHÁT TRIỂN VIỆT NAM"
    },
    "Tên_gói_thầu": {
        "description": "Tên gói thầu/dự án",
        "type": "text",
        "example": "Mua sắm trang thiết bị phục vụ công tác lý lịch tư pháp"
    },
    "Giá_trúng_thầu": {
        "description": "Giá trị trúng thầu (tỷ VND)",
        "type": "double precision",
        "example": "0.2818005, 1.9646627",
        "note": "Dùng cột này để tính toán, sắp xếp, tổng hợp"
    },
    "Hình_thức_LCNT": {
        "description": "Hình thức lựa chọn nhà thầu",
        "type": "text",
        "example": "Tham gia thực hiện cộng đồng, Đàm phán giá, Chỉ định thầu rút gọn"
    },
    "Mã_tỉnh_cũ": {
        "description": "Mã tỉnh cũ của khách hàng",
        "type": "text",
        "example": "HNI, QBH"
    },
    "Mã_tỉnh_mới": {
        "description": "Mã tỉnh mới của khách hàng",
        "type": "text",
        "example": "HNI, HUE"
    },
    "Lĩnh_vực_Khách_hàng": {
        "description": "Lĩnh vực của khách hàng",
        "type": "text",
        "unique values": "GDS, BQP, TW, CQT, YTS, KHDN",
        "values": {
            "GDS": "Giáo dục số",
            "BQP": "Bộ quốc phòng",
            "TW": "Trung ương/Bộ ngành",
            "CQT": "Chính quyền tỉnh",
            "YTS": "Y tế số",
            "KHDN": "Khách hàng doanh nghiệp",
        }
    },
    "Đơn_vị_kinh_doanh(VTS)": {
        "description": "Đơn vị kinh doanh của VTS",
        "type": "text",
        "unique values": "TT CQĐT, P KHHN, TT GPYTS, TT DTTM, TT GPGDS, TT KHDN, TT QPAN, TT GPMN",
        "values": {
            "TT CQĐT": "Trung tâm Chính quyền điện tử",
            "TT GPYTS": "Trung tâm Giải pháp Y tế số",
            "TT DTTM": "Trung tâm Đô thị thông minh",
            "TT GPGDS": "Trung tâm Giải pháp Giáo dục số",
            "TT KHDN": "Trung tâm Khách hàng doanh nghiệp",
            "TT QPAN": "Trung tâm Quốc phòng an ninh",
            "TT GPMN": "Trung tâm Giải pháp miền Nam"
        }
    },
    "Phân_loại_sản_phẩm": {
        "description": "Loại sản phẩm/dịch vụ",
        "type": "text",
        "unique values": "Phần mềm, Kênh truyền, dịch vụ, phần cứng, [null]"
    },
    "Nhóm_mời_thầu": {
        "description": "Nhóm phân loại bên mời thầu",
        "type": "text",
        "unique values": "dịch vụ đặc thù, Khác, X1"
    },
    "Nhóm_trúng_thầu": {
        "description": "Tên công ty trúng thầu",
        "type": "text",
        "example": "FPT, Viettel-IDC, Viettel-VCC, VNPT, Viettel-Khác",
        "note": "Các giá trị liên quan Viettel cần lọc bằng ILIKE '%Viettel%', không dùng = 'VTS'"
    },
    "Nhóm_trúng_thầu_shortlist": {
        "description": "Tên công ty trúng thầu, group thành 4 nhóm chính.",
        "type": "text",
        "unique values": "FPT, Viettel, VNPT, khác"
    },
    "Năm_phát_hành_TBMT": {
        "description": "Năm phát hành thông báo mời thầu",
        "type": "text",
        "unique values": "2022, 2023, 2024, 2025"
    },
    "Thoi_gian_phe_duyet": {
        "description": "Thời gian phê duyệt (datetime format)",
        "type": "datetime",
        "pg_type": "timestamp without time zone",
        "validation": "datetime",
        "example": "2024-10-01 00:00:00",
        "note": "Dùng để filter theo tháng/năm chính xác"
    }
}


###############################################################################
# UNIFIED GUIDE
###############################################################################

GENERAL_GUIDE_COMBINED = """
CÁC TÌNH HUỐNG MẪU (Intent Examples) DÀNH CHO TRỢ LÝ HỒ SƠ THẦU (HST)

1. **Phân tích thị phần (market_share)**
- Hỏi: "Thị phần của Viettel so với FPT trong tháng 10/2025 là bao nhiêu?"
- Hướng dẫn: dùng WHERE "Nhóm_trúng_thầu" ILIKE '%Viettel%' GROUP BY "Nhóm_trúng_thầu",
  SUM("Giá_trúng_thầu"), tính tổng và %.

2. **Phân tích đối thủ (competitor_analysis)**
- Hỏi: "So sánh kết quả đấu thầu giữa Viettel, VNPT và FPT"
- Hướng dẫn: nhóm theo "Nhóm_trúng_thầu_shortlist", tính tổng giá trị và đếm số gói.

3. **Phân tích theo thời gian (time_series)**
- Hỏi: "Xu hướng giá trị trúng thầu qua các tháng năm 2025"
- Hướng dẫn: GROUP BY "Năm_phê_duyệt_KQLCNT", "Thời_gian_phê_duyệt_KQLCNT", SUM("Giá_trúng_thầu").

4. **Phân tích theo đơn vị (unit_performance)**
- Hỏi: "Trung tâm nào của Viettel có giá trị trúng thầu cao nhất?"
- Hướng dẫn: GROUP BY "Đơn_vị_kinh_doanh(VTS)", SUM("Giá_trúng_thầu").

5. **Phân tích NSNN (nsnn_analysis)**
- Hỏi: "Thị phần Viettel trong lĩnh vực NSNN 10 tháng đầu năm 2025"
- Hướng dẫn: WHERE "Lĩnh_vực_Khách_hàng" IN ('YTS','GDS','CQT'),
  dùng "Thoi_gian_phe_duyet" để lọc thời gian, tính tổng thị trường và Viettel.

6. **Top hợp đồng (top_contracts)**
- Hỏi: "Top 5 gói thầu có giá trị cao nhất trong tháng 9"
- Hướng dẫn: ORDER BY "Giá_trúng_thầu" DESC LIMIT 5.

7. **Báo cáo tổng quan thị trường (market_overview)**
- Hỏi: "Báo cáo tổng quan thị trường thầu lũy kế 10 tháng" hoặc 
       "Tổng quan thị trường đấu thầu năm 2025 đến nay"
- Gợi ý SQL:
  ```sql
  SELECT 
      "Nhóm_trúng_thầu_shortlist",
      COUNT(*) AS tong_so_goi,
      SUM("Giá_trúng_thầu") AS tong_gia_tri_thi_truong
  FROM thau_2025
  WHERE "Thoi_gian_phe_duyet" >= DATE_TRUNC('year', CURRENT_DATE)
      AND "Thoi_gian_phe_duyet" < DATE_TRUNC('month', CURRENT_DATE)
  GROUP BY "Nhóm_trúng_thầu_shortlist"
  ORDER BY tong_gia_tri_thi_truong DESC;
  ```
- Gợi ý hiển thị: bảng tổng hợp thị phần từng nhóm (FPT, Viettel, VNPT, Khác) kèm số lượng gói và tổng giá trị.

8. **Báo cáo thị phần tháng cụ thể (monthly_market_share_report)**
- Hỏi: "Báo cáo thị phần thầu tháng 10/2025"
- Hướng dẫn: Tổng hợp giá trị trúng thầu theo "Nhóm_trúng_thầu", lọc theo tháng 10 và năm 2025.
- Gợi ý SQL:
  ```sql
    SELECT 
        "Nhóm_trúng_thầu",
        SUM("Giá_trúng_thầu") AS tong_gia_tri_trung_thau,
        COUNT(*) AS so_goi_thau,
        ROUND(
            CAST(SUM("Giá_trúng_thầu") * 100.0 /
            SUM(SUM("Giá_trúng_thầu")) OVER () AS numeric), 
            2
        ) AS thi_phan_phan_tram
    FROM thau_2025
    WHERE 
        "Năm_phê_duyệt_KQLCNT" = 2025
        AND LOWER("Thời_gian_phê_duyệt_KQLCNT") IN ('tháng 10')
        AND "Giá_trúng_thầu" IS NOT NULL
        AND "Giá_trúng_thầu" > 0
        AND "Nhóm_trúng_thầu" != 'Khác'
    GROUP BY "Nhóm_trúng_thầu"
    ORDER BY tong_gia_tri_trung_thau DESC
    LIMIT 10;

9. **So sánh giá trị thầu giữa các tháng (month_comparison)**
- Hỏi ví dụ: "So sánh giá trị thầu trong tháng 9 và 10 với trung bình 6 tháng đầu năm"
- Lưu ý: Phải tự hiểu là chỉ tính cho Viettel
- Gợi ý SQL:
  ```sql
    WITH 
    -- Tổng giá trị theo tháng (chỉ Viettel)
    thau_theo_thang AS (
        SELECT 
            EXTRACT(MONTH FROM "Thoi_gian_phe_duyet") AS thang,
            SUM(CAST("Giá_trúng_thầu" AS DECIMAL)) AS tong_gia_tri,
            COUNT(*) AS so_goi
        FROM thau_2025
        WHERE 
            EXTRACT(YEAR FROM "Thoi_gian_phe_duyet") = 2025
            AND "Nhóm_trúng_thầu_shortlist" = 'Viettel'
        GROUP BY EXTRACT(MONTH FROM "Thoi_gian_phe_duyet")
    ),

    -- Trung bình 6 tháng đầu năm
    tb_6_thang_dau AS (
        SELECT 
            AVG(tong_gia_tri) AS tb_6_thang_dau_nam
        FROM thau_theo_thang
        WHERE thang BETWEEN 1 AND 6
    )

    SELECT 
        t10.so_goi AS so_goi_thang_10,
        t10.tong_gia_tri AS gia_tri_thang_10,
        t9.tong_gia_tri AS gia_tri_thang_9,
        tb.tb_6_thang_dau_nam,
        ROUND((t10.tong_gia_tri - t9.tong_gia_tri) / NULLIF(t9.tong_gia_tri, 0) * 100, 2) AS ty_le_tang_vs_thang9,
        ROUND((t10.tong_gia_tri - tb.tb_6_thang_dau_nam) / NULLIF(tb.tb_6_thang_dau_nam, 0) * 100, 2) AS ty_le_tang_vs_tb6
    FROM thau_theo_thang t10
    JOIN thau_theo_thang t9 ON t9.thang = 9
    CROSS JOIN tb_6_thang_dau tb
    WHERE t10.thang = 10;
  ```
- Dùng để so sánh quy mô giá trị và tốc độ tăng trưởng giữa tháng hiện tại, tháng trước và trung bình 6T đầu năm

10. Báo cáo thị phần lĩnh vực Chính quyền Tỉnh (provincial_gov_market_share)
- Hỏi ví dụ: "Báo cáo thị phần lĩnh vực chính quyền tỉnh"
- Gợi ý SQL:
    ```sql
    SELECT 
        "Nhóm_trúng_thầu_shortlist" AS nhom,
        COUNT(*) AS so_goi,
        SUM(CAST("Giá_trúng_thầu" AS DECIMAL)) AS gia_tri,
        ROUND(SUM(CAST("Giá_trúng_thầu" AS DECIMAL)) 
            / NULLIF(
                (SELECT SUM(CAST("Giá_trúng_thầu" AS DECIMAL)) 
                FROM thau_2025 
                WHERE "Lĩnh_vực_Khách_hàng" = 'CQT' 
                AND EXTRACT(YEAR FROM "Thoi_gian_phe_duyet") = 2025
                AND "Thoi_gian_phe_duyet" < DATE_TRUNC('month', CURRENT_DATE)
                ), 0
            ) * 100, 2) AS thi_phan_phan_tram
    FROM thau_2025
    WHERE 
        "Lĩnh_vực_Khách_hàng" = 'CQT'
        AND EXTRACT(YEAR FROM "Thoi_gian_phe_duyet") = 2025
        AND "Thoi_gian_phe_duyet" < DATE_TRUNC('month', CURRENT_DATE)
    GROUP BY "Nhóm_trúng_thầu_shortlist"
    ORDER BY gia_tri DESC;
    ```
- Dùng để tạo báo cáo chi tiết thị phần Viettel, VNPT, FPT trong lĩnh vực chính quyền tỉnh.

11. **Gói thầu lớn nhất (largest_contract)**
- Hỏi ví dụ: 
  - "Gói thầu lớn nhất của VNPT là gì?"
  - "Cho tôi thông tin gói thầu có giá trị cao nhất của Viettel năm 2025"
- Hướng dẫn:
  - Lọc theo "Nhóm_trúng_thầu_shortlist" tương ứng ('Viettel', 'VNPT', 'FPT', 'Khác')
  - Nếu người dùng chỉ nói "Viettel", có thể match bằng ILIKE '%Viettel%' trên "Nhóm_trúng_thầu"
  - Có thể thêm điều kiện theo năm nếu được nhắc đến.
  - Sắp xếp giảm dần theo "Giá_trúng_thầu" và lấy LIMIT 1.
- Gợi ý SQL:
  ```sql
  SELECT 
      "Số_thông_báo_mời_thầu",
      "Tên_gói_thầu",
      "Tên_bên_trúng_thầu",
      "Tên_bên_mời_thầu",
      "Giá_trúng_thầu",
      "Lĩnh_vực_Khách_hàng",
      "Đơn_vị_kinh_doanh(VTS)",
      "Phân_loại_sản_phẩm",
      "Hình_thức_LCNT",
      "Thoi_gian_phe_duyet",
      "Năm_phê_duyệt_KQLCNT",
      "Thời_gian_phê_duyệt_KQLCNT"
  FROM thau_2025
  WHERE 
      "Nhóm_trúng_thầu_shortlist" = 'VNPT'
      AND "Giá_trúng_thầu" IS NOT NULL
      AND "Giá_trúng_thầu" > 0
  ORDER BY "Giá_trúng_thầu" DESC
  LIMIT 1;
    ```

12. **So sánh kết quả theo quý có phụ thuộc lịch sử hội thoại (quarter_comparison_with_history)**

CASE MẪU:

Lượt 1 — User hỏi:
"so sánh số gói và tổng giá trị trúng thầu của VTS quý 3 năm 2025 với cùng kỳ năm ngoái"

→ SQL chuẩn phải tạo:
SELECT 
    EXTRACT(YEAR FROM "Thoi_gian_phe_duyet") AS nam,
    COUNT(*) AS so_goi_thau,
    SUM("Giá_trúng_thầu") AS tong_gia_tri
FROM thau_2025
WHERE 
    "Nhóm_trúng_thầu_shortlist" = 'Viettel'
    AND EXTRACT(MONTH FROM "Thoi_gian_phe_duyet") IN (7,8,9)
    AND EXTRACT(YEAR FROM "Thoi_gian_phe_duyet") IN (2024,2025)
    AND "Giá_trúng_thầu" IS NOT NULL
    AND "Giá_trúng_thầu" > 0
GROUP BY nam
ORDER BY nam;

Giải thích:
- User nói “VTS” → mapping chính xác phải là "Nhóm_trúng_thầu_shortlist" = 'Viettel'
- Quý 3 = tháng 7–9 → dùng EXTRACT(MONTH) IN (7,8,9)
- “cùng kỳ năm ngoái” → luôn lấy năm hiện tại trong câu hỏi và năm hiện tại - 1
- Dùng Thoi_gian_phe_duyet (datetime) để lọc thời gian.
- GROUP BY theo năm để có 2 dòng: 2024 & 2025.

***

Lượt 2 — User hỏi:
"thế còn quý 2"

→ Agent phải hiểu:
- User KHÔNG nhắc lại “VTS” vì đã nói ở lượt 1 → tiếp tục dùng Viettel
- Không nhắc lại “2025” nhưng phải hiểu: vẫn so năm 2025 và 2024
- Chỉ thay đổi quý → dùng tháng 4–6

→ SQL chuẩn:
SELECT 
    EXTRACT(YEAR FROM "Thoi_gian_phe_duyet") AS nam,
    COUNT(*) AS so_goi_thau,
    SUM("Giá_trúng_thầu") AS tong_gia_tri
FROM thau_2025
WHERE 
    "Nhóm_trúng_thầu_shortlist" = 'Viettel'
    AND EXTRACT(MONTH FROM "Thoi_gian_phe_duyet") IN (4,5,6)
    AND EXTRACT(YEAR FROM "Thoi_gian_phe_duyet") IN (2024,2025)
    AND "Giá_trúng_thầu" IS NOT NULL
    AND "Giá_trúng_thầu" > 0
GROUP BY nam
ORDER BY nam;

Nguyên tắc cần ghi nhớ cho mọi trường hợp tương tự:
- Nếu user ở lượt sau chỉ thay đổi một phần câu hỏi (ví dụ: “thế còn quý 2”, “còn tháng 8 thì sao”, “còn FPT?”), agent phải:
  1. Kế thừa toàn bộ cấu trúc logic từ câu hỏi trước đó  
  2. Chỉ thay đổi duy nhất phần mà user hỏi lại  
  3. Tuyệt đối không reset ý nghĩa, không hiểu sang ngữ cảnh mới  

👉 Trong mọi trường hợp, tuân thủ các quy tắc SQL và quy trình ReAct chuẩn:
- Dùng cột "Giá_trúng_thầu" (numeric) để tính toán.
- Dùng "Thoi_gian_phe_duyet" cho điều kiện thời gian (NOT "Thời_gian_phê_duyệt_KQLCNT").
- Các nhóm nhà thầu chuẩn: FPT, Viettel, VNPT, Khác.
- Kết quả trả lời phải có số liệu cụ thể, không placeholder.
"""


# ============================================================================
# StreamBuffer, ErrorHandler, 
# ============================================================================

class StreamBuffer:
    """Buffer chunks for optimized streaming"""
    def __init__(self, buffer_size: int = 5):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, chunk: str) -> Optional[str]:
        """Add chunk, return combined if buffer full"""
        self.buffer.append(chunk)
        if len(self.buffer) >= self.buffer_size:
            result = "".join(self.buffer)
            self.buffer = []
            return result
        return None

    def flush(self) -> str:
        """Flush remaining buffer"""
        result = "".join(self.buffer)
        self.buffer = []
        return result


class ErrorHandler:
    """Centralized error handling với mã lỗi"""
    
    @staticmethod
    def get_user_friendly_message(error: Exception, source_name: str = "") -> Tuple[str, str]:
        """
        Convert exception to user-friendly message + error code
        Returns: (message, error_code)
        """
        error_str = str(error).lower()
        
        # SQL Error
        if isinstance(error, (SQLAlchemyError, OperationalError, SQLTimeoutError)):
            logger.error(f"SQL error for {source_name}: {error}")
            return (
                f"Lỗi truy vấn cơ sở dữ liệu. Vui lòng thử lại. (Mã lỗi: {ERROR_CODES['sql_error']})",
                ERROR_CODES['sql_error']
            )
        
        # Timeout Error
        if "timeout" in error_str or isinstance(error, asyncio.TimeoutError):
            logger.warning(f"Timeout for {source_name}: {error}")
            return (
                f"Truy vấn mất quá nhiều thời gian. Vui lòng thử lại. (Mã lỗi: {ERROR_CODES['timeout']})",
                ERROR_CODES['timeout']
            )
        
        # Rate Limit Error
        if isinstance(error, RateLimitError) or "rate limit" in error_str or "429" in error_str:
            logger.warning(f"Rate limit hit for {source_name}: {error}")
            return (
                f"Hệ thống đang quá tải. Vui lòng thử lại sau ít phút. (Mã lỗi: {ERROR_CODES['rate_limit']})",
                ERROR_CODES['rate_limit']
            )
        
        # Authentication Error
        if isinstance(error, AuthenticationError) or "authentication" in error_str or "401" in error_str:
            logger.error(f"Authentication error for {source_name}: {error}")
            return (
                f"Lỗi xác thực hệ thống. Vui lòng liên hệ quản trị viên. (Mã lỗi: {ERROR_CODES['authentication']})",
                ERROR_CODES['authentication']
            )
        
        # Connection Error
        if isinstance(error, APIConnectionError) or "connection" in error_str:
            logger.error(f"Connection error for {source_name}: {error}")
            return (
                f"Không thể kết nối tới hệ thống. Vui lòng kiểm tra kết nối mạng. (Mã lỗi: {ERROR_CODES['connection']})",
                ERROR_CODES['connection']
            )
        
        # General Error
        logger.error(f"General error for {source_name}: {error}")
        return (
            f"Có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại sau. (Mã lỗi: {ERROR_CODES['general']})",
            ERROR_CODES['general']
        )
    
    @staticmethod
    def should_retry(error: Exception) -> Tuple[bool, float]:
        """
        Determine if should retry and wait time
        Returns: (should_retry, wait_seconds)
        """
        error_str = str(error).lower()
        
        # Rate limit - extract wait time from error message
        if isinstance(error, RateLimitError) or "rate limit" in error_str:
            import re
            match = re.search(r'try again in (\d+)ms', error_str)
            if match:
                wait_ms = int(match.group(1))
                wait_seconds = (wait_ms / 1000.0) + 0.5
                return True, min(wait_seconds, 10.0)
            return True, 2.0
        
        # Connection errors - quick retry
        if isinstance(error, (APIConnectionError, OperationalError)) or "connection" in error_str or "timeout" in error_str:
            return True, 1.0
        
        # Don't retry auth errors
        return False, 0.0


# ============================================================================
# HSTAgent - Main Agent Class
# ============================================================================

class HSTAgent:
    """HST Agent with Text2SQL and ReAct mechanism"""
    
    DEFAULT_MODEL = "gpt-4.1"
    TIMEOUT_SECONDS = 120
    MAX_RETRIES = 3
    
    def __init__(
        self,
        source_name: str,
        db_connection_string: str,
        table_name: str,
        redis_client: redis.Redis,
        system_prompt: str = None,
        model: str = None
    ):
        """
        Initialize HST Agent
        
        Args:
            source_name: Name of the data source
            db_connection_string: Database connection string
            table_name: Table name to query
            redis_client: Redis client for caching
            system_prompt: Custom system prompt
            model: Model to use (default: gpt-4.1)
        """
        self.source_name = source_name
        self.db_connection_string = db_connection_string
        self.table_name = table_name
        self.redis_client = redis_client
        self.model = model or self.DEFAULT_MODEL
        self.model = self.model.split("/")[1] if "/" in self.model else self.model
        self.vector_store_id = "hst"
        logger.warning(f"Model for hst agent is {self.model}")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize database engine
        self.engine = create_engine(
            db_connection_string,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        # System prompt
        self.system_prompt = system_prompt or self._default_system_prompt()
    
        # Thời gian hiện tại 
        now = datetime.now()
        self.current_date = now
        self.current_year = now.year
        self.current_month = now.month
        self.current_day = now.day

        logger.info(f"[TIME CONTEXT] Current date context initialized: {self.current_date}")

        # Initialize schema
        self._initialize_schema()
        
        logger.info(f"HST Agent initialized for {source_name} with table {table_name}")
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for HST Agent"""
        return """Bạn là trợ lý AI chuyên trả lời câu hỏi về hồ sơ thầu (HST).
Bạn có khả năng chuyển đổi câu hỏi tiếng Việt thành SQL query để truy vấn database.

Quy trình ReAct:
1. THOUGHT: Phân tích câu hỏi và xác định thông tin cần thiết
2. ACTION: Quyết định hành động tiếp theo (execute_query, final_answer)
3. OBSERVATION: Phân tích kết quả từ hành động
4. Lặp lại cho đến khi bạn CHO LÀ đã đủ thông tin để trả lời

💡 Lưu ý:
- Bạn tự quyết định khi nào dùng final_answer. Nếu thấy đủ dữ liệu/thông tin, hãy trả lời.
- Khi viết final_answer, PHẢI chèn các con số cụ thể (tổng giá trị, % thị phần, số hợp đồng, v.v.)
- Nếu chưa có số, hãy thực hiện thêm execute_query hoặc phép tính trung gian (tổng, chia, %).
    Ví dụ:
    ✅ "Tổng giá trị NSNN là 1.230 tỷ VND, VTS đạt 615 tỷ (50%)."
    ❌ "Tổng giá trị là X đồng, VTS đạt Y đồng, chiếm Z%."

🧭 QUY TẮC CHO AGENT REACT:
- Bạn KHÔNG cần viết câu trả lời tự nhiên trong final_answer.
- Khi bạn đã xác định được SQL đúng, đã thực thi query và dữ liệu trả về hợp lý (có kết quả, không lỗi),
  hãy kết thúc bằng:
  ACTION: final_answer("ready")
- Agent phía sau sẽ tự tổng hợp báo cáo chi tiết từ dữ liệu.

QUAN TRỌNG (PostgreSQL):
1. TÊN CỘT: Sử dụng CHÍNH XÁC tên cột từ schema (có dấu, chữ hoa/thường đúng)
   
   ⚠️ PostgreSQL yêu cầu DOUBLE QUOTES cho column names có dấu/mixed case
   
   DANH SÁCH TÊN CỘT ĐÚNG (luôn wrap trong ""):
   ✅ "Giá_trúng_thầu" (số, dùng để tính toán)
   ✅ "Giá_trúng_thầu" (text, KHÔNG dùng để tính)
   ✅ "Lĩnh_vực_Khách_hàng"
   ✅ "Thời_gian_phê_duyệt_KQLCNT" (có _KQLCNT ở cuối)
   ✅ "Năm_phê_duyệt_KQLCNT" (có _KQLCNT ở cuối)
   ✅ "Đơn_vị_kinh_doanh(VTS)" (có (VTS) ở cuối)
   ✅ "Nhóm_trúng_thầu"
   ✅ "Tên_bên_trúng_thầu"
   ✅ "Tên_bên_mời_thầu"
   ✅ "Tên_gói_thầu"
   
   VÍ DỤ SQL ĐÚNG (với double quotes):
   SELECT "Giá_trúng_thầu", "Nhóm_trúng_thầu" FROM table
   WHERE "Thời_gian_phê_duyệt_KQLCNT" = 'Tháng 10'
   
   SAI THƯỜNG GẶP:
   ❌ Giá_trúng_thầu (no quotes) → SYNTAX ERROR
   ❌ 'Giá_trúng_thầu' (single quotes) → ERROR
   ✅ "Giá_trúng_thầu" (double quotes) → CORRECT
   
2. STRING LITERALS: Dùng dấu nháy đơn cho values (NOT column names)
   ✅ ĐÚNG: WHERE "Nhóm_trúng_thầu" = 'VTS'
   ❌ SAI: WHERE Nhóm_trúng_thầu = 'VTS' - missing quotes on column
   
3. ACTION FORMAT: Tên cột KHÔNG CẦN quotes trong action parameter
   ✅ ĐÚNG: get_distinct_values("Thời_gian_phê_duyệt_KQLCNT")
   ⚠️ NOTE: Code sẽ tự thêm double quotes khi generate SQL
   
4. KIỂM TRA SQL TRƯỚC KHI EXECUTE:
   - Column names wrapped trong double quotes ""
   - String values wrapped trong single quotes ''
   - Không có empty column name
""" + "\n\n" + GENERAL_GUIDE_COMBINED
    
    def _initialize_schema(self):
        """Initialize database schema information with metadata"""
        try:
            # Get schema from database
            inspector = inspect(self.engine)
            columns = inspector.get_columns(self.table_name)
            
            # Enrich with metadata
            enriched_columns = []
            for col in columns:
                col_name = col["name"]
                col_info = {
                    "name": col_name,
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True)
                }
                
                # Add metadata if available
                if col_name in SCHEMA_METADATA:
                    col_info.update(SCHEMA_METADATA[col_name])
                
                enriched_columns.append(col_info)
            
            self.schema_info = {
                "table_name": self.table_name,
                "columns": enriched_columns
            }
            
            logger.info(f"Initialized enriched schema for {self.source_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            self.schema_info = {"table_name": self.table_name, "columns": []}
    
    def _get_sample_rows(self, limit: int = 5) -> List[Dict]:
        """Get sample rows from database"""
        try:
            # Query database
            query = text(f"SELECT * FROM {self.table_name} LIMIT {limit}")
            logger.info(f"[SQL SAMPLES] SELECT * FROM {self.table_name} LIMIT {limit}")
            
            with self.engine.connect() as conn:
                result = conn.execute(query)
                samples = []
                for row in result:
                    mapped = {}
                    for k, v in dict(row._mapping).items():
                        if isinstance(v, datetime):
                            mapped[k] = v.isoformat(sep=" ", timespec="seconds")
                        else:
                            mapped[k] = v
                    samples.append(mapped)
            
            logger.info(f"[SQL SAMPLES SUCCESS] Retrieved {len(samples)} sample rows")
            return samples
            
        except Exception as e:
            logger.error(f"[SQL SAMPLES FAILED] Failed to get sample rows: {e}")
            return []
    
    def _get_distinct_values(self, column_name: str, limit: int = 50) -> List[Any]:
        """Get distinct values for a column"""
        try:
            # PostgreSQL requires double quotes for column names with special chars or mixed case
            # Wrap column name in double quotes
            quoted_column = f'"{column_name}"'
            query = text(f'SELECT DISTINCT {quoted_column} FROM {self.table_name} WHERE {quoted_column} IS NOT NULL LIMIT {limit}')
            
            # Log the query
            logger.info(f'[SQL DISTINCT] SELECT DISTINCT {quoted_column} FROM {self.table_name} WHERE {quoted_column} IS NOT NULL LIMIT {limit}')
            
            with self.engine.connect() as conn:
                result = conn.execute(query)
                values = [row[0] for row in result]
            
            logger.info(f"[SQL DISTINCT SUCCESS] Found {len(values)} distinct values for '{column_name}'")
            return values
            
        except Exception as e:
            logger.error(f"[SQL DISTINCT FAILED] Failed to get distinct values for {column_name}: {e}")
            return []
    
    def _validate_sql(self, sql_query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query before execution
        Returns: (is_valid, error_message)
        """
        try:
            # Check for empty/whitespace query
            if not sql_query or not sql_query.strip():
                return False, "Empty SQL query"
            
            # Check parentheses balance
            if sql_query.count('(') != sql_query.count(')'):
                return False, "Unbalanced parentheses"
            
            # Check for common column name mistakes (case-insensitive patterns)
            
            # Pattern 1: Wrong column names
            common_mistakes = {
                'giá_trúng_thầu': 'Giá_trúng_thầu',
                'gia_trung_thau': 'Giá_trúng_thầu',  # Missing dấu
                'lĩnh_vực_khách_hàng': 'Lĩnh_vực_Khách_hàng',
                'linh_vuc_khach_hang': 'Lĩnh_vực_Khách_hàng',
                'thời_gian_phê_duyệt_kqlcnt': 'Thời_gian_phê_duyệt_KQLCNT',
                'thoi_gian_phe_duyet_kqlcnt': 'Thời_gian_phê_duyệt_KQLCNT',
                'thời_gian_phê_duyệt': 'Thời_gian_phê_duyệt_KQLCNT',  # Thiếu _KQLCNT
                'thoi_gian_phe_duyet': 'Thời_gian_phê_duyệt_KQLCNT',
                'đơn_vị_kinh_doanh': 'Đơn_vị_kinh_doanh(VTS)',
                'don_vi_kinh_doanh': 'Đơn_vị_kinh_doanh(VTS)',
                'năm_phê_duyệt_kqlcnt': 'Năm_phê_duyệt_KQLCNT',
                'nam_phe_duyet_kqlcnt': 'Năm_phê_duyệt_KQLCNT'
            }
            
            for wrong, correct in common_mistakes.items():
                # Use word boundary to avoid false positives
                import re
                pattern = r'\b' + re.escape(wrong) + r'\b'
                if re.search(pattern, sql_query):
                    return False, f"Wrong column name: use '{correct}' instead of '{wrong}'"
            
            # Gợi ý thay thế bằng DATE_TRUNC
            if 'Thoi_gian_phe_duyet' not in sql_query and 'CURRENT_DATE' in sql_query:
                logger.warning("[SQL VALIDATION] Cảnh báo: có thể cần dùng Thoi_gian_phe_duyet cho điều kiện thời gian.")
            
            return True, None
            
        except Exception as e:
            logger.warning(f"SQL validation error: {e}")
            return True, None  # Don't block if validator fails
    
    def _execute_sql(self, sql_query: str) -> Tuple[List[Dict], Optional[str]]:
        """Execute SQL query with validation and automatic scalar handling"""
        is_valid, validation_error = self._validate_sql(sql_query)
        if not is_valid:
            error_msg = f"SQL Validation Error: {validation_error}"
            logger.error(error_msg)
            return [], error_msg

        logger.info(f"[SQL EXECUTING] {sql_query}")

        try:
            with self.engine.connect() as conn:
                # Clean escape characters if accidentally added
                sql_query = sql_query.replace('\\"', '"').replace("\\'", "'")

                # Detect if it's a single-value aggregate
                is_scalar_query = bool(
                    re.search(r'\b(SUM|AVG|COUNT|MAX|MIN)\b', sql_query, re.IGNORECASE)
                    and not re.search(r'\bGROUP\s+BY\b', sql_query, re.IGNORECASE)
                )

                if is_scalar_query:
                    scalar_val = conn.scalar(text(sql_query))
                    if scalar_val is None:
                        scalar_val = 0.0
                    try:
                        scalar_val = float(scalar_val)
                    except Exception:
                        scalar_val = float(str(scalar_val).replace(",", "")) if scalar_val else 0.0
                    logger.info(f"[SQL SCALAR] Result: {scalar_val:,.2f}")
                    return ([{"column": "value", "value": scalar_val, "formatted": f"{scalar_val:,.2f}"}], None)

                # Normal multi-row query
                result = conn.execute(text(sql_query))
                rows = []
                for row in result:
                    mapped = {}
                    for k, v in dict(row._mapping).items():
                        # Force convert Decimal / memoryview / bytearray / None to float
                        if isinstance(v, (Decimal, memoryview, bytearray)):
                            try:
                                mapped[k] = float(str(v))
                            except Exception:
                                mapped[k] = None
                        elif isinstance(v, (int, float)):
                            mapped[k] = float(v)
                        elif v is None:
                            mapped[k] = 0.0
                        else:
                            try:
                                mapped[k] = float(v) if str(v).replace('.', '', 1).isdigit() else v
                            except Exception:
                                mapped[k] = v
                    rows.append(mapped)

                # Handle single-row numeric fallback
                if len(rows) == 1 and len(rows[0]) == 1:
                    key, val = list(rows[0].items())[0]
                    try:
                        val = float(val or 0)
                    except Exception:
                        val = 0.0
                    logger.info(f"[SQL SINGLE NUMERIC] {key}: {val:,.2f}")
                    return ([{"column": key, "value": val, "formatted": f"{val:,.2f}"}], None)

                logger.info(f"[SQL SUCCESS] Returned {len(rows)} rows")
                return rows, None

        except Exception as e:
            err = f"SQL Execution Error: {e}"
            logger.error(f"[SQL FAILED] {err}")
            return [], err
        
    async def _execute_sql_async(self, sql: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._execute_sql(sql))
    
    async def run_queries_parallel(self, queries: list[dict]):
        tasks = []
        for q in queries:
            tasks.append(self._execute_sql_async(q["sql"]))
        results = await asyncio.gather(*tasks)
        merged = []
        for idx, (rows, error) in enumerate(results):
            merged.append({
                "id": queries[idx].get("id"),
                "description": queries[idx].get("description"),
                "error": error,
                "rows": rows
            })
        return merged

    
    def _create_react_prompt(self, question: str, react_history: List[Dict] = None) -> str:
        """Create ReAct prompt with schema metadata and guides"""
        
        # Schema information với metadata
        schema_str = json.dumps(self.schema_info, ensure_ascii=False, indent=2)
        
        # Sample rows
        samples = self._get_sample_rows(3)
        samples_str = json.dumps(samples, ensure_ascii=False, indent=2)

        # Dùng toàn bộ hướng dẫn chung
        general_guides = GENERAL_GUIDE_COMBINED

        # ReAct history
        history_str = ""
        if react_history:
            history_str = "\n\nLịch sử các bước đã thực hiện:\n"
            for i, step in enumerate(react_history, 1):
                history_str += f"Bước {i}:\n"
                history_str += f"  THOUGHT: {step.get('thought', '')}\n"
                history_str += f"  ACTION: {step.get('action', '')}\n"
                history_str += f"  OBSERVATION: {step.get('observation', '')}\n"
        
        prompt = f"""Database Schema (có metadata mô tả ý nghĩa từng cột):
{schema_str}

Sample Data (3 dòng mẫu):
{samples_str}

{general_guides}

Available Actions:
- get_distinct_values(column_name): Lấy các giá trị unique của một cột
- execute_query(sql): Thực thi SQL query
- final_answer(answer): Đưa ra câu trả lời cuối cùng

Question: {question}
{history_str}

Hãy sử dụng quy trình ReAct để trả lời câu hỏi. Với mỗi bước:
1. THOUGHT: Suy nghĩ về những gì cần làm dựa trên metadata và general guide
2. ACTION: Quyết định hành động tiếp theo (execute_query, final_answer)
3. OBSERVATION: Phân tích kết quả từ hành động

Hãy bắt đầu với THOUGHT đầu tiên:
"""
        
        return prompt
    
    def _parse_react_response(self, response_text: str) -> Tuple[str, str, str]:
        """
        Parse ReAct response - handles multiline ACTION
        Returns: (action_type, action_param, thought)
        """
        import re
        
        thought = ""
        action_type = ""
        action_param = ""
        
        # Extract THOUGHT
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|$)', response_text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Extract ACTION (may span multiple lines)
        action_match = re.search(r'ACTION:\s*(.+?)(?=OBSERVATION:|THOUGHT:|$)', response_text, re.DOTALL)
        if not action_match:
            return "", "", thought
        
        action_text = action_match.group(1).strip()
        
        # Parse action type and parameter
        if "get_distinct_values" in action_text:
            action_type = "get_distinct_values"
            # Extract column name
            match = re.search(r'get_distinct_values\s*\(\s*["\']([^"\']+)["\']\s*\)', action_text)
            if match:
                action_param = match.group(1).strip()
            else:
                # Without quotes
                match = re.search(r'get_distinct_values\s*\(\s*([^\)]+)\s*\)', action_text)
                if match:
                    action_param = match.group(1).strip()
            
            # Validate
            if not action_param or not action_param.strip():
                logger.warning(f"Empty column name parsed from: {action_text[:100]}")
                action_type = ""
                
        elif "execute_query" in action_text:
            action_type = "execute_query"
            
            # Log what we're trying to parse
            logger.info(f"[REACT PARSE] Attempting to parse execute_query from: {action_text[:200]}...")
            
            # Clean action_text - remove extra whitespace/newlines between function call
            cleaned = re.sub(r'execute_query\s*\(\s*', 'execute_query(', action_text)
            
            # Pattern 1: execute_query("SQL") or execute_query('SQL')
            match = re.search(r'execute_query\(["\'](.+?)["\']\)', cleaned, re.DOTALL)
            if match:
                action_param = match.group(1).strip()
                logger.info(f"[REACT PARSE] Pattern 1 matched, SQL length: {len(action_param)}")
            else:
                # Pattern 2: execute_query( "SQL" ) with spaces
                match = re.search(r'execute_query\(\s*["\'](.+?)["\']\s*\)', action_text, re.DOTALL)
                if match:
                    action_param = match.group(1).strip()
                    logger.info(f"[REACT PARSE] Pattern 2 matched, SQL length: {len(action_param)}")
                else:
                    # Pattern 3: Try greedy match - everything between ( and last )
                    match = re.search(r'execute_query\s*\((.+)\)', action_text, re.DOTALL)
                    if match:
                        sql = match.group(1).strip()
                        # Remove surrounding quotes if any
                        if (sql.startswith('"') and sql.endswith('"')) or (sql.startswith("'") and sql.endswith("'")):
                            sql = sql[1:-1]
                        action_param = sql.strip()
                        logger.info(f"[REACT PARSE] Pattern 3 (greedy) matched, SQL length: {len(action_param)}")
                    else:
                        # Pattern 4: Incomplete - missing closing paren
                        logger.error(f"[REACT PARSE] Failed to parse execute_query")
                        logger.error(f"[REACT PARSE] Action text: {action_text}")
            
            # Validate
            if not action_param or not action_param.strip():
                logger.warning(f"Empty SQL parsed from: {action_text[:200]}")
                action_type = ""
            else:
                logger.info(f"[REACT PARSE SUCCESS] Extracted SQL: {action_param[:100]}...")
                
        elif "final_answer" in action_text:
            action_type = "final_answer"
            # Extract answer
            match = re.search(r'final_answer\s*\(\s*["\'](.+?)["\']\s*\)', action_text, re.DOTALL)
            if match:
                action_param = match.group(1).strip()
            else:
                # Without quotes
                match = re.search(r'final_answer\s*\(\s*(.+?)\s*\)\s*$', action_text, re.DOTALL)
                if match:
                    answer = match.group(1).strip()
                    if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
                        answer = answer[1:-1]
                    action_param = answer.strip()
        
        return action_type, action_param, thought


    async def query_agentic(self, question: str):
        """
        Agentic V2:
        1. Planner draft kế hoạch
        2. SQL Agent chạy song song
        3. Summarizer viết báo cáo
        """

        # 1. PLANNER
        planner = PlannerAgent(model=self.model)
        plan = planner.plan(question, self.schema_info)

        if "queries" not in plan:
            return "❌ Planner lỗi: không thể lập kế hoạch.", plan

        queries = plan["queries"]

        # 2. SQL EXECUTOR (parallel)
        sql_results = await self.run_queries_parallel(queries)

        # 3. SUMMARIZER
        summarizer = SummarizerAgent(model=self.model)
        final_report = summarizer.summarize(
            question=question,
            sql_query=json.dumps(queries, ensure_ascii=False),
            sql_results=sql_results,
            scenario=plan.get("scenario")
        )

        return final_report, {
            "plan": plan,
            "sql_results": sql_results
        }

    def query(
        self,
        question: str,
        conversation_history: List[Dict[str, Any]] = None,
        model: str = None,
        max_react_steps: int = 20
    ) -> Generator[str, None, Dict[str, Any]]:
        """
        Query with ReAct mechanism - streaming response
        
        Args:
            question: User question
            conversation_history: Conversation history
            model: Model to use
            max_react_steps: Maximum ReAct steps
            
        Yields:
            Response chunks (text)
            
        Returns:
            Final metadata dict with usage info
        """
        start_time = time.time()
        # ============================================================
        # AUTO-DETECT: Nếu câu hỏi cần multi-query → chuyển sang agentic
        # ============================================================
        planner = PlannerAgent(model=self.model or self.DEFAULT_MODEL)
        scenario = planner.classify(question)
        
        # Nếu là scenario_1, scenario_2, hoặc scenario_3 → dùng agentic mode
        if scenario in ["scenario_1", "scenario_2", "scenario_3"]:
            logger.info(f"[AUTO-DETECT] Question matches {scenario} → switching to AGENTIC mode")
            
            # Stream thông báo cho user
            yield "🔍 Đang phân tích câu hỏi và lập kế hoạch thực thi...\n\n"
            import asyncio
            import concurrent.futures
            try:
                # Check if there's a running loop
                try:
                    loop = asyncio.get_running_loop()
                    # Loop is running - use thread pool
                    use_thread = True
                except RuntimeError:
                    # No running loop - we can create one
                    use_thread = False
                
                if use_thread:
                    # Run in a separate thread with its own event loop
                    def run_in_new_loop():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(self.query_agentic(question))
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_new_loop)
                        final_report, metadata = future.result()
                        # ✅ FIX 1: Stream report ra user
                        buffer = StreamBuffer(buffer_size=5)
                        for char in final_report:
                            buffered = buffer.add(char)
                            if buffered:
                                yield buffered
                        remaining = buffer.flush()
                        if remaining:
                            yield remaining

                        # ✅ FIX 2: RETURN để dừng execution (CRITICAL!)
                        return metadata  # ← THÊM DÒNG NÀY!
                else:
                    # No running loop, create and use one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        final_report, metadata = loop.run_until_complete(self.query_agentic(question))
                    finally:
                        loop.close()
                        
            except Exception as e:
                logger.error(f"Event loop error: {e}")
                yield "⚠️ Có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại sau. (Mã lỗi: 99)\n"
                return {"error": str(e), "error_code": "99"}
        
        # Nếu không phải multi-query scenario → tiếp tục ReAct mode
        logger.info(f"[AUTO-DETECT] Question is '{scenario}' → using REACT mode")

        react_history = []
        full_response = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
    
        try:
            for step in range(max_react_steps):
                logger.info(f"[REACT STEP {step+1}/{max_react_steps}] Starting...")
                
                # Create prompt
                prompt = self._create_react_prompt(question, react_history)
                
                # Call LLM
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                response_text = response.choices[0].message.content
                logger.debug(f"[REACT RAW STEP OUTPUT]\n{response_text}")
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens
                
                # Parse ReAct response
                action_type, action_param, thought = self._parse_react_response(response_text)
                
                logger.info(f"[REACT STEP {step+1}] Action: {action_type}, Thought: {thought[:100]}...")
                
                if action_type == "get_distinct_values":
                    # Get distinct values
                    logger.info(f"[REACT ACTION] get_distinct_values('{action_param}')")
                    values = self._get_distinct_values(action_param)
                    observation = f"Distinct values: {values[:20]}"  # Limit to 20
                    react_history.append({
                        "thought": thought,
                        "action": f"get_distinct_values({action_param})",
                        "observation": observation
                    })
                    
                elif action_type == "execute_query":
                    logger.info(f"[REACT ACTION] execute_query('{action_param[:100]}...')")
                    results, error = self._execute_sql(action_param)
                    if error:
                        observation = f"Error: {error}"
                        logger.warning(f"[REACT ACTION FAILED] SQL error: {error}")
                    else:
                        # Không cắt chỉ "First row", giữ toàn bộ dữ liệu
                        observation = f"Query returned {len(results)} rows with full dataset attached."
                        logger.info(f"[REACT ACTION SUCCESS] Returned {len(results)} rows (full data retained).")
                    
                    react_history.append({
                        "thought": thought,
                        "action": f"execute_query({action_param})",
                        "observation": observation,
                        "results": results if not error else []
                    })
                        
                elif action_type == "final_answer":
                    logger.info("[REACT FINAL] Detected final_answer trigger — skipping LLM-generated text.")
                    
                    # Luôn tìm kết quả query cuối cùng có dữ liệu
                    last_exec = next(
                        (s for s in reversed(react_history)
                        if s.get("action", "").startswith("execute_query") and s.get("results")),
                        None
                    )

                    summarizer = SummarizerAgent(model=self.model)
                    if last_exec:
                        full_results = last_exec.get("results", [])
                        logger.info(f"[REACT FINAL] Feeding {len(full_results)} rows to summarizer for final report.")
                        try:
                            full_response = summarizer.summarize(
                                question,
                                last_exec.get("action", ""),
                                full_results,
                                scenario=None   # ReAct không có Planner
                            )
                            logger.info("[REACT FINAL] SummarizerAgent successfully generated final report.")
                        except Exception as e:
                            logger.error(f"[REACT FINAL] SummarizerAgent failed: {e}")
                            full_response = "Không thể sinh báo cáo tổng hợp do lỗi Summarizer."
                    else:
                        logger.warning("[REACT FINAL] Không có dữ liệu execute_query — không thể tổng hợp báo cáo.")
                        full_response = "Không tìm thấy dữ liệu truy vấn để tổng hợp báo cáo."

                    # Stream ra user
                    buffer = StreamBuffer(buffer_size=5)
                    for char in full_response:
                        buffered = buffer.add(char)
                        if buffered:
                            yield buffered
                    remaining = buffer.flush()
                    if remaining:
                        yield remaining
                    break
            
            # Create response ID
            response_id = f"hst_{int(time.time())}_{hashlib.md5(question.encode()).hexdigest()[:8]}"
            
            # Calculate usage
            usage = TokenUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
                model=self.model
            )
            
            # Log
            log_openai_agent_response(
                response_id=response_id,
                source_name=self.source_name,
                vector_store_id="hst",
                user_query=question,
                assistant_response=full_response,
                model=self.model,
                usage=usage
            )
            
            return {
                "response_id": response_id,
                "usage": usage,
                "duration": time.time() - start_time,
                "source_name": self.source_name,
                "model": model,
                "react_steps": len(react_history)
            }
            
        except Exception as e:
            error_msg, error_code = ErrorHandler.get_user_friendly_message(e, self.source_name)
            logger.error(f"Query failed: {e}")
            
            yield f"\n\n⚠️ {error_msg}"
            
            log_openai_agent_error(
                source_name=self.source_name,
                vector_store_id="hst",
                model=self.model,
                user_query=question,
                error_message=str(e)
            )
            
            return {
                "error": error_msg,
                "error_code": error_code,
                "duration": time.time() - start_time,
                "source_name": self.source_name,
                "model": model
            }
    
    async def query_async(self, *args, **kwargs):
        """Async wrapper around sync query for API compatibility"""
        if kwargs.get("mode") == "agentic":
            async for out in self.query_agentic(*args, **kwargs):
                yield out
            return

        loop = asyncio.get_event_loop()
        def run_sync():
            results = []
            for chunk in self.query(*args, **kwargs):
                results.append(chunk)
            return results

        # Chạy sync query trong thread executor
        chunks = await loop.run_in_executor(None, run_sync)
        for c in chunks:
            yield c
    
    async def generate_title_from_message(self, message: str, model: str = None) -> Tuple[str, TokenUsage]:
        """Generate conversation title - async"""
        
        try:
            async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = await async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tạo tiêu đề ngắn (≤10 từ) cho cuộc hội thoại về hồ sơ thầu. Chỉ trả tiêu đề."},
                    {"role": "user", "content": f"Tiêu đề: {message[:200]}"}
                ],
                max_tokens=40,
                temperature=0.3
            )
            
            title = response.choices[0].message.content.strip().strip('"')
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model=self.model
            )
            
            return title, usage
            
        except Exception as e:
            logger.error(f"Title generation failed: {str(e)}")
            return f"Hội thoại {self.source_name.title()}", TokenUsage(model=self.model)
    
    async def generate_next_turn_suggestions(
        self, 
        conversation_history: List[Dict[str, Any]], 
        model: str = None
    ) -> Tuple[List[str], TokenUsage]:
        """Generate next turn suggestions - async"""
        
        try:
            async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            recent_history = conversation_history[-2:]
            formatted_history = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content'][:300]}"
                for msg in recent_history
            ])
            
            system_prompt = (
                "Gợi ý 3-5 câu hỏi tiếp theo về hồ sơ thầu.\n"
                "JSON array. Không giải thích.\n"
                "Nếu không phù hợp → []."
            )
            
            response = await async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_history}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            suggestions_text = response.choices[0].message.content.strip()
            try:
                suggestions = json.loads(suggestions_text)
                if not isinstance(suggestions, list):
                    suggestions = []
            except:
                suggestions = []
            
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model=self.model
            )
            
            return suggestions, usage
            
        except Exception as e:
            logger.error(f"Suggestions failed: {str(e)}")
            return [], TokenUsage(model=self.model)

class SummarizerAgent:
    """
    Agent tóm tắt kết quả SQL thành báo cáo tự nhiên.
    - Tập trung mô tả, so sánh dựa trên số liệu.
    - Không đưa ra nhận định chủ quan, dự đoán, khuyến nghị hay phần ký tên.
    - Trọng tâm là góc nhìn của Viettel Solutions (VTS), so với FPT và VNPT.
    """

    def __init__(self, model: str = "gpt-4.1"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def sanitize_json(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return obj

    def summarize(self, question: str, sql_query: str, sql_results: list[dict], scenario: str = None):
        """
        Sinh báo cáo tiếng Việt, khách quan, tập trung vào số liệu thực tế.
        """
        logger.info("=== DEBUG SUMMARIZER START ===")
        logger.info(f"SQL_RESULTS_RAW: {sql_results}")
        try:
            logger.info(f"SQL_RESULTS_KEYS: {[list(r.keys()) for r in sql_results]}")
        except Exception as e:
            logger.info(f"FAILED TO EXTRACT KEYS: {e}")

        if scenario:
            logger.info(f"[SUMMARIZER] Using forwarded scenario: {scenario}")
        else:
            scenario = self.detect_template(sql_results)

        if scenario == "scenario_1":
            template = open("src/agents/hst/templates/scenario_1.txt").read() 
        elif scenario == "scenario_2":
            template = open("src/agents/hst/templates/scenario_2.txt").read()
        elif scenario == "scenario_3":
            template = open("src/agents/hst/templates/scenario_3.txt").read()
        else:
            template = """
            Hãy tóm tắt ngắn gọn dựa trên dữ liệu.
            YÊU CẦU TRÌNH BÀY:
1. **Tổng quan thị trường**: mô tả quy mô, xu hướng chính (nếu có thể). LƯU Ý không nói về nhóm 'KHÁC'.
2. **Chi tiết từng bên**: trình bày kết quả theo bảng (số gói, giá trị, tỷ trọng).
3. **Kết luận**: Kết luận ngắn gọn (không suy diễn, hạn chế nhắc lại ý ở phần 1). LƯU Ý không nói về nhóm 'KHÁC'.
            """
            
        prompt = f"""
Bạn là chuyên gia phân tích dữ liệu đấu thầu của Viettel Solutions (VTS).
Hãy viết báo cáo tóm tắt dựa hoàn toàn trên dữ liệu được cung cấp — KHÔNG được suy diễn, dự đoán hoặc đưa ra nhận định chủ quan.

Câu hỏi người dùng: {question}

Hãy trả về báo cáo đúng format template sau:

{template}

SQL được thực thi:
{sql_query}

Kết quả dữ liệu SQL (JSON):
{json.dumps(sql_results, default=self.sanitize_json, ensure_ascii=False, indent=2)}

HƯỚNG DẪN BỔ SUNG:
- Ưu tiên trình bày kết quả ở dạng bảng, dễ đọc.
- Các số liệu đầu ra đều tính theo **tỷ Việt Nam Đồng (tỷ VND)**.
- Trọng tâm là hiệu quả và vị thế của Viettel Solutions (VTS), so với FPT và VNPT nếu có dữ liệu.
- Nhóm “Khác” chỉ cần nêu tổng giá trị và tỷ trọng, không đi sâu chi tiết.
- Tuyệt đối không thêm phần "Khuyến nghị", "Ghi chú", hoặc "Phòng Phân tích dữ liệu".

QUY TẮC ĐỊNH DẠNG SỐ:
- Dữ liệu đầu vào có dấu thập phân là "." (ví dụ: 100100.1).
- Khi hiển thị trong báo cáo, chuyển sang định dạng tiếng Việt:
  + Dấu phân cách phần thập phân là ",".
  + Dấu phân cách hàng nghìn, hàng triệu, tỷ là ".".
  Ví dụ: 100100.1 → 100.100,1
- Đảm bảo định dạng này áp dụng nhất quán cho tất cả số liệu trong báo cáo.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia phân tích dữ liệu đấu thầu của Viettel Solutions, chỉ mô tả và so sánh số liệu, không được đưa ra nhận định chủ quan hay khuyến nghị."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.15,
            max_tokens=900
        )
        return response.choices[0].message.content.strip()

    def detect_template(self, sql_results):
            """
            Nhận diện template dựa vào metadata của planner hoặc cấu trúc dữ liệu từ ReAct,
            KHÔNG bắt buộc phải có field 'id' trong từng row SQL.
            
            Logic:
            - Nếu có field 'id' → dùng logic cũ (Planner mode)
            - Nếu không có 'id' → phân tích cấu trúc dữ liệu (ReAct mode)
            """
            logger.info("=== DEBUG detect_template START ===")
            logger.info(f"INPUT sql_results: {sql_results}")

            if not sql_results:
                logger.error("detect_template: EMPTY sql_results → return 'other'")
                return "other"

            try:
                all_keys = [list(r.keys()) for r in sql_results]
                logger.info(f"detect_template: KEYS OF ROWS → {all_keys}")
            except Exception as e:
                logger.error(f"detect_template: FAILED TO LIST KEYS: {e}")
                return "other"

            # Kiểm tra xem có field 'id' không
            first_row_keys = list(sql_results[0].keys()) if sql_results else []
            has_id_field = "id" in first_row_keys
            
            logger.info(f"detect_template: has_id_field = {has_id_field}")

            # ============================================================
            # PLANNER MODE: Nếu có field 'id', dùng logic cũ
            # ============================================================
            if has_id_field:
                logger.info("detect_template: Using PLANNER MODE (id-based detection)")
                
                try:
                    if any(r.get("id") == "nsnn" for r in sql_results):
                        logger.info("MATCH scenario_1")
                        return "scenario_1"
                except Exception as e:
                    logger.error(f"detect_template ERROR at scenario_1: {e}")

                try:
                    if any(r.get("id") == "viettel_overview" for r in sql_results):
                        logger.info("MATCH scenario_2")
                        return "scenario_2"
                except Exception as e:
                    logger.error(f"detect_template ERROR at scenario_2: {e}")

                try:
                    if any(str(r.get("id", "")).startswith("obj_") for r in sql_results):
                        logger.info("MATCH scenario_3")
                        return "scenario_3"
                except Exception as e:
                    logger.error(f"detect_template ERROR at scenario_3: {e}")

                logger.info("detect_template: PLANNER MODE → RETURN 'other'")
                return "other"

            # ============================================================
            # REACT MODE: Phân tích dựa trên cấu trúc dữ liệu
            # ============================================================
            logger.info("detect_template: Using REACT MODE (structure-based detection)")
            
            # Lấy tất cả các keys từ kết quả SQL
            all_column_names = set()
            for row in sql_results:
                all_column_names.update(row.keys())
            
            logger.info(f"detect_template: All column names found: {all_column_names}")
            
            # Scenario 1: Market overview - có nhiều nhóm trúng thầu và phân khúc
            # Đặc điểm: có trường "Nhóm_trúng_thầu_shortlist" hoặc nhiều rows với các nhóm khác nhau
            scenario_1_indicators = {
                "Nhóm_trúng_thầu_shortlist",
                "market_total",
                "nsnn",
                "khdn"
            }
            
            # Scenario 2: Viettel detail analysis
            # Đặc điểm: có trường liên quan đến tháng, ĐVKD, hoặc so sánh với FPT/VNPT
            scenario_2_indicators = {
                "thang",
                "dvkd", 
                "Đơn_vị_kinh_doanh(VTS)",
                "by_month",
                "by_center"
            }
            
            # Scenario 3: Specific object analysis (province, sector, unit)
            # Đặc điểm: có trường tỉnh, lĩnh vực, hoặc phân tích theo đối tượng cụ thể
            scenario_3_indicators = {
                "Mã_tỉnh_mới",
                "Mã_tỉnh_cũ",
                "Lĩnh_vực_Khách_hàng",
                "obj_monthly"
            }
            
            # Kiểm tra overlap với các indicators
            s1_match = len(all_column_names & scenario_1_indicators)
            s2_match = len(all_column_names & scenario_2_indicators)
            s3_match = len(all_column_names & scenario_3_indicators)
            
            logger.info(f"detect_template: scenario_1 matches: {s1_match}")
            logger.info(f"detect_template: scenario_2 matches: {s2_match}")
            logger.info(f"detect_template: scenario_3 matches: {s3_match}")
            
            # Quyết định scenario dựa trên số lượng match
            if s1_match > 0 and s1_match >= s2_match and s1_match >= s3_match:
                logger.info("detect_template: REACT MODE → scenario_1 (market overview)")
                return "scenario_1"
            elif s2_match > 0 and s2_match > s1_match and s2_match >= s3_match:
                logger.info("detect_template: REACT MODE → scenario_2 (viettel detail)")
                return "scenario_2"
            elif s3_match > 0:
                logger.info("detect_template: REACT MODE → scenario_3 (specific object)")
                return "scenario_3"
            
            # Kiểm tra dựa trên số lượng rows và có GROUP BY
            # Nếu có nhiều rows với "Nhóm_trúng_thầu" → likely market share query
            if "Nhóm_trúng_thầu" in all_column_names and len(sql_results) > 2:
                logger.info("detect_template: REACT MODE → scenario_1 (detected market share pattern)")
                return "scenario_1"
            
            # Nếu có "thang" (month) → time series analysis, likely scenario 2
            if "thang" in all_column_names:
                logger.info("detect_template: REACT MODE → scenario_2 (detected time series)")
                return "scenario_2"
            
            logger.info("detect_template: REACT MODE → RETURN 'other'")
            return "other"


class PlannerAgent:
    """
    Planner Agent:
    - Nhận user query
    - Phân loại vào 4 scenario
    - Sinh danh sách SQL queries tương ứng
    """

    def __init__(self, model="gpt-4.1"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.allowed_scenarios = ["scenario_1", "scenario_2", "scenario_3", "other"]

    # =========================
    # 1. LLM-based classifier
    # =========================
    def _classify_llm(self, question: str) -> str:
        """
        Dùng LLM để phân loại intent.
        Chỉ được trả về 1 trong 4 chuỗi:
        - scenario_1
        - scenario_2
        - scenario_3
        - other
        """
        try:
            system_prompt = """
Bạn là hệ thống phân loại intent cho trợ lý hồ sơ thầu (HST). 
Nhiệm vụ: Dựa trên câu hỏi tiếng Việt của người dùng, phân loại vào đúng MỘT trong 4 nhóm sau:

===========================================================
🎯 **scenario_1 — Báo cáo THỊ PHẦN TỔNG QUAN / TOÀN THỊ TRƯỜNG**
===========================================================
Miêu tả:
- Người dùng hỏi về toàn thị trường nói chung, *không* tập trung vào Viettel.
- Thời gian có thể là tháng cụ thể, lũy kế, hoặc nhiều tháng.
- Có thể yêu cầu tổng hợp, cập nhật, xu hướng chung của thị trường.

Dấu hiệu:
- “thị phần thầu nói chung”, “tổng quan thị trường”, “báo cáo thị phần”, 
  “tổng hợp thị phần”, “tình hình thị phần”, “toàn thị trường”.
- KHÔNG nhắc tới Viettel hoặc DVKD/tỉnh cụ thể.

Ví dụ đúng scenario_1:
- “Báo cáo thị phần thầu lũy kế 10 tháng”
- “Tổng hợp thị phần thầu tháng 9/2025”
- “Báo cáo thị phần các tháng 6 7 8”
- “Cập nhật thị phần thầu 34 tỉnh”
- “Xu hướng thị phần toàn thị trường năm 2025”

Ví dụ KHÔNG phải scenario_1:
- “Báo cáo thị phần Viettel tháng 10” → scenario_2
- “Top ĐVKD của Viettel” → scenario_3
- “Thị phần tỉnh Hà Nội tháng 9” → scenario_3


===========================================================
🎯 **scenario_2 — Báo cáo THỊ PHẦN CHI TIẾT CHO VIETTEL**
===========================================================
Miêu tả:
- Người dùng hỏi về kết quả hoặc thị phần của **Viettel (VTS)**.
- Trọng tâm là Viettel so với đối thủ (FPT, VNPT, GAET,…).
- Câu hỏi chỉ nhắm vào Viettel, không nhắm vào một đơn vị/tỉnh/lĩnh vực cụ thể.

Dấu hiệu:
- “Viettel”, “VTS”, “Viettel Solutions”, “thị phần của Viettel”.
- Hỏi riêng về Viettel hoặc so sánh Viettel với đơn vị khác.

Ví dụ đúng scenario_2:
- “Báo cáo thị phần thầu tháng 10 của Viettel”
- “Hiệu suất của Viettel trong quý 2”
- “So sánh thị phần Viettel với FPT và VNPT”
- “Tổng giá trị trúng thầu của Viettel lũy kế 9 tháng”

Ví dụ KHÔNG phải scenario_2:
- “Báo cáo thị phần ĐVKD miền Nam của Viettel” → scenario_3
- “Thị phần tỉnh Hà Nội của Viettel” → scenario_3
- “Báo cáo thị phần toàn thị trường” → scenario_1


===========================================================
🎯 **scenario_3 — Báo cáo THEO ĐỐI TƯỢNG CỤ THỂ**
===========================================================
Miêu tả:
- Câu hỏi nhắm vào một **dimension cụ thể** như:
  ▸ Đơn vị kinh doanh (ĐVKD)  
  ▸ Tỉnh / Thành phố  
  ▸ Lĩnh vực khách hàng   
- Dù có hoặc không nhắc tới Viettel.

Dấu hiệu:
- “tỉnh”, “thành phố”, “Hà Nội”, “Đà Nẵng”
- “ĐVKD”, “trung tâm”, “TT CQĐT”, “KHDN”
- “lĩnh vực khách hàng”, “YTS”, “GDS”, “CQT”, “BQP”

Ví dụ đúng scenario_3:
- “Báo cáo thị phần thầu tháng 10 của Hà Nội”
- “Báo cáo thị phần nhóm ĐVKD TT CQĐT”
- “Thị phần lĩnh vực YTS năm 2025”
- “Báo cáo thị phần Đà Nẵng tháng 9”
- “Thị phần phân khúc CQT của Viettel”

Ví dụ KHÔNG phải scenario_3:
- “Báo cáo thị phần Viettel năm 2025” → scenario_2
- “Tổng quan thị phần lũy kế” → scenario_1


===========================================================
🎯 **other — Không thuộc 3 nhóm trên**
===========================================================
Miêu tả:
- Mọi câu hỏi không thuộc về 3 nhóm trên.

===========================================================
⚠️ QUY TẮC ƯU TIÊN PHÂN LOẠI (VERY IMPORTANT)
===========================================================
1. Nếu câu hỏi nhắc rõ Viettel → ưu tiên scenario_2  
   *Trừ khi nhắc rõ một đối tượng cụ thể (tỉnh/ĐVKD/lĩnh vực) → scenario_3.*

2. Nếu câu hỏi có tỉnh/ĐVKD/lĩnh vực → scenario_3  
   *Dù có hoặc không nhắc Viettel.*

3. Nếu không nhắc đối tượng cụ thể & không nhấn mạnh Viettel → scenario_1.

4. Luôn trả về duy nhất một chuỗi:  
   👉 “scenario_1”, “scenario_2”, “scenario_3” hoặc “other”.

5. Không giải thích thêm bất kỳ nội dung nào.


===========================================================
CHỈ TRẢ VỀ:
scenario_1
hoặc
scenario_2
hoặc
scenario_3
hoặc
other
===========================================================
"""

            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=10,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            )

            label = resp.choices[0].message.content.strip()
            # Chuẩn hóa
            label = label.split()[0]  # phòng trường hợp model lỡ nói thêm gì đó
            if label not in self.allowed_scenarios:
                return "other"
            return label

        except Exception as e:
            logger.error(f"[PLANNER] LLM classify failed: {e}")
            return "other"

    # =========================
    # 2. Public API
    # =========================
    def classify(self, question: str) -> str:
        scenario = self._classify_llm(question)
        return scenario


    # =========================
    # 3. Planner logic (giữ nguyên)
    # =========================
    def plan(self, question: str, schema):
        scenario = self.classify(question)

        if scenario == "scenario_1":
            return self._plan_scenario_1()

        if scenario == "scenario_2":
            return self._plan_scenario_2()

        if scenario == "scenario_3":
            return self._plan_scenario_3(question)

        return {"scenario": "other", "queries": []}
    
    def _plan_scenario_1(self):
        queries = [
            {
                "id": "market_total",
                "description": "Tổng số gói + tổng giá trị toàn thị trường",
                "sql": """
                    SELECT 
                        COUNT(*) AS so_goi,
                        SUM("Giá_trúng_thầu") AS tong_gia_tri
                    FROM thau_2025
                    WHERE "Giá_trúng_thầu" > 0
                    AND "Nhóm_trúng_thầu_shortlist" != 'Khác'
                """
            },
            {
                "id": "market_by_vendor",
                "description": "Tổng giá trị theo nhóm trúng thầu",
                "sql": """
                    SELECT 
                        "Nhóm_trúng_thầu_shortlist",
                        COUNT(*) AS so_goi,
                        SUM("Giá_trúng_thầu") AS gia_tri
                    FROM thau_2025
                    WHERE "Giá_trúng_thầu" > 0
                    AND "Nhóm_trúng_thầu_shortlist" != 'Khác'
                    GROUP BY "Nhóm_trúng_thầu_shortlist"
                    ORDER BY gia_tri DESC
                """
            },
            {
                "id": "nsnn",
                "description": "Giá trị theo nhóm NSNN",
                "sql": """
                    SELECT 
                        "Nhóm_trúng_thầu_shortlist",
                        COUNT(*) AS so_goi,
                        SUM("Giá_trúng_thầu") AS gia_tri
                    FROM thau_2025
                    WHERE "Giá_trúng_thầu" > 0
                        AND "Nhóm_trúng_thầu_shortlist" != 'Khác'
                        AND "Lĩnh_vực_Khách_hàng" IN ('TW','BQP','CQT','YTS','GDS')
                    GROUP BY "Nhóm_trúng_thầu_shortlist"
                    ORDER BY gia_tri DESC
                """
            },
            {
                "id": "khdn",
                "description": "Khối doanh nghiệp",
                "sql": """
                    SELECT 
                        "Nhóm_trúng_thầu_shortlist",
                        COUNT(*) AS so_goi,
                        SUM("Giá_trúng_thầu") AS gia_tri
                    FROM thau_2025
                    WHERE "Giá_trúng_thầu" > 0
                        AND "Nhóm_trúng_thầu_shortlist" != 'Khác'
                        AND "Lĩnh_vực_Khách_hàng" = 'KHDN'
                    GROUP BY "Nhóm_trúng_thầu_shortlist"
                    ORDER BY gia_tri DESC
                """
            },
            {
                "id": "kenh_truyen",
                "description": "Dịch vụ kênh truyền",
                "sql": """
                    SELECT 
                        "Nhóm_trúng_thầu_shortlist",
                        COUNT(*) AS so_goi,
                        SUM("Giá_trúng_thầu") AS gia_tri
                    FROM thau_2025
                    WHERE "Giá_trúng_thầu" > 0
                        AND "Nhóm_trúng_thầu_shortlist" != 'Khác'
                        AND "Phân_loại_sản_phẩm" = 'Kênh truyền'
                    GROUP BY "Nhóm_trúng_thầu_shortlist"
                    ORDER BY gia_tri DESC
                """
            }
        ]
        return {"scenario": "scenario_1", "queries": queries}
    
    def _plan_scenario_2(self):
        """
        Scenario 2: Báo cáo thị phần chi tiết của Viettel
        Yêu cầu đầy đủ các queries theo template scenario_2.txt
        """
        return {
            "scenario": "scenario_2",
            "queries": [
                # 1. TỔNG QUAN VIETTEL
                {
                    "id": "viettel_overview",
                    "description": "Tổng quan Viettel: số gói, tổng giá trị, thị phần, xếp hạng",
                    "sql": """
                        WITH market_total AS (
                            SELECT 
                                SUM("Giá_trúng_thầu") AS tong_thi_truong
                            FROM thau_2025
                            WHERE "Giá_trúng_thầu" > 0
                                AND "Nhóm_trúng_thầu_shortlist" != 'Khác'
                        ),
                        viettel_data AS (
                            SELECT 
                                COUNT(*) AS so_goi,
                                SUM("Giá_trúng_thầu") AS gia_tri
                            FROM thau_2025
                            WHERE "Nhóm_trúng_thầu_shortlist" = 'Viettel'
                                AND "Giá_trúng_thầu" > 0
                        ),
                        vendor_ranking AS (
                            SELECT 
                                "Nhóm_trúng_thầu_shortlist",
                                SUM("Giá_trúng_thầu") AS gia_tri,
                                RANK() OVER (ORDER BY SUM("Giá_trúng_thầu") DESC) AS rank
                            FROM thau_2025
                            WHERE "Giá_trúng_thầu" > 0
                                AND "Nhóm_trúng_thầu_shortlist" != 'Khác'
                            GROUP BY "Nhóm_trúng_thầu_shortlist"
                        )
                        SELECT 
                            vd.so_goi,
                            vd.gia_tri,
                            ROUND(CAST(vd.gia_tri * 100.0 / mt.tong_thi_truong AS NUMERIC), 1) AS market_share,
                            vr.rank
                        FROM viettel_data vd
                        CROSS JOIN market_total mt
                        LEFT JOIN vendor_ranking vr ON vr."Nhóm_trúng_thầu_shortlist" = 'Viettel'
                    """
                },
                
                # 2. LĨNH VỰC VIETTEL ĐỨNG SỐ 1
                {
                    "id": "fields_rank1",
                    "description": "Danh sách lĩnh vực Viettel đứng hạng 1",
                    "sql": """
                        WITH field_ranking AS (
                            SELECT 
                                "Lĩnh_vực_Khách_hàng",
                                "Nhóm_trúng_thầu_shortlist",
                                SUM("Giá_trúng_thầu") AS gia_tri,
                                RANK() OVER (
                                    PARTITION BY "Lĩnh_vực_Khách_hàng" 
                                    ORDER BY SUM("Giá_trúng_thầu") DESC
                                ) AS rank
                            FROM thau_2025
                            WHERE "Giá_trúng_thầu" > 0
                                AND "Nhóm_trúng_thầu_shortlist" != 'Khác'
                                AND "Lĩnh_vực_Khách_hàng" IS NOT NULL
                            GROUP BY "Lĩnh_vực_Khách_hàng", "Nhóm_trúng_thầu_shortlist"
                        )
                        SELECT 
                            "Lĩnh_vực_Khách_hàng" AS linh_vuc,
                            gia_tri
                        FROM field_ranking
                        WHERE "Nhóm_trúng_thầu_shortlist" = 'Viettel'
                            AND rank = 1
                        ORDER BY gia_tri DESC
                    """
                },
                
                # 3. LĨNH VỰC VIETTEL CHƯA ĐỨNG SỐ 1
                {
                    "id": "fields_not_rank1",
                    "description": "Danh sách lĩnh vực Viettel chưa đứng hạng 1",
                    "sql": """
                        WITH field_ranking AS (
                            SELECT 
                                "Lĩnh_vực_Khách_hàng",
                                "Nhóm_trúng_thầu_shortlist",
                                SUM("Giá_trúng_thầu") AS gia_tri,
                                RANK() OVER (
                                    PARTITION BY "Lĩnh_vực_Khách_hàng" 
                                    ORDER BY SUM("Giá_trúng_thầu") DESC
                                ) AS rank
                            FROM thau_2025
                            WHERE "Giá_trúng_thầu" > 0
                                AND "Nhóm_trúng_thầu_shortlist" != 'Khác'
                                AND "Lĩnh_vực_Khách_hàng" IS NOT NULL
                            GROUP BY "Lĩnh_vực_Khách_hàng", "Nhóm_trúng_thầu_shortlist"
                        ),
                        viettel_fields AS (
                            SELECT 
                                "Lĩnh_vực_Khách_hàng" AS linh_vuc,
                                gia_tri AS viettel_gia_tri,
                                rank AS viettel_rank
                            FROM field_ranking
                            WHERE "Nhóm_trúng_thầu_shortlist" = 'Viettel'
                        ),
                        top_vendor AS (
                            SELECT 
                                "Lĩnh_vực_Khách_hàng" AS linh_vuc,
                                "Nhóm_trúng_thầu_shortlist" AS top_vendor_name,
                                gia_tri AS top_gia_tri
                            FROM field_ranking
                            WHERE rank = 1
                        )
                        SELECT 
                            vf.linh_vuc,
                            vf.viettel_gia_tri,
                            vf.viettel_rank,
                            tv.top_vendor_name,
                            tv.top_gia_tri
                        FROM viettel_fields vf
                        LEFT JOIN top_vendor tv ON vf.linh_vuc = tv.linh_vuc
                        WHERE vf.viettel_rank > 1
                        ORDER BY vf.viettel_rank, vf.viettel_gia_tri DESC
                    """
                },
                
                # 4. TOP 3 TỈNH CÓ THỊ PHẦN CAO NHẤT
                {
                    "id": "top_provinces",
                    "description": "Top 3 tỉnh có thị phần Viettel cao nhất",
                    "sql": """
                        WITH province_total AS (
                            SELECT 
                                "Mã_tỉnh_mới",
                                SUM("Giá_trúng_thầu") AS tong_tinh
                            FROM thau_2025
                            WHERE "Giá_trúng_thầu" > 0
                                AND "Nhóm_trúng_thầu_shortlist" != 'Khác'
                                AND "Mã_tỉnh_mới" IS NOT NULL
                            GROUP BY "Mã_tỉnh_mới"
                        ),
                        viettel_by_province AS (
                            SELECT 
                                "Mã_tỉnh_mới",
                                SUM("Giá_trúng_thầu") AS viettel_gia_tri
                            FROM thau_2025
                            WHERE "Nhóm_trúng_thầu_shortlist" = 'Viettel'
                                AND "Giá_trúng_thầu" > 0
                                AND "Mã_tỉnh_mới" IS NOT NULL
                            GROUP BY "Mã_tỉnh_mới"
                        )
                        SELECT 
                            vp."Mã_tỉnh_mới" AS tinh,
                            ROUND(CAST(vp.viettel_gia_tri * 100.0 / pt.tong_tinh AS NUMERIC), 1) AS thi_phan
                        FROM viettel_by_province vp
                        LEFT JOIN province_total pt ON vp."Mã_tỉnh_mới" = pt."Mã_tỉnh_mới"
                        WHERE pt.tong_tinh > 0
                        ORDER BY thi_phan DESC
                        LIMIT 3
                    """
                },
                
                # 5. TOP 5 ĐVKD
                {
                    "id": "top_dvkd",
                    "description": "Top 5 ĐVKD theo giá trị trúng thầu",
                    "sql": """
                        SELECT 
                            "Đơn_vị_kinh_doanh(VTS)" AS dvkd,
                            COUNT(*) AS so_goi,
                            SUM("Giá_trúng_thầu") AS gia_tri
                        FROM thau_2025
                        WHERE "Nhóm_trúng_thầu_shortlist" = 'Viettel'
                            AND "Giá_trúng_thầu" > 0
                        GROUP BY dvkd
                        ORDER BY gia_tri DESC
                        LIMIT 5
                    """
                },
                
                # 6. GIÁ TRỊ THEO THÁNG
                {
                    "id": "by_month",
                    "description": "Giá trị Viettel theo từng tháng",
                    "sql": """
                        SELECT 
                            EXTRACT(MONTH FROM "Thoi_gian_phe_duyet") AS thang,
                            SUM("Giá_trúng_thầu") AS gia_tri,
                            COUNT(*) AS so_goi
                        FROM thau_2025
                        WHERE "Nhóm_trúng_thầu_shortlist" = 'Viettel'
                            AND "Giá_trúng_thầu" > 0
                        GROUP BY thang
                        ORDER BY thang
                    """
                },
                
                # 7. GIÁ TRỊ LŨY KẾ THEO THÁNG
                {
                    "id": "by_month_lk",
                    "description": "Giá trị lũy kế theo tháng",
                    "sql": """
                        WITH monthly_data AS (
                            SELECT 
                                EXTRACT(MONTH FROM "Thoi_gian_phe_duyet") AS thang,
                                SUM("Giá_trúng_thầu") AS gia_tri
                            FROM thau_2025
                            WHERE "Nhóm_trúng_thầu_shortlist" = 'Viettel'
                                AND "Giá_trúng_thầu" > 0
                            GROUP BY thang
                        )
                        SELECT 
                            thang,
                            SUM(gia_tri) OVER (ORDER BY thang) AS gia_tri_luy_ke
                        FROM monthly_data
                        ORDER BY thang
                    """
                }
            ]
        }

    def _extract_scenario3_object_via_llm(self, question: str):
        """
        Dùng LLM để trích xuất đối tượng cho scenario_3:
        - ĐVKD (Đơn vị kinh doanh Viettel)
        - Tỉnh / Thành phố
        - Lĩnh vực khách hàng

        Trả về:
            target_name: tên hiển thị cho user (Hà Nội, TT CQĐT, lĩnh vực YTS, ...)
            search_field: 'dvkd' | 'province' | 'field' | 'unknown'
            search_value: giá trị dùng để search (đã lower + escape quote)
        """
        import json
        import re

        system_prompt = """
Bạn đang hỗ trợ hệ thống phân tích hồ sơ thầu Viettel (HST Agent).

Nhiệm vụ: Từ câu hỏi tiếng Việt của người dùng, hãy xác định ĐỐI TƯỢNG chính
cho báo cáo scenario_3, thuộc một trong các nhóm:

1. Đơn vị kinh doanh Viettel (ĐVKD)
   - Lưu trong cột: "Đơn_vị_kinh_doanh(VTS)"
   - Ví dụ: "TT CQĐT", "TT DTTM", "Trung tâm miền Bắc", ...

2. Tỉnh / Thành phố
   - Lưu trong cột: "Mã_tỉnh_mới"
   - Giá trị là mã tỉnh, ví dụ:
     - Hà Nội  -> HNI
     - Hồ Chí Minh / TP HCM -> HCM
     - Đà Nẵng -> DNG
     - Hải Phòng -> HPG
     - Cần Thơ -> CTO
   - Nếu không chắc mã tỉnh, có thể dùng tên thường (không dấu hoặc có dấu),
     miễn là dễ dùng để search.

3. Lĩnh vực khách hàng
   - Lưu trong cột: "Lĩnh_vực_Khách_hàng"
   - Ví dụ: "YTS", "GDS", "CQT", "BQP", "KHDN", ...

Hãy TRẢ VỀ DUY NHẤT một JSON với schema:

{
  "target_name": "string",    // tên hiển thị cho người dùng: "Hà Nội", "TT CQĐT", "lĩnh vực YTS", ...
  "search_field": "dvkd" | "province" | "field" | "unknown",
  "search_value": "string"    // giá trị dùng để search (không cần thêm %)
}

QUY TẮC:
- Nếu câu hỏi nói rõ về tỉnh / thành phố:
  + Ví dụ: "Hà Nội", "TP HCM", "Đà Nẵng", ...
  => search_field = "province"
  => search_value = mã tỉnh (HNI, HCM, DNG, ...) nếu bạn biết,
     nếu không biết thì dùng tên thường (vd: "hà nội" hoặc "ha noi").

- Nếu câu hỏi nói về ĐVKD:
  + Ví dụ: "TT CQĐT", "trung tâm CQĐT", "ĐVKD miền Nam", ...
  => search_field = "dvkd"
  => search_value = tên/viết tắt ĐVKD (vd: "tt cqđt").

- Nếu câu hỏi nói về lĩnh vực khách hàng:
  + Ví dụ: "lĩnh vực YTS", "lĩnh vực CQT", "phân khúc KHDN", ...
  => search_field = "field"
  => search_value = giá trị dùng trong cột "Lĩnh_vực_Khách_hàng" (vd: "yts", "cqt", "khdn").

- Nếu không xác định rõ được loại đối tượng:
  => search_field = "unknown"
  => search_value = từ khóa quan trọng nhất liên quan đến đối tượng,
     ưu tiên từ/cụm từ ở cuối câu.

CHỈ TRẢ VỀ JSON, KHÔNG THÊM GIẢI THÍCH.
"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            )
            content = resp.choices[0].message.content.strip()

            # Cố gắng bóc riêng phần JSON nếu model lỡ nói thêm
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                json_str = m.group(0)
            else:
                json_str = content

            data = json.loads(json_str)

            target_name = (data.get("target_name") or "").strip()
            search_field = (data.get("search_field") or "").strip().lower()
            search_value = (data.get("search_value") or "").strip().lower()

        except Exception as e:
            logger.error(f"[PLANNER] LLM extract scenario_3 object failed: {e}")
            # Fallback đơn giản: lấy từ cuối cùng trong câu hỏi
            words = question.split()
            target_name = words[-1] if words else ""
            search_field = "unknown"
            search_value = target_name.lower()

        if not target_name:
            target_name = search_value or "đối tượng"

        if search_field not in {"dvkd", "province", "field", "unknown"}:
            search_field = "unknown"

        # Escape dấu nháy đơn cho an toàn SQL
        search_value = search_value.replace("'", "''")

        return target_name, search_field, search_value
    
    def _plan_scenario_3(self, question: str):
        """
        Scenario 3: Báo cáo theo đối tượng cụ thể (ĐVKD/Tỉnh/Lĩnh vực)

        Trả về 4 query:
        - viettel_overall: Tổng quan Viettel toàn thị trường
        - obj_overview: Tổng quan riêng cho đối tượng
        - obj_by_month: Giá trị theo tháng của đối tượng
        - obj_by_month_lk: Giá trị lũy kế theo tháng của đối tượng
        """

        # 1. Nhờ LLM trích xuất đối tượng
        target_name, search_field, raw_search_value = self._extract_scenario3_object_via_llm(question)

        logger.info(
            f"[PLANNER S3] target_name='{target_name}', "
            f"search_field='{search_field}', search_value='{raw_search_value}'"
        )

        # 2. Build WHERE clause dựa trên loại đối tượng
        if search_field == "dvkd":
            # Đơn vị kinh doanh Viettel
            where_clause = (
                f'LOWER("Đơn_vị_kinh_doanh(VTS)") ILIKE \'%{raw_search_value}%\''
            )
        elif search_field == "province":
            # Tỉnh / Thành phố (dùng Mã_tỉnh_mới, nhưng vẫn cho phép ILIKE để linh hoạt)
            where_clause = (
                f'LOWER("Mã_tỉnh_mới") ILIKE \'%{raw_search_value}%\''
            )
        elif search_field == "field":
            # Lĩnh vực khách hàng
            where_clause = (
                f'LOWER("Lĩnh_vực_Khách_hàng") ILIKE \'%{raw_search_value}%\''
            )
        else:
            # Fallback: tìm trong cả 3 field
            where_clause = f"""(
                LOWER("Đơn_vị_kinh_doanh(VTS)") ILIKE '%{raw_search_value}%'
                OR LOWER("Mã_tỉnh_mới") ILIKE '%{raw_search_value}%'
                OR LOWER("Lĩnh_vực_Khách_hàng") ILIKE '%{raw_search_value}%'
            )"""

        # 3. Trả về bộ queries giống logic cũ nhưng không dùng regex nữa
        return {
            "scenario": "scenario_3",
            "queries": [
                # Query 1: TỔNG QUAN VIETTEL TOÀN THỊ TRƯỜNG
                {
                    "id": "viettel_overall",
                    "description": "Tổng quan Viettel toàn thị trường",
                    "sql": """
                        WITH market_total AS (
                            SELECT 
                                SUM("Giá_trúng_thầu") AS tong_thi_truong
                            FROM thau_2025
                            WHERE "Giá_trúng_thầu" > 0
                                AND "Nhóm_trúng_thầu_shortlist" != 'Khác'
                        ),
                        viettel_data AS (
                            SELECT 
                                COUNT(*) AS so_goi,
                                SUM("Giá_trúng_thầu") AS gia_tri
                            FROM thau_2025
                            WHERE "Nhóm_trúng_thầu_shortlist" = 'Viettel'
                                AND "Giá_trúng_thầu" > 0
                        ),
                        vendor_ranking AS (
                            SELECT 
                                "Nhóm_trúng_thầu_shortlist",
                                SUM("Giá_trúng_thầu") AS gia_tri,
                                RANK() OVER (ORDER BY SUM("Giá_trúng_thầu") DESC) AS rank
                            FROM thau_2025
                            WHERE "Giá_trúng_thầu" > 0
                                AND "Nhóm_trúng_thầu_shortlist" != 'Khác'
                            GROUP BY "Nhóm_trúng_thầu_shortlist"
                        )
                        SELECT 
                            vd.so_goi,
                            vd.gia_tri,
                            ROUND(
                                CAST(vd.gia_tri * 100.0 / mt.tong_thi_truong AS NUMERIC),
                                1
                            ) AS share,
                            vr.rank
                        FROM viettel_data vd
                        CROSS JOIN market_total mt
                        LEFT JOIN vendor_ranking vr 
                            ON vr."Nhóm_trúng_thầu_shortlist" = 'Viettel'
                    """
                },

                # Query 2: TỔNG QUAN RIÊNG CHO ĐỐI TƯỢNG
                {
                    "id": "obj_overview",
                    "description": f"Tổng quan riêng cho {target_name}",
                    "sql": f"""
                        WITH obj_total AS (
                            SELECT 
                                SUM("Giá_trúng_thầu") AS tong_obj
                            FROM thau_2025
                            WHERE "Giá_trúng_thầu" > 0
                                AND "Nhóm_trúng_thầu_shortlist" != 'Khác'
                                AND {where_clause}
                        ),
                        viettel_obj AS (
                            SELECT 
                                COUNT(*) AS so_goi,
                                SUM("Giá_trúng_thầu") AS gia_tri
                            FROM thau_2025
                            WHERE "Nhóm_trúng_thầu_shortlist" = 'Viettel'
                                AND "Giá_trúng_thầu" > 0
                                AND {where_clause}
                        )
                        SELECT 
                            vo.so_goi,
                            vo.gia_tri,
                            CASE 
                                WHEN ot.tong_obj > 0 THEN 
                                    ROUND(
                                        CAST(vo.gia_tri * 100.0 / ot.tong_obj AS NUMERIC),
                                        1
                                    )
                                ELSE 0
                            END AS share
                        FROM viettel_obj vo
                        CROSS JOIN obj_total ot
                    """
                },

                # Query 3: GIÁ TRỊ THEO THÁNG CỦA ĐỐI TƯỢNG
                {
                    "id": "obj_by_month",
                    "description": f"Giá trị theo tháng của {target_name}",
                    "sql": f"""
                        SELECT 
                            EXTRACT(MONTH FROM "Thoi_gian_phe_duyet") AS thang,
                            SUM("Giá_trúng_thầu") AS gia_tri,
                            COUNT(*) AS so_goi
                        FROM thau_2025
                        WHERE "Nhóm_trúng_thầu_shortlist" = 'Viettel'
                            AND "Giá_trúng_thầu" > 0
                            AND {where_clause}
                        GROUP BY thang
                        ORDER BY thang
                    """
                },

                # Query 4: GIÁ TRỊ LŨY KẾ THEO THÁNG CỦA ĐỐI TƯỢNG
                {
                    "id": "obj_by_month_lk",
                    "description": f"Giá trị lũy kế theo tháng của {target_name}",
                    "sql": f"""
                        WITH monthly_data AS (
                            SELECT 
                                EXTRACT(MONTH FROM "Thoi_gian_phe_duyet") AS thang,
                                SUM("Giá_trúng_thầu") AS gia_tri
                            FROM thau_2025
                            WHERE "Nhóm_trúng_thầu_shortlist" = 'Viettel'
                                AND "Giá_trúng_thầu" > 0
                                AND {where_clause}
                            GROUP BY thang
                        )
                        SELECT 
                            thang,
                            SUM(gia_tri) OVER (ORDER BY thang) AS gia_tri_luy_ke
                        FROM monthly_data
                        ORDER BY thang
                    """
                },
            ]
        }
