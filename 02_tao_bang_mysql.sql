-- BƯỚC 3 – Tạo cơ sở dữ liệu và bảng giá cho tiểu luận
-- Chạy trong MySQL Workbench (File > Open SQL Script, rồi bấm biểu tượng tia sét).

CREATE DATABASE IF NOT EXISTS tieu_luan_khdl CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE tieu_luan_khdl;

-- Giá theo ngày của 100 mã trong rổ và VN-Index (giá tính bằng nghìn đồng, chỉ số tính bằng điểm)
CREATE TABLE IF NOT EXISTS gia_ngay (
    ma            VARCHAR(10)   NOT NULL,
    ngay          DATE          NOT NULL,
    mo            DECIMAL(12,3),
    cao           DECIMAL(12,3),
    thap          DECIMAL(12,3),
    dong_cua_dc   DECIMAL(12,3) COMMENT 'Giá đóng cửa đã điều chỉnh (CafeF)',
    dong_cua_goc  DECIMAL(12,3) COMMENT 'Giá đóng cửa chưa điều chỉnh (CafeF)',
    khoi_luong    BIGINT,
    PRIMARY KEY (ma, ngay)
);

-- Rổ 100 mã: nạp bằng Table Data Import Wizard từ tệp data/processed/ro_100_ma.csv
CREATE TABLE IF NOT EXISTS ro_co_phieu (
    ma                    VARCHAR(10) PRIMARY KEY,
    hang                  SMALLINT,
    gtgd_bq_2019_ty       DECIMAL(14,3) COMMENT 'Giá trị giao dịch bình quân phiên năm 2019, tỷ đồng',
    ty_le_phien_2019      DECIMAL(5,3),
    san_hien_tai          VARCHAR(10),
    ngay_cuoi_co_du_lieu  DATE
);

-- Truy vấn kiểm tra sau khi nạp
-- 1) Mỗi mã có bao nhiêu phiên, từ ngày nào đến ngày nào
SELECT ma, COUNT(*) AS so_phien, MIN(ngay) AS ngay_dau, MAX(ngay) AS ngay_cuoi
FROM gia_ngay
GROUP BY ma
ORDER BY so_phien;

-- 2) Phiên có giá đã điều chỉnh bằng 0 hoặc trống (không được có)
SELECT * FROM gia_ngay WHERE dong_cua_dc IS NULL OR dong_cua_dc <= 0;

-- 3) Mã có trong rổ nhưng chưa có giá
SELECT r.ma FROM ro_co_phieu r LEFT JOIN gia_ngay g ON r.ma = g.ma WHERE g.ma IS NULL;

-- 4) Ngày thứ Hai đến thứ Sáu không có phiên VN-Index, kèm số mã có giá trong ngày
--    Kết quả 16/11/2026 (ket_qua/eda/kiem_tra_lich_phien.csv): 77 ngày nghỉ lễ;
--    17/05/2024 có giá 96 mã nhưng thiếu VN-Index; 03/02/2025 CafeF mất cả phiên.
SET SESSION cte_max_recursion_depth = 3000;

WITH RECURSIVE lich AS (
    SELECT DATE('2019-01-02') AS ngay
    UNION ALL
    SELECT ngay + INTERVAL 1 DAY FROM lich WHERE ngay < '2025-12-31'
)
SELECT l.ngay, DAYNAME(l.ngay) AS thu,
       (SELECT COUNT(*) FROM tieu_luan_khdl.gia_ngay g WHERE g.ngay = l.ngay) AS so_ma_co_gia
FROM lich l
WHERE WEEKDAY(l.ngay) < 5
  AND l.ngay NOT IN (SELECT ngay FROM tieu_luan_khdl.gia_ngay WHERE ma = 'VNINDEX')
ORDER BY l.ngay;

