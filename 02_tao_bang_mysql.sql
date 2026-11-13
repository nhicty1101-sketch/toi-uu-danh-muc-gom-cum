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

