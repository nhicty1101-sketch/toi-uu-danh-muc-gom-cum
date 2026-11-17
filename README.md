# Tối ưu hóa danh mục đầu tư bằng gom cụm và học máy (HOSE 2019–2025)

Tiểu luận cuối kỳ môn Khoa học dữ liệu và Phân tích kinh doanh, Trường Đại học Kinh tế – Luật (UEL). Kho này hiện chứa phần lấy và chuẩn bị dữ liệu. Phần phân tích (gom cụm, phân bổ tỷ trọng, kiểm tra ngoài mẫu) sẽ được bổ sung sau.

## Dữ liệu

- **Giá cổ phiếu và VN-Index:** CafeF, bộ tệp Upto cho MetaStock/AmiBroker (giá đã điều chỉnh, giá chưa điều chỉnh, chỉ số). Tệp giá chốt ngày 31/12/2025, tệp chỉ số chốt ngày 11/11/2026, tải ngày 11/11/2026
- **Rổ cổ phiếu:** 100 mã thuộc HOSE ngày 31/12/2019 có giá trị giao dịch bình quân năm 2019 cao nhất và giao dịch ít nhất 80% số phiên năm 2019. Các mã bị hủy niêm yết về sau vẫn được giữ để tránh thiên lệch sống sót.
- **Sàn niêm yết ngày 31/12/2019 của các mã phải tra tay:** `data/raw/kiem_tra_san_2019.csv` (có cột nguồn).
- **Sự kiện quyền CafeF bỏ sót, điều chỉnh bổ sung:** `data/raw/dieu_chinh_bo_sung.csv` (có cột nguồn).
- **Lãi suất phi rủi ro:** lợi suất thứ cấp trái phiếu Chính phủ Việt Nam kỳ hạn 10 năm theo ngày, Investing.com, tải ngày 13/11/2026. Bảng đối chiếu với lãi suất trúng thầu cuối quý: `phu_luc/doi_chieu_rf_trung_thau_vs_thu_cap.csv`.

Kho không kèm các tệp gốc của CafeF và Investing.com, cũng không kèm các tệp đã xử lý chứa nguyên chuỗi giá hoặc chuỗi lợi suất (`gia_dong_cua.csv`, `gia_ngay_dai.csv`, `rf_ngay.csv`). Làm theo mục dưới đây sẽ tạo lại đủ các tệp đó.

## Chạy lại

1. Cài Python 3.12, tạo môi trường ảo và cài thư viện:
   ```
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```
2. Tải ba tệp Upto của CafeF, giữ nguyên dạng .zip, đặt vào `data/raw/cafef_dieu_chinh`, `data/raw/cafef_chua_dieu_chinh` và `data/raw/cafef_index`.
3. Tải dữ liệu lịch sử lợi suất trái phiếu Chính phủ Việt Nam 10 năm từ Investing.com cho giai đoạn 01/01/2019–31/12/2025, lưu thành `data/raw/loi_suat_tpcp_10n_investing.csv`.
4. Chạy `.venv/bin/python 01_lay_du_lieu_cafef.py`. Báo cáo kiểm tra được ghi vào `data/processed/bao_cao_du_lieu.txt`.
5. (Tùy chọn) Nạp vào MySQL:
   - chạy `02_tao_bang_mysql.sql` trong MySQL Workbench;
   - tạo tệp `.env` ở thư mục gốc với `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB` (tệp này không được đưa lên GitHub);
   - đặt `NAP_MYSQL = True` trong `01_lay_du_lieu_cafef.py` rồi chạy lại để nạp bảng giá;
   - chạy `.venv/bin/python 03_nap_ro_mysql.py` để nạp bảng rổ.
6. Chạy `.venv/bin/python 04_kham_pha_du_lieu.py`; bảng và hình ghi vào ket_qua/eda/.
7. Chạy `.venv/bin/python 05_gom_cum.py`; kết quả ghi vào ket_qua/gom_cum/.
## Các tệp trong kho

| Tệp | Nội dung |
|---|---|
| `01_lay_du_lieu_cafef.py` | Đọc tệp CafeF, xác định sàn 2019, chọn rổ, kiểm tra điều chỉnh giá, tạo lãi suất phi rủi ro, xuất bảng |
| `02_tao_bang_mysql.sql` | Tạo cơ sở dữ liệu, bảng và các truy vấn kiểm tra |
| `03_nap_ro_mysql.py` | Nạp rổ 100 mã vào MySQL |
| `dien_san_2019.py` | Điền kết quả tra tay sàn niêm yết ngày 31/12/2019 |
| `nhat_ky_du_lieu.md` | Nhật ký dữ liệu: nguồn, ngày tải, kết quả kiểm tra, các thay đổi |
| `data/processed/` | Rổ 100 mã, danh sách ứng viên, kết quả kiểm tra điều chỉnh giá, độ phủ dữ liệu, lãi suất phi rủi ro bình quân năm và theo quý, báo cáo dữ liệu |
| `phu_luc/` | Bảng đối chiếu lãi suất trúng thầu và lợi suất thứ cấp |
| `04_kham_pha_du_lieu.py` | Khám phá dữ liệu: chất lượng dữ liệu, thống kê mô tả, diễn biến thị trường, đặc trưng gom cụm, tương quan giữa các mã |
| `ket_qua/eda/` | Bảng và hình của bước khám phá dữ liệu |
| `cong_cu.py` | Hàm dùng chung: lịch tái cân bằng, điều kiện chọn mã, đặc trưng, hai trục, K-Means |
| `05_gom_cum.py` | Gom cụm: chọn k, K-Means hai trục qua 20 kỳ, so sánh với K-Means bốn đặc trưng và gom cụm phân cấp theo tương quan |
| `ket_qua/gom_cum/` | Bảng và hình của bước gom cụm |
| `06_kiem_tra_ngoai_mau.py` | Kiểm tra ngoài mẫu S1–S5 và biến thể đối chứng S4đc, kiểm định chênh lệch Sharpe; 20 kỳ tái cân bằng, phí 0,2% mỗi chiều |
| `ket_qua/kiem_tra/` | Chỉ tiêu hiệu quả, lợi suất theo năm, tỷ trọng và vòng quay từng kỳ, kiểm định Sharpe, hình 11–14 | 
| `07_do_vung.py` | Kiểm định độ vững: rf = 0, phí 0%/0,4%, mất trắng 5 mã đình chỉ, cửa sổ 252/504/756, tái cân bằng 21/63/126 phiên, Ledoit–Wolf, K-Means bốn đặc trưng, Sharpe > 0, loại mã đứng giá; đóng góp lợi suất theo mã |
| `ket_qua/do_vung/` | Chỉ tiêu và kiểm định Sharpe theo từng cấu hình, đóng góp theo mã, hình 15–16 |