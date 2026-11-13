


# Nhật ký dữ liệu

## Nguồn và ngày tải
- Trang: https://cafef.vn/du-lieu/du-lieu-download.chn
- Bộ dùng chính: tệp chốt 31/12/2025, tải ngày 12/11/2026 từ
  https://cafefnew.mediacdn.vn/data/ami_data/20251231/
  (CafeF.SolieuGD.Upto31122025.zip, CafeF.SolieuGD.Raw.Upto31122025.zip, CafeF.Index.Upto31122025.zip)
- Bộ đối chiếu: tệp chốt 11/11/2026, để ở data/raw/cafef_ban_11112026
- Tệp chốt 31/12/2019: không còn trên máy chủ
- Tệp Index dùng bộ 11/11/2026, vì tệp Index chốt 31/12/2025 thiếu phiên 31/12/2025.
  Đã đối chiếu: giá chưa điều chỉnh của FPT, HPG, SSI, VCB, VNM ngày 30–31/12/2025 trùng khít giữa hai bộ.

## Vì sao dùng bộ chốt 31/12/2025
- Bộ 11/11/2026 chỉ giữ lịch sử ở sàn hiện tại hoặc sàn cuối cùng:
  ACB từ 14/12/2020, VIB từ 10/11/2020, SHB từ 11/10/2021.
- BCG chỉ còn 8 phiên trong bộ 11/11/2026 (bị hủy niêm yết năm 2026);
  trong bộ 31/12/2025 có 2.551 phiên (16/07/2015–08/10/2025). TCD tương tự.
- ROS, FLC: hai bộ như nhau; dữ liệu dừng ở 08–09/2022, không có phần giao dịch sau hủy niêm yết
  -> bắt buộc kiểm tra độ vững "vị thế mất trắng".
- 12/11/2026: TNA 20/08/2020 (giá gốc 17,3 -> 15,9): biên độ giá phiên GDKHQ khớp tỷ lệ chia cổ phiếu 6,5-16,4%, chọn 15% theo ĐTCK 11/08/2020 và thông báo VSD (ĐKCC 21/08/2020).
- IJC, SCR 05/07/2021: giá đóng cửa trong tệp CafeF (31,1; 11,4) cao hơn giá tham chiếu suy ra từ giá sàn phiên 06/07 (30,60; 11,25). Lỗi chỉ dời lợi suất giữa hai phiên liền nhau, lợi suất hai phiên cộng lại không đổi. Giữ nguyên, không sửa.
- 12/11/2026: CafeF có điều chỉnh cổ tức tiền mặt (IJC GDKHQ 14/07/2021: hệ số 0,94, khớp 1.500 đ/cp; 343/644 bậc điều chỉnh có hệ số >= 0,95). Chuỗi giá đã điều chỉnh là lợi suất tổng, VN-Index là chỉ số giá.
- TPB 13/05/2025 (tiền mặt 1.000 đ): CafeF bỏ sót, bổ sung qua dieu_chinh_bo_sung.csv.
- Giới hạn: quy tắc kiểm tra chỉ bắt được sự kiện bị bỏ sót khiến giá giảm quá 7,5% trong một phiên; sai sót nhỏ hơn không phát hiện được.
## Định dạng tệp
- Tệp con: CafeF.RAW_HSX/HNX/UPCOM (chưa điều chỉnh), CafeF.HSX/HNX/UPCOM (đã điều chỉnh).
- Cột: Ticker, DTYYYYMMDD, Open, High, Low, Close, Volume; giá nghìn đồng, 2 chữ số thập phân.

## Lãi suất phi rủi ro
- Định nghĩa: lợi suất thứ cấp TPCP kỳ hạn 10 năm theo từng phiên, nguồn Investing.com. rf ngày = (1 + y/100)^(1/252) − 1.
- Đường dẫn: https://vn.investing.com/rates-bonds/vietnam-10-year-bond-yield-historical-data   Ngày tải: 13/11/2026
- Xử lý (lan_chay_7, 13/11/2026): bỏ 1 dòng Chủ nhật 03/10/2021 (2,071%, nghi báo giá lỗi); bỏ 70 ngày Investing có số nhưng HOSE nghỉ; 0 phiên HOSE phải lấy số phiên trước.
- Thấp nhất 2,038% (15/12/2021), cao nhất 5,209% (03/11/2022). Bình quân năm 2019–2025: 4,37; 2,85; 2,20; 3,64; 3,24; 2,77; 3,49%.
- Độ vững: chạy lại với rf = 0.

## Thay đổi so với kế hoạch
- 12/11: chốt lãi suất trúng thầu TPCP 10 năm cuối quý.
- 13/11: đổi sang lợi suất thứ cấp 10 năm theo ngày. Lý do: có số cho từng phiên; phản ánh giá thị trường; lãi suất trúng thầu bị trần phát hành kéo thấp (quý 3/2022: 3,00% so với 5,02% trên thị trường thứ cấp) và còn thiếu 2 quý. Bảng trúng thầu 28 quý đưa vào phụ lục (phu_luc/doi_chieu_rf_trung_thau_vs_thu_cap.csv).

## MySQL (13/11/2026)
- CSDL tieu_luan_khdl, tài khoản riêng khdl; thông tin kết nối trong .env (không đưa lên GitHub).
- gia_ngay: 171.411 dòng, khớp bảng dài. ro_co_phieu: 100 dòng.
- Số mã theo năm 2019–2025: 100, 100, 100, 100, 97, 96, 94.
- Bảng rổ nạp bằng 03_nap_ro_mysql.py vì Import Wizard của Workbench trên Mac lỗi với tệp CSV (cả bản có và không có BOM).
- Cột san_hien_tai lấy theo tệp HSX của CafeF nên vẫn ghi HOSE với các mã đã hủy niêm yết (ví dụ ROS); dùng cột ngay_cuoi_co_du_lieu để biết mã dừng giao dịch.