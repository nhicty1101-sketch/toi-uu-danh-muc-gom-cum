# -*- coding: utf-8 -*-
"""
Điền sàn niêm yết ngày 31/12/2019 cho các mã còn phải tra tay, kèm nguồn.

Cách dùng:
  1. Với mỗi dòng đang bị chú thích (#), mở hồ sơ mã đó trên Vietstock (finance.vietstock.vn, gõ mã vào
     ô tìm kiếm), xem sàn hiện tại và "Ngày giao dịch đầu tiên".
       - Sàn là HOSE và ngày giao dịch đầu tiên trước 31/12/2019  -> ghi "HOSE".
       - Ngày giao dịch đầu tiên trên HOSE sau 31/12/2019, hoặc mã đang ở HNX/UPCoM -> ghi sàn mã
         đang giao dịch ngày 31/12/2019 ("HNX" hoặc "UPCOM"); cần thì tìm thêm tin chuyển sàn.
  2. Sửa giá trị trong ngoặc cho đúng điều bạn thấy (ngày, sàn), rồi bỏ dấu # ở đầu dòng.
     Cột thứ hai là sàn tôi dự đoán, KHÔNG phải nguồn; chỉ bỏ # sau khi đã tự xem.
  3. Chạy:  python dien_san_2019.py   rồi chạy lại 01_lay_du_lieu_cafef.py.

Giữ tệp này trong dự án: nó là bằng chứng cho cột nguon_kiem_tra và đưa lên GitHub cùng phụ lục.
"""
import pandas as pd

TEP = "data/raw/kiem_tra_san_2019.csv"
NGAY_TRA = "12/11/2026"

DIEN = {
    # Đã tra (Claude, 12/11/2026) – bạn mở lại đường dẫn để tự xác nhận
    "AST": ("HOSE", "Vietstock hồ sơ AST: ngày giao dịch đầu tiên 04/01/2018, sàn HOSE"),
    "DBD": ("HOSE", "Vietstock hồ sơ DBD: ngày giao dịch đầu tiên 15/06/2018, sàn HOSE; trước đó UPCoM từ 16/01/2017 (VnBusiness)"),
    "VPG": ("HOSE", "Vietstock hồ sơ VPG: ngày giao dịch đầu tiên 18/01/2018, sàn HOSE"),
    "D2D": ("HOSE", "Niêm yết HOSE từ 08/2009 (báo cáo FireAnt 09/05/2025; Simplize)"),
    "YEG": ("HOSE", "Nhadautu.vn 06/12/2018: YEG lên HOSE từ 26/06/2018"),

    # Bạn tra rồi bỏ dấu #. Sàn trong ngoặc là dự đoán của tôi, chưa phải nguồn.
     "VNM": ("HOSE", "Vietstock hồ sơ VNM: ngày giao dịch đầu tiên 19/01/2006, sàn HOSE"),
     "VCB": ("HOSE", "Vietstock hồ sơ VCB: ngày giao dịch đầu tiên 30/06/2009, sàn HOSE"),
     "VIC": ("HOSE", "Vietstock hồ sơ VIC: ngày giao dịch đầu tiên 19/09/2007, sàn HOSE"),
     "VHM": ("HOSE", "Vietstock hồ sơ VHM: ngày giao dịch đầu tiên 17/05/2018, sàn HOSE"),
     "VJC": ("HOSE", "Vietstock hồ sơ VJC: ngày giao dịch đầu tiên 28/02/2017, sàn HOSE"),
     "MWG": ("HOSE", "Vietstock hồ sơ MWG: ngày giao dịch đầu tiên 14/07/2014, sàn HOSE"),
     "MSN": ("HOSE", "Vietstock hồ sơ MSN: ngày giao dịch đầu tiên 05/11/2009, sàn HOSE"),
     "PLX": ("HOSE", "Vietstock hồ sơ PLX: ngày giao dịch đầu tiên 21/04/2017, sàn HOSE"),
     "PNJ": ("HOSE", "Vietstock hồ sơ PNJ: ngày giao dịch đầu tiên 23/03/2009, sàn HOSE"),
     "GAS": ("HOSE", "Vietstock hồ sơ GAS: ngày giao dịch đầu tiên 21/05/2012, sàn HOSE"),
     "SAB": ("HOSE", "Vietstock hồ sơ SAB: ngày giao dịch đầu tiên 06/12/2016, sàn HOSE"),
     "NVL": ("HOSE", "Vietstock hồ sơ NVL: ngày giao dịch đầu tiên 28/12/2016, sàn HOSE"),
     "BVH": ("HOSE", "Vietstock hồ sơ BVH: ngày giao dịch đầu tiên 25/06/2009, sàn HOSE"),
     "VHC": ("HOSE", "Vietstock hồ sơ VHC: ngày giao dịch đầu tiên 24/12/2007, sàn HOSE"),
     "CTD": ("HOSE", "Vietstock hồ sơ CTD: ngày giao dịch đầu tiên 20/01/2010, sàn HOSE"),
     "PTB": ("HOSE", "Vietstock hồ sơ PTB: ngày giao dịch đầu tiên 22/07/2011, sàn HOSE"),
     "DHG": ("HOSE", "Vietstock hồ sơ DHG: ngày giao dịch đầu tiên 21/12/2006, sàn HOSE"),
     "VCS": ("HNX", "Vietstock hồ sơ VCS: sàn HNX, ngày giao dịch đầu tiên 17/12/2007, sàn HNX"),
     "VTP": ("UPCOM", "Vietstock/tin chuyển sàn: VTP giao dịch UPCoM từ 23/11/2018, lên HOSE ngày 12/03/2024"),
     "ACV": ("UPCOM", "Vietstock hồ sơ ACV: sàn UPCoM, ngày giao dịch đầu tiên 21/11/2016"),
}

k = pd.read_csv(TEP, dtype=str, sep=None, engine="python", encoding="utf-8-sig").fillna("")
for ma, (san, nguon) in DIEN.items():
    if "<...>" in nguon:
        raise SystemExit(f"{ma}: còn chỗ <...> trong nguồn, hãy điền ngày bạn tra được.")
    if ma not in set(k["ma"]):
        print(f"Bỏ qua {ma}: không có trong {TEP}")
        continue
    k.loc[k["ma"] == ma, ["san_ngay_31_12_2019", "nguon_kiem_tra"]] = [san, f"{nguon}; tra {NGAY_TRA}"]
k.to_csv(TEP, index=False, encoding="utf-8-sig")

con = k.loc[k["san_ngay_31_12_2019"].str.strip() == "", "ma"].tolist()
print(f"Đã điền {len(k) - len(con)}/{len(k)} mã.", f"Còn thiếu: {', '.join(con)}" if con else "Đủ cả.")
