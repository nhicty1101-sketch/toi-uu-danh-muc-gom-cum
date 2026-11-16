# -*- coding: utf-8 -*-
"""
BƯỚC 3 – LẤY VÀ CHUẨN BỊ DỮ LIỆU GIÁ
Tiểu luận: Tối ưu hóa danh mục đầu tư bằng gom cụm và học máy (HOSE, 2019–2025)

Nguồn dữ liệu: CafeF, trang "Dữ liệu cho phần mềm phân tích kỹ thuật MetaStock, AmiBroker"
    https://cafef.vn/du-lieu/du-lieu-download.chn

Tải tay 3 tệp nén của CÙNG MỘT NGÀY rồi đặt vào đúng thư mục (để nguyên .zip hoặc giải nén đều được).
Nên dùng tệp chốt ngày 31/12/2025 (cuối kỳ nghiên cứu): tệp của những ngày sau đó đã mất lịch sử
của một số mã bị hủy niêm yết năm 2026 (ví dụ BCG). Tệp cũ tải theo mẫu đường dẫn
    https://cafefnew.mediacdn.vn/data/ami_data/20251231/CafeF.SolieuGD.Upto31122025.zip
Mỗi thư mục chỉ để tệp của một ngày chốt:
    data/raw/cafef_dieu_chinh/        <- cột "Upto 3 sàn", dòng "Đã điều chỉnh"
    data/raw/cafef_chua_dieu_chinh/   <- cột "Upto 3 sàn", dòng "Chưa điều chỉnh"
    data/raw/cafef_index/             <- cột "Upto", dòng "Index"
Tùy chọn, nếu tải được: tệp "Upto 3 sàn" chốt ngày 31/12/2019 đặt vào data/raw/cafef_san_2019/
để script tự lấy sàn niêm yết cuối 2019, bớt phần kiểm tra tay.

Script chạy từ thư mục gốc của dự án (nơi có thư mục data/). Mỗi ô "# %%" chạy riêng được
trong PyCharm hoặc VS Code; trên Colab thì chạy cả tệp bằng lệnh %run.

Kết quả chính (data/processed/):
    ung_vien_ro_2019.csv   danh sách ứng viên xếp theo giá trị giao dịch năm 2019
    ro_100_ma.csv          rổ 100 mã cuối cùng
    gia_dong_cua.csv       giá đóng cửa đã điều chỉnh, dòng = phiên HOSE, cột = mã + VNINDEX
    gia_ngay_dai.csv       bảng dài để nạp MySQL
    kiem_tra_dieu_chinh.csv, do_phu_du_lieu.csv, bao_cao_du_lieu.txt
"""

# %% 0. Cấu hình
import io
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(".")                      # thư mục gốc dự án
RAW = GOC / "data" / "raw"
OUT = GOC / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

TU_NGAY, DEN_NGAY = "2019-01-01", "2025-12-31"
NAM_CHON_RO = 2019
SO_MA_TRONG_RO = 100
SO_UNG_VIEN = 140                    # lấy dư để bù các mã không thuộc HOSE ngày 31/12/2019
TY_LE_PHIEN_TOI_THIEU = 0.80         # mã phải có giao dịch ở ít nhất 80% số phiên HOSE năm 2019
NGUONG_BIEN_DO = 0.075               # biên độ HOSE ±7%, cộng phần dư do làm tròn bước giá
SO_PHIEN_XET_BUOC_GIA = 20           # cần ít nhất 20 phiên giá dưới 50.000 đồng mới xét bước giá
SO_PHIEN_CUOI_NAM = 20               # số phiên cuối năm 2019 dùng làm bằng chứng đang ở HOSE
SO_GIA_LE_BANG_CHUNG = 3             # cần ít nhất 3 mức giá lẻ bước 10/50 đồng trong các phiên đó
THU_MUC_SAN_2019 = RAW / "cafef_san_2019"   # tùy chọn: tệp Upto của CafeF chốt cuối năm 2019
NAP_MYSQL = False                  # đổi thành True khi đã tạo bảng bằng 02_tao_bang_mysql.sql

BAO_CAO = []                         # các dòng ghi vào bao_cao_du_lieu.txt


def ghi(dong=""):
    print(dong)
    BAO_CAO.append(str(dong))


# %% 1. Hàm đọc tệp CafeF
TEN_COT = {
    "ticker": "ma", "symbol": "ma", "ma": "ma", "mack": "ma", "mã": "ma",
    "dtyyyymmdd": "ngay", "date": "ngay", "ngay": "ngay", "ngày": "ngay",
    "open": "mo", "high": "cao", "low": "thap", "close": "dong",
    "volume": "kl", "vol": "kl",
}
COT_MAC_DINH = ["ma", "ngay", "mo", "cao", "thap", "dong", "kl"]


def _chuan_hoa_ten_cot(ten):
    ten = str(ten).strip().lower().replace("<", "").replace(">", "").replace(" ", "")
    return TEN_COT.get(ten, ten)


def _san_tu_ten_tep(ten_tep):
    t = ten_tep.upper()
    for khoa, san in (("INDEX", "INDEX"), ("HSX", "HOSE"), ("HOSE", "HOSE"),
                      ("HNX", "HNX"), ("UPCOM", "UPCOM")):
        if khoa in t:
            return san
    return "KHONG_RO"


def _doc_bytes(du_lieu, ten_tep):
    """Đọc nội dung một tệp CSV/TXT của CafeF, tự nhận dấu phân cách và dòng tiêu đề."""
    dong_dau = du_lieu.split(b"\n", 1)[0].decode("utf-8-sig", errors="ignore").strip()
    sep = ";" if dong_dau.count(";") > dong_dau.count(",") else ","
    o_dau = [o.strip().strip("<>") for o in dong_dau.split(sep)]
    co_tieu_de = not (len(o_dau) > 1 and o_dau[1].isdigit() and len(o_dau[1]) == 8)

    if co_tieu_de:
        df = pd.read_csv(io.BytesIO(du_lieu), sep=sep, encoding="utf-8-sig")
        df.columns = [_chuan_hoa_ten_cot(c) for c in df.columns]
    else:
        df = pd.read_csv(io.BytesIO(du_lieu), sep=sep, header=None)
        df.columns = COT_MAC_DINH + [f"phu_{i}" for i in range(df.shape[1] - len(COT_MAC_DINH))]

    thieu = [c for c in COT_MAC_DINH if c not in df.columns]
    if thieu:
        raise ValueError(f"{ten_tep}: thiếu cột {thieu}. Các cột đọc được: {list(df.columns)}. "
                         "Sửa từ điển TEN_COT cho khớp tên cột thực tế.")
    df = df[COT_MAC_DINH].copy()
    df["ma"] = df["ma"].astype(str).str.strip().str.upper()
    chuoi_ngay = df["ngay"].astype(str).str.strip().str[:8]
    ngay = pd.to_datetime(chuoi_ngay, format="%Y%m%d", errors="coerce")
    if ngay.isna().mean() > 0.5:      # phòng trường hợp ngày dạng dd/mm/yyyy
        ngay = pd.to_datetime(df["ngay"].astype(str), dayfirst=True, errors="coerce")
    df["ngay"] = ngay
    for c in ("mo", "cao", "thap", "dong", "kl"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["san"] = _san_tu_ten_tep(ten_tep)
    return df.dropna(subset=["ngay", "dong"])


def doc_thu_muc(thu_muc, la_chi_so=False):
    """Đọc mọi tệp .csv/.txt (kể cả nằm trong .zip) trong một thư mục, ghép thành bảng dài."""
    thu_muc = Path(thu_muc)
    if not thu_muc.exists():
        raise SystemExit(f"Chưa có thư mục {thu_muc}. Tạo thư mục và đặt tệp CafeF vào đó.")
    bang, ngay_chot = [], set()
    for tep in sorted(thu_muc.rglob("*")):
        if tep.suffix.lower() == ".zip":
            with zipfile.ZipFile(tep) as zf:
                for ten in zf.namelist():
                    if ten.lower().endswith((".csv", ".txt")):
                        bang.append(_doc_bytes(zf.read(ten), Path(ten).name))
        elif tep.suffix.lower() in (".csv", ".txt"):
            bang.append(_doc_bytes(tep.read_bytes(), tep.name))
        else:
            continue
        m = re.search(r"Upto(\d{2})\.?(\d{2})\.?(\d{4})", tep.name)
        if m:
            ngay_chot.add(f"{m.group(1)}/{m.group(2)}/{m.group(3)}")
    if not bang:
        raise SystemExit(f"Không tìm thấy tệp .zip/.csv/.txt nào trong {thu_muc}.")
    if len(ngay_chot) > 1:
        # giá điều chỉnh của hai ngày chốt khác nhau không ghép được với nhau
        raise SystemExit(f"Thư mục {thu_muc} có tệp của nhiều ngày chốt {sorted(ngay_chot)}. "
                         "Mỗi thư mục chỉ để tệp của một ngày; chuyển tệp còn lại ra chỗ khác.")
    ghi(f"  {thu_muc.name}: tệp chốt ngày {', '.join(ngay_chot) or 'không rõ (tên tệp không có Upto)'}")
    df = pd.concat(bang, ignore_index=True)
    if la_chi_so:
        df["san"] = "INDEX"
    else:
        if (df["san"] == "KHONG_RO").all():
            raise SystemExit(f"Không nhận ra sàn từ tên tệp trong {thu_muc}. Tên tệp CafeF thường có "
                             "chữ HSX, HNX hoặc UPCOM; nếu đã đổi tên thì đặt lại cho có các chữ này.")
        df = df[df["ma"].str.fullmatch(r"[A-Z0-9]{3}")]          # chỉ giữ cổ phiếu (mã 3 ký tự)
        if df["dong"].median() > 500:                             # tệp tính bằng đồng
            df[["mo", "cao", "thap", "dong"]] = df[["mo", "cao", "thap", "dong"]] / 1000
    truoc = len(df)
    df = df.sort_values(["ma", "ngay"]).drop_duplicates(["ma", "ngay"], keep="last")
    if truoc - len(df):
        ghi(f"  {thu_muc.name}: bỏ {truoc - len(df)} dòng trùng (mã, ngày)")
    df = df.reset_index(drop=True)
    df.attrs["ngay_chot"] = ", ".join(sorted(ngay_chot))
    return df


def xem_nhanh(thu_muc, so_dong=3):
    """In vài dòng đầu của từng tệp để kiểm tra định dạng trước khi chạy cả quy trình."""
    for tep in sorted(Path(thu_muc).rglob("*")):
        if tep.suffix.lower() == ".zip":
            with zipfile.ZipFile(tep) as zf:
                for ten in zf.namelist():
                    print(f"--- {tep.name} / {ten}")
                    print(zf.read(ten)[:300].decode("utf-8", errors="ignore"))
        elif tep.suffix.lower() in (".csv", ".txt"):
            print(f"--- {tep.name}")
            print("\n".join(tep.read_text(encoding="utf-8", errors="ignore").splitlines()[:so_dong]))


# %% 2. Đọc ba bộ dữ liệu
ghi(f"Ngày chạy: {datetime.now():%d/%m/%Y %H:%M}")
dc = doc_thu_muc(RAW / "cafef_dieu_chinh")
goc = doc_thu_muc(RAW / "cafef_chua_dieu_chinh")
chi_so = doc_thu_muc(RAW / "cafef_index", la_chi_so=True)
if dc.attrs["ngay_chot"] != goc.attrs["ngay_chot"]:
    raise SystemExit(f"Tệp đã điều chỉnh (chốt {dc.attrs['ngay_chot']}) và chưa điều chỉnh "
                     f"(chốt {goc.attrs['ngay_chot']}) phải cùng một ngày chốt.")
if chi_so.attrs["ngay_chot"] != dc.attrs["ngay_chot"]:
    ghi(f"Lưu ý: tệp Index chốt {chi_so.attrs['ngay_chot']}, khác ngày chốt của tệp giá "
        f"({dc.attrs['ngay_chot']}). Chấp nhận được vì chỉ số không bị điều chỉnh, nhưng nên dùng cùng ngày.")

for ten, bang in (("Đã điều chỉnh", dc), ("Chưa điều chỉnh", goc)):
    dem = bang.groupby("san")["ma"].nunique().to_dict()
    ghi(f"{ten}: {bang['ma'].nunique()} mã {dem}, từ {bang['ngay'].min():%d/%m/%Y} "
        f"đến {bang['ngay'].max():%d/%m/%Y}")

ma_vni = chi_so["ma"].str.replace("-", "", regex=False).str.upper()
vni = chi_so[ma_vni == "VNINDEX"].set_index("ngay")["dong"].sort_index()
if vni.empty:
    raise SystemExit(f"Không thấy VNINDEX trong tệp Index. Các mã có trong tệp: "
                     f"{sorted(chi_so['ma'].unique())[:20]}")
lich = vni.loc[TU_NGAY:DEN_NGAY].index          # lịch phiên giao dịch HOSE
ghi(f"VN-Index: {len(lich)} phiên từ {lich.min():%d/%m/%Y} đến {lich.max():%d/%m/%Y}")


# %% 3. Xếp hạng ứng viên theo giá trị giao dịch năm 2019 (dùng giá CHƯA điều chỉnh)
so_phien_2019 = int((lich.year == NAM_CHON_RO).sum())
g = goc[goc["ngay"].dt.year == NAM_CHON_RO].copy()
g["gtgd_ty"] = g["dong"] * g["kl"] / 1e6        # nghìn đồng × cổ phiếu / 1e6 = tỷ đồng
g["thay_doi"] = g.groupby("ma")["dong"].pct_change()

# Bước giá năm 2019: HOSE dùng bước 10 đồng (giá dưới 10.000) và 50 đồng (10.000–49.950),
# HNX và UPCoM chỉ có bước 100 đồng. Mã nào cả năm luôn tròn 100 đồng thì gần như chắc không ở HOSE.
gia_vnd = (g[["mo", "cao", "thap", "dong"]] * 1000).round()
co_gia = gia_vnd.notna() & (gia_vnd > 0)
le_100 = co_gia & (gia_vnd % 100 != 0)
duoi_50 = g["dong"] < 50
g["phien_duoi_50"] = duoi_50.astype(int)
g["so_gia_xet"] = co_gia.sum(axis=1).where(duoi_50, 0)
g["so_gia_le"] = le_100.sum(axis=1).where(duoi_50, 0)
# Bằng chứng trực tiếp: mức giá lẻ bước 10/50 đồng trong các phiên cuối năm 2019 chỉ có thể khớp trên HOSE
phien_cuoi_nam = lich[lich.year == NAM_CHON_RO][-SO_PHIEN_CUOI_NAM:]
g["so_gia_le_cuoi_nam"] = le_100.sum(axis=1).where(g["ngay"].isin(phien_cuoi_nam) & duoi_50, 0)

tk = g.groupby("ma").agg(san_hien_tai=("san", "last"),
                         tong_gtgd=("gtgd_ty", "sum"),
                         so_phien=("ngay", "nunique"),
                         tang_max=("thay_doi", "max"),
                         phien_duoi_50=("phien_duoi_50", "sum"),
                         so_gia_xet=("so_gia_xet", "sum"),
                         so_gia_le=("so_gia_le", "sum"),
                         so_gia_le_cuoi_nam=("so_gia_le_cuoi_nam", "sum"))
tk["gtgd_bq_2019_ty"] = tk["tong_gtgd"] / so_phien_2019
tk["ty_le_phien_2019"] = tk["so_phien"] / so_phien_2019
tk["ty_le_gia_le_100"] = tk["so_gia_le"] / tk["so_gia_xet"].replace(0, np.nan)
tk["ngay_cuoi_co_du_lieu"] = goc.groupby("ma")["ngay"].max()

uv = (tk[tk["ty_le_phien_2019"] >= TY_LE_PHIEN_TOI_THIEU]
      .sort_values("gtgd_bq_2019_ty", ascending=False)
      .head(SO_UNG_VIEN).copy())
uv.insert(0, "hang", range(1, len(uv) + 1))

uv["khac_san"] = uv["san_hien_tai"] != "HOSE"
uv["vuot_bien_do_2019"] = uv["tang_max"] > NGUONG_BIEN_DO
uv["ngung_truoc_cuoi_ky"] = uv["ngay_cuoi_co_du_lieu"] < pd.Timestamp(DEN_NGAY) - pd.Timedelta(days=30)
# Xét theo thứ tự: quá ít phiên giá dưới 50.000 → không xét được; nhiều mức giá lệch bước 100 đồng
# → khớp với HOSE; đủ phiên mà gần như không có mức lệch → dấu hiệu HNX/UPCoM; còn lại → không rõ.
uv["buoc_gia"] = np.select(
    [uv["phien_duoi_50"] < 5,
     uv["ty_le_gia_le_100"] >= 0.25,
     uv["phien_duoi_50"] < SO_PHIEN_XET_BUOC_GIA,
     uv["ty_le_gia_le_100"] < 0.02],
    ["khong_xet_duoc", "giong_hose", "khong_xet_duoc", "chi_tron_100"], default="khong_ro")
# Tự xác định được sàn khi có bằng chứng đủ mạnh từ bước giá:
#   HOSE       – có ít nhất 3 mức giá lẻ bước 10/50 đồng trong 20 phiên cuối năm 2019
#   NGOAI_HOSE – ít nhất 20 phiên giá dưới 50.000 mà gần như không có mức giá lẻ nào cả năm 2019
uv["bang_chung"] = np.select(
    [uv["so_gia_le_cuoi_nam"] >= SO_GIA_LE_BANG_CHUNG,
     (uv["phien_duoi_50"] >= SO_PHIEN_XET_BUOC_GIA) & (uv["ty_le_gia_le_100"] < 0.005)],
    ["HOSE", "NGOAI_HOSE"], default="")
co_dau_hieu = uv["khac_san"] | uv["vuot_bien_do_2019"] | (uv["buoc_gia"] != "giong_hose")
uv["can_kiem_tra"] = co_dau_hieu & (uv["bang_chung"] == "")


def _ly_do(d):
    ly_do = []
    if d["khac_san"]:
        ly_do.append(f"hiện ở {d['san_hien_tai']}: có thể đã bị hủy niêm yết hoặc chuyển khỏi HOSE")
    if d["vuot_bien_do_2019"]:
        ly_do.append(f"có phiên tăng {d['tang_max']:.1%} năm 2019, vượt biên độ HOSE")
    if d["buoc_gia"] == "chi_tron_100":
        ly_do.append("giá năm 2019 luôn tròn 100 đồng, trong khi HOSE có bước 10/50 đồng cho giá dưới "
                     "50.000: dấu hiệu khi đó ở HNX hoặc UPCoM")
    elif d["buoc_gia"] == "khong_ro":
        ly_do.append(f"chỉ {d['ty_le_gia_le_100']:.0%} mức giá năm 2019 lệch bước 100 đồng: không rõ sàn")
    elif d["buoc_gia"] == "khong_xet_duoc":
        ly_do.append("giá năm 2019 hầu hết từ 50.000 đồng trở lên, bước giá không phân biệt được sàn")
    return "; ".join(ly_do)


uv["ly_do"] = uv.apply(_ly_do, axis=1)
uv.reset_index().to_csv(OUT / "ung_vien_ro_2019.csv", index=False, encoding="utf-8-sig")
ghi(f"Năm {NAM_CHON_RO} có {so_phien_2019} phiên HOSE; {len(uv)} ứng viên; {int(co_dau_hieu.sum())} mã có dấu hiệu "
    f"cần xét sàn, trong đó tự xác định được từ bước giá: "
    f"{uv.loc[co_dau_hieu, 'bang_chung'].replace('', 'chưa').value_counts().to_dict()}.")
ghi(f"  Còn {int(uv['can_kiem_tra'].sum())} mã phải kiểm tra tay sàn niêm yết.")
ghi(f"  Phân loại theo bước giá năm 2019: {uv['buoc_gia'].value_counts().to_dict()}")
if (uv.loc[~uv["khac_san"], "buoc_gia"] == "chi_tron_100").mean() > 0.5:
    ghi("CẢNH BÁO: quá nửa số mã đang ở HOSE có giá 2019 luôn tròn 100 đồng. Có thể tệp đã làm tròn giá; "
        "khi đó dấu hiệu bước giá không dùng được, hãy kiểm tra tay hoặc dùng tệp chốt cuối 2019.")


# %% 4. Sàn niêm yết ngày 31/12/2019: tệp chốt cuối 2019 (nếu có), rồi kiểm tra tay các mã còn lại
san_tu_tep = pd.Series(dtype=object)
if THU_MUC_SAN_2019.exists() and any(THU_MUC_SAN_2019.iterdir()):
    snap = doc_thu_muc(THU_MUC_SAN_2019)
    ngay_chot = snap["ngay"].max()
    if pd.Timestamp("2019-12-24") <= ngay_chot <= pd.Timestamp("2020-01-10"):
        cuoi = snap[snap["ngay"] >= ngay_chot - pd.Timedelta(days=10)]
        san_tu_tep = cuoi.groupby("ma")["san"].last()
        ghi(f"Tệp chốt cuối 2019 có dữ liệu đến {ngay_chot:%d/%m/%Y}: lấy sàn của {len(san_tu_tep)} mã "
            f"{san_tu_tep.value_counts().to_dict()}.")
    else:
        ghi(f"CẢNH BÁO: tệp trong {THU_MUC_SAN_2019.name} có dữ liệu đến {ngay_chot:%d/%m/%Y}, "
            "không phải cuối 2019; bỏ qua tệp này.")
    del snap
co_trong_tep = uv.index.isin(san_tu_tep.index)

tep_kiem_tra = RAW / "kiem_tra_san_2019.csv"
can = uv[uv["can_kiem_tra"] & ~co_trong_tep].reset_index()[["ma", "san_hien_tai", "ly_do"]]
if tep_kiem_tra.exists():
    # sep=None: đọc được cả tệp do Excel lưu bằng dấu chấm phẩy
    da_kt = pd.read_csv(tep_kiem_tra, dtype=str, sep=None, engine="python", encoding="utf-8-sig").fillna("")
    moi = can[~can["ma"].isin(da_kt["ma"])]
    da_kt = pd.concat([da_kt, moi.assign(san_ngay_31_12_2019="", nguon_kiem_tra="")], ignore_index=True)
else:
    da_kt = can.assign(san_ngay_31_12_2019="", nguon_kiem_tra="")
# chỉ giữ mã còn phải tra và các dòng đã điền từ trước
da_kt = da_kt[da_kt["ma"].isin(can["ma"]) | (da_kt["san_ngay_31_12_2019"].str.strip() != "")]
da_kt.to_csv(tep_kiem_tra, index=False, encoding="utf-8-sig")

chua_dien = da_kt[da_kt["ma"].isin(can["ma"]) & (da_kt["san_ngay_31_12_2019"].str.strip() == "")]
if len(chua_dien):
    raise SystemExit(
        f"Còn {len(chua_dien)} mã chưa điền sàn niêm yết ngày 31/12/2019: {', '.join(chua_dien['ma'])}.\n"
        f"Mở {tep_kiem_tra}, điền cột san_ngay_31_12_2019 (HOSE, HNX hoặc UPCOM) và cột nguon_kiem_tra, "
        "rồi chạy lại từ ô 4.")

# Thứ tự ưu tiên: kiểm tra tay > tệp chốt cuối 2019 > bằng chứng bước giá > sàn hiện tại
san_2019 = uv["san_hien_tai"].copy()
nguon = pd.Series("sàn hiện tại, không có dấu hiệu khác", index=uv.index)
bc_hose = uv["bang_chung"] == "HOSE"
bc_ngoai = uv["bang_chung"] == "NGOAI_HOSE"
san_2019[bc_hose] = "HOSE"
nguon[bc_hose] = f"có giá lẻ bước 10/50 đồng trong {SO_PHIEN_CUOI_NAM} phiên cuối 2019"
san_2019[bc_ngoai] = "NGOAI_HOSE"
nguon[bc_ngoai] = "giá 2019 luôn tròn 100 đồng"
san_2019.update(san_tu_tep[san_tu_tep.index.isin(san_2019.index)])
nguon[co_trong_tep] = "tệp CafeF chốt cuối 2019"
sua = da_kt.set_index("ma")["san_ngay_31_12_2019"].str.strip().str.upper()
sua = sua[sua.index.isin(san_2019.index) & (sua != "")]
san_2019.update(sua)
nguon[sua.index] = "kiểm tra tay"
uv["san_2019"] = san_2019
uv["nguon_san_2019"] = nguon
uv.reset_index().to_csv(OUT / "ung_vien_ro_2019.csv", index=False, encoding="utf-8-sig")
ghi(f"Nguồn xác định sàn 31/12/2019 của {len(uv)} ứng viên: {uv['nguon_san_2019'].value_counts().to_dict()}")

ro = uv[uv["san_2019"] == "HOSE"].head(SO_MA_TRONG_RO).copy()
if len(ro) < SO_MA_TRONG_RO:
    ghi(f"CẢNH BÁO: chỉ được {len(ro)} mã. Tăng SO_UNG_VIEN rồi chạy lại từ ô 3.")
ro["hang"] = range(1, len(ro) + 1)
cot_ro = ["hang", "gtgd_bq_2019_ty", "ty_le_phien_2019", "san_hien_tai", "ngay_cuoi_co_du_lieu"]
ro[cot_ro].reset_index().to_csv(OUT / "ro_100_ma.csv", index=False, encoding="utf-8-sig")

ngung = ro[ro["ngung_truoc_cuoi_ky"]]
ghi(f"Rổ cuối cùng: {len(ro)} mã. Loại vì không ở HOSE cuối 2019: "
    f"{', '.join(uv.index[uv['san_2019'] != 'HOSE']) or 'không có'}.")
ghi("Mã trong rổ ngừng có dữ liệu trước cuối kỳ (giữ lại, không loại): " +
    (", ".join(f"{m} ({d:%d/%m/%Y})" for m, d in ngung["ngay_cuoi_co_du_lieu"].items()) or "không có"))


# %% 5. Bảng giá đã điều chỉnh theo lịch phiên HOSE
# 5a. Bổ sung điều chỉnh cho sự kiện quyền mà tệp CafeF bỏ sót (phát hiện ở ô 6, tra nguồn rồi điền tay).
# Tệp data/raw/dieu_chinh_bo_sung.csv, mỗi dòng một sự kiện:
#   ma, ngay_gdkhq (dd/mm/yyyy), ty_le_co_phieu (% cổ tức/thưởng bằng cổ phiếu, để trống nếu không có),
#   tien_mat_dong (đồng/cổ phiếu, để trống nếu không có), nguon
# Hệ số = (P - D) / (1 + r) / P, với P là giá đóng cửa chưa điều chỉnh phiên liền trước ngày GDKHQ.
# Mọi giá đã điều chỉnh của mã TRƯỚC ngày GDKHQ được nhân với hệ số này.
# Nếu CafeF đã điều chỉnh ngày đó: hệ số của CafeF khớp hệ số tính từ nguồn (lệch không quá 0,5%) thì bỏ qua;
# lệch hơn 0,5% thì thay hệ số của CafeF bằng hệ số tính từ nguồn (ví dụ MWG 24/07/2025: CafeF dùng 0,8989
# cho cổ tức tiền mặt 1.000 đồng, hệ số đúng là 0,9858).
tep_bo_sung = RAW / "dieu_chinh_bo_sung.csv"
if tep_bo_sung.exists():
    bs = pd.read_csv(tep_bo_sung, dtype=str, sep=None, engine="python", encoding="utf-8-sig").fillna("")
    so = lambda x: float(str(x).replace(",", ".")) if str(x).strip() else 0.0
    for _, r in bs.iterrows():
        ma = r["ma"].strip().upper()
        ngay_gd = pd.to_datetime(r["ngay_gdkhq"].strip(), dayfirst=True)
        g_ma = (dc.loc[dc["ma"] == ma, ["ngay", "dong"]]
                .merge(goc.loc[goc["ma"] == ma, ["ngay", "dong"]], on="ngay", suffixes=("_dc", "_goc"))
                .sort_values("ngay"))
        truoc_gd, tu_gd = g_ma[g_ma["ngay"] < ngay_gd], g_ma[g_ma["ngay"] >= ngay_gd]
        if truoc_gd.empty or tu_gd.empty:
            raise SystemExit(f"dieu_chinh_bo_sung.csv: {ma} không có giá quanh ngày {ngay_gd:%d/%m/%Y}")
        # Tỷ số giá đã điều chỉnh/giá gốc nhảy bậc ở ngày này nghĩa là CafeF đã điều chỉnh, với hệ số he_so_cafef.
        ts_truoc = truoc_gd["dong_dc"].iloc[-1] / truoc_gd["dong_goc"].iloc[-1]
        ts_sau = tu_gd["dong_dc"].iloc[0] / tu_gd["dong_goc"].iloc[0]
        he_so_cafef = ts_truoc / ts_sau
        P = truoc_gd["dong_goc"].iloc[-1]
        he_so = (P - so(r["tien_mat_dong"]) / 1000) / (1 + so(r["ty_le_co_phieu"]) / 100) / P
        truoc = (dc["ma"] == ma) & (dc["ngay"] < ngay_gd)
        if abs(he_so_cafef - 1) > 0.005:
            if abs(he_so_cafef / he_so - 1) <= 0.005:
                ghi(f"Bỏ qua bổ sung {ma} {ngay_gd:%d/%m/%Y}: CafeF đã điều chỉnh đúng ngày này "
                    f"(hệ số CafeF {he_so_cafef:.4f}, theo nguồn {he_so:.4f})")
                continue
            dc.loc[truoc, ["mo", "cao", "thap", "dong"]] *= he_so / he_so_cafef
            ghi(f"Thay hệ số CafeF {ma} {ngay_gd:%d/%m/%Y}: {he_so_cafef:.4f} -> {he_so:.4f} cho {int(truoc.sum())} "
                f"phiên trước đó (nguồn: {r['nguon']})")
            continue
        dc.loc[truoc, ["mo", "cao", "thap", "dong"]] *= he_so
        ghi(f"Bổ sung điều chỉnh {ma} {ngay_gd:%d/%m/%Y}: hệ số {he_so:.4f} cho {int(truoc.sum())} phiên trước đó "
            f"(nguồn: {r['nguon']})")

d = dc[dc["ma"].isin(ro.index) & dc["ngay"].between(TU_NGAY, DEN_NGAY)]
gia = d.pivot(index="ngay", columns="ma", values="dong")
ngoai_lich = gia.index.difference(lich)
if len(ngoai_lich):
    ghi(f"Bỏ {len(ngoai_lich)} ngày có giá nhưng không có phiên VN-Index (ví dụ {ngoai_lich[0]:%d/%m/%Y})")
gia = gia.reindex(index=lich, columns=ro.index)
gia["VNINDEX"] = vni.reindex(lich)
gia.index.name = "ngay"
gia.to_csv(OUT / "gia_dong_cua.csv")

khong_co = [m for m in ro.index if gia[m].isna().all()]
if khong_co:
    ghi(f"CẢNH BÁO: không có giá đã điều chỉnh cho {khong_co}")


# %% 6. Kiểm tra: giá đã được điều chỉnh theo sự kiện quyền chưa
cot = ["ma", "ngay", "dong"]
ss = (d[cot].merge(goc[cot], on=["ma", "ngay"], suffixes=("_dc", "_goc"))
      .sort_values(["ma", "ngay"]))
ss["td_goc"] = ss.groupby("ma")["dong_goc"].pct_change()
ss["td_dc"] = ss.groupby("ma")["dong_dc"].pct_change()
# Số phiên HOSE giữa hai lần có giá liên tiếp: >1 nghĩa là mã nghỉ giao dịch ở giữa (tạm ngừng,
# chỉ được giao dịch phiên thứ Sáu, ...), khi đó mức giảm gộp nhiều phiên vượt 7% là chuyện bình thường.
vi_tri = pd.Series(np.arange(len(lich)), index=lich)
ss["cach_phien"] = ss["ngay"].map(vi_tri).groupby(ss["ma"]).diff()
su_kien = ss[ss["td_goc"] < -NGUONG_BIEN_DO].copy()     # giá gốc giảm quá biên độ sàn: thường là ngày GDKHQ
giam_dc = su_kien["td_dc"] < -NGUONG_BIEN_DO
su_kien["nghi_van"] = giam_dc & (su_kien["cach_phien"] == 1)
su_kien["giam_qua_nhieu_phien"] = giam_dc & (su_kien["cach_phien"] != 1)
su_kien.to_csv(OUT / "kiem_tra_dieu_chinh.csv", index=False, encoding="utf-8-sig")
# Lịch các lần CafeF (và tệp bổ sung) đã điều chỉnh: ngày tỷ số giá điều chỉnh/giá gốc nhảy bậc.
# Bỏ các bậc nhỏ hơn sai số làm tròn 2 chữ số thập phân của giá điều chỉnh.
ss["ty_so"] = ss["dong_dc"] / ss["dong_goc"]
ss["he_so_buoc"] = ss.groupby("ma")["ty_so"].shift() / ss["ty_so"]
sai_so = 0.005 / ss["dong_dc"] + 0.005 / ss.groupby("ma")["dong_dc"].shift()
lich_dc = ss[((ss["he_so_buoc"] - 1).abs() > np.maximum(0.005, 2 * sai_so))][["ma", "ngay", "dong_goc", "he_so_buoc"]]
lich_dc.to_csv(OUT / "lich_dieu_chinh_cafef.csv", index=False, encoding="utf-8-sig")
ghi(f"Số lần giá đã điều chỉnh nhảy bậc so với giá gốc: {len(lich_dc)} ở {lich_dc['ma'].nunique()} mã; "
    f"hệ số từ 0,95 trở lên (cỡ cổ tức tiền mặt): {int((lich_dc['he_so_buoc'] >= 0.95).sum())}")
ghi(f"Ngày giá gốc giảm quá {NGUONG_BIEN_DO:.1%} (thường là ngày GDKHQ): {len(su_kien)}; trong đó giá điều chỉnh "
    f"vẫn giảm mạnh: {int(giam_dc.sum())}, gồm {int(su_kien['giam_qua_nhieu_phien'].sum())} lần sau nhiều phiên "
    f"không giao dịch (bình thường) và {int(su_kien['nghi_van'].sum())} lần giữa hai phiên liền nhau (cần xem lại)")
if su_kien["nghi_van"].any():
    ghi("  Mã có ngày cần xem lại: " + ", ".join(
        f"{m} ({n})" for m, n in su_kien[su_kien["nghi_van"]].groupby("ma").size()
        .sort_values(ascending=False).items()))


# %% 7. Độ phủ dữ liệu từng mã
bang_ma = gia.drop(columns="VNINDEX")
phu = pd.DataFrame({
    "ngay_dau": bang_ma.apply(pd.Series.first_valid_index),
    "ngay_cuoi": bang_ma.apply(pd.Series.last_valid_index),
    "ty_le_thieu_ca_ky": bang_ma.isna().mean(),
})
phu["ty_le_thieu_khi_niem_yet"] = [
    bang_ma.loc[r.ngay_dau:r.ngay_cuoi, m].isna().mean() if pd.notna(r.ngay_dau) else 1.0
    for m, r in phu.iterrows()
]
phu.sort_values("ty_le_thieu_ca_ky", ascending=False).to_csv(OUT / "do_phu_du_lieu.csv", encoding="utf-8-sig")
ghi(f"Tỷ lệ ô thiếu trung bình: {phu['ty_le_thieu_ca_ky'].mean():.2%}; "
    f"mã thiếu nhiều nhất: {phu['ty_le_thieu_ca_ky'].idxmax()} ({phu['ty_le_thieu_ca_ky'].max():.1%})")


# %% 8. Lãi suất phi rủi ro
# Nguồn chính: lợi suất thứ cấp TPCP 10 năm theo ngày tải từ Investing.com, lưu thành
# data/raw/loi_suat_tpcp_10n_investing.csv (giữ nguyên tệp tải về, không đưa lên GitHub).
# Mỗi phiên HOSE dùng lợi suất của chính phiên đó; phiên nào Investing không có số thì
# lấy số gần nhất trước đó. rf ngày = (1 + y/100)^(1/252) - 1.
# Không có tệp Investing thì script dùng tệp điền tay data/raw/lai_suat_phi_rui_ro.csv:
#   - theo quý: nam, quy, ngay_cuoi_quy, loi_suat_phan_tram, loai_lai_suat, nguon
#     -> lấy bình quân giản đơn 4 quý của mỗi năm;
#   - theo năm: nam, rf_nam_phan_tram, loai_lai_suat, nguon.
tep_inv = RAW / "loi_suat_tpcp_10n_investing.csv"
tep_rf = RAW / "lai_suat_phi_rui_ro.csv"
if tep_inv.exists():
    ls = pd.read_csv(tep_inv, dtype=str, encoding="utf-8-sig")
    if {"Ngày", "Lần cuối"} <= set(ls.columns):
        ls["ngay"] = pd.to_datetime(ls["Ngày"].str.strip(), format="%d/%m/%Y")
        ls["y"] = ls["Lần cuối"]
    elif {"Date", "Price"} <= set(ls.columns):                 # bản tải từ trang tiếng Anh
        ls["ngay"] = pd.to_datetime(ls["Date"].str.strip(), format="%m/%d/%Y")
        ls["y"] = ls["Price"]
    else:
        raise SystemExit(f"{tep_inv} cần cột Ngày và Lần cuối (hoặc Date và Price); đang có {list(ls.columns)}")
    ls["y"] = pd.to_numeric(ls["y"].str.strip().str.replace(",", ".", regex=False), errors="coerce")
    ls = ls.dropna(subset=["y"]).drop_duplicates("ngay").set_index("ngay")["y"].sort_index()
    cuoi_tuan = ls.index[ls.index.day_of_week >= 5]
    if len(cuoi_tuan):
        ghi("Bỏ dòng lợi suất rơi vào cuối tuần: " + ", ".join(f"{n:%d/%m/%Y}" for n in cuoi_tuan))
    ls = ls.drop(cuoi_tuan)
    if ls.index.min() > lich.min() or ls.index.max() < lich.max():
        raise SystemExit(f"Tệp lợi suất chỉ phủ {ls.index.min():%d/%m/%Y}–{ls.index.max():%d/%m/%Y}, "
                         f"chưa phủ hết {lich.min():%d/%m/%Y}–{lich.max():%d/%m/%Y}")
    y = ls.asof(lich)
    y.index = lich
    bu = lich[~lich.isin(ls.index)]
    ghi(f"Lợi suất TPCP 10 năm (Investing): {len(ls)} dòng ngày thường; "
        f"{int((~ls.loc[lich.min():lich.max()].index.isin(lich)).sum())} ngày Investing có số nhưng HOSE nghỉ (bỏ); "
        f"{len(bu)} phiên HOSE không có số, lấy số phiên trước"
        + (": " + ", ".join(f"{n:%d/%m/%Y}" for n in bu) if 0 < len(bu) <= 20 else ""))
    nhay = y.diff().abs()
    if (nhay > 0.5).any():
        ghi("CẢNH BÁO: lợi suất đổi hơn 0,5 điểm % trong một phiên: " +
            ", ".join(f"{n:%d/%m/%Y} ({v:+.3f})" for n, v in y.diff()[nhay > 0.5].items()))
    if ((y <= 0) | (y > 15)).any():
        ghi("CẢNH BÁO: có lợi suất ngoài khoảng 0–15%, kiểm tra lại đơn vị.")
    ghi(f"Lợi suất 10 năm thấp nhất {y.min():.3f}% ({y.idxmin():%d/%m/%Y}), "
        f"cao nhất {y.max():.3f}% ({y.idxmax():%d/%m/%Y})")
    rf = y.groupby(y.index.year).mean()
    rf.rename("rf_nam_phan_tram").round(4).to_csv(OUT / "lai_suat_phi_rui_ro_nam.csv", index_label="nam")
    ghi("Lợi suất 10 năm bình quân năm theo phiên HOSE (%): " + ", ".join(f"{n}: {v:.2f}" for n, v in rf.items()))
    quy = pd.DataFrame({"cuoi_quy_pt": y.resample("QE").last(), "binh_quan_quy_pt": y.resample("QE").mean()})
    quy.index = quy.index.to_period("Q").astype(str)
    quy.round(4).to_csv(OUT / "lai_suat_phi_rui_ro_quy.csv", index_label="quy")
    pd.DataFrame({"loi_suat_10n_pt": y, "rf_ngay": (1 + y / 100) ** (1 / 252) - 1}).to_csv(OUT / "rf_ngay.csv")
    ghi("Đã ghi data/processed/rf_ngay.csv (rf theo từng phiên)")
elif not tep_rf.exists():
    quy_cuoi = {1: "31/03", 2: "30/06", 3: "30/09", 4: "31/12"}
    pd.DataFrame([{"nam": n, "quy": q, "ngay_cuoi_quy": f"{quy_cuoi[q]}/{n}", "loi_suat_phan_tram": "",
                   "loai_lai_suat": "", "nguon": ""} for n in range(2019, 2026) for q in range(1, 5)]
                 ).to_csv(tep_rf, index=False, encoding="utf-8-sig")
    ghi(f"Đã tạo mẫu {tep_rf}: điền lãi suất cuối mỗi quý (%) và nguồn, rồi chạy lại ô 8.")
else:
    bang_rf = pd.read_csv(tep_rf, dtype=str, sep=None, engine="python", encoding="utf-8-sig").fillna("")
    so_pt = lambda c: pd.to_numeric(bang_rf[c].str.strip().str.replace(",", ".", regex=False), errors="coerce")
    nam_can = range(pd.Timestamp(TU_NGAY).year, pd.Timestamp(DEN_NGAY).year + 1)
    if "quy" in bang_rf.columns:
        bang_rf["ls"] = so_pt("loi_suat_phan_tram")
        thieu = bang_rf[bang_rf["ls"].isna()]
        if len(thieu):
            ghi(f"Còn thiếu lãi suất phi rủi ro {len(thieu)} quý: " +
                ", ".join(f"{r.nam}Q{r.quy}" for r in thieu.itertuples()) + f" trong {tep_rf}.")
        nhom = bang_rf.groupby(bang_rf["nam"].astype(int))["ls"]
        rf = nhom.mean().reindex(nam_can)
        rf[nhom.count().reindex(nam_can).fillna(0) < 4] = np.nan     # năm chưa đủ 4 quý thì chưa tính
        bat_thuong = bang_rf[(bang_rf["ls"] <= 0) | (bang_rf["ls"] > 15)]
        if len(bat_thuong):
            ghi("CẢNH BÁO: lãi suất ngoài khoảng 0–15%, kiểm tra lại đơn vị: " +
                ", ".join(f"{r.nam}Q{r.quy}={r.ls}" for r in bat_thuong.itertuples()))
        if "loai_lai_suat" in bang_rf.columns:
            ghi("Loại lãi suất đã dùng (số quý): " + str(bang_rf.loc[bang_rf["ls"].notna(), "loai_lai_suat"]
                                                        .str.strip().value_counts().to_dict()))
    else:
        rf = so_pt("rf_nam_phan_tram")
        rf.index = bang_rf["nam"].astype(int)
        rf = rf.reindex(nam_can)
    if rf.isna().any():
        ghi(f"Chưa đủ lãi suất phi rủi ro năm {list(rf.index[rf.isna()])}; chưa ghi rf_ngay.csv.")
    else:
        rf.rename("rf_nam_phan_tram").round(4).to_csv(OUT / "lai_suat_phi_rui_ro_nam.csv", index_label="nam")
        ghi("Lãi suất phi rủi ro bình quân năm (%): " + ", ".join(f"{n}: {v:.2f}" for n, v in rf.items()))
        rf_ngay = (1 + rf / 100) ** (1 / 252) - 1
        pd.Series(lich.year.map(rf_ngay), index=lich, name="rf_ngay").to_csv(OUT / "rf_ngay.csv")
        ghi("Đã ghi data/processed/rf_ngay.csv")


# %% 9. Bảng dài cho MySQL và báo cáo
dai = (d.merge(goc[["ma", "ngay", "dong"]].rename(columns={"dong": "dong_cua_goc"}),
               on=["ma", "ngay"], how="left")
       .rename(columns={"dong": "dong_cua_dc", "kl": "khoi_luong"}))
vni_dai = chi_so[(ma_vni == "VNINDEX") & chi_so["ngay"].between(TU_NGAY, DEN_NGAY)]
vni_dai = vni_dai.assign(ma="VNINDEX").rename(columns={"dong": "dong_cua_dc", "kl": "khoi_luong"})
vni_dai["dong_cua_goc"] = vni_dai["dong_cua_dc"]
cot_sql = ["ma", "ngay", "mo", "cao", "thap", "dong_cua_dc", "dong_cua_goc", "khoi_luong"]
dai = pd.concat([dai[cot_sql], vni_dai[cot_sql]], ignore_index=True)
dai["khoi_luong"] = dai["khoi_luong"].round().astype("Int64")
dai.to_csv(OUT / "gia_ngay_dai.csv", index=False)
ghi(f"Bảng dài: {len(dai):,} dòng ({dai['ma'].nunique()} mã)")

(OUT / "bao_cao_du_lieu.txt").write_text("\n".join(BAO_CAO), encoding="utf-8")
print(f"\nĐã ghi báo cáo vào {OUT / 'bao_cao_du_lieu.txt'}")


# %% 10. (Tùy chọn) nạp vào MySQL – chạy trên máy có MySQL, không chạy trên Colab
def nap_mysql(bang, kich_thuoc_lo=5000):
    import os
    import mysql.connector                       # pip install mysql-connector-python
    try:
        from dotenv import load_dotenv           # pip install python-dotenv; đọc mật khẩu từ tệp .env
        load_dotenv()
    except ImportError:
        pass
    kn = mysql.connector.connect(host=os.environ.get("MYSQL_HOST", "localhost"),
                                 user=os.environ["MYSQL_USER"],
                                 password=os.environ["MYSQL_PASSWORD"],
                                 database=os.environ.get("MYSQL_DB", "tieu_luan_khdl"))
    sql = ("INSERT INTO gia_ngay (ma, ngay, mo, cao, thap, dong_cua_dc, dong_cua_goc, khoi_luong) "
           "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
           "ON DUPLICATE KEY UPDATE mo=VALUES(mo), cao=VALUES(cao), thap=VALUES(thap), "
           "dong_cua_dc=VALUES(dong_cua_dc), dong_cua_goc=VALUES(dong_cua_goc), "
           "khoi_luong=VALUES(khoi_luong)")
    b = bang.copy()
    b["ngay"] = b["ngay"].dt.date
    b = b.astype(object).where(b.notna(), None)
    dong = [tuple(x) for x in b.itertuples(index=False)]
    con_tro = kn.cursor()
    for i in range(0, len(dong), kich_thuoc_lo):
        con_tro.executemany(sql, dong[i:i + kich_thuoc_lo])
        kn.commit()
    con_tro.close()
    kn.close()
    print(f"Đã nạp {len(dong):,} dòng vào bảng gia_ngay")


if NAP_MYSQL:
    nap_mysql(dai)
