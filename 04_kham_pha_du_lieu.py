# -*- coding: utf-8 -*-
"""
BƯỚC 4a – KHÁM PHÁ DỮ LIỆU (EDA)
Tiểu luận: Tối ưu hóa danh mục đầu tư bằng gom cụm và học máy (HOSE, 2019–2025)

Đầu vào (do 01_lay_du_lieu_cafef.py tạo ra):
    bảng gia_ngay trong MySQL (đọc khi có tệp .env), nếu không thì data/processed/gia_dong_cua.csv
    data/processed/ro_100_ma.csv
    data/raw/ngay_huy_niem_yet.csv
Đầu ra: thư mục ket_qua/eda/ (bảng .csv, hình .png và bao_cao_eda.txt)

Chạy từ thư mục gốc dự án:  .venv/bin/python 04_kham_pha_du_lieu.py
Thư viện cần thêm so với Bước 3: matplotlib, scikit-learn, scipy.

Quy ước lợi suất: lợi suất giản đơn theo phiên, tính trên giá đóng cửa đã điều chỉnh (đã gồm cổ tức).
Trong khoảng mã còn có dữ liệu, phiên trống được giữ giá phiên trước (lợi suất 0), và phiên có giá
trở lại mang cả biến động của các phiên trống. Đây cũng là quy ước "vị thế đóng băng" của bước
kiểm tra ngoài mẫu, nên đặc trưng tính ở đây khớp với đặc trưng mô hình sẽ dùng.
"""

# %% 0. Cấu hình
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")                     # chỉ ghi hình ra tệp, không mở cửa sổ
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

GOC = Path(".")
PROC = GOC / "data" / "processed"
RAW = GOC / "data" / "raw"
KQ = GOC / "ket_qua" / "eda"
KQ.mkdir(parents=True, exist_ok=True)

DOC_TU_MYSQL = True          # False: chỉ đọc tệp CSV
TU_NGAY, DEN_NGAY = "2019-01-01", "2025-12-31"
CUA_SO = 504                 # số phiên của cửa sổ ước lượng
BUOC = 63                    # số phiên giữa hai lần tái cân bằng
NGUONG_THIEU = 0.10          # mã thiếu quá 10% số phiên trong cửa sổ thì không được chọn
CUA_SO_TQ = 126              # cửa sổ trượt cho tương quan bình quân giữa các mã
BUOC_TQ = 5                  # tính tương quan trượt cách nhau 5 phiên
NGUONG_BIEN_DO = 0.075       # biên độ HOSE ±7% cộng phần dư do làm tròn bước giá
PHIEN_NAM = 252
# Phiên giao dịch thật nhưng không có trong lịch dữ liệu (kiểm tra lịch phiên bằng SQL ngày 16/11/2026):
#   17/05/2024: tệp Index của CafeF thiếu VN-Index, script 01 bỏ cả ngày;
#   03/02/2025: phiên đầu năm Ất Tỵ, CafeF không có dữ liệu của mã nào.
# Lợi suất của phiên liền sau mỗi ngày này là lợi suất gộp hai phiên.
NGAY_BO_KHOI_LICH = ["2024-05-17", "2025-02-03"]
DAC_TRUNG = {"ln_nam": "Lợi suất năm hóa", "bd_nam": "Độ biến động năm hóa",
             "beta": "Beta với VN-Index", "sgtd": "Sụt giảm tối đa"}

BAO_CAO = []


def ghi(dong=""):
    print(dong)
    BAO_CAO.append(str(dong))


def pt(x, so_le=1):
    """Định dạng tỷ lệ thành phần trăm kiểu Việt Nam (dấu phẩy thập phân)."""
    return f"{x * 100:.{so_le}f}%".replace(".", ",")


def so(x, so_le=2):
    return f"{x:.{so_le}f}".replace(".", ",")


plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


# %% 1. Đọc dữ liệu
ro = pd.read_csv(PROC / "ro_100_ma.csv", encoding="utf-8-sig", dtype={"ma": str})
DS_MA = ro.sort_values("hang")["ma"].tolist()
NGAY_BO = set(pd.to_datetime(NGAY_BO_KHOI_LICH))   # thêm các ngày phát hiện khi đọc MySQL


def doc_csv():
    tep = PROC / "gia_dong_cua.csv"
    if not tep.exists():
        return None
    g = pd.read_csv(tep, index_col="ngay", parse_dates=["ngay"])
    g.columns = g.columns.astype(str)
    return g


def doc_mysql():
    """Đọc giá đã điều chỉnh từ bảng gia_ngay, xoay thành bảng rộng theo lịch phiên của VN-Index."""
    import os
    try:
        import mysql.connector
        from dotenv import load_dotenv
    except ImportError as e:
        ghi(f"Không đọc được MySQL ({e}). Dùng tệp CSV.")
        return None
    load_dotenv()
    if not os.environ.get("MYSQL_USER") or not os.environ.get("MYSQL_PASSWORD"):
        ghi("Không thấy MYSQL_USER hoặc MYSQL_PASSWORD trong .env. Dùng tệp CSV.")
        return None
    try:
        kn = mysql.connector.connect(host=os.environ.get("MYSQL_HOST", "localhost"),
                                     user=os.environ["MYSQL_USER"],
                                     password=os.environ["MYSQL_PASSWORD"],
                                     database=os.environ.get("MYSQL_DB", "tieu_luan_khdl"))
    except Exception as e:
        ghi(f"Không kết nối được MySQL ({e}). Dùng tệp CSV.")
        return None
    con_tro = kn.cursor()
    con_tro.execute("SELECT ma, ngay, dong_cua_dc FROM gia_ngay WHERE ngay BETWEEN %s AND %s",
                    (TU_NGAY, DEN_NGAY))
    dai = pd.DataFrame(con_tro.fetchall(), columns=["ma", "ngay", "dong_cua_dc"])
    con_tro.close()
    kn.close()
    dai["ngay"] = pd.to_datetime(dai["ngay"])
    dai["dong_cua_dc"] = pd.to_numeric(dai["dong_cua_dc"].astype(object), errors="coerce").astype(float)
    lich_vni = pd.DatetimeIndex(sorted(dai.loc[dai["ma"] == "VNINDEX", "ngay"].unique()))
    ngoai = dai[~dai["ngay"].isin(lich_vni)]
    if len(ngoai):
        ngay_ngoai = sorted(ngoai["ngay"].unique())
        NGAY_BO.update(pd.to_datetime(ngay_ngoai))
        ghi(f"MySQL: bỏ {len(ngoai)} dòng giá cổ phiếu ở {len(ngay_ngoai)} ngày không có phiên VN-Index "
            f"({', '.join(f'{d:%d/%m/%Y}' for d in ngay_ngoai[:5])})")
    g = dai.pivot(index="ngay", columns="ma", values="dong_cua_dc")
    g = g.reindex(index=lich_vni, columns=DS_MA + ["VNINDEX"])
    g.index.name = "ngay"
    ghi(f"Đọc từ MySQL: {len(dai):,} dòng, {dai['ma'].nunique()} mã (gồm VN-Index)")
    return g


gia_csv = doc_csv()
gia_sql = doc_mysql() if DOC_TU_MYSQL else None
if gia_sql is None and gia_csv is None:
    raise SystemExit("Không có dữ liệu: chạy 01_lay_du_lieu_cafef.py trước.")
gia = gia_sql if gia_sql is not None else gia_csv[DS_MA + ["VNINDEX"]]
ghi(f"Nguồn dùng để phân tích: {'MySQL (bảng gia_ngay)' if gia_sql is not None else 'tệp gia_dong_cua.csv'}")

# Khi có cả hai nguồn: đối chiếu để chắc dữ liệu trong MySQL trùng với tệp CSV.
if gia_sql is not None and gia_csv is not None:
    a, b = gia_sql, gia_csv.reindex(index=gia_sql.index, columns=gia_sql.columns)
    lech_lich = len(gia_sql.index.symmetric_difference(gia_csv.index))
    lech_trong = int((a.isna() != b.isna()).sum().sum())
    lech_gia = float((a - b).abs().max().max())
    ghi(f"Đối chiếu MySQL với gia_dong_cua.csv: lệch lịch phiên {lech_lich} ngày; lệch ô trống {lech_trong} ô; "
        f"chênh giá lớn nhất {so(lech_gia, 4)} (MySQL lưu 3 chữ số thập phân)")
    if lech_lich or lech_trong or lech_gia > 0.0006:
        ghi("CẢNH BÁO: hai nguồn không khớp. Nạp lại MySQL hoặc chạy lại script 01 trước khi phân tích.")

LICH = gia.index
gia_cp = gia[DS_MA]
vni = gia["VNINDEX"]
ghi(f"Bảng giá: {len(LICH)} phiên từ {LICH[0]:%d/%m/%Y} đến {LICH[-1]:%d/%m/%Y}; {len(DS_MA)} mã và VN-Index")
ghi("Phiên giao dịch thật không có trong dữ liệu: " + ", ".join(f"{d:%d/%m/%Y}" for d in sorted(NGAY_BO))
    + " (phiên liền sau mang lợi suất gộp hai phiên)")
if vni.isna().any():
    raise SystemExit("VN-Index có phiên trống: kiểm tra lại dữ liệu.")

# Ngày giao dịch cuối trên HOSE của các mã hủy niêm yết (quy tắc cách B, chốt 13/11/2026)
tep_hny = RAW / "ngay_huy_niem_yet.csv"
NGAY_CUOI_HOSE = pd.Series(pd.NaT, index=DS_MA, dtype="datetime64[ns]")
hny = pd.DataFrame()
if tep_hny.exists():
    hny = pd.read_csv(tep_hny, encoding="utf-8-sig", dtype={"ma": str},
                      parse_dates=["ngay_gd_cuoi_hose", "ngay_huy_niem_yet", "ngay_gd_dau_upcom"])
    hny = hny[hny["ma"].isin(DS_MA)].set_index("ma")
    NGAY_CUOI_HOSE.loc[hny.index] = hny["ngay_gd_cuoi_hose"].values
    ghi(f"Mã hủy niêm yết trong rổ: {len(hny)} ({', '.join(hny.index)})")
else:
    ghi("CẢNH BÁO: không có data/raw/ngay_huy_niem_yet.csv; coi như mọi mã còn niêm yết đến cuối kỳ.")


# %% 2. Lợi suất theo phiên
# Giữ giá phiên trước cho các phiên trống, nhưng chỉ đến phiên cuối cùng mã còn có giá.
gia_ff = gia.ffill().where(gia.bfill().notna())
r = gia_ff.pct_change(fill_method=None)
r_cp, r_vni = r[DS_MA], r["VNINDEX"]

# Lịch tái cân bằng: phiên thứ CUA_SO + 1 trở đi, cách nhau BUOC phiên.
VT_TCB = list(range(CUA_SO, len(LICH), BUOC))
lich_tcb = pd.DataFrame({
    "ky": np.arange(1, len(VT_TCB) + 1),
    "ngay_quyet_dinh": [LICH[k - 1] for k in VT_TCB],
    "phien_dau_nam_giu": [LICH[k] for k in VT_TCB],
    "phien_cuoi_nam_giu": [LICH[min(k + BUOC, len(LICH)) - 1] for k in VT_TCB],
    "so_phien_nam_giu": [min(k + BUOC, len(LICH)) - k for k in VT_TCB],
})


def du_dieu_kien(k):
    """Mã được chọn ở kỳ bắt đầu từ phiên k: thiếu không quá 10% cửa sổ và còn giao dịch trên HOSE."""
    cua_so = gia_cp.iloc[k - CUA_SO:k]
    thieu = cua_so.isna().mean()
    con_niem_yet = NGAY_CUOI_HOSE.isna() | (NGAY_CUOI_HOSE > LICH[k - 1])
    return pd.DataFrame({"ty_le_thieu": thieu, "con_niem_yet": con_niem_yet,
                         "co_gia_ngay_quyet_dinh": cua_so.iloc[-1].notna(),
                         "du": (thieu <= NGUONG_THIEU) & con_niem_yet})


DK = {k: du_dieu_kien(k) for k in VT_TCB}
lich_tcb["so_ma_du_dieu_kien"] = [int(DK[k]["du"].sum()) for k in VT_TCB]
lich_tcb["so_ma_du_nhung_trong_gia_ngay_qd"] = [
    int((DK[k]["du"] & ~DK[k]["co_gia_ngay_quyet_dinh"]).sum()) for k in VT_TCB]
lich_tcb.to_csv(KQ / "lich_tai_can_bang.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")


# %% 3. Phần 1 – Rà chất lượng dữ liệu lần cuối
ghi("\n=== PHẦN 1. CHẤT LƯỢNG DỮ LIỆU ===")
so_phien_kt = len(LICH) - CUA_SO
ghi(f"Lịch tái cân bằng: {len(VT_TCB)} kỳ. Phiên đầu nắm giữ {LICH[VT_TCB[0]]:%d/%m/%Y} "
    f"(quyết định dựa trên dữ liệu đến {LICH[VT_TCB[0] - 1]:%d/%m/%Y}); {so_phien_kt} phiên kiểm tra; "
    f"kỳ cuối có {lich_tcb['so_phien_nam_giu'].iloc[-1]} phiên")

# 1a. Ngày giao dịch cuối trên HOSE có khớp dữ liệu không
if len(hny):
    dong = []
    for m, h in hny.iterrows():
        s = gia_cp[m]
        cuoi = s.loc[:h["ngay_gd_cuoi_hose"]].last_valid_index()
        sau = s.loc[h["ngay_gd_cuoi_hose"] + pd.Timedelta(days=1):].first_valid_index()
        ky_vong_sau = h["ngay_gd_dau_upcom"] if h["nhom"] == "chuyen_upcom" else pd.NaT
        khop = (cuoi == h["ngay_gd_cuoi_hose"]) and (
            (pd.isna(sau) and pd.isna(ky_vong_sau)) or (pd.notna(sau) and sau == ky_vong_sau))
        dong.append(f"{m} [{h['nhom']}]: phiên có giá cuối trên HOSE {cuoi:%d/%m/%Y}, "
                    f"phiên có giá kế tiếp {'không có' if pd.isna(sau) else f'{sau:%d/%m/%Y}'} -> "
                    f"{'khớp' if khop else 'LỆCH, xem lại'}")
    ghi("Đối chiếu ngày hủy niêm yết với dữ liệu giá:")
    for d in dong:
        ghi("  " + d)

# 1b. Phiên có lợi suất vượt biên độ HOSE, xét cả chiều tăng và chiều giảm
vi_tri = np.arange(len(LICH), dtype=float)
vt = pd.DataFrame(np.where(gia_cp.notna(), vi_tri[:, None], np.nan), index=LICH, columns=DS_MA)
cach_phien = vt - vt.ffill().shift()
gia_truoc = gia_ff[DS_MA].shift()
vuot = (r_cp.abs() > NGUONG_BIEN_DO) & gia_cp.notna()
pv = vuot.stack()
pv = pv[pv].index.to_frame(index=False)
pv.columns = ["ngay", "ma"]
if len(pv):
    lay = lambda bang: [bang.at[d, m] for d, m in zip(pv["ngay"], pv["ma"])]
    pv["loi_suat"] = lay(r_cp)
    pv["gia_truoc"] = lay(gia_truoc)
    pv["gia"] = lay(gia_cp)
    pv["so_phien_tu_gia_truoc"] = lay(cach_phien)
    sau_hose = [pd.notna(NGAY_CUOI_HOSE[m]) and d > NGAY_CUOI_HOSE[m] for d, m in zip(pv["ngay"], pv["ma"])]
    # Phiên ngay sau một ngày bị bỏ khỏi lịch mang lợi suất của hai phiên giao dịch
    phien_truoc = pd.Series(LICH[:-1], index=LICH[1:])
    qua_ngay_bo = [any(phien_truoc.get(d, d) < b < d for b in NGAY_BO) for d in pv["ngay"]]
    pv["phan_loai"] = np.select(
        [np.array(sau_hose), pv["so_phien_tu_gia_truoc"] > 1, np.array(qua_ngay_bo)],
        ["sau_khi_roi_HOSE", "sau_nhieu_phien_khong_giao_dich", "gop_hai_phien_do_thieu_phien"],
        default="can_xem")
    pv = pv.sort_values(["phan_loai", "ma", "ngay"])
pv.to_csv(KQ / "phien_vuot_bien_do.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
ghi(f"Phiên có lợi suất vượt ±{pt(NGUONG_BIEN_DO)}: {len(pv)}"
    + (f" (tăng {int((pv['loi_suat'] > 0).sum())}, giảm {int((pv['loi_suat'] < 0).sum())})" if len(pv) else ""))
if len(pv):
    for loai, n in pv["phan_loai"].value_counts().items():
        ghi(f"  {loai}: {n}")
    cx = pv[pv["phan_loai"] == "can_xem"]
    for _, x in cx.head(40).iterrows():
        ghi(f"    cần xem: {x['ma']} {x['ngay']:%d/%m/%Y} {pt(x['loi_suat'])} "
            f"(giá trước {so(x['gia_truoc'])}, giá {so(x['gia'])})")
    if len(cx) > 40:
        ghi(f"    ... còn {len(cx) - 40} dòng, xem phien_vuot_bien_do.csv")
ghi(f"VN-Index: phiên giảm mạnh nhất {pt(r_vni.min(), 2)} ({r_vni.idxmin():%d/%m/%Y}), "
    f"tăng mạnh nhất {pt(r_vni.max(), 2)} ({r_vni.idxmax():%d/%m/%Y})")

# 1c. Số mã đủ điều kiện ở từng kỳ
dk0 = DK[VT_TCB[0]]
ghi(f"Kỳ 1: {int(dk0['du'].sum())}/{len(DS_MA)} mã đủ điều kiện")
loai_thieu = dk0[(dk0["ty_le_thieu"] > NGUONG_THIEU) & dk0["con_niem_yet"]]
if len(loai_thieu):
    ghi("  Không đủ dữ liệu trong cửa sổ đầu: " + ", ".join(
        f"{m} (thiếu {pt(t, 0)})" for m, t in loai_thieu["ty_le_thieu"].sort_values().items()))
ghi(f"Số mã đủ điều kiện qua các kỳ: thấp nhất {lich_tcb['so_ma_du_dieu_kien'].min()}, "
    f"cao nhất {lich_tcb['so_ma_du_dieu_kien'].max()}")
trong = lich_tcb[lich_tcb["so_ma_du_nhung_trong_gia_ngay_qd"] > 0]
for _, x in trong.iterrows():
    k = VT_TCB[int(x["ky"]) - 1]
    ds = DK[k].index[DK[k]["du"] & ~DK[k]["co_gia_ngay_quyet_dinh"]]
    ghi(f"  Kỳ {int(x['ky'])} ({x['ngay_quyet_dinh']:%d/%m/%Y}): đủ điều kiện nhưng không có giá ngày "
        f"quyết định: {', '.join(ds)}")

# 1d. Mã đủ điều kiện ở một kỳ nhưng rời HOSE ngay trong kỳ nắm giữ đó (vướng hạn chế của quy tắc B)
for k, x in zip(VT_TCB, lich_tcb.itertuples()):
    for m in DK[k].index[DK[k]["du"]]:
        nc = NGAY_CUOI_HOSE[m]
        if pd.notna(nc) and x.phien_dau_nam_giu <= nc <= x.phien_cuoi_nam_giu:
            ghi(f"  {m}: được chọn ở kỳ {x.ky} (quyết định {x.ngay_quyet_dinh:%d/%m/%Y}) nhưng giao dịch "
                f"phiên cuối trên HOSE ngày {nc:%d/%m/%Y}, trong kỳ nắm giữ")


# %% 4. Phần 2 – Thống kê mô tả lợi suất theo phiên
ghi("\n=== PHẦN 2. THỐNG KÊ MÔ TẢ LỢI SUẤT THEO PHIÊN ===")


def mo_ta(x):
    x = x.dropna()
    return pd.Series({
        "so_phien": len(x),
        "trung_binh_pt": x.mean() * 100,
        "trung_vi_pt": x.median() * 100,
        "do_lech_chuan_pt": x.std() * 100,
        "do_lech": x.skew(),
        "do_nhon_vuot": x.kurt(),
        "nho_nhat_pt": x.min() * 100,
        "lon_nhat_pt": x.max() * 100,
        "ln_nam_hoa_pt": x.mean() * PHIEN_NAM * 100,
        "bd_nam_hoa_pt": x.std() * np.sqrt(PHIEN_NAM) * 100,
        "jarque_bera_p": stats.jarque_bera(x).pvalue,
    })


tk_ma = r_cp.apply(mo_ta).T
tk_ma.index.name = "ma"
# Tỷ lệ phiên giá đứng yên: chỉ xét những cặp phiên liền nhau mà cả hai đều có giá thật.
# Tỷ lệ cao cho thấy cổ phiếu ít thanh khoản, khiến độ biến động và tương quan đo được thấp giả tạo.
lien_tiep = gia_cp.notna() & gia_cp.shift().notna()
dung_yen = (gia_cp == gia_cp.shift()) & lien_tiep
tk_ma["ty_le_phien_dung_gia_pt"] = dung_yen.sum() / lien_tiep.sum() * 100
tk_ma.round(4).to_csv(KQ / "thong_ke_loi_suat_tung_ma.csv", encoding="utf-8-sig")
tk_vni = mo_ta(r_vni)
bang = pd.DataFrame({
    "VN-Index": tk_vni,
    "100 mã: nhỏ nhất": tk_ma.min(),
    "100 mã: Q1": tk_ma.quantile(0.25),
    "100 mã: trung vị": tk_ma.median(),
    "100 mã: Q3": tk_ma.quantile(0.75),
    "100 mã: lớn nhất": tk_ma.max(),
})
bang.round(4).to_csv(KQ / "bang_thong_ke_mo_ta_loi_suat.csv", encoding="utf-8-sig")
ghi(bang.drop(index="jarque_bera_p").round(3).to_string())
ghi(f"Độ lệch âm: {int((tk_ma['do_lech'] < 0).sum())}/{len(tk_ma)} mã; VN-Index {so(tk_vni['do_lech'])}")
ghi(f"Độ nhọn vượt: trung vị {so(tk_ma['do_nhon_vuot'].median())}; VN-Index {so(tk_vni['do_nhon_vuot'])} "
    f"(phân phối chuẩn bằng 0)")
ghi(f"Kiểm định Jarque–Bera bác bỏ phân phối chuẩn ở mức 5%: {int((tk_ma['jarque_bera_p'] < 0.05).sum())}"
    f"/{len(tk_ma)} mã; VN-Index p = {tk_vni['jarque_bera_p']:.2g}")
top_dung = tk_ma["ty_le_phien_dung_gia_pt"].sort_values(ascending=False).head(10)
ghi("Tỷ lệ phiên giá đứng yên cả kỳ: trung vị " + so(tk_ma["ty_le_phien_dung_gia_pt"].median(), 1)
    + "%; 10 mã cao nhất: " + ", ".join(f"{m} {so(v, 1)}%" for m, v in top_dung.items()))

# Hình 2: phân phối lợi suất VN-Index so với phân phối chuẩn cùng trung bình, độ lệch chuẩn
x = r_vni.dropna()
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.hist(x * 100, bins=80, density=True, color="#4C78A8", alpha=0.8, label="Lợi suất VN-Index")
luoi = np.linspace(x.min(), x.max(), 400)
ax.plot(luoi * 100, stats.norm.pdf(luoi, x.mean(), x.std()) / 100, color="#E45756", lw=1.4,
        label="Phân phối chuẩn tương ứng")
ax.set_xlabel("Lợi suất theo phiên (%)")
ax.set_ylabel("Mật độ")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(KQ / "hinh_2_phan_phoi_loi_suat_vnindex.png")
plt.close(fig)


# %% 5. Phần 3 – Diễn biến thị trường 2019–2025
ghi("\n=== PHẦN 3. DIỄN BIẾN THỊ TRƯỜNG ===")
sut = vni / vni.cummax() - 1
day = sut.idxmin()
dinh = vni.loc[:day].idxmax()
hoi_phuc = vni.loc[day:][vni.loc[day:] >= vni[dinh]]
ghi(f"Sụt giảm lớn nhất của VN-Index: {pt(sut.min())}, từ đỉnh {dinh:%d/%m/%Y} ({so(vni[dinh])} điểm) "
    f"đến đáy {day:%d/%m/%Y} ({so(vni[day])} điểm); "
    + (f"lấy lại đỉnh ngày {hoi_phuc.index[0]:%d/%m/%Y}" if len(hoi_phuc) else "chưa lấy lại đỉnh đến cuối kỳ"))

# Chỉ số bình quân đều của các mã có giá mỗi phiên (chỉ để mô tả, không phải chiến lược 1/N)
r_bqd = r_cp.mean(axis=1, skipna=True).fillna(0)
chi_so_bqd = (1 + r_bqd).cumprod()
nam = []
for y, nhom in r.groupby(LICH.year):
    rv = nhom["VNINDEX"].dropna()
    v = vni[vni.index.year == y]
    nam.append({
        "nam": y,
        "so_phien": len(v),
        "vnindex_dau_nam": v.iloc[0],
        "vnindex_cuoi_nam": v.iloc[-1],
        "loi_suat_vnindex_pt": ((1 + rv).prod() - 1) * 100,
        "bien_dong_vnindex_pt": rv.std() * np.sqrt(PHIEN_NAM) * 100,
        "sut_giam_toi_da_trong_nam_pt": (v / v.cummax() - 1).min() * 100,
        "loi_suat_bq_deu_100_ma_pt": ((1 + r_bqd[r_bqd.index.year == y]).prod() - 1) * 100,
        "so_ma_co_gia": int(gia_cp[gia_cp.index.year == y].notna().any().sum()),
    })
bang_nam = pd.DataFrame(nam).set_index("nam")
bang_nam.round(2).to_csv(KQ / "dien_bien_theo_nam.csv", encoding="utf-8-sig")
ghi(bang_nam.round(2).to_string())
ghi("Ghi chú: lợi suất VN-Index năm 2019 tính từ phiên đầu tiên có dữ liệu; VN-Index là chỉ số giá, "
    "không gồm cổ tức, còn bình quân đều 100 mã dùng giá đã điều chỉnh nên gồm cổ tức.")

# Hình 1: VN-Index, chỉ số bình quân đều và mức sụt giảm từ đỉnh
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
a1.plot(vni.index, vni, color="#4C78A8", lw=1.1, label="VN-Index")
a1.plot(chi_so_bqd.index, chi_so_bqd * vni.iloc[0], color="#F58518", lw=1.0,
        label="Bình quân đều 100 mã (quy về điểm VN-Index đầu kỳ)")
a1.axvline(LICH[VT_TCB[0]], color="grey", ls="--", lw=0.8)
a1.text(LICH[VT_TCB[0]], a1.get_ylim()[0], " bắt đầu kiểm tra ngoài mẫu", va="bottom", fontsize=8, color="grey")
a1.set_ylabel("Điểm")
a1.legend(frameon=False, loc="upper left")
a2.fill_between(sut.index, sut * 100, 0, color="#E45756", alpha=0.6)
a2.set_ylabel("Sụt giảm từ đỉnh (%)")
fig.tight_layout()
fig.savefig(KQ / "hinh_1_vnindex_va_sut_giam.png")
plt.close(fig)


# %% 6. Phần 4 – Bốn đặc trưng gom cụm
ghi("\n=== PHẦN 4. ĐẶC TRƯNG GOM CỤM ===")


def tinh_dac_trung(k):
    """Đặc trưng của các mã đủ điều kiện, tính trên CUA_SO phiên ngay trước phiên k."""
    ma = DK[k].index[DK[k]["du"]]
    rr = r_cp.iloc[k - CUA_SO:k][ma]
    rm = r_vni.iloc[k - CUA_SO:k]
    p = gia_ff[ma].iloc[k - CUA_SO:k]
    beta = {m: rr[m].cov(rm) / rm[rr[m].notna()].var() for m in ma}
    return pd.DataFrame({
        "ln_nam": rr.mean() * PHIEN_NAM,
        "bd_nam": rr.std() * np.sqrt(PHIEN_NAM),
        "beta": pd.Series(beta),
        "sgtd": -(p / p.cummax() - 1).min(),
    })


DT = {k: tinh_dac_trung(k) for k in VT_TCB}
dt0 = DT[VT_TCB[0]]
cs0 = slice(VT_TCB[0] - CUA_SO, VT_TCB[0])
dung0 = dung_yen.iloc[cs0][dt0.index].sum() / lien_tiep.iloc[cs0][dt0.index].sum()
dt0.assign(ty_le_phien_dung_gia=dung0).round(4).to_csv(
    KQ / "dac_trung_cua_so_dau.csv", encoding="utf-8-sig", index_label="ma")
mo_ta_dt = dt0.describe().T
mo_ta_dt["do_lech"] = dt0.skew()
mo_ta_dt.round(4).to_csv(KQ / "mo_ta_dac_trung_cua_so_dau.csv", encoding="utf-8-sig")
ghi(f"Cửa sổ đầu ({LICH[VT_TCB[0] - CUA_SO]:%d/%m/%Y}–{LICH[VT_TCB[0] - 1]:%d/%m/%Y}), {len(dt0)} mã:")
ghi(mo_ta_dt.round(3).to_string())
thap = dt0.nsmallest(8, "bd_nam")
ghi("8 mã biến động thấp nhất ở cửa sổ đầu (độ biến động; beta; tỷ lệ phiên giá đứng yên): " + "; ".join(
    f"{m} {pt(x.bd_nam)}, {so(x.beta)}, {pt(dung0[m])}" for m, x in thap.iterrows()))
ghi(f"Tương quan giữa độ biến động và tỷ lệ phiên giá đứng yên ở cửa sổ đầu: {so(dt0['bd_nam'].corr(dung0))}; "
    f"giữa beta và tỷ lệ phiên giá đứng yên: {so(dt0['beta'].corr(dung0))}")

# Ngoại lệ theo quy tắc IQR
nl = []
for c in DAC_TRUNG:
    q1, q3 = dt0[c].quantile([0.25, 0.75])
    duoi, tren = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    for m, v in dt0[c].items():
        if v < duoi or v > tren:
            nl.append({"dac_trung": c, "ma": m, "gia_tri": v, "can_duoi": duoi, "can_tren": tren})
nl = pd.DataFrame(nl, columns=["dac_trung", "ma", "gia_tri", "can_duoi", "can_tren"])
nl.round(4).to_csv(KQ / "ngoai_le_dac_trung.csv", index=False, encoding="utf-8-sig")
for c in DAC_TRUNG:
    x = nl[nl["dac_trung"] == c]
    ghi(f"Ngoại lệ IQR – {DAC_TRUNG[c]}: {len(x)}" +
        (f" ({', '.join(f'{m} {so(v, 2)}' for m, v in zip(x['ma'], x['gia_tri']))})" if len(x) else ""))

# Tương quan giữa các đặc trưng
tq_p = dt0.corr()
tq_s = dt0.corr(method="spearman")
pd.concat({"pearson": tq_p, "spearman": tq_s}).round(3).to_csv(
    KQ / "tuong_quan_dac_trung_cua_so_dau.csv", encoding="utf-8-sig")
ghi("Tương quan Pearson giữa các đặc trưng (cửa sổ đầu):")
ghi(tq_p.round(2).to_string())
ghi("Tương quan Spearman:")
ghi(tq_s.round(2).to_string())

pca0 = PCA().fit(StandardScaler().fit_transform(dt0))
ghi("PCA trên đặc trưng đã chuẩn hóa z-score, tỷ lệ phương sai giải thích: " +
    ", ".join(f"PC{i + 1} {pt(v)}" for i, v in enumerate(pca0.explained_variance_ratio_)))

# Tương quan giữa các đặc trưng qua mọi kỳ tái cân bằng
cap = [("bd_nam", "sgtd"), ("bd_nam", "beta"), ("beta", "sgtd"),
       ("ln_nam", "bd_nam"), ("ln_nam", "beta"), ("ln_nam", "sgtd")]
theo_ky = []
for k, x in zip(VT_TCB, lich_tcb.itertuples()):
    d = DT[k]
    c = d.corr()
    dong = {"ky": x.ky, "ngay_quyet_dinh": x.ngay_quyet_dinh, "so_ma": len(d)}
    dong.update({f"{a}~{b}": c.loc[a, b] for a, b in cap})
    dong["pc1_pt"] = PCA().fit(StandardScaler().fit_transform(d)).explained_variance_ratio_[0] * 100
    theo_ky.append(dong)
theo_ky = pd.DataFrame(theo_ky)
theo_ky.set_index(["ky", "ngay_quyet_dinh"]).round(3).to_csv(
    KQ / "tuong_quan_dac_trung_theo_ky.csv", encoding="utf-8-sig", date_format="%Y-%m-%d")
ghi(f"Tương quan Pearson giữa các đặc trưng qua {len(theo_ky)} kỳ (trung vị; thấp nhất – cao nhất):")
for a, b in cap:
    s = theo_ky[f"{a}~{b}"]
    ghi(f"  {DAC_TRUNG[a]} – {DAC_TRUNG[b]}: {so(s.median())} ({so(s.min())} – {so(s.max())})")
ghi(f"  PC1 giải thích: trung vị {so(theo_ky['pc1_pt'].median(), 1)}% "
    f"({so(theo_ky['pc1_pt'].min(), 1)}% – {so(theo_ky['pc1_pt'].max(), 1)}%)")

# Hình 3: ma trận phân tán của bốn đặc trưng (cửa sổ đầu)
ve = dt0.rename(columns=DAC_TRUNG)
axs = scatter_matrix(ve, figsize=(7, 7), diagonal="hist", alpha=0.7, s=12, color="#4C78A8",
                     hist_kwds={"bins": 20, "color": "#4C78A8"})
for a in axs.ravel():
    a.xaxis.label.set_size(8)
    a.yaxis.label.set_size(8)
plt.gcf().tight_layout()
plt.gcf().savefig(KQ / "hinh_3_ma_tran_phan_tan_dac_trung.png")
plt.close("all")

# Hình 4: rủi ro – lợi suất của các mã ở cửa sổ đầu, màu theo beta
fig, ax = plt.subplots(figsize=(6.5, 4.5))
sc = ax.scatter(dt0["bd_nam"] * 100, dt0["ln_nam"] * 100, c=dt0["beta"], cmap="viridis", s=22)
fig.colorbar(sc, ax=ax, label="Beta với VN-Index")
for m in set(nl["ma"]):
    ax.annotate(m, (dt0.at[m, "bd_nam"] * 100, dt0.at[m, "ln_nam"] * 100), fontsize=7,
                xytext=(3, 3), textcoords="offset points")
ax.axhline(0, color="grey", lw=0.6)
ax.set_xlabel("Độ biến động năm hóa (%)")
ax.set_ylabel("Lợi suất năm hóa (%)")
fig.tight_layout()
fig.savefig(KQ / "hinh_4_rui_ro_loi_suat_cua_so_dau.png")
plt.close(fig)


# %% 7. Phần 5 – Tương quan giữa các mã
ghi("\n=== PHẦN 5. TƯƠNG QUAN GIỮA CÁC MÃ ===")
k0 = VT_TCB[0]
ma0 = dt0.index
C = r_cp.iloc[k0 - CUA_SO:k0][ma0].corr(min_periods=int(CUA_SO * (1 - 2 * NGUONG_THIEU)))
tren_cheo = C.values[np.triu_indices(len(C), 1)]
tren_cheo = tren_cheo[~np.isnan(tren_cheo)]
ghi(f"Cửa sổ đầu, {len(tren_cheo)} cặp mã: tương quan trung bình {so(tren_cheo.mean())}, trung vị "
    f"{so(np.median(tren_cheo))}, Q1 {so(np.quantile(tren_cheo, .25))}, Q3 {so(np.quantile(tren_cheo, .75))}, "
    f"thấp nhất {so(tren_cheo.min())}, cao nhất {so(tren_cheo.max())}, cặp âm {pt((tren_cheo < 0).mean())}")
tq_vni = r_cp.iloc[k0 - CUA_SO:k0][ma0].corrwith(r_vni.iloc[k0 - CUA_SO:k0])
ghi(f"Tương quan của từng mã với VN-Index: trung vị {so(tq_vni.median())} "
    f"({so(tq_vni.min())} – {so(tq_vni.max())})")

# Tương quan bình quân theo cửa sổ trượt
truot = []
for e in range(CUA_SO_TQ, len(LICH) + 1, BUOC_TQ):
    cs = gia_cp.iloc[e - CUA_SO_TQ:e]
    ma = cs.columns[(cs.isna().mean() <= NGUONG_THIEU)]
    c = r_cp.iloc[e - CUA_SO_TQ:e][ma].corr(min_periods=int(CUA_SO_TQ * (1 - 2 * NGUONG_THIEU))).values
    v = c[np.triu_indices(len(ma), 1)]
    truot.append({"ngay": LICH[e - 1], "so_ma": len(ma), "tuong_quan_bq": np.nanmean(v),
                  "bien_dong_vnindex_pt": r_vni.iloc[e - CUA_SO_TQ:e].std() * np.sqrt(PHIEN_NAM) * 100})
truot = pd.DataFrame(truot).set_index("ngay")
truot.round(4).to_csv(KQ / "tuong_quan_binh_quan_truot.csv", encoding="utf-8-sig", date_format="%Y-%m-%d")
ghi(f"Tương quan bình quân trên cửa sổ trượt {CUA_SO_TQ} phiên: thấp nhất "
    f"{so(truot['tuong_quan_bq'].min())} ({truot['tuong_quan_bq'].idxmin():%d/%m/%Y}), cao nhất "
    f"{so(truot['tuong_quan_bq'].max())} ({truot['tuong_quan_bq'].idxmax():%d/%m/%Y}); hệ số tương quan "
    f"giữa tương quan bình quân và biến động VN-Index: {so(truot['tuong_quan_bq'].corr(truot['bien_dong_vnindex_pt']))}")

# Hình 5: tương quan bình quân và biến động VN-Index theo thời gian
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(truot.index, truot["tuong_quan_bq"], color="#4C78A8", lw=1.2, label="Tương quan bình quân giữa các mã")
ax.set_ylabel("Hệ số tương quan")
ax2 = ax.twinx()
ax2.plot(truot.index, truot["bien_dong_vnindex_pt"], color="#E45756", lw=1.0, alpha=0.8,
         label="Biến động năm hóa VN-Index (%)")
ax2.set_ylabel("Biến động VN-Index (%)")
ax2.spines["right"].set_visible(True)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=8)
ax.set_title(f"Cửa sổ trượt {CUA_SO_TQ} phiên", fontsize=9)
fig.tight_layout()
fig.savefig(KQ / "hinh_5_tuong_quan_binh_quan_truot.png")
plt.close(fig)

# Hình 6: ma trận tương quan cửa sổ đầu, sắp xếp theo gom cụm phân cấp (liên kết đơn như HRP)
c0 = C.fillna(0).to_numpy(copy=True)
np.fill_diagonal(c0, 1.0)
kc = np.sqrt(np.clip(0.5 * (1 - c0), 0, None))
np.fill_diagonal(kc, 0)
thu_tu = leaves_list(linkage(squareform(kc, checks=False), method="single"))
Cs = pd.DataFrame(c0, index=C.index, columns=C.columns).iloc[thu_tu, thu_tu]
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(Cs.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(Cs)))
ax.set_yticks(range(len(Cs)))
ax.set_xticklabels(Cs.columns, rotation=90, fontsize=4)
ax.set_yticklabels(Cs.index, fontsize=4)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Hệ số tương quan")
fig.tight_layout()
fig.savefig(KQ / "hinh_6_ma_tran_tuong_quan_sap_xep.png")
plt.close(fig)
pd.Series(Cs.columns, name="ma").to_csv(KQ / "thu_tu_ma_theo_gom_cum_phan_cap.csv", index=False,
                                        encoding="utf-8-sig")


# %% 8. Ghi báo cáo
(KQ / "bao_cao_eda.txt").write_text("\n".join(BAO_CAO), encoding="utf-8")
print(f"\nĐã ghi {len(list(KQ.iterdir()))} tệp vào {KQ}")
