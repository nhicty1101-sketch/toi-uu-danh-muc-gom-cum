# -*- coding: utf-8 -*-
"""
Các hàm dùng chung cho bước gom cụm (05) và kiểm tra ngoài mẫu (06).

Mọi quy ước đã chốt nằm ở đây để hai script dùng giống hệt nhau:
  - lịch tái cân bằng: cửa sổ 504 phiên, tái cân bằng mỗi 63 phiên;
  - điều kiện chọn mã ở mỗi kỳ: thiếu không quá 10% cửa sổ, còn giao dịch trên HOSE sau ngày quyết định
    (quy tắc B về mã hủy niêm yết, chốt 13/11/2026), có giá ở ngày quyết định (thêm 16/11/2026);
  - bốn đặc trưng: lợi suất năm hóa, độ biến động năm hóa, beta với VN-Index, sụt giảm tối đa;
  - hai trục cho K-Means (chốt 16/11/2026): lợi suất và chỉ số rủi ro tổng hợp.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(".")
PROC = GOC / "data" / "processed"
RAW = GOC / "data" / "raw"

TU_NGAY, DEN_NGAY = "2019-01-01", "2025-12-31"
CUA_SO = 504                 # số phiên của cửa sổ ước lượng
BUOC = 63                    # số phiên giữa hai lần tái cân bằng
NGUONG_THIEU = 0.10          # thiếu quá 10% số phiên trong cửa sổ thì không được chọn
PHIEN_NAM = 252
CAT_DUOI = (0.05, 0.95)      # cắt đuôi từng đặc trưng ở phân vị 5% và 95% trong mỗi kỳ
HAT_GIONG = 2026             # hạt giống ngẫu nhiên cho K-Means, để chạy lại ra cùng kết quả
TEN_DAC_TRUNG = {"ln_nam": "Lợi suất năm hóa", "bd_nam": "Độ biến động năm hóa",
                 "beta": "Beta với VN-Index", "sgtd": "Sụt giảm tối đa"}


def doc_ro():
    ro = pd.read_csv(PROC / "ro_100_ma.csv", encoding="utf-8-sig", dtype={"ma": str})
    return ro.sort_values("hang")["ma"].tolist()


def doc_gia(ds_ma, ghi=print, doc_tu_mysql=True):
    """Bảng rộng giá đóng cửa đã điều chỉnh: dòng là phiên VN-Index, cột là các mã và VNINDEX.
    Đọc bảng gia_ngay trong MySQL nếu kết nối được, nếu không thì đọc data/processed/gia_dong_cua.csv."""
    if doc_tu_mysql:
        try:
            import mysql.connector
            from dotenv import load_dotenv
            load_dotenv()
            kn = mysql.connector.connect(host=os.environ.get("MYSQL_HOST", "localhost"),
                                         user=os.environ["MYSQL_USER"],
                                         password=os.environ["MYSQL_PASSWORD"],
                                         database=os.environ.get("MYSQL_DB", "tieu_luan_khdl"))
            con_tro = kn.cursor()
            con_tro.execute("SELECT ma, ngay, dong_cua_dc FROM gia_ngay WHERE ngay BETWEEN %s AND %s",
                            (TU_NGAY, DEN_NGAY))
            dai = pd.DataFrame(con_tro.fetchall(), columns=["ma", "ngay", "dong_cua_dc"])
            con_tro.close()
            kn.close()
            dai["ngay"] = pd.to_datetime(dai["ngay"])
            dai["dong_cua_dc"] = pd.to_numeric(dai["dong_cua_dc"].astype(object), errors="coerce").astype(float)
            lich = pd.DatetimeIndex(sorted(dai.loc[dai["ma"] == "VNINDEX", "ngay"].unique()))
            g = dai.pivot(index="ngay", columns="ma", values="dong_cua_dc").reindex(
                index=lich, columns=ds_ma + ["VNINDEX"])
            g.index.name = "ngay"
            ghi(f"Đọc giá từ MySQL (bảng gia_ngay): {len(lich)} phiên")
            return g
        except Exception as e:
            ghi(f"Không đọc được MySQL ({type(e).__name__}: {e}). Dùng tệp gia_dong_cua.csv.")
    tep = PROC / "gia_dong_cua.csv"
    if not tep.exists():
        raise SystemExit("Không có dữ liệu: chạy 01_lay_du_lieu_cafef.py trước.")
    g = pd.read_csv(tep, index_col="ngay", parse_dates=["ngay"])
    g.columns = g.columns.astype(str)
    ghi(f"Đọc giá từ {tep}: {len(g)} phiên")
    return g[ds_ma + ["VNINDEX"]]


def doc_ngay_cuoi_hose(ds_ma, ghi=print):
    """Ngày giao dịch cuối trên HOSE của các mã hủy niêm yết; NaT nghĩa là còn niêm yết đến cuối kỳ."""
    s = pd.Series(pd.NaT, index=ds_ma, dtype="datetime64[ns]")
    tep = RAW / "ngay_huy_niem_yet.csv"
    if not tep.exists():
        ghi("CẢNH BÁO: không có data/raw/ngay_huy_niem_yet.csv; coi như mọi mã còn niêm yết.")
        return s
    h = pd.read_csv(tep, encoding="utf-8-sig", dtype={"ma": str}, parse_dates=["ngay_gd_cuoi_hose"])
    h = h[h["ma"].isin(ds_ma)].set_index("ma")
    s.loc[h.index] = h["ngay_gd_cuoi_hose"].values
    return s


def loi_suat(gia):
    """Giữ giá phiên trước cho phiên trống (chỉ đến phiên cuối có giá), rồi tính lợi suất giản đơn."""
    gia_ff = gia.ffill().where(gia.bfill().notna())
    return gia_ff, gia_ff.pct_change(fill_method=None)


def lich_tai_can_bang(lich):
    vt = list(range(CUA_SO, len(lich), BUOC))
    bang = pd.DataFrame({
        "ky": np.arange(1, len(vt) + 1),
        "vi_tri": vt,
        "ngay_quyet_dinh": [lich[k - 1] for k in vt],
        "phien_dau_nam_giu": [lich[k] for k in vt],
        "phien_cuoi_nam_giu": [lich[min(k + BUOC, len(lich)) - 1] for k in vt],
        "so_phien_nam_giu": [min(k + BUOC, len(lich)) - k for k in vt],
    })
    return vt, bang


def du_dieu_kien(gia_cp, k, ngay_cuoi_hose):
    """Mã được chọn ở kỳ bắt đầu từ phiên k (quyết định bằng dữ liệu đến phiên k - 1)."""
    lich = gia_cp.index
    cua_so = gia_cp.iloc[k - CUA_SO:k]
    thieu = cua_so.isna().mean()
    con_niem_yet = ngay_cuoi_hose.isna() | (ngay_cuoi_hose > lich[k - 1])
    co_gia = cua_so.iloc[-1].notna()
    return pd.DataFrame({"ty_le_thieu": thieu, "con_niem_yet": con_niem_yet, "co_gia_ngay_quyet_dinh": co_gia,
                         "du": (thieu <= NGUONG_THIEU) & con_niem_yet & co_gia})


def tinh_dac_trung(r_cp, r_vni, gia_ff_cp, k, ds_ma):
    """Bốn đặc trưng của các mã trong ds_ma, tính trên CUA_SO phiên ngay trước phiên k."""
    rr = r_cp.iloc[k - CUA_SO:k][ds_ma]
    rm = r_vni.iloc[k - CUA_SO:k]
    p = gia_ff_cp.iloc[k - CUA_SO:k][ds_ma]
    beta = {m: rr[m].cov(rm) / rm[rr[m].notna()].var() for m in ds_ma}
    return pd.DataFrame({
        "ln_nam": rr.mean() * PHIEN_NAM,
        "bd_nam": rr.std() * np.sqrt(PHIEN_NAM),
        "beta": pd.Series(beta),
        "sgtd": -(p / p.cummax() - 1).min(),
    })


def _z(x):
    return (x - x.mean()) / x.std(ddof=0)


def cat_duoi(dt):
    """Cắt đuôi từng cột ở phân vị CAT_DUOI, tính trong chính kỳ đó (không dùng dữ liệu kỳ khác)."""
    duoi, tren = dt.quantile(CAT_DUOI[0]), dt.quantile(CAT_DUOI[1])
    return dt.clip(lower=duoi, upper=tren, axis=1)


def hai_truc(dt):
    """Trục lợi suất và trục rủi ro tổng hợp, mỗi trục có trung bình 0, độ lệch chuẩn 1.
    Rủi ro tổng hợp = trung bình z-score của độ biến động, beta, sụt giảm tối đa, rồi chuẩn hóa lại:
    trung bình của ba biến tương quan có phương sai nhỏ hơn 1, không chuẩn hóa lại thì trục lợi suất
    sẽ nặng ký hơn trong khoảng cách của K-Means."""
    w = cat_duoi(dt)
    rui_ro = (_z(w["bd_nam"]) + _z(w["beta"]) + _z(w["sgtd"])) / 3
    return pd.DataFrame({"truc_loi_suat": _z(w["ln_nam"]), "truc_rui_ro": _z(rui_ro)})


def bon_dac_trung(dt):
    """Phương án kiểm định độ vững: bốn đặc trưng cắt đuôi rồi z-score, như đề cương ban đầu."""
    return cat_duoi(dt).apply(_z)


def kmeans_co_thu_tu(X, k, cot_rui_ro):
    """K-Means rồi đánh số lại cụm theo mức rủi ro của tâm cụm, từ thấp đến cao (cụm 1 rủi ro thấp nhất),
    để số hiệu cụm có cùng nghĩa giữa các kỳ."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=50, random_state=HAT_GIONG).fit(X.values)
    tam = pd.DataFrame(km.cluster_centers_, columns=X.columns)
    diem_rr = tam[cot_rui_ro].mean(axis=1) if isinstance(cot_rui_ro, list) else tam[cot_rui_ro]
    thu_tu = {cu: moi + 1 for moi, cu in enumerate(diem_rr.sort_values().index)}
    nhan = pd.Series([thu_tu[c] for c in km.labels_], index=X.index, name="cum")
    tam.index = [thu_tu[c] for c in tam.index]
    return nhan, tam.sort_index(), km.inertia_
