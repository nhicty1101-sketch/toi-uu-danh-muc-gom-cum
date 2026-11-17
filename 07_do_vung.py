# -*- coding: utf-8 -*-
"""
BƯỚC 6 – KIỂM ĐỊNH ĐỘ VỮNG
Tiểu luận: Tối ưu hóa danh mục đầu tư bằng gom cụm và học máy (HOSE, 2019–2025)

Chạy lại sáu chiến lược của 06_kiem_tra_ngoai_mau.py (S1 VN-Index, S2 1/N, S3 Markowitz, S4 K-Means + Markowitz,
S4đc đối chứng không gom cụm, S5 HRP) dưới các giả định khác, mỗi lần đổi đúng một yếu tố so với bản chính:
  A. Chi phí và dữ liệu (cùng giai đoạn bản chính 06/01/2021–31/12/2025):
     rf = 0; phí 0% và 0,4% mỗi chiều; vị thế 5 mã bị đình chỉ rồi hủy niêm yết mất trắng.
  B. Độ dài cửa sổ ước lượng 252 / 504 / 756 phiên, CÙNG giao dịch từ phiên thứ 757 (10/01/2022) để so sánh sạch
     (người dùng chọn 16/11/2026); bản 504 phiên ở đây chính là kỳ 5–20 của bản chính.
  C. Tần suất tái cân bằng: mỗi 21 phiên (tháng), 63 phiên (bản chính), 126 phiên (nửa năm).
  D. Cách ước lượng và chọn mã: hiệp phương sai co Ledoit–Wolf; K-Means bốn đặc trưng; S4 và S4đc chỉ chọn mã
     có Sharpe > 0; loại khỏi rổ các mã có trên 15% số phiên giá đứng yên trong cửa sổ.
Kèm theo: bảng đóng góp lợi suất theo mã của S4 và S4đc từng năm (bản chính).

Số cụm k giữ như bản chính (đọc từ ket_qua/gom_cum/chon_k_ky_1.csv). Bản chính chạy lại ở đây phải khớp 06.
Đầu ra: ket_qua/do_vung/ (bảng .csv, hình 15–16, bao_cao_do_vung.txt)
Chạy từ thư mục gốc dự án:  .venv/bin/python 07_do_vung.py   (mất vài phút)
"""

# %% 0. Cấu hình
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cong_cu import (BUOC, CUA_SO, GOC, PHI_GIAO_DICH, PHIEN_NAM, RAW, SO_MA_MOI_CUM, TRAN_S3, TRAN_S4,
                     bon_dac_trung, chi_tieu, doc_gia, doc_ngay_cuoi_hose, doc_rf, doc_ro, du_dieu_kien, hai_truc,
                     hrp, kiem_dinh_sharpe, kmeans_co_thu_tu, loi_suat, mo_phong, phuong_sai_nho_nhat,
                     sharpe_cua_so, tinh_dac_trung)

KQ = GOC / "ket_qua" / "do_vung"
KQ.mkdir(parents=True, exist_ok=True)
K_MIN, K_MAX = 3, 8
NGUONG_DUNG_GIA = 0.15
BAT_DAU_CHUNG = 756          # phiên giao dịch đầu tiên khi so độ dài cửa sổ (cửa sổ dài nhất 756 phiên)
CHIEN_LUOC = {"S1": "S1 VN-Index", "S2": "S2 1/N", "S3": "S3 Markowitz", "S4": "S4 K-Means + Markowitz",
              "S4dc": "S4đc Top-8 Sharpe (không gom cụm)", "S5": "S5 HRP"}
DS_CL = ["S2", "S3", "S4", "S4dc", "S5"]          # các chiến lược có danh mục (S1 là chỉ số)
CAP_SO_SANH = [("S4", "S1"), ("S4", "S2"), ("S4", "S4dc"), ("S5", "S2"), ("S3", "S2")]
MAU = {"S1": "#7F7F7F", "S2": "#4C78A8", "S3": "#F58518", "S4": "#54A24B", "S4dc": "#B279A2", "S5": "#E45756"}
DAU = {"S1": "s", "S2": "o", "S3": "D", "S4": "*", "S4dc": "P", "S5": "^"}

BAO_CAO = []


def ghi(dong=""):
    print(dong)
    BAO_CAO.append(str(dong))


def pt(x, so_le=1):
    return "" if pd.isna(x) else f"{x * 100:.{so_le}f}%".replace(".", ",")


def so(x, so_le=2):
    return "" if pd.isna(x) else f"{x:.{so_le}f}".replace(".", ",")


def ngan(s):
    return CHIEN_LUOC[s].split()[0]


@dataclass(frozen=True)
class CauHinh:
    ten: str
    mo_ta: str
    nhom: str
    cua_so: int = CUA_SO
    buoc: int = BUOC
    bat_dau: int = CUA_SO            # vị trí phiên giao dịch đầu tiên trong lịch
    phi: float = PHI_GIAO_DICH
    rf0: bool = False
    mat_trang: bool = False
    hiep_phuong_sai: str = "mau"     # "mau" hoặc "ledoit_wolf"
    dac_trung: str = "hai_truc"      # "hai_truc" hoặc "bon"
    sharpe_duong: bool = False
    loc_dung_gia: bool = False


BAN_CHINH = CauHinh("ban_chinh", "Bản chính (cửa sổ 504, mỗi 63 phiên, phí 0,2%)", "A")
BIEN_THE = [
    BAN_CHINH,
    replace(BAN_CHINH, ten="rf0", mo_ta="rf = 0", rf0=True),
    replace(BAN_CHINH, ten="phi0", mo_ta="Phí 0%", phi=0.0),
    replace(BAN_CHINH, ten="phi04", mo_ta="Phí 0,4% mỗi chiều", phi=0.004),
    replace(BAN_CHINH, ten="mat_trang", mo_ta="Mất trắng 5 mã bị đình chỉ", mat_trang=True),
    CauHinh("cs252", "Cửa sổ 252 phiên", "B", cua_so=252, bat_dau=BAT_DAU_CHUNG),
    CauHinh("cs504", "Cửa sổ 504 phiên (= kỳ 5–20 bản chính)", "B", bat_dau=BAT_DAU_CHUNG),
    CauHinh("cs756", "Cửa sổ 756 phiên", "B", cua_so=756, bat_dau=BAT_DAU_CHUNG),
    CauHinh("thang", "Tái cân bằng mỗi 21 phiên", "C", buoc=21),
    replace(BAN_CHINH, ten="quy", mo_ta="Tái cân bằng mỗi 63 phiên (bản chính)", nhom="C"),
    CauHinh("nua_nam", "Tái cân bằng mỗi 126 phiên", "C", buoc=126),
    CauHinh("ledoit_wolf", "Hiệp phương sai co Ledoit–Wolf", "D", hiep_phuong_sai="ledoit_wolf"),
    CauHinh("kmeans4", "K-Means bốn đặc trưng", "D", dac_trung="bon"),
    CauHinh("sharpe_duong", "S4, S4đc chỉ chọn mã Sharpe > 0", "D", sharpe_duong=True),
    CauHinh("loc_dung_gia", "Loại mã đứng giá > 15% số phiên", "D", loc_dung_gia=True),
]
TEN_NHOM = {"A": "A. Chi phí và dữ liệu (06/01/2021–31/12/2025)",
            "B": "B. Độ dài cửa sổ (cùng giai đoạn từ phiên thứ 757)",
            "C": "C. Tần suất tái cân bằng (06/01/2021–31/12/2025)",
            "D": "D. Cách ước lượng và chọn mã (06/01/2021–31/12/2025)"}

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


# %% 1. Dữ liệu
DS_MA = doc_ro()
gia = doc_gia(DS_MA, ghi)
LICH = gia.index
NGAY_CUOI = doc_ngay_cuoi_hose(DS_MA, ghi)
gia_ff, r = loi_suat(gia)
gia_cp, r_cp, r_vni, gia_ff_cp = gia[DS_MA], r[DS_MA], r["VNINDEX"], gia_ff[DS_MA]
RF = doc_rf(LICH, ghi)
RF0 = RF * 0.0
tep_k = GOC / "ket_qua" / "gom_cum" / "chon_k_ky_1.csv"
if not tep_k.exists():
    raise SystemExit("Chưa có ket_qua/gom_cum/chon_k_ky_1.csv: chạy 05_gom_cum.py trước.")
K = int(pd.read_csv(tep_k, index_col="k").loc[K_MIN:K_MAX, "silhouette"].idxmax())
lien_tiep = gia_cp.notna() & gia_cp.shift().notna()
dung_yen = (gia_cp == gia_cp.shift()) & lien_tiep
R = r_cp.fillna(0.0)
ghi(f"{len(LICH)} phiên, {len(DS_MA)} mã, k = {K}; {len(BIEN_THE)} cấu hình")
ghi(f"Giai đoạn so độ dài cửa sổ: giao dịch từ {LICH[BAT_DAU_CHUNG]:%d/%m/%Y}")

# Mất trắng: phiên ngay sau ngày giao dịch cuối trên HOSE, vị thế của mã bị đình chỉ mất toàn bộ giá trị
R_MT = R.copy()
hnl = pd.read_csv(RAW / "ngay_huy_niem_yet.csv", encoding="utf-8-sig", dtype={"ma": str},
                  parse_dates=["ngay_gd_cuoi_hose"])
DINH_CHI = hnl[(hnl["nhom"] == "dinh_chi") & hnl["ma"].isin(DS_MA)]
for _, d in DINH_CHI.iterrows():
    vt = LICH.searchsorted(d["ngay_gd_cuoi_hose"], side="right")
    if vt < len(LICH):
        R_MT.iat[vt, R_MT.columns.get_loc(d["ma"])] = -1.0
        ghi(f"  Mất trắng {d['ma']}: giao dịch cuối {d['ngay_gd_cuoi_hose']:%d/%m/%Y}, ghi lỗ 100% ở phiên "
            f"{LICH[vt]:%d/%m/%Y}")


# %% 2. Tỷ trọng mục tiêu ở mỗi phiên giao dịch (có nhớ để các cấu hình dùng lại)
NHO = {}


def ty_trong_ky(ch, k):
    khoa = (ch.cua_so, ch.rf0, ch.hiep_phuong_sai, ch.dac_trung, ch.sharpe_duong, ch.loc_dung_gia, k)
    if khoa in NHO:
        return NHO[khoa]
    cs = ch.cua_so
    rf = RF0 if ch.rf0 else RF
    dk = du_dieu_kien(gia_cp, k, NGAY_CUOI, cs)
    ma = list(dk.index[dk["du"]])
    if ch.loc_dung_gia:
        ty_dung = dung_yen.iloc[k - cs:k][ma].sum() / lien_tiep.iloc[k - cs:k][ma].sum()
        ma = [m for m in ma if ty_dung[m] <= NGUONG_DUNG_GIA]
    rr = r_cp.iloc[k - cs:k][ma]
    rr0 = rr.fillna(0.0)
    if ch.hiep_phuong_sai == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        S = LedoitWolf().fit(rr0.values).covariance_ * PHIEN_NAM
        sd = np.sqrt(np.diag(S))
        C = S / np.outer(sd, sd)
    else:
        S = rr0.cov().values * PHIEN_NAM
        C = rr0.corr().values

    w = {"S2": pd.Series(1.0 / len(ma), index=ma),
         "S3": pd.Series(phuong_sai_nho_nhat(S, TRAN_S3), index=ma),
         "S5": pd.Series(hrp(S, C), index=ma)}
    dt = tinh_dac_trung(r_cp, r_vni, gia_ff_cp, k, ma, cs)
    if ch.dac_trung == "bon":
        nhan, _, _ = kmeans_co_thu_tu(bon_dac_trung(dt), K, ["bd_nam", "beta", "sgtd"])
    else:
        nhan, _, _ = kmeans_co_thu_tu(hai_truc(dt), K, "truc_rui_ro")
    sh = sharpe_cua_so(rr, rf.iloc[k - cs:k])
    ung_vien = sh[sh > 0] if ch.sharpe_duong else sh
    chon = []
    for cu in range(1, K + 1):
        nhom = [m for m in nhan.index[nhan == cu] if m in ung_vien.index]
        chon += list(ung_vien[nhom].sort_values(ascending=False).head(SO_MA_MOI_CUM).index)
    chon_dc = list(ung_vien.sort_values(ascending=False).head(len(chon)).index)

    def toi_uu(ds):
        if not ds:
            return pd.Series(dtype=float)       # không có mã nào đạt: giữ tiền mặt, lợi suất 0
        idx = [ma.index(m) for m in ds]
        return pd.Series(phuong_sai_nho_nhat(S[np.ix_(idx, idx)], TRAN_S4), index=ds)

    w["S4"], w["S4dc"] = toi_uu(chon), toi_uu(chon_dc)
    NHO[khoa] = (w, chon, len(ma))
    return NHO[khoa]


# %% 3. Chạy từng cấu hình
CHI_TIEU, KIEM_DINH, CHON_S4, SO_MA_RO, VONG_QUAY = [], [], {}, {}, {}
DONG_GOP, LS_GOC = [], None
bat_dau_dong_ho = time.time()
for ch in BIEN_THE:
    VT = list(range(ch.bat_dau, len(LICH), ch.buoc))
    theo_ky = {s: {} for s in DS_CL}
    CHON_S4[ch.ten], SO_MA_RO[ch.ten] = {}, []
    for k in VT:
        w, chon, n_ma = ty_trong_ky(ch, k)
        for s in DS_CL:
            theo_ky[s][k] = w[s]
        CHON_S4[ch.ten][k] = chon
        SO_MA_RO[ch.ten].append(n_ma)
    R_dung = R_MT if ch.mat_trang else R
    LS = {"S1": r_vni.iloc[ch.bat_dau:].copy()}
    LS["S1"].iloc[0] = 0.0
    vq = {}
    for s in DS_CL:
        lay_dong_gop = ch.ten == "ban_chinh" and s in ("S4", "S4dc")
        LS[s], vq[s], dg = mo_phong(theo_ky[s], R_dung, ch.bat_dau, ch.phi, dong_gop=lay_dong_gop)
        if dg is not None:
            DONG_GOP.append(dg.assign(chien_luoc=ngan(s)))
    LS = pd.DataFrame(LS)
    if ch.ten == "ban_chinh":
        LS_GOC = LS
    rf_kt = (RF0 if ch.rf0 else RF).iloc[ch.bat_dau:]
    for s in LS.columns:
        ct = chi_tieu(LS[s], rf_kt)
        ct["vong_quay_hai_chieu_bq"] = vq[s]["vong_quay_hai_chieu"].iloc[1:].mean() if s in vq else np.nan
        ct["tong_phi"] = vq[s]["phi"].sum() if s in vq else np.nan
        ct["so_ky_tien_mat"] = int(sum(len(theo_ky[s][k]) == 0 for k in VT)) if s in theo_ky else 0
        CHI_TIEU.append(pd.concat([pd.Series({"bien_the": ch.ten, "nhom": ch.nhom, "mo_ta": ch.mo_ta,
                                              "chien_luoc": ngan(s), "so_ky": len(VT),
                                              "tu_ngay": LS.index[0].date(), "den_ngay": LS.index[-1].date()}), ct]))
    vuot = LS.sub(rf_kt, axis=0)
    for a, b in CAP_SO_SANH:
        chenh, ss, z, p = kiem_dinh_sharpe(vuot[a].values, vuot[b].values)
        KIEM_DINH.append({"bien_the": ch.ten, "nhom": ch.nhom, "cap": f"{ngan(a)} − {ngan(b)}",
                          "chenh_lech_sharpe": chenh, "sai_so_chuan": ss, "z": z, "p_value": p})
    print(f"  xong {ch.ten:13s} ({len(VT)} kỳ, {time.time() - bat_dau_dong_ho:.0f} giây)")

CT = pd.DataFrame(CHI_TIEU)
KD = pd.DataFrame(KIEM_DINH)
CT.to_csv(KQ / "chi_tieu_theo_bien_the.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
KD.to_csv(KQ / "kiem_dinh_sharpe_theo_bien_the.csv", index=False, encoding="utf-8-sig", float_format="%.4f")


# %% 4. Đối chiếu bản chính với 06
ghi("\n=== KIỂM TRA: BẢN CHÍNH CHẠY LẠI SO VỚI 06 ===")
tep06 = GOC / "ket_qua" / "kiem_tra" / "chi_tieu_ca_giai_doan.csv"
if tep06.exists():
    b06 = pd.read_csv(tep06, index_col=0)
    b07 = CT[CT["bien_the"] == "ban_chinh"].set_index("chien_luoc")
    b06.index = [ten.split()[0] for ten in b06.index]
    cot = ["loi_suat_nam_hoa", "do_bien_dong_nam_hoa", "sharpe", "sut_giam_toi_da"]
    lech = (b07.loc[b06.index, cot].astype(float) - b06[cot]).abs().max().max()
    ghi(f"Chênh lệch lớn nhất so với ket_qua/kiem_tra/chi_tieu_ca_giai_doan.csv: {lech:.6f} "
        f"({'khớp; 06 lưu 4 chữ số thập phân' if lech < 1e-3 else 'KHÔNG KHỚP – chạy lại 06 rồi chạy lại 07'})")
else:
    ghi("Không có ket_qua/kiem_tra/chi_tieu_ca_giai_doan.csv để đối chiếu (chạy 06 trước).")


# %% 5. Bảng tóm tắt
def bang_rong(cot, dinh_dang):
    t = CT.pivot(index="bien_the", columns="chien_luoc", values=cot)
    t = t.reindex(index=[ch.ten for ch in BIEN_THE], columns=[ngan(s) for s in ["S1"] + DS_CL]).astype(float)
    return t, t.map(dinh_dang)


def in_theo_nhom(tieu_de, hien):
    ghi(f"\n{tieu_de}")
    for nhom, ten_nhom in TEN_NHOM.items():
        ds = [ch for ch in BIEN_THE if ch.nhom == nhom]
        ghi(f"  {ten_nhom}")
        for ch in ds:
            ghi(f"    {ch.mo_ta:48s} " + "  ".join(f"{c} {hien.at[ch.ten, c]:>7s}" for c in hien.columns))


ghi("\n=== KẾT QUẢ THEO CẤU HÌNH ===")
for ch in BIEN_THE:
    d = CT[CT["bien_the"] == ch.ten].iloc[0]
    if ch.ten in ("ban_chinh", "cs252", "thang", "nua_nam"):
        ghi(f"{ch.mo_ta}: {d['so_ky']} kỳ, {pd.Timestamp(d['tu_ngay']):%d/%m/%Y}–{pd.Timestamp(d['den_ngay']):%d/%m/%Y}")
sharpe, sharpe_hien = bang_rong("sharpe", so)
in_theo_nhom("Sharpe:", sharpe_hien)
_, ln_hien = bang_rong("loi_suat_nam_hoa", pt)
in_theo_nhom("Lợi suất năm hóa:", ln_hien)
_, sg_hien = bang_rong("sut_giam_toi_da", pt)
in_theo_nhom("Sụt giảm tối đa:", sg_hien)
_, vq_hien = bang_rong("vong_quay_hai_chieu_bq", pt)
in_theo_nhom("Vòng quay hai chiều bình quân mỗi kỳ:", vq_hien.drop(columns="S1"))
sharpe.to_csv(KQ / "sharpe_theo_bien_the.csv", encoding="utf-8-sig", float_format="%.4f")

ghi("\nThứ hạng Sharpe của S4 trong 6 chiến lược, và chiến lược có Sharpe cao nhất:")
for ch in BIEN_THE:
    hang = sharpe.loc[ch.ten].rank(ascending=False)
    ghi(f"    {ch.mo_ta:48s} S4 hạng {hang['S4']:.0f}/6; cao nhất: {sharpe.loc[ch.ten].idxmax()}")

ghi("\nKiểm định chênh lệch Sharpe (Ledoit & Wolf, 2008, HAC): chênh lệch năm hóa (p-value); ** p < 0,05; * p < 0,10")
for nhom, ten_nhom in TEN_NHOM.items():
    ghi(f"  {ten_nhom}")
    for ch in [c for c in BIEN_THE if c.nhom == nhom]:
        d = KD[KD["bien_the"] == ch.ten]
        ghi(f"    {ch.mo_ta:48s} " + "; ".join(
            f"{x.cap} {so(x.chenh_lech_sharpe)} ({so(x.p_value, 2)}){'**' if x.p_value < 0.05 else '*' if x.p_value < 0.10 else ''}"
            for x in d.itertuples()))

# Ghi chú riêng từng cấu hình
ghi("\n=== GHI CHÚ RIÊNG ===")
goc_vt = sorted(CHON_S4["ban_chinh"])
doi = [len(set(CHON_S4["ban_chinh"][k]) - set(CHON_S4["rf0"][k])) for k in goc_vt]
pd.DataFrame({"ky": range(1, len(goc_vt) + 1), "ngay_giao_dich": [LICH[k].date() for k in goc_vt],
              "ma_s4_ban_chinh": [" ".join(CHON_S4["ban_chinh"][k]) for k in goc_vt],
              "ma_s4_rf0": [" ".join(CHON_S4["rf0"][k]) for k in goc_vt], "so_ma_doi": doi}).to_csv(
    KQ / "rf0_ma_s4_doi.csv", index=False, encoding="utf-8-sig")
ghi(f"rf = 0: số mã S4 đổi so với bản chính: tổng {sum(doi)} lượt trên {len(goc_vt)} kỳ, "
    f"{sum(x > 0 for x in doi)} kỳ có đổi (tối đa {max(doi)} mã/kỳ)")
for ten in ("sharpe_duong",):
    so_ma = [len(CHON_S4[ten][k]) for k in sorted(CHON_S4[ten])]
    tm = CT[(CT["bien_the"] == ten) & (CT["chien_luoc"] == "S4")]["so_ky_tien_mat"].iloc[0]
    ghi(f"Sharpe > 0: S4 giữ {min(so_ma)}–{max(so_ma)} mã (trung vị {np.median(so_ma):.0f}); "
        f"{int(tm)} kỳ không có mã nào đạt (giữ tiền mặt)")
so_ma_loc = SO_MA_RO["loc_dung_gia"]
ghi(f"Loại mã đứng giá: rổ còn {min(so_ma_loc)}–{max(so_ma_loc)} mã mỗi kỳ (bản chính "
    f"{min(SO_MA_RO['ban_chinh'])}–{max(SO_MA_RO['ban_chinh'])})")
mt = CT[CT["bien_the"].isin(["ban_chinh", "mat_trang"])].pivot(index="chien_luoc", columns="bien_the",
                                                                 values="loi_suat_tich_luy")
ghi("Mất trắng: lợi suất tích lũy bản chính → mất trắng: " + "; ".join(
    f"{c} {pt(mt.at[c, 'ban_chinh'])} → {pt(mt.at[c, 'mat_trang'])}" for c in [ngan(s) for s in DS_CL]))


# %% 6. Đóng góp lợi suất theo mã (bản chính, S4 và S4đc)
ghi("\n=== ĐÓNG GÓP LỢI SUẤT THEO MÃ (bản chính) ===")
DG = pd.concat(DONG_GOP).groupby(["chien_luoc", "nam", "ma"], as_index=False)["dong_gop"].sum()
DG.to_csv(KQ / "dong_gop_theo_ma.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
nam_ls = LS_GOC.groupby(LS_GOC.index.year).apply(lambda d: (1 + d).prod() - 1)
for cl in ["S4", "S4đc"]:
    s_key = "S4" if cl == "S4" else "S4dc"
    for nam in sorted(DG["nam"].unique()):
        d = DG[(DG["chien_luoc"] == cl) & (DG["nam"] == nam)]
        tong = d["dong_gop"].sum()
        assert abs(tong - nam_ls.at[nam, s_key]) < 1e-9, "tổng đóng góp phải bằng lợi suất năm"
        ma_d = d[d["ma"] != "(phí giao dịch)"].sort_values("dong_gop", ascending=False)
        top = ma_d.head(3)
        ghi(f"  {cl} {nam}: lợi suất {pt(tong)}; 3 mã đóng góp lớn nhất "
            + ", ".join(f"{x.ma} {pt(x.dong_gop)}" for x in top.itertuples())
            + f"; mã kéo lùi nhiều nhất {ma_d.iloc[-1]['ma']} {pt(ma_d.iloc[-1]['dong_gop'])}"
            + f"; bỏ mã đóng góp lớn nhất còn {pt(tong - top.iloc[0]['dong_gop'])}")
ghi("Ghi chú: đóng góp = lãi/lỗ của mã trong năm chia cho giá trị danh mục đầu năm; tổng các mã trừ phí đúng bằng "
    "lợi suất năm. \"Bỏ mã đóng góp lớn nhất\" là phép trừ đơn giản, không mô phỏng lại danh mục.")


# %% 7. Hình
fig, ax = plt.subplots(figsize=(7.5, 6.2))
nhan_y, y, vi_tri = [], 0, {}
for nhom in TEN_NHOM:
    ds = [ch for ch in BIEN_THE if ch.nhom == nhom]
    for ch in ds:
        vi_tri[ch.ten] = y
        nhan_y.append(ch.mo_ta)
        y += 1
    if nhom != list(TEN_NHOM)[-1]:
        ax.axhline(y - 0.5, color="#DDDDDD", lw=0.6)
for s in ["S1"] + DS_CL:
    c = ngan(s)
    ax.scatter(sharpe[c].values, [vi_tri[t] for t in sharpe.index], marker=DAU[s], color=MAU[s],
               s=60 if s == "S4" else 28, label=CHIEN_LUOC[s], zorder=3)
ax.set_yticks(range(len(nhan_y)), nhan_y, fontsize=7.5)
ax.invert_yaxis()
ax.axvline(0, color="black", lw=0.5)
ax.set_xlabel("Sharpe năm hóa")
ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.4, 1.0))
fig.tight_layout()
fig.savefig(KQ / "hinh_15_sharpe_theo_bien_the.png")
plt.close(fig)

nam_ve = [n for n in [2021, 2024] if n in DG["nam"].unique()]
fig, axes = plt.subplots(len(nam_ve), 2, figsize=(7.5, 2.6 * len(nam_ve)), squeeze=False)
for i, nam in enumerate(nam_ve):
    for j, (cl, s) in enumerate([("S4", "S4"), ("S4đc", "S4dc")]):
        d = DG[(DG["chien_luoc"] == cl) & (DG["nam"] == nam) & (DG["ma"] != "(phí giao dịch)")]
        d = pd.concat([d.nlargest(5, "dong_gop"), d.nsmallest(2, "dong_gop")]).drop_duplicates("ma")
        d = d.sort_values("dong_gop")
        ax = axes[i, j]
        ax.barh(d["ma"], d["dong_gop"] * 100, color=[MAU[s] if v >= 0 else "#BBBBBB" for v in d["dong_gop"]])
        ax.axvline(0, color="black", lw=0.5)
        ax.set_title(f"{CHIEN_LUOC[s]}, năm {nam}: {pt(nam_ls.at[nam, s])}", fontsize=8)
        ax.set_xlabel("Đóng góp vào lợi suất năm (điểm %)", fontsize=7.5)
fig.tight_layout()
fig.savefig(KQ / "hinh_16_dong_gop_theo_ma.png")
plt.close(fig)


# %% 8. Ghi báo cáo
(KQ / "bao_cao_do_vung.txt").write_text("\n".join(BAO_CAO), encoding="utf-8")
print(f"\nĐã ghi {len(list(KQ.iterdir()))} tệp vào {KQ} ({time.time() - bat_dau_dong_ho:.0f} giây)")
