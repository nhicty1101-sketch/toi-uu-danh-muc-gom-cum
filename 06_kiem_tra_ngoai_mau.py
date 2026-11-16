# -*- coding: utf-8 -*-
"""
BƯỚC 5 – KIỂM TRA NGOÀI MẪU (BẢN CHÍNH)
Tiểu luận: Tối ưu hóa danh mục đầu tư bằng gom cụm và học máy (HOSE, 2019–2025)

Năm chiến lược, cùng lịch 20 kỳ tái cân bằng (cửa sổ 504 phiên, mỗi kỳ 63 phiên):
  S1 VN-Index: mua và giữ chỉ số, không phí.
  S2 1/N: chia đều cho các mã đủ điều kiện.
  S3 Markowitz: phương sai nhỏ nhất, hiệp phương sai mẫu, không bán khống, trần 10% mỗi mã.
  S4 K-Means + Markowitz: gom 4 cụm trên hai trục (lợi suất, rủi ro tổng hợp), mỗi cụm lấy 2 mã có Sharpe
     cao nhất, rồi phương sai nhỏ nhất với trần 25% mỗi mã.
  S5 HRP: Hierarchical Risk Parity (López de Prado, 2016), liên kết đơn.

Quy ước đã chốt:
  - tỷ trọng tính bằng dữ liệu đến giá đóng cửa ngày quyết định; mua bán ở giá đóng cửa phiên kế tiếp;
  - trong kỳ mua và giữ (tỷ trọng tự trôi theo giá); phí 0,2% trên giá trị mua và trên giá trị bán;
  - mã rời HOSE giữa kỳ (quy tắc B): giữ đến kỳ sau; mã chuyển UPCoM theo giá UPCoM thật, mã bị đình chỉ
    hoặc sáp nhập giữ giá phiên cuối (lợi suất 0) rồi bán ở giá đó khi tái cân bằng;
  - rf theo phiên từ lợi suất TPCP 10 năm (data/processed/rf_ngay.csv).

Đầu vào: như 05_gom_cum.py, thêm data/processed/rf_ngay.csv và ket_qua/gom_cum/chon_k_ky_1.csv (số cụm k).
Đầu ra: ket_qua/kiem_tra/ (bảng .csv, hình 11–14, bao_cao_kiem_tra.txt)
Chạy từ thư mục gốc dự án:  .venv/bin/python 06_kiem_tra_ngoai_mau.py
"""

# %% 0. Cấu hình
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cong_cu import (CUA_SO, GOC, PHI_GIAO_DICH, PHIEN_NAM, SO_MA_MOI_CUM, TRAN_S3, TRAN_S4, doc_gia,
                     doc_ngay_cuoi_hose, doc_rf, doc_ro, du_dieu_kien, hai_truc, hrp, kmeans_co_thu_tu,
                     lich_tai_can_bang, loi_suat, phuong_sai_nho_nhat, sharpe_cua_so, tinh_dac_trung)

KQ = GOC / "ket_qua" / "kiem_tra"
KQ.mkdir(parents=True, exist_ok=True)
K_MIN, K_MAX = 3, 8
CHIEN_LUOC = {"S1": "S1 VN-Index", "S2": "S2 1/N", "S3": "S3 Markowitz", "S4": "S4 K-Means + Markowitz",
              "S5": "S5 HRP"}
GIAI_DOAN_CON = {"Kỳ 1–4": (1, 4), "Kỳ 5–8": (5, 8), "Kỳ 9–20": (9, 20)}
NGUONG_DUNG_GIA = 0.15       # mã có trên 15% phiên giá đứng yên trong cửa sổ: theo dõi riêng ở S3

BAO_CAO = []


def ghi(dong=""):
    print(dong)
    BAO_CAO.append(str(dong))


def pt(x, so_le=1):
    return f"{x * 100:.{so_le}f}%".replace(".", ",")


def so(x, so_le=2):
    return f"{x:.{so_le}f}".replace(".", ",")


plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
MAU = {"S1": "#7F7F7F", "S2": "#4C78A8", "S3": "#F58518", "S4": "#54A24B", "S5": "#E45756"}


# %% 1. Dữ liệu
DS_MA = doc_ro()
gia = doc_gia(DS_MA, ghi)
LICH = gia.index
NGAY_CUOI = doc_ngay_cuoi_hose(DS_MA, ghi)
gia_ff, r = loi_suat(gia)
gia_cp, r_cp, r_vni, gia_ff_cp = gia[DS_MA], r[DS_MA], r["VNINDEX"], gia_ff[DS_MA]
rf = doc_rf(LICH, ghi)
VT, lich_tcb = lich_tai_can_bang(LICH)
# Phiên giao dịch của kỳ = phiên ngay sau ngày quyết định = vị trí k trong lịch
tep_k = GOC / "ket_qua" / "gom_cum" / "chon_k_ky_1.csv"
if not tep_k.exists():
    raise SystemExit("Chưa có ket_qua/gom_cum/chon_k_ky_1.csv: chạy 05_gom_cum.py trước.")
K = int(pd.read_csv(tep_k, index_col="k").loc[K_MIN:K_MAX, "silhouette"].idxmax())
ghi(f"{len(LICH)} phiên, {len(DS_MA)} mã, {len(VT)} kỳ; k = {K} (từ 05_gom_cum.py); phí {pt(PHI_GIAO_DICH)} mỗi chiều")
ghi(f"Kỳ 1: quyết định bằng dữ liệu đến {LICH[VT[0] - 1]:%d/%m/%Y}, giao dịch ở giá đóng cửa {LICH[VT[0]]:%d/%m/%Y}; "
    f"lợi suất danh mục tính đến {LICH[-1]:%d/%m/%Y}")

# Phiên giá đứng yên (để theo dõi nghi vấn thanh khoản ở S3)
lien_tiep = gia_cp.notna() & gia_cp.shift().notna()
dung_yen = (gia_cp == gia_cp.shift()) & lien_tiep


# %% 2. Tỷ trọng mục tiêu của từng chiến lược ở mỗi kỳ
ghi("\n=== PHẦN 1. TỶ TRỌNG MỖI KỲ ===")
TY_TRONG = {s: {} for s in ["S2", "S3", "S4", "S5"]}
nhat_ky_s4, tap_trung = [], []
for i, k in enumerate(VT):
    ky = i + 1
    dk = du_dieu_kien(gia_cp, k, NGAY_CUOI)
    ma = list(dk.index[dk["du"]])
    rr = r_cp.iloc[k - CUA_SO:k][ma]
    rr0 = rr.fillna(0.0)                       # phiên trước khi niêm yết trong cửa sổ: lợi suất 0
    S = rr0.cov().values * PHIEN_NAM
    C = rr0.corr().values

    TY_TRONG["S2"][k] = pd.Series(1.0 / len(ma), index=ma)
    TY_TRONG["S3"][k] = pd.Series(phuong_sai_nho_nhat(S, TRAN_S3), index=ma)
    TY_TRONG["S5"][k] = pd.Series(hrp(S, C), index=ma)

    # S4: gom cụm hai trục, mỗi cụm lấy SO_MA_MOI_CUM mã Sharpe cao nhất, rồi phương sai nhỏ nhất
    dt = tinh_dac_trung(r_cp, r_vni, gia_ff_cp, k, ma)
    nhan, _, _ = kmeans_co_thu_tu(hai_truc(dt), K, "truc_rui_ro")
    sh = sharpe_cua_so(rr, rf.iloc[k - CUA_SO:k])
    chon = []
    for cu in range(1, K + 1):
        top = sh[nhan.index[nhan == cu]].sort_values(ascending=False).head(SO_MA_MOI_CUM)
        chon += list(top.index)
        nhat_ky_s4.append({"ky": ky, "cum": cu, "so_ma_cum": int((nhan == cu).sum()),
                           "ma_chon": " ".join(top.index), "sharpe": " ".join(so(v) for v in top.values),
                           "ln_nam_trung_vi_cum": dt.loc[nhan == cu, "ln_nam"].median()})
    idx = [ma.index(m) for m in chon]
    TY_TRONG["S4"][k] = pd.Series(phuong_sai_nho_nhat(S[np.ix_(idx, idx)], TRAN_S4), index=chon)

    # Mức tập trung và mức dồn vào mã ít biến động giá
    ty_dung = dung_yen.iloc[k - CUA_SO:k][ma].sum() / lien_tiep.iloc[k - CUA_SO:k][ma].sum()
    nghi_van = ty_dung[ty_dung > NGUONG_DUNG_GIA].index
    for s in ["S2", "S3", "S4", "S5"]:
        w = TY_TRONG[s][k]
        top5 = w.sort_values(ascending=False).head(5)
        tap_trung.append({"ky": ky, "chien_luoc": s, "so_ma": int((w > 1e-4).sum()),
                          "ty_trong_lon_nhat": w.max(), "hhi": float((w ** 2).sum()),
                          "ty_trong_ma_dung_gia_nhieu": float(w.reindex(nghi_van).fillna(0).sum()),
                          "top5": "; ".join(f"{m} {pt(v)} (đứng giá {pt(ty_dung[m], 0)})" for m, v in top5.items())})

tap_trung = pd.DataFrame(tap_trung)
tap_trung.to_csv(KQ / "tap_trung_ty_trong.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
pd.DataFrame(nhat_ky_s4).to_csv(KQ / "s4_ma_chon_theo_cum.csv", index=False, encoding="utf-8-sig",
                                float_format="%.4f")
dai = []
for s, theo_ky in TY_TRONG.items():
    for i, k in enumerate(VT):
        w = theo_ky[k]
        dai.append(pd.DataFrame({"ky": i + 1, "ngay_giao_dich": LICH[k], "chien_luoc": s, "ma": w.index,
                                 "ty_trong": w.values}))
pd.concat(dai).to_csv(KQ / "ty_trong_theo_ky.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d",
                      float_format="%.6f")

for s in ["S2", "S3", "S4", "S5"]:
    t = tap_trung[tap_trung["chien_luoc"] == s]
    ghi(f"{CHIEN_LUOC[s]}: số mã nắm giữ {t['so_ma'].min()}–{t['so_ma'].max()}; tỷ trọng lớn nhất trung vị "
        f"{pt(t['ty_trong_lon_nhat'].median())}; HHI trung vị {so(t['hhi'].median(), 3)}; tỷ trọng ở mã đứng giá trên "
        f"{pt(NGUONG_DUNG_GIA, 0)} số phiên: trung vị {pt(t['ty_trong_ma_dung_gia_nhieu'].median())}")
t3 = tap_trung[tap_trung["chien_luoc"] == "S3"].set_index("ky")
for ky in [1, 5, 9, 13, 17, 20]:
    ghi(f"  S3 kỳ {ky}, 5 mã lớn nhất: {t3.at[ky, 'top5']}")
tb3 = pd.concat([TY_TRONG["S3"][k] for k in VT], axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
ghi("  S3, tỷ trọng bình quân 20 kỳ cao nhất: " + ", ".join(f"{m} {pt(v)}" for m, v in tb3.head(10).items()))
s4 = pd.DataFrame(nhat_ky_s4)
lo = s4[s4["ln_nam_trung_vi_cum"] < 0]
ghi(f"S4: {len(lo)} lượt cụm có lợi suất năm hóa trung vị âm vẫn góp 2 mã, ở các kỳ "
    + ", ".join(str(v) for v in sorted(lo["ky"].unique())))


# %% 3. Mô phỏng giá trị danh mục
ghi("\n=== PHẦN 2. MÔ PHỎNG ===")
R = r_cp.fillna(0.0)                 # mã bị đình chỉ hoặc chưa có giá: lợi suất 0 (vị thế đóng băng)
BAT_DAU = VT[0]
TCB = {k: i + 1 for i, k in enumerate(VT)}


def mo_phong(theo_ky):
    """Mua và giữ trong kỳ; tái cân bằng ở giá đóng cửa các phiên giao dịch, trừ phí trên phần mua bán."""
    h = pd.Series(dtype=float)       # giá trị nắm giữ từng mã
    gia_tri_truoc = 1.0
    loi_suat_ngay, vong_quay = [], []
    for t in range(BAT_DAU, len(LICH)):
        if len(h):
            h = h * (1.0 + R.iloc[t][h.index])
        gia_tri = h.sum() if len(h) else gia_tri_truoc
        if t in TCB:
            w_cu = h / gia_tri if len(h) else pd.Series(dtype=float)
            w_moi = theo_ky[t]
            chi_so = w_moi.index.union(w_cu.index)
            vq = float((w_moi.reindex(chi_so, fill_value=0) - w_cu.reindex(chi_so, fill_value=0)).abs().sum())
            gia_tri = gia_tri * (1 - PHI_GIAO_DICH * vq)
            h = w_moi * gia_tri
            vong_quay.append({"ky": TCB[t], "vong_quay_hai_chieu": vq, "phi": PHI_GIAO_DICH * vq})
        loi_suat_ngay.append(gia_tri / gia_tri_truoc - 1)
        gia_tri_truoc = gia_tri
    return pd.Series(loi_suat_ngay, index=LICH[BAT_DAU:]), pd.DataFrame(vong_quay)


LS = {"S1": r_vni.iloc[BAT_DAU:].copy()}
LS["S1"].iloc[0] = 0.0               # mua chỉ số ở giá đóng cửa phiên giao dịch đầu tiên
VQ = {}
for s in ["S2", "S3", "S4", "S5"]:
    LS[s], VQ[s] = mo_phong(TY_TRONG[s])
LS = pd.DataFrame(LS)
LS.to_csv(KQ / "loi_suat_ngay_chien_luoc.csv", encoding="utf-8-sig", date_format="%Y-%m-%d", float_format="%.8f")
GT = (1 + LS).cumprod()
vq = pd.concat({s: d.set_index("ky") for s, d in VQ.items()}, axis=1)
vq.to_csv(KQ / "vong_quay_theo_ky.csv", encoding="utf-8-sig", float_format="%.4f")
ghi(f"{len(LS)} phiên từ {LS.index[0]:%d/%m/%Y} đến {LS.index[-1]:%d/%m/%Y}")


# %% 4. Chỉ tiêu hiệu quả
def chi_tieu(x, rf_x):
    n = len(x)
    vuot = x - rf_x
    gt = (1 + x).cumprod()
    duoi = np.sqrt((np.minimum(vuot, 0) ** 2).mean())
    sgtd = (gt / gt.cummax() - 1).min()
    ln = gt.iloc[-1] ** (PHIEN_NAM / n) - 1
    return pd.Series({
        "loi_suat_tich_luy": gt.iloc[-1] - 1,
        "loi_suat_nam_hoa": ln,
        "do_bien_dong_nam_hoa": x.std() * np.sqrt(PHIEN_NAM),
        "sharpe": vuot.mean() / vuot.std() * np.sqrt(PHIEN_NAM),
        "sortino": vuot.mean() * PHIEN_NAM / (duoi * np.sqrt(PHIEN_NAM)),
        "sut_giam_toi_da": sgtd,
        "calmar": ln / abs(sgtd) if sgtd < 0 else np.nan,
        "so_phien": n,
    })


rf_kt = rf.iloc[BAT_DAU:]
bang = pd.DataFrame({CHIEN_LUOC[s]: chi_tieu(LS[s], rf_kt) for s in LS.columns}).T
for s in ["S2", "S3", "S4", "S5"]:
    bang.loc[CHIEN_LUOC[s], "vong_quay_hai_chieu_bq"] = VQ[s]["vong_quay_hai_chieu"].iloc[1:].mean()
    bang.loc[CHIEN_LUOC[s], "tong_phi"] = VQ[s]["phi"].sum()
bang.to_csv(KQ / "chi_tieu_ca_giai_doan.csv", encoding="utf-8-sig", float_format="%.4f")
ghi("\n=== PHẦN 3. KẾT QUẢ CẢ GIAI ĐOẠN ===")
hien = bang.copy()
for c in ["loi_suat_tich_luy", "loi_suat_nam_hoa", "do_bien_dong_nam_hoa", "sut_giam_toi_da",
          "vong_quay_hai_chieu_bq", "tong_phi"]:
    hien[c] = hien[c].map(lambda v: "" if pd.isna(v) else pt(v))
for c in ["sharpe", "sortino", "calmar"]:
    hien[c] = hien[c].map(so)
ghi(hien.drop(columns="so_phien").to_string())
ghi("Ghi chú: vòng quay hai chiều bình quân không tính lần mua đầu tiên; S1 không tính phí; VN-Index là chỉ số giá, "
    "các chiến lược còn lại dùng giá đã điều chỉnh (gồm cổ tức).")

# Theo năm và theo giai đoạn con
ky_cua_phien = pd.Series(np.searchsorted(VT, np.arange(BAT_DAU, len(LICH)), side="right"), index=LS.index)
theo_nam = LS.groupby(LS.index.year).apply(lambda d: (1 + d).prod() - 1)
theo_nam.index.name = "nam"
theo_nam.columns = [CHIEN_LUOC[s] for s in theo_nam.columns]
theo_nam.to_csv(KQ / "loi_suat_theo_nam.csv", encoding="utf-8-sig", float_format="%.4f")
ghi("\nLợi suất theo năm:")
ghi(theo_nam.map(pt).to_string())

con = []
for ten, (a, b) in GIAI_DOAN_CON.items():
    chon_p = ky_cua_phien.between(a, b)
    for s in LS.columns:
        con.append(chi_tieu(LS.loc[chon_p, s], rf_kt[chon_p]).rename((ten, CHIEN_LUOC[s])))
con = pd.DataFrame(con)
con.index = pd.MultiIndex.from_tuples(con.index, names=["giai_doan", "chien_luoc"])
con.to_csv(KQ / "chi_tieu_giai_doan_con.csv", encoding="utf-8-sig", float_format="%.4f")
ghi("\nGiai đoạn con (lợi suất năm hóa | Sharpe | sụt giảm tối đa):")
for ten, (a, b) in GIAI_DOAN_CON.items():
    d = con.loc[ten]
    ngay = LS.index[ky_cua_phien.between(a, b)]
    ghi(f"  {ten} ({ngay[0]:%d/%m/%Y}–{ngay[-1]:%d/%m/%Y}, VN-Index {pt((1 + LS.loc[ngay, 'S1']).prod() - 1)}): " + "; ".join(f"{s.split()[0]} {pt(x.loi_suat_nam_hoa)} | {so(x.sharpe)} | {pt(x.sut_giam_toi_da)}"
                                 for s, x in d.iterrows()))


# %% 5. Hình
fig, ax = plt.subplots(figsize=(7.5, 4))
for s in LS.columns:
    ax.plot(GT.index, GT[s], color=MAU[s], lw=1.3 if s != "S1" else 1.0, label=CHIEN_LUOC[s])
ax.set_yscale("log")
thap, cao = ax.get_ylim()
vach = [v for v in [0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5] if thap <= v <= cao]
ax.set_yticks(vach, [f"{v:g}".replace(".", ",") for v in vach])
ax.set_yticks([], minor=True)
ax.set_ylabel("Giá trị danh mục (bắt đầu = 1, thang log)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(KQ / "hinh_11_gia_tri_danh_muc.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 3.4))
for s in LS.columns:
    ax.plot(GT.index, (GT[s] / GT[s].cummax() - 1) * 100, color=MAU[s], lw=1.0, label=CHIEN_LUOC[s])
ax.set_ylabel("Sụt giảm từ đỉnh (%)")
ax.legend(frameon=False, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
fig.savefig(KQ / "hinh_12_sut_giam_tu_dinh.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
rong = 0.16
for j, s in enumerate(LS.columns):
    ax.bar(theo_nam.index + (j - 2) * rong, theo_nam[CHIEN_LUOC[s]] * 100, width=rong, color=MAU[s],
           label=CHIEN_LUOC[s])
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(theo_nam.index)
ax.set_ylabel("Lợi suất năm (%)")
ax.legend(frameon=False, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
fig.savefig(KQ / "hinh_13_loi_suat_theo_nam.png")
plt.close(fig)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True)
for s in ["S3", "S4", "S5"]:
    t = tap_trung[tap_trung["chien_luoc"] == s].set_index("ky")
    a1.plot(t.index, t["ty_trong_lon_nhat"] * 100, marker="o", color=MAU[s], label=CHIEN_LUOC[s])
    a2.plot(t.index, t["ty_trong_ma_dung_gia_nhieu"] * 100, marker="o", color=MAU[s], label=CHIEN_LUOC[s])
t2 = tap_trung[tap_trung["chien_luoc"] == "S2"].set_index("ky")
a2.plot(t2.index, t2["ty_trong_ma_dung_gia_nhieu"] * 100, ls="--", color=MAU["S2"], label=CHIEN_LUOC["S2"])
a1.set_ylabel("Tỷ trọng lớn nhất (%)")
a2.set_ylabel(f"Tỷ trọng ở mã đứng giá\n> {pt(NGUONG_DUNG_GIA, 0)} số phiên (%)")
a2.set_xlabel("Kỳ tái cân bằng")
a2.set_xticks(list(t.index))
a1.legend(frameon=False, fontsize=8)
a2.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(KQ / "hinh_14_tap_trung_ty_trong.png")
plt.close(fig)


# %% 6. Ghi báo cáo
(KQ / "bao_cao_kiem_tra.txt").write_text("\n".join(BAO_CAO), encoding="utf-8")
print(f"\nĐã ghi {len(list(KQ.iterdir()))} tệp vào {KQ}")
