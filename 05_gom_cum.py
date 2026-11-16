# -*- coding: utf-8 -*-
"""
BƯỚC 4b – GOM CỤM
Tiểu luận: Tối ưu hóa danh mục đầu tư bằng gom cụm và học máy (HOSE, 2019–2025)

Quyết định đã chốt (16/11/2026):
  - K-Means trên hai trục: lợi suất năm hóa và rủi ro tổng hợp (trung bình z-score của độ biến động,
    beta, sụt giảm tối đa), cắt đuôi 5%/95% trước khi chuẩn hóa;
  - số cụm k chọn MỘT LẦN trên cửa sổ ước lượng đầu tiên (02/01/2019–05/01/2021) bằng Elbow và
    Silhouette, trong khoảng 3–8 (k = 2 làm S4 chỉ còn 4 mã x trần 25%, tỷ trọng bị ép chia đều),
    rồi giữ cố định cho cả 20 kỳ;
  - K-Means bốn đặc trưng (như đề cương) giữ làm kiểm định độ vững;
  - gom cụm phân cấp theo tương quan (khoảng cách sqrt(0,5(1 - rho)), liên kết Ward) để so sánh.
    Khoảng cách này tỷ lệ với khoảng cách Euclid giữa hai chuỗi lợi suất đã chuẩn hóa, nên dùng được Ward.
    Liên kết trung bình (bản đầu) cho ra một nhóm lớn và vài mã lẻ loi, nên không dùng để so sánh được.

Đầu vào: bảng gia_ngay (MySQL) hoặc data/processed/gia_dong_cua.csv, ro_100_ma.csv, ngay_huy_niem_yet.csv
Đầu ra: ket_qua/gom_cum/ (bảng .csv, hình 7–10, bao_cao_gom_cum.txt)
Chạy từ thư mục gốc dự án:  .venv/bin/python 05_gom_cum.py
"""

# %% 0. Cấu hình
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from cong_cu import (CUA_SO, GOC, HAT_GIONG, NGUONG_THIEU, bon_dac_trung, doc_gia, doc_ngay_cuoi_hose, doc_ro,
                     du_dieu_kien, hai_truc, kmeans_co_thu_tu, lich_tai_can_bang, loi_suat, tinh_dac_trung)

KQ = GOC / "ket_qua" / "gom_cum"
KQ.mkdir(parents=True, exist_ok=True)
K_THU = range(1, 11)         # thử k từ 1 đến 10 cho Elbow (Silhouette từ 2)
K_MIN, K_MAX = 3, 8          # khoảng được phép chọn
LIEN_KET = "ward"            # liên kết cho gom cụm theo tương quan
TEN_LIEN_KET = {"ward": "Ward", "average": "trung bình", "single": "đơn", "complete": "đầy đủ"}

BAO_CAO = []


def ghi(dong=""):
    print(dong)
    BAO_CAO.append(str(dong))


def so(x, so_le=2):
    return f"{x:.{so_le}f}".replace(".", ",")


plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
MAU = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]


# %% 1. Đọc dữ liệu
DS_MA = doc_ro()
gia = doc_gia(DS_MA, ghi)
NGAY_CUOI = doc_ngay_cuoi_hose(DS_MA, ghi)
gia_ff, r = loi_suat(gia)
gia_cp, r_cp, r_vni, gia_ff_cp = gia[DS_MA], r[DS_MA], r["VNINDEX"], gia_ff[DS_MA]
VT, lich_tcb = lich_tai_can_bang(gia.index)
ghi(f"{len(gia)} phiên, {len(DS_MA)} mã, {len(VT)} kỳ tái cân bằng")


# %% 2. Đặc trưng và hai trục ở mỗi kỳ
ghi("\n=== PHẦN 1. ĐẶC TRƯNG VÀ HAI TRỤC ===")
DT, X2, X4 = {}, {}, {}
khong_gia = []
for k in VT:
    dk = du_dieu_kien(gia_cp, k, NGAY_CUOI)
    ma = dk.index[dk["du"]]
    bi_loai = dk.index[(dk["ty_le_thieu"] <= NGUONG_THIEU) & dk["con_niem_yet"] & ~dk["co_gia_ngay_quyet_dinh"]]
    if len(bi_loai):
        khong_gia.append(f"kỳ {VT.index(k) + 1}: {', '.join(bi_loai)}")
    DT[k] = tinh_dac_trung(r_cp, r_vni, gia_ff_cp, k, list(ma))
    X2[k] = hai_truc(DT[k])
    X4[k] = bon_dac_trung(DT[k])
lich_tcb["so_ma"] = [len(DT[k]) for k in VT]
ghi(f"Số mã được gom cụm mỗi kỳ: {lich_tcb['so_ma'].min()}–{lich_tcb['so_ma'].max()}")
if khong_gia:
    ghi("Loại vì không có giá ở ngày quyết định: " + "; ".join(khong_gia))

# Căn cứ cho chỉ số rủi ro tổng hợp: ba thành phần tương quan cùng chiều và đủ chặt
tin_cay = []
for k in VT:
    w = DT[k][["bd_nam", "beta", "sgtd"]]
    c = w.corr().values[np.triu_indices(3, 1)]
    r_tb = c.mean()
    alpha = 3 * r_tb / (1 + 2 * r_tb)          # Cronbach's alpha của ba biến đã chuẩn hóa
    tq = DT[k].join(X2[k])[["truc_rui_ro", "bd_nam", "beta", "sgtd", "truc_loi_suat"]].corr()
    tin_cay.append({"ky": VT.index(k) + 1, "tuong_quan_tb_3_bien": r_tb, "cronbach_alpha": alpha,
                    "rui_ro~bd_nam": tq.loc["truc_rui_ro", "bd_nam"], "rui_ro~beta": tq.loc["truc_rui_ro", "beta"],
                    "rui_ro~sgtd": tq.loc["truc_rui_ro", "sgtd"],
                    "rui_ro~loi_suat": tq.loc["truc_rui_ro", "truc_loi_suat"]})
tin_cay = pd.DataFrame(tin_cay).set_index("ky")
tin_cay.round(3).to_csv(KQ / "do_tin_cay_chi_so_rui_ro.csv", encoding="utf-8-sig")
t1 = tin_cay.iloc[0]
ghi(f"Kỳ 1: tương quan bình quân giữa ba biến rủi ro {so(t1['tuong_quan_tb_3_bien'])}, Cronbach's alpha "
    f"{so(t1['cronbach_alpha'])}; chỉ số rủi ro tương quan với độ biến động {so(t1['rui_ro~bd_nam'])}, beta "
    f"{so(t1['rui_ro~beta'])}, sụt giảm tối đa {so(t1['rui_ro~sgtd'])}; với trục lợi suất {so(t1['rui_ro~loi_suat'])}")
ghi(f"Qua {len(VT)} kỳ: Cronbach's alpha trung vị {so(tin_cay['cronbach_alpha'].median())} "
    f"({so(tin_cay['cronbach_alpha'].min())}–{so(tin_cay['cronbach_alpha'].max())}); tương quan giữa hai trục "
    f"trung vị {so(tin_cay['rui_ro~loi_suat'].median())} ({so(tin_cay['rui_ro~loi_suat'].min())}–"
    f"{so(tin_cay['rui_ro~loi_suat'].max())})")


# %% 3. Chọn k trên cửa sổ đầu tiên
ghi("\n=== PHẦN 2. CHỌN SỐ CỤM k (KỲ 1) ===")
k0 = VT[0]
X0 = X2[k0].values
chon = []
for kk in K_THU:
    km = KMeans(n_clusters=kk, n_init=50, random_state=HAT_GIONG).fit(X0)
    chon.append({"k": kk, "sse": km.inertia_,
                 "silhouette": silhouette_score(X0, km.labels_) if kk >= 2 else np.nan})
chon = pd.DataFrame(chon).set_index("k")
# Điểm khuỷu: k có khoảng cách lớn nhất tới đường thẳng nối hai đầu đường SSE (đã quy về thang 0–1)
x = (chon.index.values - chon.index.min()) / (chon.index.max() - chon.index.min())
y = (chon["sse"].values - chon["sse"].min()) / (chon["sse"].max() - chon["sse"].min())
chon["khoang_cach_khuyu"] = np.abs((1 - x) - y) / np.sqrt(2)
k_khuyu = int(chon["khoang_cach_khuyu"].idxmax())
trong_khoang = chon.loc[K_MIN:K_MAX]
K = int(trong_khoang["silhouette"].idxmax())
chon.round(4).to_csv(KQ / "chon_k_ky_1.csv", encoding="utf-8-sig")
ghi(chon[["sse", "silhouette"]].round(3).to_string())
ghi(f"Silhouette cao nhất trong khoảng {K_MIN}–{K_MAX}: k = {K} ({so(chon.at[K, 'silhouette'], 3)}). "
    f"Điểm khuỷu của đường SSE: k = {k_khuyu}. "
    f"Silhouette cao nhất nếu tính cả k = 2: k = {int(chon['silhouette'].idxmax())}.")
ghi(f"=> CHỌN k = {K}, giữ cố định cho {len(VT)} kỳ (quy tắc: Silhouette cao nhất trong {K_MIN}–{K_MAX}; "
    "Elbow dùng để đối chiếu).")
if K in (K_MIN, K_MAX):
    ghi(f"LƯU Ý: k = {K} nằm ở biên khoảng {K_MIN}–{K_MAX}; cần xem lại đường Silhouette trước khi chốt.")
if k_khuyu != K:
    ghi(f"LƯU Ý: Elbow (k = {k_khuyu}) và Silhouette (k = {K}) không trùng nhau; cần bàn trước khi chốt.")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
a1.plot(chon.index, chon["sse"], marker="o", color=MAU[0])
a1.axvline(k_khuyu, color="grey", ls="--", lw=0.8)
a1.set_xlabel("Số cụm k")
a1.set_ylabel("Tổng bình phương khoảng cách trong cụm (SSE)")
a1.set_title("Elbow", fontsize=9)
a2.plot(chon.index[1:], chon["silhouette"].iloc[1:], marker="o", color=MAU[1])
a2.axvspan(K_MIN - 0.5, K_MAX + 0.5, color="grey", alpha=0.08)
a2.scatter([K], [chon.at[K, "silhouette"]], s=80, facecolors="none", edgecolors="black", zorder=3)
a2.set_xlabel("Số cụm k")
a2.set_ylabel("Hệ số Silhouette")
a2.set_title(f"Silhouette (vùng xám: k = {K_MIN}–{K_MAX})", fontsize=9)
a1.set_xticks(list(K_THU))
a2.set_xticks(list(K_THU)[1:])
fig.tight_layout()
fig.savefig(KQ / "hinh_7_chon_so_cum.png")
plt.close(fig)


# %% 4. Gom cụm ở mọi kỳ với k cố định
ghi("\n=== PHẦN 3. GOM CỤM 20 KỲ ===")
nhan_dai, dac_diem, on_dinh = [], [], []
nhan_truoc = None
for k, x in zip(VT, lich_tcb.itertuples()):
    X, dt = X2[k], DT[k]
    nhan, tam, _ = kmeans_co_thu_tu(X, K, "truc_rui_ro")
    nhan4, _, _ = kmeans_co_thu_tu(X4[k], K, ["bd_nam", "beta", "sgtd"])
    # Gom cụm phân cấp theo tương quan lợi suất trong cùng cửa sổ
    C = r_cp.iloc[k - CUA_SO:k][list(dt.index)].corr(min_periods=int(CUA_SO * (1 - 2 * NGUONG_THIEU)))
    c = C.fillna(0).to_numpy(copy=True)
    np.fill_diagonal(c, 1.0)
    kc = np.sqrt(np.clip(0.5 * (1 - c), 0, None))
    np.fill_diagonal(kc, 0)
    Z = linkage(squareform(kc, checks=False), method=LIEN_KET)
    nhan_tq = pd.Series(fcluster(Z, t=K, criterion="maxclust"), index=dt.index)
    # Silhouette của k đã chọn và của k tốt nhất riêng kỳ này (chỉ để kiểm tra, không dùng để chọn)
    sil = silhouette_score(X.values, nhan.values)
    sil_k = {kk: silhouette_score(X.values, KMeans(n_clusters=kk, n_init=20, random_state=HAT_GIONG)
                                  .fit(X.values).labels_) for kk in range(K_MIN, K_MAX + 1)}
    k_tot = max(sil_k, key=sil_k.get)
    chung = nhan.index.intersection(nhan_truoc.index) if nhan_truoc is not None else []
    co_cum = nhan.value_counts()
    co_cum_tq = nhan_tq.value_counts()
    toan_lo = [cu for cu in co_cum.index if (dt.loc[nhan == cu, "ln_nam"] < 0).all()]
    on_dinh.append({
        "ky": x.ky, "ngay_quyet_dinh": x.ngay_quyet_dinh, "so_ma": len(dt),
        "silhouette_k_chon": sil, "k_tot_nhat_ky_nay": k_tot, "silhouette_k_tot_nhat": sil_k[k_tot],
        "co_cum_nho_nhat": int(co_cum.min()), "co_cum_lon_nhat": int(co_cum.max()),
        "co_nhom_tq_nho_nhat": int(co_cum_tq.min()), "co_nhom_tq_lon_nhat": int(co_cum_tq.max()),
        "so_cum_toan_ma_lo": len(toan_lo),
        "ari_voi_ky_truoc": adjusted_rand_score(nhan_truoc[chung], nhan[chung]) if len(chung) else np.nan,
        "ari_hai_truc_vs_bon_dac_trung": adjusted_rand_score(nhan, nhan4),
        "ari_hai_truc_vs_tuong_quan": adjusted_rand_score(nhan, nhan_tq),
    })
    for cu in range(1, K + 1):
        thanh_vien = dt[nhan == cu]
        dac_diem.append({"ky": x.ky, "cum": cu, "so_ma": len(thanh_vien),
                         "tam_truc_loi_suat": tam.at[cu, "truc_loi_suat"], "tam_truc_rui_ro": tam.at[cu, "truc_rui_ro"],
                         "ln_nam_trung_vi": thanh_vien["ln_nam"].median(),
                         "bd_nam_trung_vi": thanh_vien["bd_nam"].median(),
                         "beta_trung_vi": thanh_vien["beta"].median(), "sgtd_trung_vi": thanh_vien["sgtd"].median(),
                         "toan_ma_lo": cu in toan_lo, "cac_ma": " ".join(sorted(thanh_vien.index))})
    nhan_dai.append(dt.join(X).assign(ky=x.ky, ngay_quyet_dinh=x.ngay_quyet_dinh, cum=nhan,
                                      cum_bon_dac_trung=nhan4, cum_tuong_quan=nhan_tq))
    nhan_truoc = nhan
    if x.ky == 1:
        Z0, ma_z0, nhan_tq0 = Z, list(dt.index), nhan_tq

nhan_dai = pd.concat(nhan_dai).reset_index(names="ma")
cot = ["ky", "ngay_quyet_dinh", "ma", "ln_nam", "bd_nam", "beta", "sgtd", "truc_loi_suat", "truc_rui_ro",
       "cum", "cum_bon_dac_trung", "cum_tuong_quan"]
nhan_dai[cot].to_csv(KQ / "nhan_cum_theo_ky.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d",
                     float_format="%.4f")
dac_diem = pd.DataFrame(dac_diem)
dac_diem.to_csv(KQ / "dac_diem_cum_theo_ky.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
on_dinh = pd.DataFrame(on_dinh).set_index("ky")
on_dinh.to_csv(KQ / "on_dinh_cum_theo_ky.csv", encoding="utf-8-sig", date_format="%Y-%m-%d", float_format="%.4f")

ghi(f"Silhouette với k = {K} qua các kỳ: trung vị {so(on_dinh['silhouette_k_chon'].median(), 3)} "
    f"({so(on_dinh['silhouette_k_chon'].min(), 3)}–{so(on_dinh['silhouette_k_chon'].max(), 3)})")
ghi("k có Silhouette cao nhất từng kỳ (chỉ để kiểm tra): "
    + ", ".join(f"k={kk}: {n} kỳ" for kk, n in on_dinh["k_tot_nhat_ky_nay"].value_counts().sort_index().items()))
ghi(f"Silhouette mất đi khi giữ k = {K} so với k tốt nhất từng kỳ: trung vị "
    f"{so((on_dinh['silhouette_k_tot_nhat'] - on_dinh['silhouette_k_chon']).median(), 3)}, lớn nhất "
    f"{so((on_dinh['silhouette_k_tot_nhat'] - on_dinh['silhouette_k_chon']).max(), 3)}")
ghi(f"Cỡ cụm nhỏ nhất qua các kỳ: {on_dinh['co_cum_nho_nhat'].min()} mã; số kỳ có cụm dưới 2 mã: "
    f"{int((on_dinh['co_cum_nho_nhat'] < 2).sum())}")
ghi(f"Số kỳ có ít nhất một cụm mà mọi mã đều có lợi suất năm hóa âm: {int((on_dinh['so_cum_toan_ma_lo'] > 0).sum())}"
    f" (" + ", ".join(f"kỳ {i}" for i in on_dinh.index[on_dinh['so_cum_toan_ma_lo'] > 0]) + ")")
ghi(f"Độ ổn định thành viên giữa hai kỳ liền nhau (ARI): trung vị {so(on_dinh['ari_voi_ky_truoc'].median())} "
    f"({so(on_dinh['ari_voi_ky_truoc'].min())}–{so(on_dinh['ari_voi_ky_truoc'].max())})")
ghi(f"Mức trùng giữa K-Means hai trục và K-Means bốn đặc trưng (ARI): trung vị "
    f"{so(on_dinh['ari_hai_truc_vs_bon_dac_trung'].median())}")
ghi(f"Gom cụm phân cấp theo tương quan (liên kết {TEN_LIEN_KET[LIEN_KET]}, {K} nhóm): nhóm nhỏ nhất "
    f"{on_dinh['co_nhom_tq_nho_nhat'].min()}–{on_dinh['co_nhom_tq_nho_nhat'].max()} mã, nhóm lớn nhất "
    f"{on_dinh['co_nhom_tq_lon_nhat'].min()}–{on_dinh['co_nhom_tq_lon_nhat'].max()} mã qua các kỳ")
ghi(f"Mức trùng giữa K-Means hai trục và gom cụm phân cấp theo tương quan (ARI): trung vị "
    f"{so(on_dinh['ari_hai_truc_vs_tuong_quan'].median())} "
    f"({so(on_dinh['ari_hai_truc_vs_tuong_quan'].min())}–{so(on_dinh['ari_hai_truc_vs_tuong_quan'].max())})")

ghi(f"\nCác cụm ở kỳ 1 (cửa sổ {gia.index[k0 - CUA_SO]:%d/%m/%Y}–{gia.index[k0 - 1]:%d/%m/%Y}), "
    "đánh số theo rủi ro tăng dần:")
for _, d in dac_diem[dac_diem["ky"] == 1].iterrows():
    ghi(f"  Cụm {d['cum']} ({d['so_ma']} mã): lợi suất năm hóa trung vị {so(d['ln_nam_trung_vi'] * 100, 1)}%, "
        f"độ biến động {so(d['bd_nam_trung_vi'] * 100, 1)}%, beta {so(d['beta_trung_vi'])}, "
        f"sụt giảm tối đa {so(d['sgtd_trung_vi'] * 100, 1)}%: {d['cac_ma']}")
ghi(f"\nCác nhóm theo tương quan ở kỳ 1 (liên kết {TEN_LIEN_KET[LIEN_KET]}):")
for nh, ds in nhan_tq0.groupby(nhan_tq0).groups.items():
    cheo = nhan_dai[(nhan_dai["ky"] == 1) & nhan_dai["ma"].isin(ds)]["cum"].value_counts().sort_index()
    ghi(f"  Nhóm {nh} ({len(ds)} mã; thuộc cụm K-Means " + ", ".join(f"{c}: {n}" for c, n in cheo.items())
        + f"): {' '.join(sorted(ds))}")


# %% 5. Hình
# Hình 8: các cụm ở kỳ 1 trên mặt phẳng rủi ro – lợi suất
n0 = nhan_dai[nhan_dai["ky"] == 1]
tam0 = dac_diem[dac_diem["ky"] == 1]
fig, ax = plt.subplots(figsize=(7, 5.2))
for cu in range(1, K + 1):
    d = n0[n0["cum"] == cu]
    ax.scatter(d["truc_rui_ro"], d["truc_loi_suat"], s=20, color=MAU[(cu - 1) % len(MAU)], label=f"Cụm {cu}")
    for _, h in d.iterrows():
        ax.annotate(h["ma"], (h["truc_rui_ro"], h["truc_loi_suat"]), fontsize=5, xytext=(2, 2),
                    textcoords="offset points", color="#444444")
ax.scatter(tam0["tam_truc_rui_ro"], tam0["tam_truc_loi_suat"], marker="X", s=120, color="black", label="Tâm cụm")
ax.axhline(0, color="grey", lw=0.5)
ax.axvline(0, color="grey", lw=0.5)
ax.set_xlabel("Trục rủi ro tổng hợp (z-score)")
ax.set_ylabel("Trục lợi suất (z-score)")
ax.legend(frameon=False, fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(KQ / "hinh_8_cac_cum_ky_1.png")
plt.close(fig)

# Hình 9: độ ổn định của cụm qua các kỳ
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True)
a1.plot(on_dinh.index, on_dinh["silhouette_k_chon"], marker="o", color=MAU[0], label=f"k = {K} (cố định)")
a1.plot(on_dinh.index, on_dinh["silhouette_k_tot_nhat"], marker=".", ls="--", color=MAU[1],
        label=f"k tốt nhất từng kỳ trong {K_MIN}–{K_MAX}")
a1.set_ylabel("Silhouette")
a1.legend(frameon=False, fontsize=8)
a2.plot(on_dinh.index, on_dinh["ari_voi_ky_truoc"], marker="o", color=MAU[2], label="So với kỳ trước")
a2.plot(on_dinh.index, on_dinh["ari_hai_truc_vs_tuong_quan"], marker=".", ls="--", color=MAU[3],
        label="So với gom cụm theo tương quan")
a2.set_ylabel("Chỉ số Rand điều chỉnh (ARI)")
a2.set_xlabel("Kỳ tái cân bằng")
a2.set_xticks(list(on_dinh.index))
a2.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(KQ / "hinh_9_on_dinh_cum.png")
plt.close(fig)

# Hình 10: cây phân cấp theo tương quan ở kỳ 1, cắt thành K nhóm
nguong = (Z0[-K, 2] + Z0[-(K - 1), 2]) / 2
fig, ax = plt.subplots(figsize=(9, 4))
dendrogram(Z0, labels=ma_z0, color_threshold=nguong, leaf_font_size=4, ax=ax, above_threshold_color="grey")
ax.axhline(nguong, color="black", ls="--", lw=0.7)
ax.set_ylabel("Khoảng cách sqrt(0,5(1 − ρ))")
ax.set_title(f"Gom cụm phân cấp theo tương quan, liên kết {TEN_LIEN_KET[LIEN_KET]}, cắt thành {K} nhóm (kỳ 1)",
             fontsize=9)
fig.tight_layout()
fig.savefig(KQ / "hinh_10_cay_phan_cap_ky_1.png")
plt.close(fig)


# %% 6. Ghi báo cáo
(KQ / "bao_cao_gom_cum.txt").write_text("\n".join(BAO_CAO), encoding="utf-8")
print(f"\nĐã ghi {len(list(KQ.iterdir()))} tệp vào {KQ}")
