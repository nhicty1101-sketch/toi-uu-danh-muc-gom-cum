# Nạp rổ 100 mã (data/processed/ro_100_ma.csv) vào bảng ro_co_phieu trong MySQL.
# Dùng thay cho Table Data Import Wizard của Workbench trên Mac, vốn lỗi với tệp CSV có BOM.
# Thông tin kết nối đọc từ tệp .env ở thư mục gốc dự án; code không chứa mật khẩu.
# Chạy lại nhiều lần không sinh dòng trùng: mã đã có sẽ được cập nhật.
import os
from pathlib import Path

import mysql.connector
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
TEP = Path("data") / "processed" / "ro_100_ma.csv"
COT = ["ma", "hang", "gtgd_bq_2019_ty", "ty_le_phien_2019", "san_hien_tai", "ngay_cuoi_co_du_lieu"]

ro = pd.read_csv(TEP, encoding="utf-8-sig", dtype={"ma": str, "san_hien_tai": str})
thieu = [c for c in COT if c not in ro.columns]
if thieu:
    raise SystemExit(f"{TEP} thiếu cột {thieu}; đang có {list(ro.columns)}")
ro = ro[COT].copy()
ro["ngay_cuoi_co_du_lieu"] = pd.to_datetime(ro["ngay_cuoi_co_du_lieu"]).dt.date
ro["gtgd_bq_2019_ty"] = ro["gtgd_bq_2019_ty"].round(3)
ro["ty_le_phien_2019"] = ro["ty_le_phien_2019"].round(3)
dong = [tuple(None if pd.isna(v) else v for v in r) for r in ro.astype(object).itertuples(index=False)]

kn = mysql.connector.connect(host=os.environ.get("MYSQL_HOST", "localhost"),
                             user=os.environ["MYSQL_USER"],
                             password=os.environ["MYSQL_PASSWORD"],
                             database=os.environ.get("MYSQL_DB", "tieu_luan_khdl"))
sql = (f"INSERT INTO ro_co_phieu ({', '.join(COT)}) VALUES ({', '.join(['%s'] * len(COT))}) "
       "ON DUPLICATE KEY UPDATE " + ", ".join(f"{c}=VALUES({c})" for c in COT[1:]))
con_tro = kn.cursor()
con_tro.executemany(sql, dong)
kn.commit()
con_tro.execute("SELECT COUNT(*) FROM ro_co_phieu")
print(f"Đã nạp {len(dong)} dòng từ {TEP}; bảng ro_co_phieu hiện có {con_tro.fetchone()[0]} dòng")
con_tro.close()
kn.close()
