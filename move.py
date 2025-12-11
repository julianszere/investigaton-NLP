import os
import shutil

src_base = "results/SAE"
dst_base = "SAE2"

# Create target folders
invest_dst = os.path.join(dst_base, "investigathon")
long_dst = os.path.join(dst_base, "longmemeval")

os.makedirs(invest_dst, exist_ok=True)
os.makedirs(long_dst, exist_ok=True)

# --- Move investigathon_evaluation → SAE2/investigathon ---
invest_src = os.path.join(src_base, "investigathon_evaluation")
if os.path.exists(invest_src):
    print(f"Moving {invest_src} → {invest_dst}")
    for item in os.listdir(invest_src):
        shutil.move(os.path.join(invest_src, item), invest_dst)

# --- Move longmemeval → SAE2/longmemeval ---
long_src = os.path.join(src_base, "longmemeval")
if os.path.exists(long_src):
    print(f"Moving {long_src} → {long_dst}")
    for item in os.listdir(long_src):
        shutil.move(os.path.join(long_src, item), long_dst)
