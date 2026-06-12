"""
SZ Lyncis — Data Downloader
============================
Downloads all calibration and science frames for three observation nights
from SurfDrive into the folder structure expected by pipeline.py:

  data/
  ├── 20260316/
  │   ├── calibration/   bias, darks, flats (r + g)
  │   ├── szlyncis/      SZLYN run 1  (r + g)
  │   └── szlyncis2/     SZLYN-2 run 2 (r + g)
  ├── 20260414/
  │   ├── calibration/
  │   └── szlyncis/
  └── 20260305/
      ├── calibration/
      └── szlyncis/

Usage:
    python download.py
"""

import ssl, base64, urllib.request
from pathlib import Path

# ============================================================
# SurfDrive credentials
# ============================================================
TOKEN = "5z4GrbnFw5w8JKm"
SURFDRIVE_BASE = "https://surfdrive.surf.nl/public.php/webdav"

# ============================================================
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode    = ssl.CERT_NONE

_creds = base64.b64encode(f"{TOKEN}:".encode()).decode()


def download(remote_path, local_path):
    """Download one file; skip if already present; delete on failure."""
    local_path = Path(local_path)
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{SURFDRIVE_BASE}/{remote_path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {_creds}"})
    try:
        with urllib.request.urlopen(req, context=_ssl_ctx) as resp, \
             open(local_path, "wb") as f:
            f.write(resp.read())
        print(f"  Downloaded {local_path.name}")
    except Exception as e:
        local_path.unlink(missing_ok=True)
        print(f"  FAILED {local_path.name}: {e}")


# ============================================================
# 20260316
# ============================================================
print("\n=== 20260316 — calibration ===")
for i in range(1, 11):
    n = f"{i:04d}"
    download(f"20260316/bias-{n}.fit",       f"data/20260316/calibration/bias-{n}.fit")
    download(f"20260316/Dark-{n}_50.fit",    f"data/20260316/calibration/Dark-{n}_50.fit")
    download(f"20260316/Dark-{n}_60.fit",    f"data/20260316/calibration/Dark-{n}_60.fit")
    download(f"20260316/flat-{n}_r.fit",     f"data/20260316/calibration/flat-{n}_r.fit")
    download(f"20260316/flat-{n}_g.fit",     f"data/20260316/calibration/flat-{n}_g.fit")

print("\n=== 20260316 — SZLYN run 1 (r + g) ===")
for i in range(1, 18):   # frames 0001–0017
    n = f"{i:04d}"
    download(f"20260316/SZLYN/SZLYN-{n}_r60.fit", f"data/20260316/szlyncis/SZLYN-{n}_r60.fit")
    download(f"20260316/SZLYN/SZLYN-{n}_g60.fit", f"data/20260316/szlyncis/SZLYN-{n}_g60.fit")

print("\n=== 20260316 — SZLYN-2 run 2 (r + g) ===")
for i in range(1, 51):   # frames 0001–0050
    n = f"{i:04d}"
    download(f"20260316/SZLYN/SZLYN-2-{n}_r60.fit", f"data/20260316/szlyncis2/SZLYN-2-{n}_r60.fit")
    download(f"20260316/SZLYN/SZLYN-2-{n}_g60.fit", f"data/20260316/szlyncis2/SZLYN-2-{n}_g60.fit")


# ============================================================
# 20260414
# ============================================================
print("\n=== 20260414 — calibration ===")
for i in range(1, 11):
    n = f"{i:04d}"
    download(f"20260414/calibration/dark-{n}bias.fit", f"data/20260414/calibration/dark-{n}bias.fit")
    download(f"20260414/calibration/dark-{n}dark.fit", f"data/20260414/calibration/dark-{n}dark.fit")

for i in range(1, 21):
    n = f"{i:04d}"
    download(f"20260414/calibration/flat-{n}r.fit", f"data/20260414/calibration/flat-{n}r.fit")
    download(f"20260414/calibration/flat-{n}g.fit", f"data/20260414/calibration/flat-{n}g.fit")

print("\n=== 20260414 — science (r + g) ===")
for i in range(1, 6):    # frames 0001–0005
    n = f"{i:04d}"
    download(f"20260414/szlyncis/sz_lyncis-{n}r.fit", f"data/20260414/szlyncis/sz_lyncis-{n}r.fit")
    download(f"20260414/szlyncis/sz_lyncis-{n}g.fit", f"data/20260414/szlyncis/sz_lyncis-{n}g.fit")


# ============================================================
# 20260305
# ============================================================
print("\n=== 20260305 — calibration ===")
for i in range(1, 21):
    n = f"{i:04d}"
    download(f"20260305/Calibration-{n}_bias.fit", f"data/20260305/calibration/Calibration-{n}_bias.fit")

for i in range(1, 11):
    n = f"{i:04d}"
    download(f"20260305/Calibration-{n}_60s.fit", f"data/20260305/calibration/Calibration-{n}_60s.fit")
    download(f"20260305/flat-{n}_r.fit",           f"data/20260305/calibration/flat-{n}_r.fit")
    download(f"20260305/flat-{n}_g.fit",           f"data/20260305/calibration/flat-{n}_g.fit")

print("\n=== 20260305 — science r (0001–0023) ===")
for i in range(1, 24):
    n = f"{i:04d}"
    download(f"20260305/SZ_Lyncis-{n}_r.fit", f"data/20260305/szlyncis/SZ_Lyncis-{n}_r.fit")

print("\n=== 20260305 — science g (0001–0024) ===")
for i in range(1, 25):
    n = f"{i:04d}"
    download(f"20260305/SZ_Lyncis-{n}_g.fit", f"data/20260305/szlyncis/SZ_Lyncis-{n}_g.fit")


# ============================================================
print("\n=== Download complete ===")
for night in ["20260316", "20260414", "20260305"]:
    base = Path("data") / night
    cal  = len(list((base / "calibration").glob("*.fit"))) if (base / "calibration").exists() else 0
    sci  = len(list((base / "szlyncis").glob("*.fit")))    if (base / "szlyncis").exists()    else 0
    sci2 = len(list((base / "szlyncis2").glob("*.fit")))   if (base / "szlyncis2").exists()   else 0
    line = f"  {night}  calibration: {cal}  szlyncis: {sci}"
    if sci2:
        line += f"  szlyncis2: {sci2}"
    print(line)
