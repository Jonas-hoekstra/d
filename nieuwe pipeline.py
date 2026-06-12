import csv, warnings
from itertools import groupby
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy import units as u
import ccdproc as ccdp
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from scipy.ndimage import shift as nd_shift
from photutils.centroids import centroid_com

# Instellingen

FILTERS = ["r", "g"]

SZ_PERIOD       = 0.120534896 # pulsatie periode sz lyn
SZ_T0           = 2461118.405 # epoch bij maximaal gemeten magnitude
APERATURE_R     = 24          # aperature radius
SKY_IN          = 30          # inner sky annulus
SKY_OUT         = 45          # outer sky annulus
ALIGN_BOX       = 50          # halve breedte van box

POSITIONS = "positions.csv"   # csv file met posities van sterren

# Laad de coordinaten vanuit de opgeslagen csv file
# zoveel comp stars mogelijk
def load_star_coords():
    coords = {}
    if not Path(POSITIONS).exists():
        return coords
    with open(POSITIONS, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['night'], row['run_label'])
            if key in coords:
                continue
            try:
                sz = (float(row['sz_x']), float(row['sz_y']))
                comp_x_coords = sorted(k for k in row if k.startswith('comp') and k.endswith('_x'))
                comp = [(float(row[k]), float(row[k.replace('_x', '_y')])) for k in comp_x_coords]
                coords[key] = {'sz' : sz, 'comp' : comp}
            except (ValueError) as e:
                print(f"positions.csv overgeslagen {e}")
        return coords
    
STAR_COORDS = load_star_coords()

# Data mappen
DATA_DIR = Path("data")
REDUCED_DIR = Path("reduced")

# Data voor de nachten

NIGHTS = {
    "20260316": {
        "bias_glob":      "bias-*.fit",
        "dark_glob_sci":  "Dark-*_60.fit",   # 60 s
        "dark_glob_flat": "Dark-*_50.fit",   # 50 s
        "flat_glob": {"r": "flat-*_r.fit",  "g": "flat-*_g.fit"}, # flats zijn 5 sec
        "runs": [
            {"label": "SZLYN",   "subdir": "szlyncis",
             "science_glob": {"r": "SZLYN-[0-9]*_r60.fit", "g": "SZLYN-[0-9]*_g60.fit"}},
            {"label": "SZLYN-2", "subdir": "szlyncis2",
             "science_glob": {"r": "SZLYN-2-*_r60.fit",    "g": "SZLYN-2-*_g60.fit"}},
        ],
    },
    "20260414": {
        "bias_glob":      "dark-*bias.fit",
        "dark_glob_sci":  "dark-*dark.fit",
        "dark_glob_flat": None,              # darks alleen maar 60s 
        "flat_glob": {"r": "flat-*r.fit",   "g": "flat-*g.fit"},
        "runs": [
            {"label": "SZLYN", "subdir": "szlyncis",
             "science_glob": {"r": "sz_lyncis-*r.fit", "g": "sz_lyncis-*g.fit"}},
        ],
    },
    "20260305": {
        "bias_glob":      "Calibration-*_bias.fit",
        "dark_glob_sci":  "Calibration-*_60s.fit",  # 60 s 
        "dark_glob_flat": None,                       # geen 5s darks.
        "flat_glob": {"r": "flat-*_r.fit",  "g": "flat-*_g.fit"},
        "runs": [
            {"label": "SZ_Lyncis", "subdir": "szlyncis",
             "science_glob": {"r": "SZ_Lyncis-*_r.fit", "g": "SZ_Lyncis-*_g.fit"}},
        ],
    },
} 

def inv_median(a):
    return 1 / np.median(a)

# SETTING Calibratie aan/uit
RECALIBRATE = False       

all_calibrated = []

for night, cfg in NIGHTS.items():
    calibrated_path = DATA_DIR / night / "calibration"
    reduced_path = REDUCED_DIR / night

    if not RECALIBRATE:
        for run in cfg["runs"]:
            run_reduced = reduced_path / run["subdir"]
            if not run_reduced.exists():
                print(f" {night}/{run['label']}: niet opnieuw gecalibreert.")
                continue
            for filter in FILTERS:
                sci_glob = run["science_glob"].get(filter) # zoekt patroon
                if not sci_glob:
                    continue
                files = sorted(run_reduced.glob(sci_glob)) # sorteert
                if not files:
                    continue
                print(f" {night}/{run['label']}/{filter}: {len(files)} gereduceerde frames geladen")
                for f in files:
                    ccd = CCDData.read(str(f), unit='adu')
                    all_calibrated.append((night, run["label"], filter, f.name, ccd, ccd.header))
        continue

print(f" nu bezig met {night}")

print(f"combining bias")
bias_files = sorted(calibrated_path.glob(cfg["bias_glob"]))
if not bias_files:
    print(f"geen bias files gevonden {night} overgeslagen")
    continue

 # master_bias
combined_bias = ccdp.combine(
        [str(f) for f in bias_files],
        method='average',
        sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
        mem_limit=350e6, unit='adu',
    )
    combined_bias.meta['combined'] = True
    combined_bias.uncertainty = None
    combined_bias.write(red_path / 'combined_bias.fit', overwrite=True)
    print(f"  {len(bias_files)} bias frames to combined_bias.fit")

#Master_dark

    print(f"calibrating dark")
    dark_sci_files = sorted(cal_path.glob(cfg["dark_glob_sci"]))
    if not dark_sci_files:
        print(f"  no science dark files skip {night}")
        continue

    combined_dark_sci = ccdp.combine(
        [ccdp.subtract_bias(CCDData.read(str(f), unit='adu'), combined_bias)
         for f in dark_sci_files],
        method='average',
        sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
        mem_limit=350e6,
    )
    combined_dark_sci.meta['combined'] = True
    combined_dark_sci.uncertainty = None
    combined_dark_sci.write(red_path / 'combined_dark_sci.fit', overwrite=True)
    print(f"  {len(dark_sci_files)} science dark frames to combined_dark_sci.fit")

    flat_dark_glob = cfg.get("dark_glob_flat")
    dark_flat_files = sorted(cal_path.glob(flat_dark_glob)) if flat_dark_glob else []
    if dark_flat_files:
        combined_dark_flat = ccdp.combine(
            [ccdp.subtract_bias(CCDData.read(str(f), unit='adu'), combined_bias)
             for f in dark_flat_files],
            method='average',
            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
            mem_limit=350e6,
        )
        combined_dark_flat.meta['combined'] = True
        combined_dark_flat.uncertainty = None
        combined_dark_flat.write(red_path / 'combined_dark_flat.fit', overwrite=True)
        print(f"  {len(dark_flat_files)} flat dark frames to combined_dark_flat.fit")
    else:
        combined_dark_flat = combined_dark_sci
        print(f"  no separate flat darks — using science dark (scaled) for flats")

    combined_flats = {}  
#Master flats
    for filt in FILTERS:
        flat_glob = cfg["flat_glob"].get(filt)
        flat_files = sorted(cal_path.glob(flat_glob)) if flat_glob else []

        if not flat_files:
            print(f"\n  No flat files filter {filt} in {night} ")
            continue

        print(f"Combining flats {filt})")
        calibrated_flats = []
        for f in flat_files:
            ccd = CCDData.read(str(f), unit='adu')
            ccd = ccdp.subtract_bias(ccd, combined_bias)
            ccd = ccdp.subtract_dark(ccd, combined_dark_flat,
                                     exposure_time='EXPTIME', exposure_unit=u.second,
                                     scale=True)
            calibrated_flats.append(ccd)

        combined_flat = ccdp.combine(
            calibrated_flats,
            method='average', scale=inv_median,
            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
            mem_limit=350e6,
        )
        combined_flat.meta['combined'] = True
        combined_flat.uncertainty = None  
        combined_flat.write(red_path / f'combined_flat_{filt}.fit', overwrite=True)
        combined_flats[filt] = combined_flat
        print(f"  {len(flat_files)} flat frames to combined_flat_{filt}.fit")

      #calibrate total
        for run in cfg["runs"]:
            sci_glob = run["science_glob"].get(filt)
            if not sci_glob:
                continue
            sci_files = sorted((DATA_DIR / night / run["subdir"]).glob(sci_glob))
            if not sci_files:
                print(f"\n  No science files: {night}/{run['label']} filter {filt} — skipping")
                continue

            print(f"Calibrating science {night} / {run['label']} / {filt}")
            run_red = red_path / run["subdir"]
            run_red.mkdir(exist_ok=True)

            for f in sci_files:
                ccd = CCDData.read(str(f), unit='adu')
                reduced = ccdp.subtract_bias(ccd, combined_bias)
                reduced = ccdp.subtract_dark(reduced, combined_dark_sci,
                                             exposure_time='EXPTIME', exposure_unit=u.second,
                                             scale=True)
                reduced = ccdp.flat_correct(reduced, combined_flat)
                reduced.write(run_red / f.name, overwrite=True)
                all_calibrated.append((night, run["label"], filt, f.name, reduced, ccd.header))
                print(f"  Calibrated: {f.name}")
                

# Methode aangeraden door sebastian, de code maakt een box om de laatste positie, in deze box
# zoekt het programma naar de hoogste magnitude en maakt daar een nieuwe aperature cirkel omheen
# iom dit process te herhalen voor alle andere runs.
def star_centers(data, positions): 

    h, w = data.shape
    centers = []
    for x,y in positions: 
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - ALIGN_BOX), min(w, xi + ALIGN_BOX)
        y0, y1 - max(0, yi - ALIGN_BOX), min(h, yi + ALIGN_BOX)
        box = data[y0:y1, x0:x1].astype(float)
        box = box - np.median(box)
        np.clip(box, 0, None, out=box)
        cx, cy = centroid_com(box)
        centers.append((cx + x0, cy + y0))
    return centers

#magnitude uit die cirkels meten.
def do_photometry(frames, night, run_label, filters, sz_xy, comp_stars):
    all_positions = [sz_xy] + list(comp_stars)
    ref_centers   = None
    rows = []

    for fname, ccd, hdr in frames: 
        data = ccd.data.astype(float)

        cur_centers = star_centers(data, all_positions)

        if ref_centers is None:
            ref_centers = cur_centers
        else:   #hier pakken we de nieuwe box
            dy = float(np.mean([c[1] - r[1] for c, r in zip(cur_centers, ref_centers)]))
            dx = float(np.mean([c[0] - r[0] for c, r in zip(cur_centers, ref_centers)]))
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                data = nd_shift(data, (-dy, -dx), mode='nearest')
                print(f"    aligned {fname}: shift=({dx:+.1f}, {dy:+.1f}) px")

        # gebruikt de nieuwe aperature cirkels ipv voorheen opgeslagen coords.
        apertures   = CircularAperture(ref_centers, r=APERTURE_R)
        annuli      = CircularAnnulus(ref_centers, r_in=SKY_IN, r_out=SKY_OUT)
        phot        = aperture_photometry(data, apertures)
        sky         = aperture_photometry(data, annuli)
        sky_per_pix = sky['aperture_sum'] / annuli.area
        net_flux    = phot['aperture_sum'] - sky_per_pix * apertures.area
        #magnitude determining.
        inst_mag = -2.5 * np.log10(np.maximum(net_flux.value, 1.0))
        
        # voor nu goed aangezien we maar 2 comp stars per run gebruiken, even nadenken over sebastisan
        # zn laatste comment over alle sterren gebruiken en deze met elkaar vergelijken.
        if len(inst_mag) < 3:
            print(f"  {fname}  skipped — need at least 2 comparison stars")
            continue

        dif 


