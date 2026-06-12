

import csv, sys, warnings
from itertools import groupby
from pathlib import Path

# Windows-console (cp1252) kan Unicode-tekens als → × niet aan → forceer UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy import units as u
import ccdproc as ccdp
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from scipy.ndimage import shift as nd_shift
from photutils.centroids import centroid_com

warnings.filterwarnings("ignore")


FILTERS    = ["r", "g"]   # filters die zijn gebruikt

SZ_PERIOD = 0.120534896      # sz lyn pulsatie periode
SZ_T0     = 2461118.405    # epoch bij maximale magnitude
APERTURE_R = 24            # aperture radius 
SKY_IN     = 30            # inner sky annulus
SKY_OUT    = 45            # outer sky annulus
ALIGN_BOX = 50             # half width van de box die zoekt.

POSITIONS_FILE = "positions.csv"

def load_star_coords():
    #haalt data uit positions.csv, leest welke run en stercoordinaten hij moet gebruiken.
    coords = {}
    if not Path(POSITIONS_FILE).exists():
        return coords
    with open(POSITIONS_FILE, newline='') as f:
        for row in csv.DictReader(f):
            night     = row.get('night') or row.get('datum', '')
            run_label = row.get('run_label') or row.get('ronde', '')
            key = (night, run_label)
            if key in coords:
                continue   # eerste entry per run.
            try:
                sz   = (float(row['sz_x']), float(row['sz_y'])) #zoeken in positions naar eerste positie ster.
                comp = [(float(row['comp1_x']), float(row['comp1_y'])),
                        (float(row['comp2_x']), float(row['comp2_y']))]
                coords[key] = {'sz': sz, 'comp': comp}
            except (KeyError, ValueError):
                pass
    return coords

STAR_COORDS = load_star_coords()

DATA_DIR = Path("data")
RED_DIR  = Path("reduced")
RED_DIR.mkdir(exist_ok=True)


NIGHTS = {
    "20260316": {
        "bias_glob": "bias-*.fit",
        "flat_glob": {"r": "flat-*_r.fit", "g": "flat-*_g.fit"},
        "sci_exptime": 60,
        "runs": [
            {"label": "SZLYN",   "subdir": "szlyncis",
             "science_glob": {"r": "SZLYN-[0-9]*_r60.fit", "g": "SZLYN-[0-9]*_g60.fit"}},
            {"label": "SZLYN-2", "subdir": "szlyncis2",
             "science_glob": {"r": "SZLYN-2-*_r60.fit",    "g": "SZLYN-2-*_g60.fit"}},
        ],
    },
    "20260414": {
        "bias_glob": "dark-*bias.fit",
        "flat_glob": {"r": "flat-*r.fit", "g": "flat-*g.fit"},
        "sci_exptime": 60,
        "runs": [
            {"label": "SZLYN", "subdir": "szlyncis",
             "science_glob": {"r": "sz_lyncis-*r.fit", "g": "sz_lyncis-*g.fit"}},
        ],
    },
    "20260305": {
        "bias_glob": "Calibration-*_bias.fit",
        "flat_glob": {"r": "flat-*_r.fit", "g": "flat-*_g.fit"},
        "sci_exptime": 60,
        "runs": [
            {"label": "SZ_Lyncis", "subdir": "szlyncis",
             "science_glob": {"r": "SZ_Lyncis-*_r.fit", "g": "SZ_Lyncis-*_g.fit"}},
        ],
    },
}


def inv_median(a):
    return 1 / np.median(a)


# zorgen dat hij niet alles opnieuw calibreert indien het niet nodig is.
RECALIBRATE = True

all_calibrated = []

for night, cfg in NIGHTS.items():
    cal_path = DATA_DIR / night / "calibration"
    red_path = RED_DIR  / night
    red_path.mkdir(exist_ok=True)

    if not RECALIBRATE:
        for run in cfg["runs"]:
            run_red = red_path / run["subdir"]
            if not run_red.exists():
                print(f"  {night}/{run['label']}: niet opnieuw gecalibreert.")
                continue
            for filt in FILTERS:
                sci_glob = run["science_glob"].get(filt)
                if not sci_glob:
                    continue
                files = sorted(run_red.glob(sci_glob))
                if not files:
                    continue
                print(f"  {night}/{run['label']}/{filt}: {len(files)} reduced frames loaded")
                for f in files:
                    ccd = CCDData.read(str(f), unit='adu')
                    all_calibrated.append((night, run["label"], filt, f.name, ccd, ccd.header))
        continue

    print(f"\n{'='*60}")
    print(f"  Night: {night}")
    print(f"{'='*60}")

    print(f"combining bias")
    bias_files = sorted(cal_path.glob(cfg["bias_glob"]))
    if not bias_files:
        print(f"  no bias files skip {night}")
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

    # ── alle dark-frames groeperen op EXPTIME uit de header ──────────────────
    print(f"  darks inlezen en groeperen op EXPTIME...")
    dark_groups = {}   # {exptime: [Path, ...]}
    for f in sorted(cal_path.glob("*.fit")):
        try:
            hdr = CCDData.read(str(f), unit='adu').header
            imgtype = str(hdr.get('IMAGETYP', hdr.get('FRAME', ''))).strip().lower()
            if 'dark' in imgtype:
                et = float(hdr.get('EXPTIME', 0))
                if et > 0:
                    dark_groups.setdefault(et, []).append(f)
        except Exception:
            pass

    if not dark_groups:
        print(f"  geen darks gevonden — {night} overgeslagen")
        continue
    print(f"  beschikbare dark exptimes: {sorted(dark_groups.keys())}s")

    # ── science dark: exacte match op sci_exptime, anders dichtstbij ─────────
    sci_et   = cfg["sci_exptime"]
    best_sci_et = min(dark_groups, key=lambda t: abs(t - sci_et))
    dark_sci_files = dark_groups[best_sci_et]
    if best_sci_et != sci_et:
        print(f"  science dark: geen {sci_et}s — gebruik {best_sci_et}s (meest nabij)")
    else:
        print(f"  science dark: {best_sci_et}s (exacte match)")

    combined_dark_sci = ccdp.combine(
        [ccdp.subtract_bias(CCDData.read(str(f), unit='adu'), combined_bias)
         for f in dark_sci_files],
        method='average',
        sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
        mem_limit=350e6,
    )
    combined_dark_sci.meta['combined'] = True
    combined_dark_sci.meta['EXPTIME']  = best_sci_et
    combined_dark_sci.uncertainty = None
    combined_dark_sci.write(red_path / 'combined_dark_sci.fit', overwrite=True)
    print(f"  {len(dark_sci_files)} frames ({best_sci_et}s) → combined_dark_sci.fit")

    # ── flat dark: kies beste groep op basis van flat-exptime (per filterlus) ─
    # flat_exptime wordt bepaald bij de flats hieronder; dark_groups bewaard voor gebruik

    combined_flats = {}  
#Master flats
    for filt in FILTERS:
        flat_glob = cfg["flat_glob"].get(filt)
        flat_files = sorted(cal_path.glob(flat_glob)) if flat_glob else []

        if not flat_files:
            print(f"\n  No flat files filter {filt} in {night} ")
            continue

        flat_exptime = CCDData.read(str(flat_files[0]), unit='adu').header.get('EXPTIME', 1.0)

        # kies beste dark-groep voor deze flats
        best_flat_et = min(dark_groups, key=lambda t: abs(t - flat_exptime))
        dark_flat_files = dark_groups[best_flat_et]
        combined_dark_flat = ccdp.combine(
            [ccdp.subtract_bias(CCDData.read(str(f), unit='adu'), combined_bias)
             for f in dark_flat_files],
            method='average',
            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
            mem_limit=350e6,
        )
        combined_dark_flat.meta['combined'] = True
        combined_dark_flat.meta['EXPTIME']  = best_flat_et
        combined_dark_flat.uncertainty = None

        if best_flat_et != flat_exptime:
            scale_factor = flat_exptime / best_flat_et
            mdark_flat   = combined_dark_flat.multiply(scale_factor)
            print(f"  flats {filt}: {len(flat_files)} frames ({flat_exptime}s) — "
                  f"dark {best_flat_et}s × {scale_factor:.4f} → {flat_exptime}s")
        else:
            mdark_flat = combined_dark_flat
            print(f"  flats {filt}: {len(flat_files)} frames ({flat_exptime}s) — "
                  f"dark {best_flat_et}s exacte match, geen schaling")
        # dark is nu op flat-exptime; header pinnen zodat subtract_dark direct aftrekt
        mdark_flat.meta['EXPTIME'] = flat_exptime
        calibrated_flats = []
        for f in flat_files:
            ccd = CCDData.read(str(f), unit='adu')
            ccd = ccdp.subtract_bias(ccd, combined_bias)
            # dark is al handmatig geschaald → scale=False = directe aftrek
            ccd = ccdp.subtract_dark(ccd, mdark_flat,
                                     exposure_time='EXPTIME', exposure_unit=u.second,
                                     scale=False)
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


#Skypicture van bepaalde run
#if not all_calibrated:
    print("No calibrations made.")
#else:
    print(f"Generating field maps")

    seen_runs = set()
    for night, run_label, filt, fname, ccd, hdr in all_calibrated:
        key = (night, run_label)
        if key in seen_runs:
            continue
        seen_runs.add(key)

        data   = ccd.data
        median = np.median(data)
        std    = np.std(data)

        finder  = DAOStarFinder(fwhm=8, threshold=10 * std)
        sources = finder(data - median)

        tag = f"{night}_{run_label}".replace(" ", "_")
        out = f"field_stars_{tag}.png"

        _, ax = plt.subplots(figsize=(14, 14))
        vmin, vmax = np.percentile(data, [1, 99])
        ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        if sources is not None:
            ax.scatter(sources['xcentroid'], sources['ycentroid'],
                       s=40, facecolors='none', edgecolors='cyan', linewidths=0.8)
            for s in sources:
                ax.text(s['xcentroid'] + 15, s['ycentroid'],
                        f"({s['xcentroid']:.0f}, {s['ycentroid']:.0f})",
                        color='yellow', fontsize=5)
        ax.set_title(f"{night} / {run_label} — filter {filt}\n"
                     "SZ Lyn vinden en comp stars")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        n_src = len(sources) if sources else 0
        print(f"  {out}  ({n_src} stars detected)")

#Methode aangeraden door sebastian, de code maakt een box om de laatste positie, in deze box
# zoekt het programma naar de hoogste magnitude en maakt daar een nieuwe aperature cirkel omheen
# iom dit process te herhalen voor alle andere runs.
def star_centers(data, positions):

    h, w = data.shape
    centers = []
    for x, y in positions:
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - ALIGN_BOX), min(w, xi + ALIGN_BOX)
        y0, y1 = max(0, yi - ALIGN_BOX), min(h, yi + ALIGN_BOX)
        box = data[y0:y1, x0:x1].astype(float)
        box = box - np.median(box)
        np.clip(box, 0, None, out=box)
        cx, cy = centroid_com(box)
        centers.append((cx + x0, cy + y0))
    return centers

#Magnitude meten vanuit de aperature cirkels.
def do_photometry(frames, night, run_label, filt, sz_xy, comp_stars):
    all_positions = [sz_xy] + list(comp_stars)
    ref_centers   = None
    rows = []
    comp1_flux = []   # net-flux comp1 over alle frames (voor empirische run-error)
    comp2_flux = []   # net-flux comp2 over alle frames

    for fname, ccd, hdr in frames:
        data = ccd.data.astype(float)

        cur_centers = star_centers(data, all_positions)

        if ref_centers is None:
            ref_centers = cur_centers           # first frame defines reference
        else:
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
        net_flux_arr = net_flux.value

        if np.any(net_flux_arr <= 0):
            print(f"  {fname}  skipped — star outside aperture (flux <= 0)")
            continue

        inst_mag = -2.5 * np.log10(net_flux_arr)

        # voor nu goed aangezien we maar 2 comp stars per run gebruiken, even nadenken over sebastisan
        # zn laatste comment over alle sterren gebruiken en deze met elkaar vergelijken.
        if len(inst_mag) < 3:
            print(f"  {fname}  skipped — need at least 2 comparison stars")
            continue

        diff_mag     = float(inst_mag[0]) - float(inst_mag[1])
        diff_mag_ref = float(inst_mag[1]) - float(inst_mag[2])

        # comp-flux verzamelen; de error volgt na de lus uit de spreiding over de run
        comp1_flux.append(float(net_flux_arr[1]))
        comp2_flux.append(float(net_flux_arr[2]))

        hjd = hdr.get('JD-HELIO', hdr.get('JD', 0.0))
        rows.append({
            'filename': fname,
            'HJD': round(hjd, 7),
            'diff_mag': round(diff_mag, 5),
            'diff_mag_ref': round(diff_mag_ref, 5),
            'mag_err': None,            # per run, ingevuld na de lus
            'inst_mag_sz': round(float(inst_mag[0]), 5),
        })
        print(f"  {fname}  HJD={hjd:.6f}  diff_mag={diff_mag:+.4f}  diff_mag_ref={diff_mag_ref:+.4f}")

    # ── error op de magnitude: empirische spreiding van de comp-sterren over de run ──
    #   σ = 2.5/ln10 · √( (Δf₁/f₁)² + (Δf₂/f₂)² ),   Δf/f = std(flux)/mean(flux) per comp-ster
    comp1 = np.asarray(comp1_flux, dtype=float)
    comp2 = np.asarray(comp2_flux, dtype=float)
    if len(comp1) >= 2:
        rel1 = comp1.std(ddof=1) / comp1.mean()
        rel2 = comp2.std(ddof=1) / comp2.mean()
        ref_err = (2.5 / np.log(10)) * float(np.hypot(rel1, rel2))
    else:
        rel1 = rel2 = ref_err = float('nan')   # te weinig frames voor een spreiding
    for r in rows:
        r['mag_err'] = round(ref_err, 5)
    print(f"  run-error (empirisch): Δf₁/f₁={rel1:.4f}  Δf₂/f₂={rel2:.4f}  →  mag_err=±{ref_err:.4f}")

    #maakt csvfile aan met alle data voor analyse. + plot een mooie curve met genormaliseerde controle
    #magnitude/uitgezoomd om het duidelijker te maken.
    tag       = f"{night}_{run_label}_{filt}".replace(" ", "_")
    csv_dir   = Path(__file__).parent / "csv"
    csv_dir.mkdir(exist_ok=True)
    csv_name  = csv_dir / f"lightcurve_{tag}.csv"
    plot_name = f"lightcurve_{tag}.png"

    with open(csv_name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'HJD', 'diff_mag', 'diff_mag_ref', 'mag_err', 'inst_mag_sz'])
        writer.writeheader()
        writer.writerows(rows)

    hjds     = [r['HJD']        for r in rows]
    mags     = [r['diff_mag']   for r in rows]
    ref_mags = [r['diff_mag_ref'] for r in rows]
    sz_mags  = [r['inst_mag_sz'] for r in rows]
    hours    = [(t - hjds[0]) * 24 for t in hjds]

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    ax_top.scatter(hours, mags, s=60, color='steelblue', zorder=3)
    ax_top.plot(hours, mags, color='steelblue', alpha=0.5)
    ax_top.invert_yaxis()
    ax_top.set_ylabel("diffMag  (SZ Lyn − Ref)")
    ax_top.set_title(f"SZ Lyncis — {night} / {run_label} — {filt} filter")
    ax_top.grid(True, alpha=0.3)

    ax_mid.scatter(hours, sz_mags, s=60, color='darkorchid', zorder=3)
    ax_mid.plot(hours, sz_mags, color='darkorchid', alpha=0.5)
    ax_mid.invert_yaxis()
    ax_mid.set_ylabel("inst mag  (SZ Lyn only)")
    ax_mid.set_title("SZ Lyn — instrumental magnitude")
    ax_mid.grid(True, alpha=0.3)

    ax_bot.scatter(hours, ref_mags, s=60, color='tomato', zorder=3)
    ax_bot.plot(hours, ref_mags, color='tomato', alpha=0.5)
    ref_center = (max(ref_mags) + min(ref_mags)) / 2
    half_span  = (max(mags) - min(mags)) / 2 * 1.4
    ax_bot.set_ylim(ref_center - half_span, ref_center + half_span)
    ax_bot.invert_yaxis()
    ax_bot.set_ylabel("diffMagRef  (Ref Check)")
    ax_bot.set_xlabel(f"Hours since HJD {hjds[0]:.4f}")
    ax_bot.set_title("Reference star stability")
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_name, dpi=150)
    plt.close(fig)
    print(f"  Saved: {csv_name}  {plot_name}")


print("Photometry")

for (night, run_label, filt), group in groupby(all_calibrated, key=lambda x: (x[0], x[1], x[2])):
    coords = STAR_COORDS.get((night, run_label))
    if not coords or coords["sz"] is None:
        print(f"\n  Skipping {night} / {run_label} / {filt} — fill in STAR_COORDS to enable")
        continue
    frames = [(fname, ccd, hdr) for _, _, _, fname, ccd, hdr in group]
    print(f"\n  --- {night} / {run_label} / {filt} ({len(frames)} frames) ---")
    do_photometry(frames, night, run_label, filt, coords["sz"], coords["comp"])

# fase plopty
def plot_combined_phased():
    HERE          = Path(__file__).resolve().parent
    CSV_DIR       = HERE / "csv"
    ZOOM_DIR      = CSV_DIR / "zoomed_pictures"
    MIN_PTS       = 3
    FILTER_COLORS = {"r": "red", "g": "blue"}

    print(f"  csv dir  : {CSV_DIR}  (exists: {CSV_DIR.exists()})")
    print(f"  zoom dir : {ZOOM_DIR}  (exists: {ZOOM_DIR.exists()})")

    root_files = sorted(CSV_DIR.glob("lightcurve_*.csv"))

    csv_files = []
    for root in root_files:
        zoomed = sorted(ZOOM_DIR.glob(f"{root.stem}_zoom*.csv")) if ZOOM_DIR.exists() else []
        if zoomed:
            csv_files.append(zoomed[-1])
            print(f"  {root.name} → ZOOMED: {zoomed[-1].name}")
        else:
            csv_files.append(root)
            print(f"  {root.name} → no zoom found, using root")

    fig, ax = plt.subplots(figsize=(14, 6))
    plotted = 0

    for path in csv_files:
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))

        hjds = np.array([float(r['HJD'])      for r in rows])
        mags = np.array([float(r['diff_mag']) for r in rows])
        # per-point Poisson error; fall back to ref-check scatter for older CSVs
        if rows[0].get('mag_err', '') != '':
            errs = np.array([float(r['mag_err']) for r in rows])
        else:
            ref_mags = np.array([float(r['diff_mag_ref']) for r in rows])
            errs = np.full(len(mags), float(np.std(ref_mags)))

        #fase bepaling.
        phases = ((hjds - SZ_T0) / SZ_PERIOD) % 1

        ph_line  = phases.astype(float).copy()
        mag_line = mags.astype(float).copy()
        for idx in np.where(np.abs(np.diff(phases)) > 0.5)[0][::-1]:
            ph_line  = np.insert(ph_line,  idx + 1, np.nan)
            mag_line = np.insert(mag_line, idx + 1, np.nan)

        stem_clean = path.stem.split("_zoom")[0]
        parts  = stem_clean.split("_")
        filt   = parts[-1] if parts[-1] in ("r", "g") else "?"
        label  = "_".join(parts[1:])

        color  = FILTER_COLORS.get(filt, "gray")
        marker = "o" if filt == "r" else "s"

        ax.errorbar(phases, mags, yerr=errs, fmt='none',
                    ecolor=color, elinewidth=0.8, capsize=2, zorder=2)
        ax.scatter(phases, mags, s=25, color=color, marker=marker, zorder=3, label=label)
        ax.plot(ph_line, mag_line, color=color, linewidth=1.0, alpha=0.7)
        plotted += 1
        print(f"  {path.name}: {len(rows)} pts, phase {phases.min():.4f}–{phases.max():.4f}")

    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel(f"Phase  (P = {SZ_PERIOD} d,  T₀ = {SZ_T0})")
    ax.set_ylabel("diff_mag  (SZ Lyn − Ref)")
    ax.set_title("SZ Lyncis — phase-folded light curve")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("phase_plot.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: phase_plot.png  ({plotted} runs plotted)")


plot_combined_phased()
print("eindelijk klaar")
