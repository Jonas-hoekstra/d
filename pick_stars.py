"""
SZ Lyncis - Sterren Picker
Runs once per observation run (4 runs total).
For each run, shows the first frame, lets you pick SZ Lyncis and 2 comparison
stars by number, and saves the pixel coordinates to positions.csv.

Usage: python pick_stars.py
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.nddata import CCDData
from photutils.detection import DAOStarFinder


AANTAL_COMP_STERREN = 2
POSITIONS_BESTAND   = "positions.csv"
FWHM                = 8
DREMPEL             = 5          # sigma threshold (lowered to detect fainter stars)
REDUCED_MAP         = Path("reduced")

RONDE_NAMEN = {
    ("20260316", "szlyncis"):  "SZLYN",
    ("20260316", "szlyncis2"): "SZLYN-2",
    ("20260414", "szlyncis"):  "SZLYN",
    ("20260305", "szlyncis"):  "SZ_Lyncis",
}

KOLOMNAMEN = ['night', 'run_label', 'sz_x', 'sz_y',
              'comp1_x', 'comp1_y', 'comp2_x', 'comp2_y']


def laad_al_gedaan():
    """Return set of (night, run_label) pairs that already have an entry."""
    if not Path(POSITIONS_BESTAND).exists():
        return set()
    with open(POSITIONS_BESTAND, newline='') as f:
        rows = list(csv.DictReader(f))
    # support both old per-frame format and new per-run format
    done = set()
    for rij in rows:
        night     = rij.get('night') or rij.get('datum', '')
        run_label = rij.get('run_label') or rij.get('ronde', '')
        if night and run_label:
            done.add((night, run_label))
    return done


def verzamel_eerste_fotos():
    """Return the first r-filter frame for each observation run."""
    fotos = []
    seen  = set()
    for datum_map in sorted(REDUCED_MAP.iterdir()):
        if not datum_map.is_dir():
            continue
        for sub_map in sorted(datum_map.iterdir()):
            if not sub_map.is_dir():
                continue
            ronde = RONDE_NAMEN.get((datum_map.name, sub_map.name))
            if ronde is None:
                continue
            sleutel = (datum_map.name, ronde)
            if sleutel in seen:
                continue
            # take the first r-frame (fall back to g if no r exists)
            for filt_suffix in ('r', 'g'):
                frames = sorted(sub_map.glob(f'*.fit'))
                frames_filt = [f for f in frames
                               if f'_{filt_suffix}' in f.stem.lower()
                               or f.stem.lower().endswith(filt_suffix)]
                if frames_filt:
                    fotos.append((datum_map.name, ronde, frames_filt[0]))
                    seen.add(sleutel)
                    break
    return fotos


def maak_picker_afbeelding(foto_data, sterren):
    _, assen = plt.subplots(figsize=(14, 14))
    vmin = np.percentile(foto_data, 1)
    vmax = np.percentile(foto_data, 99)
    assen.imshow(foto_data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    for nummer, ster in enumerate(sterren, start=1):
        x = float(ster['xcentroid'])
        y = float(ster['ycentroid'])
        assen.plot(x, y, 'o', markersize=14, markerfacecolor='none',
                   markeredgecolor='cyan', markeredgewidth=1.2)
        assen.text(x + 22, y + 22, str(nummer), color='yellow',
                   fontsize=7, fontweight='bold')
    plt.tight_layout()
    plt.savefig("picker_current.png", dpi=150)
    plt.close()


def sla_op(night, run_label, sz_xy, comp_lijst):
    bestand_bestaat = Path(POSITIONS_BESTAND).exists()
    with open(POSITIONS_BESTAND, 'a', newline='') as bestand:
        schrijver = csv.DictWriter(bestand, fieldnames=KOLOMNAMEN)
        if not bestand_bestaat:
            schrijver.writeheader()
        rij = {
            'night': night, 'run_label': run_label,
            'sz_x': round(sz_xy[0], 2), 'sz_y': round(sz_xy[1], 2),
        }
        for i in range(AANTAL_COMP_STERREN):
            rij['comp' + str(i + 1) + '_x'] = round(comp_lijst[i][0], 2)
            rij['comp' + str(i + 1) + '_y'] = round(comp_lijst[i][1], 2)
        schrijver.writerow(rij)


def vraag_nummer(tekst, maximum):
    while True:
        invoer = input(tekst)
        if invoer.strip().isdigit():
            nummer = int(invoer.strip())
            if 1 <= nummer <= maximum:
                return nummer
        print("    Vul een getal in tussen 1 en " + str(maximum) + ".")


# ── main ─────────────────────────────────────────────────────────────────────

alle_runs  = verzamel_eerste_fotos()
al_gedaan  = laad_al_gedaan()
nog_te_doen = [(d, r, p) for d, r, p in alle_runs if (d, r) not in al_gedaan]

print(f"{len(nog_te_doen)} runs to do  "
      f"({len(al_gedaan)}/{len(alle_runs)} already done)")

for teller, (night, run_label, foto_pad) in enumerate(nog_te_doen, start=1):
    print(f"\n[{len(al_gedaan) + teller}/{len(alle_runs)}]  "
          f"{night} / {run_label}  —  {foto_pad.name}")

    foto_data = CCDData.read(str(foto_pad), unit='adu').data.astype(float)
    foto_data = foto_data - np.median(foto_data)

    sterren = DAOStarFinder(fwhm=FWHM, threshold=DREMPEL * np.std(foto_data))(foto_data)
    if sterren is None or len(sterren) == 0:
        print("  No stars detected — skipping. Try lowering DREMPEL in pick_stars.py.")
        continue

    sterren.sort('peak', reverse=True)

    maak_picker_afbeelding(foto_data, sterren)
    print(f"  picker_current.png saved  ({len(sterren)} stars, 1 = brightest)")
    print("  Open picker_current.png, find SZ Lyncis and 2 comparison stars,")
    print("  then enter their numbers below.")

    sz_nummer = vraag_nummer(f"  SZ Lyncis number [1-{len(sterren)}]: ", len(sterren))
    sz_xy = (float(sterren['xcentroid'][sz_nummer - 1]),
             float(sterren['ycentroid'][sz_nummer - 1]))

    comp_lijst = []
    for comp_nr in range(1, AANTAL_COMP_STERREN + 1):
        nr = vraag_nummer(f"  Comp {comp_nr} number [1-{len(sterren)}]: ", len(sterren))
        comp_lijst.append((float(sterren['xcentroid'][nr - 1]),
                           float(sterren['ycentroid'][nr - 1])))

    sla_op(night, run_label, sz_xy, comp_lijst)
    print("  Saved.")

Path("picker_current.png").unlink(missing_ok=True)
print("\nDone! Coordinates saved in " + POSITIONS_BESTAND)
