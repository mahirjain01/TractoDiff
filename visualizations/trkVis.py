#!/usr/bin/env python3
"""
trkVis.py  –  convert a Track‑Vis v2 file to legacy v1.

Usage:
    python3 trkVis.py  INPUT_v2.trk  [OUTPUT_v1.trk]

If OUTPUT is omitted “_v1” is added before the extension.
"""

import sys, pathlib
import nibabel as nib
from nibabel.streamlines.trk import TrkFile

def v2_to_v1(in_path, out_path):
    trk_v2 = nib.streamlines.load(str(in_path))     # ← gives TrkFile
    sft     = trk_v2.tractogram                     # StatefulTractogram
    header  = trk_v2.header.copy()

    header["version"]  = 1          # legacy format
    header["hdr_size"] = 1000

    TrkFile(sft, header=header).save(str(out_path))
    print(f"✓ wrote v1 file to {out_path}")

def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)

    inp = pathlib.Path(argv[1]).expanduser().resolve()
    out = pathlib.Path(argv[2]).expanduser().resolve() if len(argv) == 3 \
          else inp.with_name(inp.stem + "_v1.trk")

    v2_to_v1(inp, out)

if __name__ == "__main__":
    main(sys.argv)
