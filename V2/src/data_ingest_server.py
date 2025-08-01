"""
src/data_ingest_server.py
-------------------------
Loads Week7.sav directly from the Maastricht Study network volume
and writes a cached pickle in the same secure folder.

• No data leave the server share.
• Prints basic shape and missingness stats.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from pyreadstat import read_sav


def parse_args() -> argparse.Namespace:
    base = "/Volumes/education/DMS_744_S_Mirkialangaroodi/Databases"
    parser = argparse.ArgumentParser(description="Server‑side SPSS → pandas.")
    parser.add_argument(
        "--input",
        default=f"{base}/Week7.sav",
        help="Absolute path to the SPSS file on the secure volume",
    )
    parser.add_argument(
        "--out-pkl",
        default=f"{base}/Week8.pkl",
        help="Output pickle path (same secure volume)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sp_path = Path(args.input)
    out_path = Path(args.out_pkl)

    if not sp_path.exists():
        sys.exit(f"❌  Input SPSS file not found: {sp_path}")

    print(f"⏳  Reading: {sp_path}")
    df, _meta = read_sav(sp_path)
    print(f"✅  Loaded DataFrame shape: {df.shape}")

    # Basic missingness report
    miss_pct = (df.isna().mean() * 100).sort_values(ascending=False).head(10)
    print("\n🔍  Top‑10 % missing values:")
    print(miss_pct.to_string(float_format='%.1f'))

    # Cache pickle on the same secure share
    df.to_pickle(out_path, protocol=4)
    print(f"\n💾  Pickle saved to: {out_path}")

    print("\n🎉  Ingestion complete. Data remain on secure volume.")


if __name__ == "__main__":
    main()
