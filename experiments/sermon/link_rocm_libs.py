#!/usr/bin/env python3
"""Link ROCm libraries that the SDK wheel lists but does not install.

The `rocm-sdk-libraries` wheel in the halogen image omits several large
libraries (`libMIOpen.so.1`, `librocsolver.so.0`, `librocsparse.so.1`, ...)
while its RECORD still lists them. PyTorch then fails at import with
`ImportError: lib...so: cannot open shared object file`.

Install the wheel into a directory with enough space (the container overlay is
small) and run this helper to symlink the missing files into the expected
site-packages path:

    pip install --no-deps --target /data/rocm-target \
        --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
        'rocm-sdk-libraries==7.14.0'
    python experiments/sermon/link_rocm_libs.py \
        --target /data/rocm-target

The symlinks use no additional space and the pinned PyTorch weights are not
affected.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=Path("/usr/local/lib/python3.12/site-packages"),
    )
    parser.add_argument("--target", type=Path, default=Path("/data/rocm-target"))
    args = parser.parse_args()

    records = glob.glob(str(args.site_packages / "rocm_sdk_libraries-*.dist-info" / "RECORD"))
    if not records:
        raise SystemExit(f"no rocm_sdk_libraries RECORD under {args.site_packages}")

    created = 0
    missing_source: list[str] = []
    for record in records:
        with open(record, encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) < 3:
                    continue
                relative = row[0]
                destination = args.site_packages / relative
                source = args.target / relative
                if destination.exists():
                    continue
                if not source.exists():
                    missing_source.append(relative)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_symlink():
                    destination.unlink()
                destination.symlink_to(source)
                created += 1

    print(f"symlinked {created} file(s)")
    if missing_source:
        print(f"missing from target: {len(missing_source)}")
        for path in missing_source[:10]:
            print(f"  {path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
