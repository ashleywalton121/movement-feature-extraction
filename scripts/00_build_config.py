"""
00_build_config.py — Match patient EM IDs to N:\\DbData folders and build config.yaml

Phase 0 of the M1 video-EEG pipeline. Maps 28 patients from
precentral_m1_electrodes.csv to their data folders on N:\\DbData.

Usage:
    python scripts/00_build_config.py                          # full scan
    python scripts/00_build_config.py --no-scan                # name matching only (fast)
    python scripts/00_build_config.py --first-name-filter f.csv  # resolve ambiguous matches
    python scripts/00_build_config.py --patient EM1188         # single patient (debug)
"""

import argparse
import csv
import os
import re
import time
from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ELECTRODE_CSV = PROJECT_ROOT / "precentral_m1_electrodes.csv"
DATA_ROOT = Path(r"N:\DbData")
OUTPUT_DIR = PROJECT_ROOT / "output"

MATCH_REPORT_CSV = OUTPUT_DIR / "match_report.csv"
CONFIG_YAML = OUTPUT_DIR / "config.yaml"


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------
def normalize_name(name: str) -> str:
    """Lowercase, strip apostrophes/spaces/hyphens for fuzzy last-name matching.

    O'Connell  → oconnell
    de Oliveira → deoliveira
    McMahon    → mcmahon
    """
    name = name.lower()
    name = re.sub(r"['\s\-]", "", name)
    return name


def extract_last_name_from_folder(folder_name: str) -> str:
    """Extract the last name portion from 'LastName~ FirstName_UUID'.

    Returns the normalized last name, or empty string if pattern doesn't match.
    """
    # Split on '~ ' (tilde + space) to get last name
    parts = folder_name.split("~")
    if len(parts) < 2:
        return ""
    return normalize_name(parts[0].strip())


def extract_first_name_from_folder(folder_name: str) -> str:
    """Extract the first name portion from 'LastName~ FirstName_UUID'.

    Returns raw (un-normalized) first name, or empty string.
    The first name may be truncated in the folder name.
    """
    parts = folder_name.split("~")
    if len(parts) < 2:
        return ""
    after_tilde = parts[1].strip()  # " FirstName_UUID..."
    # First name ends at the last underscore before the UUID
    # Pattern: everything before _<hex>{8}-<hex>{4}-...
    uuid_match = re.search(r"_([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-)", after_tilde)
    if uuid_match:
        first_name = after_tilde[: uuid_match.start()]
        return first_name.strip()
    # Fallback: take everything before the first underscore
    parts2 = after_tilde.split("_", 1)
    return parts2[0].strip()


def extract_uuid_from_folder(folder_name: str) -> str:
    """Extract UUID from folder name."""
    uuid_match = re.search(
        r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
        folder_name,
    )
    return uuid_match.group(1) if uuid_match else ""


# ---------------------------------------------------------------------------
# Read electrode CSV
# ---------------------------------------------------------------------------
def load_patients(csv_path: Path) -> list[dict]:
    """Read precentral_m1_electrodes.csv and return list of patient dicts."""
    patients = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            electrodes_raw = row["Electrode_Names"].strip()
            electrodes = [e.strip() for e in electrodes_raw.split(";") if e.strip()]
            patients.append(
                {
                    "patient_id": row["Patient_ID"].strip(),
                    "last_name": row["Last_name"].strip(),
                    "electrode_count": int(row["Electrode_Count"]),
                    "electrodes": electrodes,
                    "normalized_name": normalize_name(row["Last_name"].strip()),
                }
            )
    return patients


# ---------------------------------------------------------------------------
# Folder listing and matching
# ---------------------------------------------------------------------------
def list_dbdata_folders(data_root: Path) -> list[str]:
    """Single os.listdir call to get all folder names from N:\\DbData."""
    print(f"Listing folders in {data_root} ...")
    t0 = time.time()
    try:
        all_entries = os.listdir(data_root)
    except OSError as e:
        print(f"ERROR: Cannot access {data_root}: {e}")
        raise
    elapsed = time.time() - t0
    print(f"  Found {len(all_entries):,} entries in {elapsed:.1f}s")
    return all_entries


def build_name_index(folder_names: list[str]) -> dict[str, list[str]]:
    """Build a dict mapping normalized last names → list of folder names."""
    index: dict[str, list[str]] = {}
    for fname in folder_names:
        norm = extract_last_name_from_folder(fname)
        if norm:
            index.setdefault(norm, []).append(fname)
    return index


def match_patients(
    patients: list[dict],
    name_index: dict[str, list[str]],
    first_name_filter: dict[str, str] | None = None,
) -> list[dict]:
    """Match each patient to their DbData folders by normalized last name.

    Returns list of match result dicts.
    """
    results = []
    for pat in patients:
        norm_name = pat["normalized_name"]
        matched_folders = name_index.get(norm_name, [])

        # Apply first-name filter if provided
        if first_name_filter and pat["patient_id"] in first_name_filter:
            expected_first = first_name_filter[pat["patient_id"]].lower()
            filtered = [
                f
                for f in matched_folders
                if extract_first_name_from_folder(f).lower().startswith(expected_first)
                or expected_first.startswith(
                    extract_first_name_from_folder(f).lower()
                )
            ]
            status = "filtered" if filtered else "no_match"
            matched_folders = filtered
        elif len(matched_folders) == 0:
            status = "no_match"
        elif _has_multiple_distinct_first_names(matched_folders):
            status = "ambiguous"
        else:
            status = "confirmed"

        # Collect first names found for the match report
        first_names_found = sorted(
            set(extract_first_name_from_folder(f) for f in matched_folders)
        )

        results.append(
            {
                "patient_id": pat["patient_id"],
                "last_name": pat["last_name"],
                "electrode_count": pat["electrode_count"],
                "electrodes": pat["electrodes"],
                "status": status,
                "folder_count": len(matched_folders),
                "first_names_found": first_names_found,
                "matched_folders": matched_folders,
            }
        )
    return results


def _has_multiple_distinct_first_names(folders: list[str]) -> bool:
    """Check if matched folders contain multiple distinct first names.

    This suggests the last name matched multiple unrelated patients.
    """
    first_names = set()
    for f in folders:
        fn = extract_first_name_from_folder(f).lower()
        if fn:
            first_names.add(fn)
    return len(first_names) > 1


# ---------------------------------------------------------------------------
# Folder content scanning
# ---------------------------------------------------------------------------
def scan_folder_contents(folder_path: Path) -> dict:
    """Use os.scandir to efficiently inventory a single study folder.

    Returns dict with counts of AVI, EEG, VTC, EDF, and other files.
    """
    avi_count = 0
    eeg_count = 0
    vtc_count = 0
    edf_count = 0
    other_count = 0

    try:
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if not entry.is_file(follow_symlinks=False):
                    continue
                name_lower = entry.name.lower()
                if name_lower.endswith(".avi"):
                    avi_count += 1
                elif name_lower.endswith(".eeg"):
                    eeg_count += 1
                elif name_lower.endswith(".vtc"):
                    vtc_count += 1
                elif name_lower.endswith(".edf"):
                    edf_count += 1
                else:
                    other_count += 1
    except OSError:
        pass  # folder may not be accessible

    return {
        "avi_count": avi_count,
        "eeg_count": eeg_count,
        "vtc_count": vtc_count,
        "edf_count": edf_count,
        "other_count": other_count,
        "has_video": avi_count > 0,
        "has_eeg": eeg_count > 0,
    }


def scan_matched_folders(
    match_results: list[dict], data_root: Path
) -> list[dict]:
    """Scan contents of all matched folders and attach inventory data."""
    total_to_scan = sum(r["folder_count"] for r in match_results)
    print(f"\nScanning contents of {total_to_scan:,} matched folders ...")
    t0 = time.time()
    scanned = 0

    for result in match_results:
        folder_details = []
        for folder_name in result["matched_folders"]:
            folder_path = data_root / folder_name
            contents = scan_folder_contents(folder_path)
            folder_details.append(
                {
                    "folder_name": folder_name,
                    "uuid": extract_uuid_from_folder(folder_name),
                    **contents,
                }
            )
            scanned += 1
            if scanned % 100 == 0:
                elapsed = time.time() - t0
                rate = scanned / elapsed if elapsed > 0 else 0
                print(
                    f"  Scanned {scanned}/{total_to_scan} "
                    f"({elapsed:.1f}s, {rate:.0f} folders/s)"
                )

        result["matched_folders_detail"] = folder_details

        # Build summary
        result["summary"] = {
            "total_folders": len(folder_details),
            "folders_with_video": sum(
                1 for d in folder_details if d["has_video"]
            ),
            "folders_with_eeg": sum(
                1 for d in folder_details if d["has_eeg"]
            ),
            "total_avi_files": sum(d["avi_count"] for d in folder_details),
            "total_eeg_files": sum(d["eeg_count"] for d in folder_details),
        }

    elapsed = time.time() - t0
    print(f"  Done scanning in {elapsed:.1f}s")
    return match_results


# ---------------------------------------------------------------------------
# Output: match_report.csv
# ---------------------------------------------------------------------------
def write_match_report(match_results: list[dict], output_path: Path) -> None:
    """Write a summary CSV for quick review of name matches."""
    rows = []
    for r in match_results:
        rows.append(
            {
                "Patient_ID": r["patient_id"],
                "Last_Name": r["last_name"],
                "Status": r["status"],
                "Folder_Count": r["folder_count"],
                "First_Names_Found": "; ".join(r["first_names_found"]),
                "Electrode_Count": r["electrode_count"],
            }
        )
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nMatch report written to: {output_path}")


# ---------------------------------------------------------------------------
# Output: config.yaml
# ---------------------------------------------------------------------------
def write_config_yaml(
    match_results: list[dict],
    data_root: Path,
    electrode_csv: Path,
    output_path: Path,
) -> None:
    """Write the full config.yaml for downstream pipeline scripts."""
    config = {
        "data_root": str(data_root),
        "electrode_csv": str(electrode_csv),
        "patients": {},
    }

    for r in match_results:
        patient_entry = {
            "last_name": r["last_name"],
            "electrode_count": r["electrode_count"],
            "electrodes": r["electrodes"],
            "status": r["status"],
        }

        # Include folder details if scanning was done
        if "matched_folders_detail" in r:
            patient_entry["matched_folders"] = r["matched_folders_detail"]
            patient_entry["summary"] = r["summary"]
        else:
            # No scan — just include folder names
            patient_entry["matched_folder_names"] = r["matched_folders"]
            patient_entry["folder_count"] = r["folder_count"]

        config["patients"][r["patient_id"]] = patient_entry

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, width=120)
    print(f"Config written to: {output_path}")


# ---------------------------------------------------------------------------
# Load first-name filter CSV
# ---------------------------------------------------------------------------
def load_first_name_filter(csv_path: str) -> dict[str, str]:
    """Load patient_first_names.csv → {EM_ID: expected_first_name}."""
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row.get("Patient_ID", "").strip()
            first_name = row.get("First_Name", "").strip()
            if patient_id and first_name:
                mapping[patient_id] = first_name
    print(f"Loaded first-name filter with {len(mapping)} entries from {csv_path}")
    return mapping


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Match patient EM IDs to N:\\DbData folders and build config.yaml"
    )
    parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Name matching only — skip folder content scanning (fast)",
    )
    parser.add_argument(
        "--first-name-filter",
        type=str,
        default=None,
        metavar="CSV",
        help="Path to patient_first_names.csv to resolve ambiguous matches",
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        metavar="EM_ID",
        help="Process a single patient (for debugging)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DATA_ROOT),
        help=f"Root data directory (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    match_report_csv = output_dir / "match_report.csv"
    config_yaml = output_dir / "config.yaml"

    # 1. Load patients
    print(f"Loading patients from {ELECTRODE_CSV} ...")
    patients = load_patients(ELECTRODE_CSV)
    print(f"  Loaded {len(patients)} patients")

    # Filter to single patient if requested
    if args.patient:
        patients = [p for p in patients if p["patient_id"] == args.patient]
        if not patients:
            print(f"ERROR: Patient {args.patient} not found in CSV")
            return
        print(f"  Filtered to single patient: {args.patient}")

    # 2. List DbData folders
    all_folders = list_dbdata_folders(data_root)

    # 3. Build name index
    name_index = build_name_index(all_folders)
    print(f"  Indexed {len(name_index):,} unique last names")

    # 4. Load first-name filter if provided
    first_name_filter = None
    if args.first_name_filter:
        first_name_filter = load_first_name_filter(args.first_name_filter)

    # 5. Match patients to folders
    print("\nMatching patients to folders ...")
    match_results = match_patients(patients, name_index, first_name_filter)

    # Print summary
    status_counts = {}
    for r in match_results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    print("  Match results:")
    for status, count in sorted(status_counts.items()):
        print(f"    {status}: {count}")

    # 6. Scan folder contents (unless --no-scan)
    if not args.no_scan:
        match_results = scan_matched_folders(match_results, data_root)

    # 7. Write outputs
    write_match_report(match_results, match_report_csv)
    if not args.no_scan:
        write_config_yaml(match_results, data_root, ELECTRODE_CSV, config_yaml)
    else:
        print("\nSkipped config.yaml (use without --no-scan to generate)")

    # 8. Print per-patient summary
    print("\n" + "=" * 70)
    print("Per-patient summary:")
    print("=" * 70)
    for r in match_results:
        line = f"  {r['patient_id']}  {r['last_name']:<15s}  {r['status']:<12s}  {r['folder_count']} folders"
        if r["first_names_found"]:
            line += f"  ({', '.join(r['first_names_found'])})"
        if "summary" in r:
            s = r["summary"]
            line += f"  | {s['total_avi_files']} AVIs, {s['folders_with_video']} w/video"
        print(line)


if __name__ == "__main__":
    main()
