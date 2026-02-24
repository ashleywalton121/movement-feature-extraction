"""
03_generate_contact_sheet.py — Generate HTML contact sheet for visual triage

Phase 1.3 of the M1 video-EEG pipeline. Reads video_metadata.csv and
thumbnail images (from 02_extract_thumbnails.py) to produce a self-contained
HTML contact sheet for reviewing camera framing, lighting, and patient
visibility across all folders.

Thumbnails are base64-embedded so the HTML file is fully portable.

Usage:
    python scripts/03_generate_contact_sheet.py                    # default
    python scripts/03_generate_contact_sheet.py --patient EM1334   # single patient
    python scripts/03_generate_contact_sheet.py --output sheet.html
"""

import argparse
import base64
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
METADATA_CSV = OUTPUT_DIR / "video_metadata.csv"
THUMBS_DIR = OUTPUT_DIR / "thumbs"
THUMB_INDEX_CSV = THUMBS_DIR / "thumbnail_index.csv"
CONFIG_YAML = OUTPUT_DIR / "config.yaml"
OUTPUT_HTML = OUTPUT_DIR / "contact_sheet.html"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_folder_metadata(metadata_csv: Path) -> pd.DataFrame:
    """Build per-folder metadata summary from video_metadata.csv."""
    df = pd.read_csv(metadata_csv, low_memory=False)
    probed = df[df["is_probed"] == True]

    folder_meta = (
        probed.groupby(["patient_id", "folder_name", "uuid"])
        .agg(
            width=("width", "first"),
            height=("height", "first"),
            fps=("fps", "first"),
            codec=("codec_fourcc", "first"),
            settings_changed=("settings_changed", "first"),
        )
        .reset_index()
    )

    folder_stats = (
        df.groupby(["patient_id", "folder_name", "uuid"])
        .agg(
            avi_count=("file_name", "count"),
            total_duration_sec=("estimated_duration_sec", "sum"),
            total_size_bytes=("file_size_bytes", "sum"),
        )
        .reset_index()
    )

    merged = folder_meta.merge(
        folder_stats, on=["patient_id", "folder_name", "uuid"], how="inner"
    )
    merged.sort_values(["patient_id", "folder_name"], inplace=True)
    return merged


def load_electrode_counts(config_yaml: Path) -> dict[str, int]:
    """Load electrode counts per patient from config.yaml."""
    with open(config_yaml, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {
        pid: info.get("electrode_count", 0)
        for pid, info in config["patients"].items()
    }


def load_thumbnail_index(index_csv: Path) -> pd.DataFrame:
    """Load thumbnail index CSV."""
    return pd.read_csv(index_csv)


def encode_image_base64(path: str) -> str:
    """Read a JPEG file and return base64-encoded data URI."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"
    except (OSError, IOError):
        return ""


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
def format_duration(seconds: float) -> str:
    """Format seconds as Xh Ym."""
    if seconds != seconds:  # NaN
        return "?"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def format_size(size_bytes: float) -> str:
    """Format bytes as human-readable."""
    if size_bytes != size_bytes:  # NaN
        return "?"
    gb = size_bytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = size_bytes / (1024 ** 2)
    return f"{mb:.0f} MB"


def generate_html(
    folder_meta: pd.DataFrame,
    thumb_index: pd.DataFrame,
    electrode_counts: dict[str, int],
    patient_filter: str | None,
) -> str:
    """Generate the full HTML contact sheet with interactive triage controls."""
    if patient_filter:
        folder_meta = folder_meta[folder_meta["patient_id"] == patient_filter]

    # Only include folders that have thumbnails
    thumb_uuids = set(thumb_index[thumb_index["success"] == True]["uuid"].unique())
    folder_meta = folder_meta[folder_meta["uuid"].isin(thumb_uuids)].copy()

    if folder_meta.empty:
        return "<html><body><h1>No folders with thumbnails found.</h1></body></html>"

    # Build thumbnail lookup: uuid -> {position -> base64_data}
    thumb_lookup: dict[str, dict[str, str]] = {}
    for _, row in thumb_index.iterrows():
        if not row["success"]:
            continue
        uuid = row["uuid"]
        pos = row["position"]
        role = row.get("avi_role", "first")
        key = f"{role}_{pos}"
        if uuid not in thumb_lookup:
            thumb_lookup[uuid] = {}
        thumb_lookup[uuid][key] = encode_image_base64(row["output_path"])

    # Group by patient
    patients = folder_meta.groupby("patient_id")

    # Overall stats
    total_patients = folder_meta["patient_id"].nunique()
    total_folders = len(folder_meta)
    total_avis = int(folder_meta["avi_count"].sum())
    total_hours = folder_meta["total_duration_sec"].sum() / 3600
    total_gb = folder_meta["total_size_bytes"].sum() / (1024 ** 3)

    # Build patient nav + cards
    nav_items = []
    patient_sections = []

    for patient_id, group in patients:
        n_folders = len(group)
        n_electrodes = electrode_counts.get(patient_id, 0)
        patient_hours = group["total_duration_sec"].sum() / 3600
        patient_avis = int(group["avi_count"].sum())

        nav_items.append(
            f'<a href="#pat-{patient_id}" class="nav-item">'
            f'{patient_id} <span class="nav-badge">{n_folders}</span></a>'
        )

        # Build folder cards
        folder_cards = []
        for _, frow in group.iterrows():
            uuid = frow["uuid"]
            thumbs = thumb_lookup.get(uuid, {})
            res = f"{int(frow['width'])}x{int(frow['height'])}"
            dur = format_duration(frow["total_duration_sec"])
            size = format_size(frow["total_size_bytes"])
            changed_badge = (
                ' <span class="badge badge-warn">settings changed</span>'
                if frow["settings_changed"] else ""
            )

            # Thumbnail strip
            thumb_imgs = []
            for pos in ["first_start", "first_mid", "first_end"]:
                src = thumbs.get(pos, "")
                label = pos.replace("first_", "")
                if src:
                    thumb_imgs.append(
                        f'<div class="thumb-cell">'
                        f'<img src="{src}" alt="{label}" loading="lazy">'
                        f'<div class="thumb-label">{label}</div></div>'
                    )
                else:
                    thumb_imgs.append(
                        f'<div class="thumb-cell">'
                        f'<div class="thumb-placeholder">no image</div>'
                        f'<div class="thumb-label">{label}</div></div>'
                    )

            # Triage controls
            triage_controls = f"""
          <div class="triage-controls" data-uuid="{uuid}" data-patient="{patient_id}"
               data-folder="{frow['folder_name']}" data-res="{res}"
               data-avi-count="{int(frow['avi_count'])}"
               data-duration="{frow['total_duration_sec']:.0f}">
            <label class="triage-usable">
              <input type="checkbox" class="cb-usable" onchange="onTriageChange(this)">
              <span class="usable-label">Usable</span>
            </label>
            <select class="sel-camera" onchange="onTriageChange(this)">
              <option value="">Camera view...</option>
              <option value="overhead">Overhead</option>
              <option value="angled">Angled</option>
              <option value="side">Side</option>
            </select>
            <select class="sel-lighting" onchange="onTriageChange(this)">
              <option value="">Lighting...</option>
              <option value="day">Day</option>
              <option value="night_ir">Night / IR</option>
              <option value="mixed">Mixed</option>
            </select>
            <select class="sel-visibility" onchange="onTriageChange(this)">
              <option value="">Patient visible...</option>
              <option value="full">Full</option>
              <option value="partial">Partial</option>
              <option value="obscured">Obscured</option>
              <option value="empty">Empty room</option>
            </select>
            <input type="text" class="inp-notes" placeholder="Notes..."
                   onchange="onTriageChange(this)">
          </div>"""

            folder_cards.append(f"""
        <div class="folder-card" id="folder-{uuid}">
          <div class="folder-header">
            <span class="uuid" title="{frow['folder_name']}">{uuid}</span>
            {changed_badge}
          </div>
          <div class="folder-meta">
            {res} &middot; {frow['fps']:.0f} fps &middot; {frow['codec']} &middot;
            {int(frow['avi_count'])} AVIs &middot; {dur} &middot; {size}
          </div>
          <div class="thumb-strip">
            {''.join(thumb_imgs)}
          </div>
          {triage_controls}
        </div>""")

        patient_sections.append(f"""
      <div class="patient-section" id="pat-{patient_id}">
        <div class="patient-header">
          <h2>{patient_id}</h2>
          <div class="patient-stats">
            {n_electrodes} M1 electrodes &middot;
            {n_folders} folders &middot;
            {patient_avis:,} AVIs &middot;
            ~{patient_hours:.0f}h
          </div>
        </div>
        {''.join(folder_cards)}
      </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Video Contact Sheet — M1 Pipeline Triage</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5; color: #333; line-height: 1.4;
  }}
  .header {{
    background: #1a1a2e; color: #fff; padding: 16px 24px;
    position: sticky; top: 0; z-index: 100;
  }}
  .header-top {{ display: flex; justify-content: space-between; align-items: center; }}
  .header h1 {{ font-size: 20px; font-weight: 600; }}
  .header-stats {{ font-size: 13px; color: #aaa; margin-top: 4px; }}
  .header-actions {{ display: flex; gap: 10px; align-items: center; }}
  .progress-text {{ font-size: 13px; color: #aaa; }}
  .progress-bar {{
    width: 120px; height: 6px; background: #333; border-radius: 3px;
    overflow: hidden;
  }}
  .progress-fill {{
    height: 100%; background: #4caf50; border-radius: 3px;
    transition: width 0.3s;
  }}
  .btn {{
    padding: 7px 16px; border: none; border-radius: 4px;
    font-size: 13px; font-weight: 600; cursor: pointer;
  }}
  .btn-export {{ background: #4caf50; color: #fff; }}
  .btn-export:hover {{ background: #43a047; }}
  .btn-export:disabled {{ background: #555; color: #888; cursor: default; }}
  .btn-clear {{ background: transparent; color: #aaa; border: 1px solid #555; }}
  .btn-clear:hover {{ color: #fff; border-color: #888; }}
  .layout {{ display: flex; min-height: 100vh; }}
  .sidebar {{
    width: 180px; background: #fff; border-right: 1px solid #ddd;
    position: sticky; top: 88px; height: calc(100vh - 88px);
    overflow-y: auto; flex-shrink: 0; padding: 8px 0;
  }}
  .nav-item {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 16px; text-decoration: none; color: #333;
    font-size: 13px; font-weight: 500;
  }}
  .nav-item:hover {{ background: #f0f0f0; }}
  .nav-badge {{
    background: #e0e0e0; border-radius: 10px; padding: 1px 7px;
    font-size: 11px; color: #666;
  }}
  .main {{ flex: 1; padding: 16px 24px; max-width: 1200px; }}
  .patient-section {{ margin-bottom: 32px; }}
  .patient-header {{
    display: flex; align-items: baseline; gap: 16px;
    border-bottom: 2px solid #1a1a2e; padding-bottom: 6px;
    margin-bottom: 12px;
  }}
  .patient-header h2 {{ font-size: 18px; color: #1a1a2e; }}
  .patient-stats {{ font-size: 13px; color: #777; }}
  .folder-card {{
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    margin-bottom: 10px; padding: 10px 14px;
    transition: border-color 0.2s, background 0.2s;
  }}
  .folder-card:hover {{ border-color: #999; }}
  .folder-card.tagged {{ border-left: 4px solid #4caf50; }}
  .folder-card.tagged-unusable {{ border-left: 4px solid #ccc; background: #fafafa; }}
  .folder-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
  .uuid {{ font-family: monospace; font-size: 12px; color: #555; }}
  .folder-meta {{ font-size: 12px; color: #888; margin-bottom: 8px; }}
  .badge {{
    display: inline-block; font-size: 10px; padding: 1px 6px;
    border-radius: 3px; font-weight: 600;
  }}
  .badge-warn {{ background: #fff3cd; color: #856404; }}
  .thumb-strip {{ display: flex; gap: 6px; }}
  .thumb-cell {{ text-align: center; }}
  .thumb-cell img {{
    width: 280px; height: auto; border-radius: 3px;
    border: 1px solid #eee; display: block;
  }}
  .thumb-label {{ font-size: 10px; color: #aaa; margin-top: 2px; }}
  .thumb-placeholder {{
    width: 280px; height: 158px; background: #f0f0f0;
    display: flex; align-items: center; justify-content: center;
    color: #ccc; font-size: 12px; border-radius: 3px;
  }}
  .triage-controls {{
    display: flex; align-items: center; gap: 8px; margin-top: 8px;
    padding-top: 8px; border-top: 1px solid #f0f0f0;
    flex-wrap: wrap;
  }}
  .triage-usable {{
    display: flex; align-items: center; gap: 5px; cursor: pointer;
    font-size: 13px; font-weight: 600; user-select: none;
  }}
  .cb-usable {{ width: 18px; height: 18px; cursor: pointer; accent-color: #4caf50; }}
  .usable-label {{ color: #888; }}
  .triage-usable:has(.cb-usable:checked) .usable-label {{ color: #4caf50; }}
  .triage-controls select {{
    padding: 4px 6px; border: 1px solid #ddd; border-radius: 3px;
    font-size: 12px; color: #555; background: #fff; cursor: pointer;
  }}
  .triage-controls select:focus {{ border-color: #999; outline: none; }}
  .inp-notes {{
    flex: 1; min-width: 150px; padding: 4px 8px; border: 1px solid #ddd;
    border-radius: 3px; font-size: 12px; color: #333;
  }}
  .inp-notes:focus {{ border-color: #999; outline: none; }}
  .toast {{
    position: fixed; bottom: 24px; right: 24px; background: #333;
    color: #fff; padding: 12px 20px; border-radius: 6px;
    font-size: 13px; opacity: 0; transition: opacity 0.3s;
    z-index: 200; pointer-events: none;
  }}
  .toast.show {{ opacity: 1; }}
  @media (max-width: 1000px) {{
    .sidebar {{ display: none; }}
    .thumb-cell img {{ width: 200px; }}
    .thumb-placeholder {{ width: 200px; height: 112px; }}
  }}
</style>
</head>
<body>
  <div class="header">
    <div class="header-top">
      <div>
        <h1>Video Contact Sheet &mdash; M1 Pipeline Triage</h1>
        <div class="header-stats">
          {total_patients} patients &middot; {total_folders} folders &middot;
          {total_avis:,} AVIs &middot; ~{total_hours:,.0f}h &middot;
          {total_gb:.0f} GB
        </div>
      </div>
      <div class="header-actions">
        <span class="progress-text" id="progress-text">0 / {total_folders} tagged</span>
        <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
        <button class="btn btn-export" id="btn-export" onclick="exportCSV()">Export CSV</button>
        <button class="btn btn-clear" onclick="clearAll()">Clear all</button>
      </div>
    </div>
  </div>
  <div class="layout">
    <div class="sidebar">
      {''.join(nav_items)}
    </div>
    <div class="main">
      {''.join(patient_sections)}
    </div>
  </div>
  <div class="toast" id="toast"></div>

<script>
const STORAGE_KEY = 'video_triage_v1';
const TOTAL = {total_folders};

// ---- State ----
function loadState() {{
  try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {{}}; }}
  catch {{ return {{}}; }}
}}
function saveState(state) {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}}

// ---- Init: restore saved state ----
function init() {{
  const state = loadState();
  document.querySelectorAll('.triage-controls').forEach(el => {{
    const uuid = el.dataset.uuid;
    const s = state[uuid];
    if (!s) return;
    const cb = el.querySelector('.cb-usable');
    const selCam = el.querySelector('.sel-camera');
    const selLight = el.querySelector('.sel-lighting');
    const selVis = el.querySelector('.sel-visibility');
    const inp = el.querySelector('.inp-notes');
    if (s.usable !== undefined) cb.checked = s.usable;
    if (s.camera) selCam.value = s.camera;
    if (s.lighting) selLight.value = s.lighting;
    if (s.visibility) selVis.value = s.visibility;
    if (s.notes) inp.value = s.notes;
    updateCardStyle(el);
  }});
  updateProgress();
}}

// ---- On any control change ----
function onTriageChange(input) {{
  const controls = input.closest('.triage-controls');
  const uuid = controls.dataset.uuid;
  const state = loadState();
  state[uuid] = {{
    usable: controls.querySelector('.cb-usable').checked,
    camera: controls.querySelector('.sel-camera').value,
    lighting: controls.querySelector('.sel-lighting').value,
    visibility: controls.querySelector('.sel-visibility').value,
    notes: controls.querySelector('.inp-notes').value,
  }};
  saveState(state);
  updateCardStyle(controls);
  updateProgress();
}}

function updateCardStyle(controls) {{
  const card = controls.closest('.folder-card');
  const cb = controls.querySelector('.cb-usable');
  const hasAny = cb.checked ||
    controls.querySelector('.sel-camera').value ||
    controls.querySelector('.sel-lighting').value ||
    controls.querySelector('.sel-visibility').value ||
    controls.querySelector('.inp-notes').value;
  card.classList.toggle('tagged', cb.checked);
  card.classList.toggle('tagged-unusable', hasAny && !cb.checked);
}}

function updateProgress() {{
  const state = loadState();
  const tagged = Object.values(state).filter(s =>
    s.usable || s.camera || s.lighting || s.visibility || s.notes
  ).length;
  document.getElementById('progress-text').textContent = tagged + ' / ' + TOTAL + ' tagged';
  document.getElementById('progress-fill').style.width = (tagged / TOTAL * 100) + '%';
}}

// ---- Export CSV ----
function exportCSV() {{
  const state = loadState();
  const rows = [['patient_id','uuid','folder_name','resolution','avi_count',
                 'duration_sec','usable','camera_view','lighting',
                 'patient_visible','notes']];
  document.querySelectorAll('.triage-controls').forEach(el => {{
    const uuid = el.dataset.uuid;
    const s = state[uuid] || {{}};
    rows.push([
      el.dataset.patient,
      uuid,
      '"' + el.dataset.folder.replace(/"/g, '""') + '"',
      el.dataset.res,
      el.dataset.aviCount,
      el.dataset.duration,
      s.usable ? 'yes' : 'no',
      s.camera || '',
      s.lighting || '',
      s.visibility || '',
      '"' + (s.notes || '').replace(/"/g, '""') + '"',
    ]);
  }});
  const csv = rows.map(r => r.join(',')).join('\\n');
  const blob = new Blob([csv], {{ type: 'text/csv' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'video_triage.csv';
  a.click();
  URL.revokeObjectURL(url);
  showToast('Exported video_triage.csv');
}}

// ---- Clear all ----
function clearAll() {{
  if (!confirm('Clear all triage annotations? This cannot be undone.')) return;
  localStorage.removeItem(STORAGE_KEY);
  document.querySelectorAll('.cb-usable').forEach(cb => cb.checked = false);
  document.querySelectorAll('.sel-camera, .sel-lighting, .sel-visibility')
    .forEach(sel => sel.selectedIndex = 0);
  document.querySelectorAll('.inp-notes').forEach(inp => inp.value = '');
  document.querySelectorAll('.folder-card').forEach(card => {{
    card.classList.remove('tagged', 'tagged-unusable');
  }});
  updateProgress();
  showToast('All annotations cleared');
}}

// ---- Toast ----
function showToast(msg) {{
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}}

// ---- Keyboard shortcut: Ctrl+E to export ----
document.addEventListener('keydown', e => {{
  if ((e.ctrlKey || e.metaKey) && e.key === 'e') {{
    e.preventDefault();
    exportCSV();
  }}
}});

init();
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML contact sheet for visual video triage"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(METADATA_CSV),
        help=f"Path to video_metadata.csv (default: {METADATA_CSV})",
    )
    parser.add_argument(
        "--thumbs-dir",
        type=str,
        default=str(THUMBS_DIR),
        help=f"Thumbnail directory (default: {THUMBS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_HTML),
        help=f"Output HTML path (default: {OUTPUT_HTML})",
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        metavar="EM_ID",
        help="Generate contact sheet for a single patient",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    thumbs_dir = Path(args.thumbs_dir)
    output_path = Path(args.output)
    index_csv = thumbs_dir / "thumbnail_index.csv"

    # 1. Load data
    print(f"Loading metadata from {metadata_path} ...")
    folder_meta = load_folder_metadata(metadata_path)
    print(f"  {len(folder_meta)} folders with probed metadata")

    print(f"Loading thumbnail index from {index_csv} ...")
    thumb_index = load_thumbnail_index(index_csv)
    successful = thumb_index[thumb_index["success"] == True]
    print(f"  {len(successful)} successful thumbnails across "
          f"{successful['uuid'].nunique()} folders")

    print(f"Loading electrode counts from {CONFIG_YAML} ...")
    electrode_counts = load_electrode_counts(CONFIG_YAML)

    # 2. Generate HTML
    print(f"\nGenerating contact sheet ...")
    t0 = time.time()
    html = generate_html(folder_meta, thumb_index, electrode_counts, args.patient)
    elapsed = time.time() - t0
    print(f"  HTML generated in {elapsed:.1f}s")

    # 3. Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {output_path} ({size_mb:.1f} MB)")
    print(f"Open in browser to review.")


if __name__ == "__main__":
    main()
