#!/usr/bin/env python3
"""Build validation/caption_review.html after Track B captioning.

Maps each captioned object to its Phase 4a crop, padded VLM crop, SAM label,
and 3D mean so mismatches can be inspected in a browser.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from scene_io import infer_run_dir  # noqa: E402

log = logging.getLogger("caption_review")


def _load_views(best_views: Path) -> Dict[int, dict]:
    if not best_views.is_file():
        return {}
    views = json.loads(best_views.read_text(encoding="utf-8").replace("Infinity", "null"))
    return {int(v["object_index"]): v for v in views if "object_index" in v}


def _padded_name_from_results(caption_results: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not caption_results.is_file():
        return out
    for row in json.loads(caption_results.read_text(encoding="utf-8")):
        ip = str(row.get("image_path") or "")
        if not ip:
            continue
        name = Path(ip).name
        if "_bbox_" in name:
            try:
                out[int(row["object_index"])] = name
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _load_vocab(vocab_file: Path) -> List[str]:
    if not vocab_file.is_file():
        return []
    return [
        ln.strip()
        for ln in vocab_file.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def _collect_padded_sources(phase4: Path) -> List[Path]:
    dirs = [
        phase4 / "padded_crops",
        Path("/tmp/farm_padded_crops"),
        phase4.parent / "validation" / "padded_crops",
    ]
    return [d for d in dirs if d.is_dir()]


def _ensure_padded_copy(name: str, sources: List[Path], dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / name
    if target.is_file():
        return True
    for src_dir in sources:
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, target)
            return True
    return False


def build_records(
    scene_state: dict,
    *,
    views_by_idx: Dict[int, dict],
    padded_by_idx: Dict[int, str],
    vocab: List[str],
    only_active: bool = True,
    only_captioned: bool = True,
) -> List[Dict[str, Any]]:
    n = int(scene_state["means"].shape[0])
    captions = scene_state.get("object_caption") or []
    categories = scene_state.get("object_category") or []
    supercats = scene_state.get("object_supercategory") or []
    attrs = scene_state.get("object_key_attributes") or []
    decisions = scene_state.get("object_caption_decision") or []
    crop_paths = scene_state.get("best_view_crop_path") or []
    active = scene_state.get("active")
    means = scene_state["means"].numpy()
    if active is not None:
        active = active.numpy().astype(bool)
    cids = scene_state.get("class_ids")
    if cids is not None:
        cids = cids.numpy()

    records: List[Dict[str, Any]] = []
    for i in range(n):
        is_act = bool(active[i]) if active is not None else True
        if only_active and not is_act:
            continue
        cap = str(captions[i] if i < len(captions) else "")
        if only_captioned and not cap:
            continue
        cid = int(cids[i]) if cids is not None and i < len(cids) else -1
        sam_label = vocab[cid] if 0 <= cid < len(vocab) else f"cls{cid}"
        v = views_by_idx.get(i, {})
        cp = str(crop_paths[i] if i < len(crop_paths) else "")
        records.append(
            {
                "id": i,
                "sam_label": sam_label,
                "category": str(categories[i] if i < len(categories) else ""),
                "supercat": str(supercats[i] if i < len(supercats) else ""),
                "attrs": list(attrs[i] if i < len(attrs) and attrs[i] else []),
                "caption": cap,
                "decision": str(decisions[i] if i < len(decisions) else ""),
                "crop": Path(cp).name if cp else "",
                "face": Path(str(v.get("rgb_path", "") or "")).name,
                "padded": padded_by_idx.get(i, ""),
                "bbox": v.get("bbox_xyxy") or [],
                "mean": [round(float(x), 2) for x in means[i]],
                "bv_ok": bool(v.get("ok", False)),
            }
        )
    return records


def _html_page(run_name: str, records: List[dict]) -> str:
    title = f"Caption Review — {run_name}"
    data_js = "const DATA = " + json.dumps(records, ensure_ascii=False) + ";\n"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #111; color: #ddd; font-family: system-ui, sans-serif; font-size: 13px; }}
  #topbar {{
    position: sticky; top: 0; z-index: 100;
    background: #1a1a1a; border-bottom: 1px solid #333;
    padding: 10px 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  }}
  #topbar h1 {{ font-size: 15px; color: #fff; white-space: nowrap; }}
  #topbar input, #topbar select {{
    background: #222; border: 1px solid #444; color: #ddd;
    padding: 4px 8px; border-radius: 4px; font-size: 12px;
  }}
  #topbar input {{ width: 260px; }}
  #stats {{ color: #888; font-size: 11px; }}
  .hint {{ color: #666; font-size: 11px; }}
  #grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 10px; padding: 14px;
  }}
  .card {{
    background: #1c1c1c; border: 1px solid #2e2e2e; border-radius: 6px;
    overflow: hidden; cursor: pointer; transition: border-color .15s;
  }}
  .card:hover {{ border-color: #555; }}
  .card.mismatch {{ border-color: #c0392b; }}
  .card.mismatch .badge {{ background: #c0392b; }}
  .card img {{
    width: 100%; aspect-ratio: 1/1; object-fit: cover;
    display: block; background: #000;
  }}
  .card-body {{ padding: 7px 8px 8px; }}
  .obj-id {{ font-size: 10px; color: #666; }}
  .badge {{
    display: inline-block; font-size: 10px; padding: 1px 5px;
    border-radius: 3px; background: #2a5a2a; color: #8fe08f; margin-bottom: 4px;
  }}
  .sam-label {{ color: #f0a040; font-size: 11px; font-weight: 600; margin-bottom: 2px; }}
  .caption {{ color: #ccc; font-size: 11px; line-height: 1.35; margin-bottom: 4px; }}
  .meta {{ font-size: 10px; color: #666; }}
  .meta span {{ color: #999; }}
  #modal-bg {{
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,.85); z-index: 200;
    align-items: center; justify-content: center;
  }}
  #modal-bg.open {{ display: flex; }}
  #modal {{
    background: #1e1e1e; border: 1px solid #444; border-radius: 8px;
    max-width: 1100px; width: 95%; max-height: 90vh; overflow-y: auto;
    padding: 20px; position: relative;
  }}
  #modal-close {{
    position: absolute; top: 10px; right: 14px;
    background: none; border: none; color: #aaa; font-size: 20px; cursor: pointer;
  }}
  #modal h2 {{ font-size: 15px; color: #fff; margin-bottom: 14px; }}
  #modal-imgs {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 16px; }}
  #modal-imgs img {{ width: 280px; height: 280px; object-fit: contain; background: #000; border-radius: 4px; }}
  #modal-imgs .img-label {{ font-size: 11px; color: #888; text-align: center; margin-top: 4px; }}
  #modal table {{ width: 100%; border-collapse: collapse; }}
  #modal td {{ padding: 5px 8px; border-bottom: 1px solid #2a2a2a; vertical-align: top; }}
  #modal td:first-child {{ color: #888; font-size: 11px; white-space: nowrap; width: 130px; }}
  #modal td:last-child {{ color: #ddd; font-size: 12px; word-break: break-word; }}
  .tag {{ display: inline-block; background: #2a3a4a; color: #8ab4d4; font-size: 10px;
         padding: 1px 5px; border-radius: 3px; margin: 1px; }}
</style>
</head>
<body>
<div id="topbar">
  <h1>{title}</h1>
  <input id="search" placeholder="Filter caption, SAM label, category…" oninput="render()">
  <select id="filter-decision" onchange="render()">
    <option value="">All decisions</option>
    <option value="keep">keep</option>
    <option value="drop">drop</option>
  </select>
  <select id="filter-mismatch" onchange="render()">
    <option value="">All</option>
    <option value="mismatch">SAM ≠ category (likely mismatch)</option>
    <option value="ok">SAM ≈ category (likely ok)</option>
  </select>
  <select id="sort-by" onchange="render()">
    <option value="id">Sort: obj id</option>
    <option value="sam">Sort: SAM label</option>
    <option value="category">Sort: category</option>
  </select>
  <span id="stats"></span>
  <span class="hint">click a card → tight crop + padded crop sent to VLM + full face. Paste 3D mean into the viewer Go to XYZ box.</span>
</div>
<div id="grid"></div>
<div id="modal-bg" onclick="closeModal(event)">
  <div id="modal">
    <button id="modal-close" onclick="this.parentElement.parentElement.classList.remove('open')">✕</button>
    <h2 id="modal-title"></h2>
    <div id="modal-imgs"></div>
    <table id="modal-table"></table>
  </div>
</div>
<script>
const CROPS_DIR = '../phase4/crops/';
const FACES_DIR = '../phase1.5/faces/';
const PADDED_DIR = '../phase4/padded_crops/';
{data_js}
function isMismatch(r) {{
  const sam = (r.sam_label || '').toLowerCase().replace(/[^a-z0-9 ]/g,'');
  const cat = (r.category || '').toLowerCase();
  const cap = (r.caption || '').toLowerCase();
  const samWords = sam.split(' ').filter(w => w.length > 3);
  if (samWords.length === 0) return false;
  return !samWords.some(w => cat.includes(w) || cap.includes(w));
}}
function render() {{
  const q = document.getElementById('search').value.toLowerCase();
  const decFilter = document.getElementById('filter-decision').value;
  const mmFilter = document.getElementById('filter-mismatch').value;
  const sortBy = document.getElementById('sort-by').value;
  let items = DATA.filter(r => {{
    if (decFilter && r.decision !== decFilter) return false;
    if (mmFilter === 'mismatch' && !isMismatch(r)) return false;
    if (mmFilter === 'ok' && isMismatch(r)) return false;
    if (q) {{
      const hay = [r.caption, r.sam_label, r.category, r.supercat, ...(r.attrs||[])].join(' ').toLowerCase();
      if (!hay.includes(q)) return false;
    }}
    return true;
  }});
  if (sortBy === 'sam') items.sort((a,b) => a.sam_label.localeCompare(b.sam_label));
  else if (sortBy === 'category') items.sort((a,b) => a.category.localeCompare(b.category));
  else items.sort((a,b) => a.id - b.id);
  document.getElementById('stats').textContent = `${{items.length}} / ${{DATA.length}} objects`;
  const grid = document.getElementById('grid');
  grid.innerHTML = items.map(r => {{
    const mm = isMismatch(r);
    const cropSrc = r.crop ? CROPS_DIR + r.crop : '';
    const attrsStr = (r.attrs||[]).map(a => `<span class="tag">${{a}}</span>`).join('');
    return `<div class="card${{mm?' mismatch':''}}" onclick="showModal(${{r.id}})">
      ${{cropSrc ? `<img src="${{cropSrc}}" loading="lazy" onerror="this.style.display='none'">` : '<div style="height:120px;background:#0a0a0a"></div>'}}
      <div class="card-body">
        <div class="obj-id">obj ${{r.id}}</div>
        <div class="sam-label">▸ ${{r.sam_label}}</div>
        <div class="badge">${{r.decision||'?'}}</div>
        <div class="caption">${{r.caption}}</div>
        <div class="meta">cat: <span>${{r.category}}</span></div>
        ${{attrsStr ? `<div style="margin-top:3px">${{attrsStr}}</div>` : ''}}
      </div>
    </div>`;
  }}).join('');
}}
function showModal(id) {{
  const r = DATA.find(x => x.id === id);
  if (!r) return;
  document.getElementById('modal-title').textContent = `obj ${{r.id}} — ${{r.sam_label}}`;
  const cropSrc = r.crop ? CROPS_DIR + r.crop : '';
  const padSrc = r.padded ? PADDED_DIR + r.padded : '';
  const faceSrc = r.face ? FACES_DIR + r.face : '';
  document.getElementById('modal-imgs').innerHTML = `
    ${{cropSrc ? `<div><img src="${{cropSrc}}" onerror="this.src=''"><div class="img-label">Phase 4a tight crop</div></div>` : ''}}
    ${{padSrc ? `<div><img src="${{padSrc}}" onerror="this.src=''"><div class="img-label">Padded crop sent to VLM (+25%)</div></div>` : ''}}
    ${{faceSrc ? `<div><img src="${{faceSrc}}" onerror="this.src=''"><div class="img-label">Best-view face (${{r.face}})</div></div>` : ''}}
  `;
  const rows = [
    ['obj id', r.id],
    ['SAM label', `<b style="color:#f0a040">${{r.sam_label}}</b>`],
    ['decision', r.decision],
    ['category', r.category],
    ['supercategory', r.supercat],
    ['attributes', (r.attrs||[]).map(a=>`<span class="tag">${{a}}</span>`).join(' ') || '—'],
    ['caption', `<b>${{r.caption}}</b>`],
    ['padded crop', r.padded || '—'],
    ['face image', r.face],
    ['bbox (xyxy)', r.bbox ? r.bbox.join(', ') : '—'],
    ['3D mean (x,y,z)', r.mean ? r.mean.join(', ') : '—'],
    ['SAM/caption match', isMismatch(r) ? '<span style="color:#e74c3c">⚠ likely mismatch</span>' : '<span style="color:#2ecc71">✓ ok</span>'],
  ];
  document.getElementById('modal-table').innerHTML = rows.map(([k,v]) =>
    `<tr><td>${{k}}</td><td>${{v}}</td></tr>`).join('');
  document.getElementById('modal-bg').classList.add('open');
}}
function closeModal(e) {{
  if (e.target === document.getElementById('modal-bg'))
    document.getElementById('modal-bg').classList.remove('open');
}}
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') document.getElementById('modal-bg').classList.remove('open');
}});
render();
</script>
</body>
</html>
"""


def write_caption_review(
    scene_state_path: Path,
    *,
    output_html: Optional[Path] = None,
    vocab_file: Optional[Path] = None,
) -> Path:
    scene_state_path = scene_state_path.resolve()
    run_dir = infer_run_dir(scene_state_path) or scene_state_path.parent.parent
    phase4 = scene_state_path.parent if scene_state_path.parent.name == "phase4" else run_dir / "phase4"
    val_dir = run_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    out_html = output_html or (val_dir / "caption_review.html")
    vocab_path = vocab_file or (_REPO / "vocab" / "construction_vocab.txt")

    log.info("Loading %s", scene_state_path)
    ss = torch.load(scene_state_path, map_location="cpu", weights_only=False)
    records = build_records(
        ss,
        views_by_idx=_load_views(phase4 / "best_views.json"),
        padded_by_idx=_padded_name_from_results(phase4 / "caption_results.json"),
        vocab=_load_vocab(vocab_path),
    )

    padded_dest = phase4 / "padded_crops"
    sources = _collect_padded_sources(phase4)
    n_ok = 0
    for rec in records:
        name = rec.get("padded") or ""
        if name and _ensure_padded_copy(name, sources, padded_dest):
            n_ok += 1

    out_html.write_text(_html_page(run_dir.name, records), encoding="utf-8")
    log.info(
        "Wrote %s (%d objects, %d padded crops, %d KB)",
        out_html,
        len(records),
        n_ok,
        out_html.stat().st_size // 1024,
    )
    return out_html


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Write validation/caption_review.html for a Track B run")
    p.add_argument("--scene-state", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None, help="HTML path (default: <run>/validation/caption_review.html)")
    p.add_argument("--vocab-file", type=Path, default=_REPO / "vocab" / "construction_vocab.txt")
    args = p.parse_args()
    if not args.scene_state.is_file():
        log.error("Missing scene state: %s", args.scene_state)
        return 2
    write_caption_review(args.scene_state, output_html=args.output, vocab_file=args.vocab_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
