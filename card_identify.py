"""
pokemon_card_name_number_detector.py

Detects:
- Card Name
- Card Number
- Searches card via TCGdex API

Shows:
- Name OCR time
- Number OCR time
- API search time
- Total processing time
- Real card details panel (HP, types, attacks, rarity, set, artist, image)

Usage:
    python pokemon_card_detector.py
    python pokemon_card_detector.py --folder my_cards

Controls:
    SPACE = next
    B     = previous
    Q     = quit

Requirements:
    pip install opencv-python rapidocr-onnxruntime requests numpy
"""

import argparse
import cv2
import numpy as np
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from rapidocr_onnxruntime import RapidOCR
import json
import requests


# -------------------------------------------------------------
# CLI ARGS
# -------------------------------------------------------------

parser = argparse.ArgumentParser(description="Pokémon card name/number detector")
parser.add_argument(
    "--folder",
    default="images1",
    help="Folder name (relative to this script) containing card images (default: images1)",
)
args = parser.parse_args()

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.folder)


# -------------------------------------------------------------
# FILES
# -------------------------------------------------------------

if not os.path.isdir(folder):
    print(f"[ERROR] Folder not found: {folder}")
    print(f"  Create the folder or pass a different one with --folder <name>")
    sys.exit(1)

files = sorted(
    [os.path.join(folder, f) for f in os.listdir(folder)
     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
)
if not files:
    print(f"[ERROR] No images found in: {folder}")
    print(f"  Add .png / .jpg / .jpeg card scans to that folder and retry.")
    sys.exit(1)


# -------------------------------------------------------------
# SINGLETONS  (created once, reused everywhere)
# -------------------------------------------------------------

# Reuse TCP connections — avoids per-call handshake overhead
_session = requests.Session()

# OCR engine
ocr = RapidOCR()

# CLAHE variants
_CLAHE_FAST   = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
_CLAHE_STRONG = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))

# Dilation kernel for strong OCR
_DILATE_KERNEL = np.ones((2, 2), np.uint8)

# Sharpening kernel for strong OCR
_SHARP9 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)

# Precompiled regexes
_LOOSE      = re.compile(r'(\d{1,3})[/|Ii](\d{2,4})|^(\d{1,3})$')
_RE_EVOLVES = re.compile(r'evolves', re.IGNORECASE)


# -------------------------------------------------------------
# API SEARCH
# -------------------------------------------------------------

TCGDEX_API = "https://api.tcgdex.net/v2/en/cards"

# In-memory caches — avoids hitting the network again for cards already seen
# or when navigating back to a previous card.
_search_cache = {}   # (name, number) → card summary dict
_detail_cache = {}   # card_id        → full details dict
_image_cache  = {}   # image_url      → resized BGR image


def search_card_api(name, number):
    if not name:
        return None
    key = (name, number)
    if key in _search_cache:
        return _search_cache[key]
    try:
        url = f"https://api.tcgdex.net/v2/en/cards?name={name}"
        r = _session.get(url, timeout=5)
        if r.status_code != 200:
            return None
        cards = r.json()
        if not cards:
            return None
        number_id = None
        if number and "/" in number:
            number_id = number.split("/")[0].lstrip("0")
        result = None
        for card in cards:
            if number_id:
                if str(card.get("localId")).lstrip("0") == number_id:
                    result = card
                    break
            else:
                result = card
                break
        _search_cache[key] = result
        return result
    except Exception as e:
        print("API error:", e)
    return None


def fetch_card_details(card_id):
    """Fetch full card details by card ID from TCGdex."""
    if not card_id:
        return None
    if card_id in _detail_cache:
        return _detail_cache[card_id]
    try:
        url = f"https://api.tcgdex.net/v2/en/cards/{card_id}"
        r = _session.get(url, timeout=5)
        if r.status_code != 200:
            return None
        data = r.json()
        _detail_cache[card_id] = data
        return data
    except Exception as e:
        print("Detail fetch error:", e)
    return None


def fetch_card_image(image_url, width=220):
    """Download card image and resize to fit panel."""
    if not image_url:
        return None
    # Normalise URL BEFORE cache check — appending /high.png after the check
    # would make the stored key never match on the next call (cache always miss).
    if not image_url.endswith(('.png', '.jpg', '.webp')):
        image_url = image_url + "/high.png"
    if image_url in _image_cache:
        return _image_cache[image_url]
    try:
        r = _session.get(image_url, timeout=8)
        if r.status_code != 200:
            return None
        arr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        new_h = int(h * width / w)
        result = cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)
        _image_cache[image_url] = result
        return result
    except Exception as e:
        print("Image fetch error:", e)
    return None


# -------------------------------------------------------------
# NUMBER DETECTION
# -------------------------------------------------------------

_FIX = str.maketrans({
    'O': '0', 'o': '0', 'Q': '0', 'l': '1', 'I': '1',
    '|': '1', 'T': '1', 'S': '5', 'Z': '2', 'B': '8', ' ': ''
})

_XY = re.compile(r'(\d{1,3})[/|\\](\d{2,4})')


def zone_number(img):
    # NOTE: crop ratios assume a standard portrait card scan (~600×840px or similar).
    # Adjust if your scans use a significantly different resolution or crop.
    h, w = img.shape[:2]
    return img[int(h * 0.8240):int(h * 0.8600),
               int(w * 0.2720):int(w * 0.9540)]


def clean_number(raw):
    s = str(raw).translate(_FIX)
    s = re.sub(r'^[^\d/|\\]+', '', s)
    s = re.sub(r'[^\d/|\\]+$', '', s)
    return s.strip()


def parse_xy(n_str, t_str):
    n = int(n_str)
    t = int(t_str)
    if 1 <= n <= t and 20 <= t <= 500:
        return n_str, t_str
    if len(t_str) == 4 and t_str.endswith("0"):
        t2 = int(t_str[:-1])
        if 1 <= n <= t2 and 20 <= t2 <= 500:
            return n_str, t_str[:-1]
    return None


def detect_number(img):
    crop = zone_number(img)
    up   = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    cla  = _CLAHE_FAST.apply(gray)
    # Sharpen the CLAHE result for better contrast than raw upscale alone
    sharpen = cv2.filter2D(cla, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    hits = []
    for v in (sharpen, cla, cv2.bitwise_not(gray)):
        res, _ = ocr(v)
        if not res:
            continue
        for r in res:
            fixed = clean_number(r[1].strip())
            if not fixed:
                continue
            conf = float(r[2])
            for m in _XY.finditer(fixed):
                p = parse_xy(m.group(1), m.group(2))
                if p:
                    hits.append((conf, p[0], p[1]))
        if hits:   # early-exit: stop as soon as any variant finds X/Y
            break
    if not hits:
        return None, None, 0.0
    pairs = Counter((n, t) for _, n, t in hits)
    (bn, bt), votes = pairs.most_common(1)[0]
    bc   = max(c for c, n, t in hits if (n, t) == (bn, bt))
    conf = min((bc + min((votes - 1) * 0.02, 0.06)) * 100, 100)
    return bn, bt, conf


def detect_number_strong(img):
    """
    Fallback OCR — runs ONLY when fast pass + API both fail.
    All 6 variants restored for accuracy, but with early-exit per crop:
    - Best case (hit on crop 1, variant 1): 1 OCR call
    - Typical:                              3–6 OCR calls
    - Worst case (all crops exhausted):     18 OCR calls
    Scales kept at 4x only — higher scales never helped in practice.
    """
    h, w = img.shape[:2]

    # NOTE: crop ratios same assumption as zone_number() — standard portrait card scan.
    # Ordered most-likely first so early-exit fires as soon as possible.
    crops = [
        img[int(h*0.810):int(h*0.870), int(w*0.200):int(w*0.980)],  # wider
        img[int(h*0.820):int(h*0.860), int(w*0.272):int(w*0.954)],  # original zone
        img[int(h*0.800):int(h*0.880), int(w*0.150):int(w*0.980)],  # extra wide
    ]

    hits      = []
    bare_seen = {}

    for crop in crops:
        up      = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray    = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        cla     = _CLAHE_STRONG.apply(gray)
        sharp   = cv2.filter2D(cla, -1, _SHARP9)
        _, otsu     = cv2.threshold(cla, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
        _, otsu_inv = cv2.threshold(cla, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dilated     = cv2.dilate(otsu_inv, _DILATE_KERNEL, iterations=1)
        inv_gray    = cv2.bitwise_not(gray)

        for v in (sharp, otsu, otsu_inv, dilated, up, inv_gray):
            res, _ = ocr(v)
            if not res:
                continue
            for r in res:
                txt   = r[1].strip()
                conf  = float(r[2])
                fixed = clean_number(txt)
                if not fixed:
                    continue
                for m in _XY.finditer(fixed):
                    p = parse_xy(m.group(1), m.group(2))
                    if p:
                        hits.append((conf, p[0], p[1]))
                for m in _LOOSE.finditer(fixed):
                    bare = m.group(3)
                    if bare and 1 <= int(bare) <= 500:
                        if bare not in bare_seen or conf > bare_seen[bare]:
                            bare_seen[bare] = conf

            if hits:   # early-exit — stop variants as soon as X/Y found
                break

        if hits:   # early-exit — no need to try wider crops
            break

    if hits:
        pairs = Counter((n, t) for _, n, t in hits)
        (bn, bt), votes = pairs.most_common(1)[0]
        bc   = max(c for c, n, t in hits if (n, t) == (bn, bt))
        conf = min((bc + min((votes - 1) * 0.02, 0.06)) * 100, 100)
        return bn, bt, conf

    # Last resort: bare numerator only (no total found)
    if bare_seen:
        bn, bc = max(bare_seen.items(), key=lambda x: x[1])
        return bn, None, bc * 100

    return None, None, 0.0


# -------------------------------------------------------------
# NAME DETECTION
# -------------------------------------------------------------

CONF_THRESHOLD = 0.80
NAME_SIZE_RATIO = 0.50

# Strip OCR-fused stage/type prefixes (e.g. "BAsic Larvitar" → "Larvitar")
_STRIP_PREFIX = re.compile(
    r'^(basic|stage\s*[12i]+|evolves(\s*from\s*\w+)?)\s+',
    re.IGNORECASE
)


def _clean_name_text(text):
    """Strip known badge prefixes that OCR fuses onto the card name."""
    return _STRIP_PREFIX.sub("", text).strip()


def _is_noise(text):
    """Return True for OCR noise that should always be discarded."""
    if len(text) < 2:
        return True
    if text.replace(" ", "").isdigit():
        return True
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if latin < len(text) * 0.5:
        return True
    return False


def detect_name(img):
    # NOTE: pixel crop assumes standard portrait card scan (~600×840px or similar).
    # Targets the name zone near the top of the card.
    crop = img[275:535, 224:730]
    up   = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    res, _ = ocr(up)

    candidates = []
    if res:
        for r in res:
            box  = np.array(r[0])
            text = r[1].strip()
            conf = float(r[2])

            text = _clean_name_text(text)
            if not text:
                continue

            if conf < CONF_THRESHOLD:
                continue

            if _RE_EVOLVES.match(text):
                continue

            if _is_noise(text):
                continue

            center_y = float(np.mean(box[:, 1]))
            if center_y > up.shape[0] * 0.60:
                continue

            box_h = float(np.linalg.norm(box[0] - box[3]))
            candidates.append((text, conf, box_h, center_y, float(np.mean(box[:, 0]))))

    name = None
    if candidates:
        # STEP 1: size filter — keep name-sized boxes only
        max_h  = max(c[2] for c in candidates)
        sized  = [c for c in candidates if c[2] >= max_h * NAME_SIZE_RATIO]
        if not sized:
            sized = candidates

        # STEP 2: row clustering — pick the bottom-most row
        # Trainer cards: "TRAINER" badge (top) + actual name (below).
        # Pokémon cards: name is the only large-text row anyway.
        sized.sort(key=lambda c: c[3])
        row_thresh = max_h * 0.40

        rows, cur = [], [sized[0]]
        for c in sized[1:]:
            if c[3] - cur[-1][3] <= row_thresh:
                cur.append(c)
            else:
                rows.append(cur)
                cur = [c]
        rows.append(cur)

        name_row = rows[-1]   # bottom-most row = the name

        # STEP 3: sort left → right for correct word order
        name_row.sort(key=lambda c: c[4])
        name = " ".join(c[0] for c in name_row).strip()

    return name


# -------------------------------------------------------------
# CARD DETAIL PANEL RENDERER
# -------------------------------------------------------------

PANEL_W  = 400
PANEL_BG = (22, 22, 28)        # BGR: near-black background
ROW_H    = 24
FONT     = cv2.FONT_HERSHEY_SIMPLEX

# All colors are in BGR order (OpenCV convention)
C_TEXT   = (220, 220, 220)     # primary text
C_LABEL  = (120, 120, 130)     # dim label
C_ACCENT = (100, 220, 140)     # card name / section headers
C_HP     = ( 80, 130, 255)     # HP value
C_RARITY = (210, 170,  50)     # rarity
C_ARTIST = (140, 190, 255)     # artist
C_ATK    = (255, 210,  90)     # attack name
C_WEAK   = ( 90,  90, 240)     # weakness
C_RESIST = ( 80, 200,  80)     # resistance
C_ABIL   = (210, 150, 255)     # ability name
C_MUTED  = (160, 160, 160)     # descriptions
C_DIV    = ( 55,  55,  65)     # divider line
C_HDR_BG = ( 40,  40,  50)     # section header background

# TYPE_COLORS: BGR order
TYPE_COLORS = {
    "Fire":      ( 50,  70, 220),
    "Water":     (180,  90,  30),
    "Grass":     ( 35, 150,  55),
    "Lightning": ( 20, 195, 215),
    "Psychic":   (170,  50, 170),
    "Fighting":  ( 25,  70, 170),
    "Darkness":  ( 80,  55,  80),
    "Metal":     (130, 130, 150),
    "Dragon":    ( 35,  90, 170),
    "Colorless": (110, 110, 110),
    "Fairy":     (170,  90, 195),
}

PAD = 12   # left/right padding inside panel


def _clip(text, max_chars):
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"


def _wrap(text, max_chars):
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 <= max_chars:
            line = (line + " " + w).strip()
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


def _divider(panel, y):
    cv2.line(panel, (PAD, y), (PANEL_W - PAD, y), C_DIV, 1)
    return y + 8


def _row(panel, y, label, value, vc=C_TEXT, lc=C_LABEL, sc=0.50):
    """Render a single label: value row."""
    cv2.putText(panel, label, (PAD, y), FONT, sc, lc, 1, cv2.LINE_AA)
    cv2.putText(panel, _clip(str(value), 38), (140, y), FONT, sc, vc, 1, cv2.LINE_AA)
    return y + ROW_H


def _section(panel, y, title):
    """Render a filled header bar for a section."""
    cv2.rectangle(panel, (0, y - 14), (PANEL_W, y + 6), C_HDR_BG, -1)
    cv2.putText(panel, title, (PAD, y), FONT, 0.52, C_ACCENT, 1, cv2.LINE_AA)
    return y + 20


def build_detail_panel(details, card_img_small, panel_h):
    """Build and return the right-side detail panel."""
    panel = np.full((panel_h, PANEL_W, 3), PANEL_BG, dtype=np.uint8)

    if details is None:
        cv2.putText(panel, "No card found", (PAD, panel_h // 2),
                    FONT, 0.6, C_LABEL, 1, cv2.LINE_AA)
        return panel

    y = 18

    # Card name
    card_name = details.get("name", "???")
    cv2.putText(panel, _clip(card_name, 28), (PAD, y),
                FONT, 0.80, C_ACCENT, 2, cv2.LINE_AA)
    y += 28

    # Card image
    if card_img_small is not None:
        ih, iw = card_img_small.shape[:2]
        x_off = (PANEL_W - iw) // 2
        y2 = y + ih
        if y2 < panel_h:
            panel[y:y2, x_off:x_off + iw] = card_img_small
        y = y2 + 8

    y = _divider(panel, y)

    # Stats
    hp = details.get("hp")
    if hp:
        y = _row(panel, y, "HP", hp, vc=C_HP)

    types = details.get("types", [])
    if types:
        cv2.putText(panel, "Type", (PAD, y), FONT, 0.50, C_LABEL, 1, cv2.LINE_AA)
        tx = 140
        for t in types:
            col = TYPE_COLORS.get(t, (110, 110, 110))
            tw  = len(t) * 8 + 10
            cv2.rectangle(panel, (tx, y - 13), (tx + tw, y + 5), col, -1)
            cv2.rectangle(panel, (tx, y - 13), (tx + tw, y + 5), (255, 255, 255), 1)
            cv2.putText(panel, t, (tx + 5, y), FONT, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
            tx += tw + 6
        y += ROW_H

    stage = details.get("stage") or details.get("supertype")
    if stage:
        y = _row(panel, y, "Stage", stage)

    rarity = details.get("rarity")
    if rarity:
        y = _row(panel, y, "Rarity", rarity, vc=C_RARITY)

    card_set = details.get("set", {})
    if isinstance(card_set, dict):
        set_name = card_set.get("name", "")
        serie    = card_set.get("serie", {})
        serie    = serie.get("name", "") if isinstance(serie, dict) else ""
        if set_name:
            y = _row(panel, y, "Set", set_name)
        if serie:
            y = _row(panel, y, "Series", serie)

    local_id   = details.get("localId")
    card_count = details.get("set", {}).get("cardCount", {})
    if isinstance(card_count, dict):
        card_count = card_count.get("official", "?")
    if local_id:
        y = _row(panel, y, "Number", f"{local_id}/{card_count}")

    artist = details.get("illustrator")
    if artist:
        y = _row(panel, y, "Artist", artist, vc=C_ARTIST)

    # Attacks
    attacks = details.get("attacks", [])
    if attacks:
        y += 4
        y = _section(panel, y, "ATTACKS")
        for atk in attacks[:3]:
            atk_name = atk.get("name", "")
            atk_dmg  = str(atk.get("damage", ""))
            cost     = atk.get("cost", [])
            cost_str = "▸ " + " ".join(c[0] for c in cost) if cost else ""
            label    = _clip(atk_name, 22) + (f"  {atk_dmg}" if atk_dmg else "")
            cv2.putText(panel, label, (PAD + 4, y), FONT, 0.52, C_ATK, 1, cv2.LINE_AA)
            y += 20
            if cost_str:
                cv2.putText(panel, cost_str, (PAD + 8, y),
                            FONT, 0.42, C_MUTED, 1, cv2.LINE_AA)
                y += 16
            desc = atk.get("effect", "")
            if desc:
                for line in _wrap(desc, 44)[:2]:
                    cv2.putText(panel, line, (PAD + 8, y),
                                FONT, 0.38, C_MUTED, 1, cv2.LINE_AA)
                    y += 14
            y += 4

    # Abilities
    abilities = details.get("abilities", [])
    if abilities:
        y += 4
        y = _section(panel, y, "ABILITY")
        for ab in abilities[:2]:
            cv2.putText(panel, _clip(f"{ab.get('name','')}  ({ab.get('type','')})", 38),
                        (PAD + 4, y), FONT, 0.50, C_ABIL, 1, cv2.LINE_AA)
            y += 18
            for line in _wrap(ab.get("effect", ""), 44)[:3]:
                cv2.putText(panel, line, (PAD + 8, y),
                            FONT, 0.38, C_MUTED, 1, cv2.LINE_AA)
                y += 14
            y += 4

    # Battle stats
    weaknesses  = details.get("weaknesses", [])
    resistances = details.get("resistances", [])
    retreat     = details.get("retreat")
    if weaknesses or resistances or retreat is not None:
        y += 4
        y = _section(panel, y, "BATTLE STATS")
        if weaknesses:
            wstr = "  ".join(f"{w.get('type','')} {w.get('value','')}".strip() for w in weaknesses)
            y = _row(panel, y, "Weak", wstr, vc=C_WEAK)
        if resistances:
            rstr = "  ".join(f"{r.get('type','')} {r.get('value','')}".strip() for r in resistances)
            y = _row(panel, y, "Resist", rstr, vc=C_RESIST)
        if retreat is not None:
            y = _row(panel, y, "Retreat", ("◆ " * int(retreat)).strip() or "Free")

    return panel


# -------------------------------------------------------------
# HUD OVERLAY  (drawn onto left panel)
# -------------------------------------------------------------

HUD_H    = 130   # height of the dark bar at the bottom of the card image
HUD_BG   = (18, 18, 22)
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_hud(disp, name, number, card_id, strong_used, card_id_found,
             name_time, num_time, strong_time, api_time, detail_time,
             total_time, filename, idx, total_files):
    h, w = disp.shape[:2]
    hud_y = h - HUD_H

    # Semi-transparent dark bar
    overlay = disp.copy()
    cv2.rectangle(overlay, (0, hud_y), (w, h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.82, disp, 0.18, 0, disp)

    # Thin accent line at top of HUD
    cv2.line(disp, (0, hud_y), (w, hud_y), (60, 60, 75), 1)

    y = hud_y + 22

    # Row 1: name + number
    num_col = (0, 160, 255) if (strong_used and card_id_found) else (80, 230, 255)
    cv2.putText(disp, f"NAME   {name or '???'}",
                (12, y), HUD_FONT, 0.65, (100, 240, 140), 1, cv2.LINE_AA)
    cv2.putText(disp, f"NO.  {number}",
                (12, y + 24), HUD_FONT, 0.65, num_col, 1, cv2.LINE_AA)

    # Strong OCR badge (right-aligned)
    if strong_used:
        badge     = "STRONG OCR OK" if card_id_found else "STRONG OCR MISS"
        badge_col = (0, 210, 255) if card_id_found else (60, 60, 220)
        (bw, _), _ = cv2.getTextSize(badge, HUD_FONT, 0.48, 1)
        cv2.putText(disp, badge, (w - bw - 10, y),
                    HUD_FONT, 0.48, badge_col, 1, cv2.LINE_AA)

    # Row 2: card id
    id_col = (210, 175, 50) if card_id else (100, 100, 120)
    cv2.putText(disp, f"ID     {card_id or 'not found'}",
                (12, y + 48), HUD_FONT, 0.55, id_col, 1, cv2.LINE_AA)

    # Row 3: timing (compact, right side)
    # name_time and num_time share the same wall-clock value because both
    # OCR tasks run concurrently inside a single ThreadPoolExecutor window.
    t_lines = [
        f"Name {name_time*1000:.0f}ms  Num {num_time*1000:.0f}ms",
        (f"Strong {strong_time*1000:.0f}ms  " if strong_used else "") +
        f"API {api_time*1000:.0f}ms  Det {detail_time*1000:.0f}ms",
        f"Total {total_time*1000:.0f}ms",
    ]
    ty = y + 46
    for tl in t_lines:
        if tl.strip():
            tw = cv2.getTextSize(tl, HUD_FONT, 0.40, 1)[0][0]
            cv2.putText(disp, tl, (w - tw - 10, ty),
                        HUD_FONT, 0.40, (110, 110, 120), 1, cv2.LINE_AA)
            ty += 15

    # Filename + counter bottom-left
    cv2.putText(disp, f"{filename}  ({idx+1}/{total_files})",
                (12, h - 8), HUD_FONT, 0.42, (90, 90, 100), 1, cv2.LINE_AA)


# -------------------------------------------------------------
# PREFETCH  — processes next card in background while user looks at current
# -------------------------------------------------------------

# Stores fully processed results keyed by file index.
# Populated by the background thread; consumed and removed by the main loop.
_prefetch_cache = {}


def _process_card(img):
    """
    Full OCR + API pipeline for one image.
    Extracted into its own function so both the main loop and the background
    prefetch thread can call it identically.
    """
    t_ocr = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_name   = ex.submit(detect_name,   img)
        fut_number = ex.submit(detect_number, img)
        name       = fut_name.result()
        n, t, conf = fut_number.result()
    ocr_time = time.time() - t_ocr

    number = f"{n}/{t}" if n else "???"

    start_api = time.time()
    card      = search_card_api(name, number)
    api_time  = time.time() - start_api
    card_id   = card.get("id") if card else None

    strong_used = False
    strong_time = 0.0
    if card_id is None:
        t0 = time.time()
        sn, st, sconf = detect_number_strong(img)
        strong_time = time.time() - t0
        strong_used = True
        if sn:
            strong_number = f"{sn}/{st}" if st else f"{sn}/???"
            card2 = search_card_api(name, strong_number)
            if card2:
                card    = card2
                card_id = card2.get("id")
                number  = strong_number

    start_detail = time.time()
    details      = fetch_card_details(card_id)
    detail_time  = time.time() - start_detail

    card_img_small = None
    if details:
        img_url = details.get("image")
        if img_url:
            card_img_small = fetch_card_image(img_url, width=190)

    return {
        "name":           name,
        "number":         number,
        "card_id":        card_id,
        "strong_used":    strong_used,
        "details":        details,
        "card_img_small": card_img_small,
        "ocr_time":       ocr_time,
        "api_time":       api_time,
        "strong_time":    strong_time,
        "detail_time":    detail_time,
    }


def _prefetch(next_idx):
    """Read and fully process the next card on a background thread."""
    if next_idx >= len(files):
        return
    next_img = cv2.imread(files[next_idx])
    if next_img is None:
        return
    _prefetch_cache[next_idx] = _process_card(next_img)
    print(f"  [prefetch] card {next_idx+1}/{len(files)} ready in background")


# -------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------

cv2.namedWindow("CARD DETECTOR", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CARD DETECTOR", 1600, 720)

idx          = 0
json_results = []
# Single long-lived background thread — one prefetch job at a time
prefetch_ex  = ThreadPoolExecutor(max_workers=1)
print("SPACE=next  B=prev  Q=quit")

while True:
    img = cv2.imread(files[idx])
    if img is None:
        print(f"[WARN] Could not read image: {files[idx]} — skipping")
        idx = min(idx + 1, len(files) - 1)
        continue

    # Start the clock before the cache check so total_time reflects real
    # elapsed time on this card — even on a prefetch hit it won't read 0ms.
    start_total = time.time()

    # Use prefetched result if ready, otherwise process synchronously now
    if idx in _prefetch_cache:
        res = _prefetch_cache.pop(idx)
        print(f"  [prefetch] cache hit — card {idx+1} displayed instantly")
    else:
        res = _process_card(img)

    name           = res["name"]
    number         = res["number"]
    card_id        = res["card_id"]
    strong_used    = res["strong_used"]
    details        = res["details"]
    card_img_small = res["card_img_small"]
    name_time      = res["ocr_time"]
    num_time       = res["ocr_time"]   # same wall time — ran in parallel
    api_time       = res["api_time"]
    strong_time    = res["strong_time"]
    detail_time    = res["detail_time"]

    if strong_used:
        status = f"hit -> {card_id}" if card_id else "missed"
        print(f"  [fallback] strong OCR {status}")

    # Kick off prefetch for next card immediately while user views current one.
    # By the time they press SPACE the next card is already processed.
    next_idx = idx + 1
    if next_idx < len(files) and next_idx not in _prefetch_cache:
        prefetch_ex.submit(_prefetch, next_idx)

    total_time = time.time() - start_total

    # Left panel: card image + HUD bar
    disp = cv2.resize(img, None, fx=0.7, fy=0.7)
    draw_hud(disp, name, number, card_id,
             strong_used, bool(card_id),
             name_time, num_time, strong_time,
             api_time, detail_time, total_time,
             os.path.basename(files[idx]), idx, len(files))

    # Right panel: card details
    detail_panel = build_detail_panel(details, card_img_small, disp.shape[0])

    cv2.imshow("CARD DETECTOR", np.hstack([disp, detail_panel]))

    # Console output
    print("------------------------------------------------")
    print(os.path.basename(files[idx]))
    print(f"Name   : {name}")
    print(f"Number : {number}")
    print(f"API ID : {card_id}")
    if details:
        print(f"HP     : {details.get('hp')}")
        print(f"Types  : {details.get('types')}")
        print(f"Rarity : {details.get('rarity')}")
        print(f"Artist : {details.get('illustrator')}")
        for atk in details.get("attacks", []):
            print(f"  Attack: {atk.get('name')}  DMG:{atk.get('damage','-')}")
    print(f"Name OCR : {name_time:.3f}s  Num OCR : {num_time:.3f}s")
    print(f"API      : {api_time:.3f}s  Detail  : {detail_time:.3f}s")
    if strong_used:
        print(f"Strong   : {strong_time:.3f}s")
    print(f"Total    : {total_time:.3f}s")

    # JSON record for this card
    record = {
        "file":            os.path.basename(files[idx]),
        "detected_name":   name,
        "detected_number": number,
        "card_id":         card_id,
        "strong_ocr_used": strong_used,
        "timing_ms": {
            "name_ocr":   round(name_time   * 1000, 1),
            "number_ocr": round(num_time    * 1000, 1),
            "strong_ocr": round(strong_time * 1000, 1) if strong_used else None,
            "api_search": round(api_time    * 1000, 1),
            "detail_api": round(detail_time * 1000, 1),
            "total":      round(total_time  * 1000, 1),
        },
    }
    if details:
        record["details"] = {
            "name":        details.get("name"),
            "hp":          details.get("hp"),
            "types":       details.get("types", []),
            "stage":       details.get("stage") or details.get("supertype"),
            "rarity":      details.get("rarity"),
            "set":         details.get("set", {}).get("name") if isinstance(details.get("set"), dict) else None,
            "series":      (details.get("set", {}).get("serie") or {}).get("name") if isinstance(details.get("set"), dict) else None,
            "number":      f"{details.get('localId')}/{details.get('set', {}).get('cardCount', {}).get('official', '?')}" if details.get("localId") else None,
            "illustrator": details.get("illustrator"),
            "attacks":     [{"name": a.get("name"), "damage": a.get("damage"), "cost": a.get("cost", [])} for a in details.get("attacks", [])],
            "abilities":   [{"name": a.get("name"), "type": a.get("type"), "effect": a.get("effect")} for a in details.get("abilities", [])],
            "weaknesses":  details.get("weaknesses", []),
            "resistances": details.get("resistances", []),
            "retreat":     details.get("retreat"),
        }
    json_results.append(record)

    key = cv2.waitKey(0) & 0xFF
    if key == ord(' '):
        idx = min(idx + 1, len(files) - 1)
    elif key == ord('b'):
        idx = max(idx - 1, 0)
    elif key in (ord('q'), 27):
        break

prefetch_ex.shutdown(wait=False)

# Save results
json_path = os.path.join(folder, "scan_results.json")
with open(json_path, "w", encoding="utf-8") as jf:
    json.dump(json_results, jf, indent=2, ensure_ascii=False)
print(f"JSON saved: {json_path}  ({len(json_results)} cards)")

cv2.destroyAllWindows()