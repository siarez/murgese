#!/usr/bin/env python3
"""
Parse the pdfplumber line dump of Table 11 into a clean JSONL stream.

Input
  - app/eqcore/maps/table11_pf5_lines.txt produced by dump_pf5_lines.py.
    Each line has the form: "p<page> y=<top> x=<x0> | <text>".

Behavior
  - Emits section headers exactly where they appear, using a canonical list
    of titles to normalize whitespace/casing.
  - Emits register rows by regex matching the table columns serialized onto
    one line (addr, page, name, bytes/format, default, description).
  - Stops before Table 12 / Process Flow 6 markers (already handled by dump).

Output
  - app/eqcore/maps/table11_pf5_from_lines.jsonl
    Interleaved: one JSON object per line, either {type:'section', title}
    or a row object with page/subaddr/name/bytes/format/default/description.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LINES = Path('app/eqcore/maps/table11_pf5_lines.txt')
OUT_JSONL = Path('app/eqcore/maps/table11_pf5_from_lines.jsonl')


CANON_TITLES = [
    'CROSS OVER BQs',
    'SUB CROSS OVER BQs',
    'DRC 1 BQ',
    'DRC 3 BQ',
    'DRC 2 BQ',
    'DPEQ SENSE BQ',
    'DPEQ LOW LEVEL PATH BQ',
    'DPEQ HIGH LEVEL PATH BQ',
    'EQ LEFT 14 BQs',
    'LEFT INTGAIN BQ',
    'EQ RIGHT 14 BQs',
    'RIGHT INTGAIN BQ',
    'VOLUME ALPHA FILTER',
    'PHASE OPTIMIZER',
    'MAIN DRC 1',
    'MAIN DRC 2',
    'MAIN DRC 3',
    'SUB DRC 1',
    'SUB DRC 2',
    'SUB DRC 3',
    'SPATIALIZER BQs',
    'DPEQ CONTROL',
    'TIME CONSTANT',
    'SUB CHANNEL CONTROL',
    'THD CLIPPER',
    'DRC MIXER GAIN',
    'GANG EQ',
    'INPUT MIXER',
    'MIX/GAIN ADJUST',
    'SPATIALIZER LEVEL',
    'OUTPUT CROSS BAR',
    'VOLUME CONTROL',
    'DPEQ GAIN SCALE',
    'AGL',
]


def norm(s: str) -> str:
    """Normalize a header/label to alnum only, lowercase.

    This makes matching PDF text (which may drop spaces) against canonical
    titles stable.
    """
    return re.sub(r'[^a-z0-9]+', '', s.lower())


CANON_MAP = {norm(t): t for t in CANON_TITLES}


ROW_RE = re.compile(
    r'^(0x[0-9A-Fa-f]{2})\s+'        # sub address
    r'(0x[0-9A-Fa-f]{2})\s+'          # page
    r'(\S+)\s+'                      # name (no spaces in TI print)
    r'(\d+)\s*/\s*([\d\.\-]+)\s+' # bytes / q-format
    r'(\S+)\s+'                      # default value (hex or number)
    r'(.*)$'                           # description (rest of line)
)


def parse_line_text(text: str):
    """Return a parsed dict for a row or section; otherwise None."""
    text = text.strip()
    if not text:
        return None
    # Row?
    m = ROW_RE.match(text)
    if m:
        subaddr, page, name, by, q, default, desc = m.groups()
        return {
            'kind': 'row',
            'page': page,
            'subaddr': subaddr,
            'name': name,
            'bytes': int(by),
            'format': q,
            'default': default,
            'description': desc or None,
        }
    # Header? match a known canonical header by normalized text
    n = norm(text)
    title = CANON_MAP.get(n)
    if title:
        return {'kind': 'section', 'title': title}
    return None


def main():
    if not LINES.exists():
        raise SystemExit(f'Missing {LINES}. Run dump first.')
    out = []
    with LINES.open() as f:
        for line in f:
            # split the visual tag and the content after the pipe
            try:
                _, after = line.split('|', 1)
            except ValueError:
                continue
            text = after.strip()
            n = norm(text)
            # Stop when Table 12 / Process Flow 6 starts
            if 'table12' in n or 'dspmemorymapforprocessflow6' in n:
                break
            entry = parse_line_text(text)
            if not entry:
                continue
            out.append(entry)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open('w') as f:
        for e in out:
            if e['kind'] == 'section':
                f.write(json.dumps({'type': 'section', 'title': e['title']}) + '\n')
            else:
                rec = {
                    'page': e['page'],
                    'subaddr': e['subaddr'],
                    'name': e['name'],
                    'bytes': e['bytes'],
                    'format': e['format'],
                    'default': e['default'],
                    'description': e['description'],
                }
                f.write(json.dumps(rec) + '\n')
    print(f'Wrote {OUT_JSONL} with {len(out)} records')


if __name__ == '__main__':
    main()
