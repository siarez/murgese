#!/usr/bin/env python3
"""
Dump a visually ordered, line-oriented view of Table 11 using pdfplumber.

Why this exists
  - pdfplumber can extract words with coordinates; grouping them by their
    y-position ("top") gives a stable reconstruction of the table's rows.
  - We serialize each visual row as: "p<page> y=<top> x=<x0> | <text>".
  - The output is the canonical input for tools/parse_pf5_lines.py.

What it does
  - Locates the page range for Table 11 (Process Flow 5) and stops before
    Table 12 (Process Flow 6).
  - Groups words per page by rounded y-position, sorts by x-position, and
    concatenates their text into a single line per visual row.

Output
  - app/eqcore/maps/table11_pf5_lines.txt
"""
from __future__ import annotations

from pathlib import Path
import pdfplumber

PDF = Path('docs/slaa799a_TAS3251_process_flows.pdf')
OUT = Path('app/eqcore/maps/table11_pf5_lines.txt')


def find_table_pages(pdf: pdfplumber.PDF) -> tuple[int, int]:
    """Return [start, end) page indices for Table 11.

    - Start: first page that mentions "Table 11" and "Process Flow 5" (or the
      more general "DSP Memory Map for Process Flow 5").
    - End: first page that contains any marker of Table 12 / Process Flow 6.
    """
    start = end = None
    for i, page in enumerate(pdf.pages):
        txt = (page.extract_text() or '').lower()
        if start is None and (
            ('table 11.' in txt and 'process flow 5' in txt) or 'dsp memory map for process flow 5' in txt
        ):
            start = i
        if start is not None:
            if ('table 12' in txt) or ('table12' in txt) or ('process flow 6' in txt) or ('dspmemorymapforprocessflow6' in txt):
                end = i  # stop before this page
                break
    if start is None:
        raise RuntimeError('Could not locate start of Table 11 in PDF')
    if end is None:
        # Safety cap if markers were not found; covers multi-page tables.
        end = min(start + 10, len(pdf.pages))
    return start, end


def main():
    with pdfplumber.open(str(PDF)) as pdf:
        p0, p1 = find_table_pages(pdf)
        entries: list[tuple[int, int, float, str]] = []
        for pi in range(p0, p1):
            page = pdf.pages[pi]
            # Extract words and group into visual rows by y-position.
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            rows: dict[int, list[dict]] = {}
            for w in words:
                top = int(round(w.get('top', 0)))
                rows.setdefault(top, []).append(w)
            for top, ws in rows.items():
                # Left-to-right within the visual row
                ws.sort(key=lambda w: w.get('x0', 0.0))
                text = ' '.join(w['text'].strip() for w in ws if w.get('text'))
                if not text:
                    continue
                x0 = ws[0].get('x0', 0.0)
                entries.append((pi, top, x0, text))

    # Sort by page, then y-position (top-down)
    entries.sort(key=lambda t: (t[0], t[1]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w') as f:
        for pi, top, x0, text in entries:
            f.write(f"p{pi+1} y={top:4d} x={x0:6.1f} | {text}\n")
    print(f'Wrote {OUT} (lines={len(entries)}) pages={p0+1}-{p1}')


if __name__ == '__main__':
    main()
