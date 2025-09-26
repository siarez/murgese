# Tools: Table 11 (Process Flow 5) extraction

This folder contains small, single-purpose scripts to extract and normalize
Table 11 from `docs/slaa799a_TAS3251_process_flows.pdf`.

Workflow

1) Dump a visual line stream using pdfplumber (stable order):
   - `python tools/dump_pf5_lines.py`
   - Output: `app/eqcore/maps/table11_pf5_lines.txt`

2) Parse the line dump into JSONL (headers + rows, in exact order):
   - `python tools/parse_pf5_lines.py`
   - Output: `app/eqcore/maps/table11_pf5_from_lines.jsonl`

Optional

- Dump the raw PDF text range for Table 11 for quick inspection:
  - `python tools/dump_pf5_text.py`
  - Output: `app/eqcore/maps/table11_pf5_raw.txt`

Notes

- The line dumper stops before Table 12/Process Flow 6, so only Table 11 pages
  are included in the dump.
- The parser recognizes section headers via a canonical list and parses rows
  with a conservative regex that matches the “flattened” table columns.
- Keep these scripts small and readable; they are designed for inspection and
  reproducibility rather than performance.

