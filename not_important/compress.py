"""
compress_ddl.py
---------------
Reads every *_schema.txt file from Spider_extracted/only_DDL_combined/
and produces a compact representation like:

  Database: aan_1
  Affiliation(affiliation_id: integer, name: varchar(255), address: varchar(255)) PK[affiliation_id]
  Author_list(paper_id: varchar(25), author_id: integer, affiliation_id: integer) PK[paper_id, author_id] FK[paper_id→Paper.paper_id; author_id→Author.author_id; affiliation_id→Affiliation.affiliation_id]

Output goes to Spider_extracted/only_DDL_compressed/<schema_name>.txt
"""

import re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
base_dir   = Path(__file__).parent
schema_dir = base_dir / "Spider_extracted" / "only_DDL_combined"
out_dir    = base_dir / "Spider_extracted" / "only_DDL_compressed"
out_dir.mkdir(parents=True, exist_ok=True)


def normalise_type(raw: str) -> str:
    """Strip extra spaces / backticks from a SQL type token."""
    return raw.strip().strip("`").strip()


def extract_create_table_block(ddl: str) -> str:
    """
    Return only the text from CREATE TABLE ... up to and including
    the matching closing parenthesis. Discards any trailing noise
    (e.g. 'Table 2: ...' text that follows in the same split chunk).
    """
    m = re.search(r'CREATE\s+TABLE\s+[`"]?\w+[`"]?\s*\(', ddl, re.IGNORECASE)
    if not m:
        return ddl  # nothing to trim

    depth = 0
    for i, ch in enumerate(ddl[m.start():], start=m.start()):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return ddl[m.start(): i + 1]  # stop right at closing paren

    return ddl  # unbalanced parens fallback


def parse_ddl_block(ddl: str):
    """
    Parse a single CREATE TABLE block and return:
      (table_name, [(col, type)], [pk_cols], [(fk_col, ref_table, ref_col)])
    """
    # ── Trim to just the CREATE TABLE block, discarding trailing noise ────────
    ddl = extract_create_table_block(ddl)

    # ── table name ────────────────────────────────────────────────────────────
    m = re.search(r'CREATE\s+TABLE\s+[`"]?(\w+)[`"]?\s*\(', ddl, re.IGNORECASE)
    if not m:
        return None
    table_name = m.group(1)

    # Strip out the CREATE TABLE header line; we just want the body
    body_start = ddl.index("(", m.start()) + 1

    # Collect lines inside the outer parentheses
    lines = ddl[body_start:].splitlines()

    columns   = []   # (col_name, col_type)
    pk_cols   = []   # list of col names
    fk_list   = []   # (col_name, ref_table, ref_col)

    for raw_line in lines:
        line = raw_line.strip().lstrip(",").strip()
        if not line or line == ")":
            continue

        # ── PRIMARY KEY ───────────────────────────────────────────────────────
        pk_match = re.match(r'PRIMARY\s+KEY\s*\((.+?)\)', line, re.IGNORECASE)
        if pk_match:
            pk_raw = pk_match.group(1)
            pk_cols = [c.strip().strip("`").strip('"') for c in pk_raw.split(",")]
            continue

        # ── FOREIGN KEY ───────────────────────────────────────────────────────
        fk_match = re.match(
            r'(?:CONSTRAINT\s+\S+\s+)?FOREIGN\s+KEY\s*\(`?(\w+)`?\)\s+REFERENCES\s+`?(\w+)`?\s*\(`?(\w+)`?\)',
            line, re.IGNORECASE
        )
        if fk_match:
            fk_col, ref_table, ref_col = fk_match.group(1), fk_match.group(2), fk_match.group(3)
            fk_list.append((fk_col, ref_table, ref_col))
            continue

        # ── CONSTRAINT / index lines (skip) ───────────────────────────────────
        if re.match(r'(CONSTRAINT|UNIQUE|INDEX|KEY)\b', line, re.IGNORECASE):
            continue

        # ── Column definition ─────────────────────────────────────────────────
        col_match = re.match(r'[`"]?(\w+)[`"]?\s+(\S+(?:\s*\([^)]*\))?)', line)
        if col_match:
            col_name = col_match.group(1).strip("`").strip('"')
            col_type = normalise_type(col_match.group(2))
            columns.append((col_name, col_type))

    return table_name, columns, pk_cols, fk_list


def compress_schema_file(txt_path: Path) -> str:
    """
    Read one *_schema.txt file and return the compressed multi-line string.
    """
    raw = txt_path.read_text(encoding="utf-8", errors="replace")

    # Derive a clean database name from the file stem (strip _schema suffix)
    db_name = txt_path.stem
    if db_name.endswith("_schema"):
        db_name = db_name[: -len("_schema")]

    lines_out = [f"Database: {db_name}"]

    # Approach: split raw text on "CREATE TABLE" anchors, then grab each block
    blocks = re.split(r'(?=CREATE\s+TABLE\b)', raw, flags=re.IGNORECASE)

    for block in blocks:
        block = block.strip()
        if not block.upper().startswith("CREATE"):
            continue

        result = parse_ddl_block(block)
        if result is None:
            continue

        table_name, columns, pk_cols, fk_list = result

        # ── build column list ─────────────────────────────────────────────────
        col_str = ", ".join(f"{col}: {typ}" for col, typ in columns)

        # ── build PK annotation ───────────────────────────────────────────────
        pk_str = ""
        if pk_cols:
            pk_str = f" PK[{', '.join(pk_cols)}]"

        # ── build FK annotation ───────────────────────────────────────────────
        fk_str = ""
        if fk_list:
            fk_parts = "; ".join(f"{col}→{ref_t}.{ref_c}" for col, ref_t, ref_c in fk_list)
            fk_str = f" FK[{fk_parts}]"

        lines_out.append(f"{table_name}({col_str}){pk_str}{fk_str}")

    return "\n".join(lines_out)


def main():
    txt_files = sorted(schema_dir.glob("*_schema.txt"))
    if not txt_files:
        print(f"No *_schema.txt files found in {schema_dir}")
        return

    print(f"Found {len(txt_files)} schema files. Compressing …\n")

    for txt_path in txt_files:
        compressed = compress_schema_file(txt_path)

        out_name = txt_path.stem  # e.g. aan_1_schema
        if out_name.endswith("_schema"):
            out_name = out_name[: -len("_schema")]  # → aan_1
        out_file = out_dir / f"{out_name}_compressed.txt"
        out_file.write_text(compressed, encoding="utf-8")

        # Print a preview
        db_line = compressed.split("\n")[0]
        table_count = compressed.count("\n")
        print(f"  ✅  {out_name:45s}  ({table_count} tables)  →  {out_file.name}")

    print(f"\nDone! Compressed files saved to:\n  {out_dir}")


if __name__ == "__main__":
    main()