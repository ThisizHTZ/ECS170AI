import csv
import os


def _normalize_rows(rows):
    if not rows:
        return []
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    ordered_keys = list(rows[0].keys()) + [k for k in all_keys if k not in rows[0]]
    normalized = []
    for row in rows:
        normalized.append({k: row.get(k, "") for k in ordered_keys})
    return normalized


def export_table(rows, output_dir, base_name, sheet_name="Sheet1"):
    os.makedirs(output_dir, exist_ok=True)
    rows = _normalize_rows(rows)
    if not rows:
        return None

    # Preferred: native Excel file if openpyxl is available.
    try:
        from openpyxl import Workbook  # type: ignore

        xlsx_path = os.path.join(output_dir, f"{base_name}.xlsx")
        wb = Workbook()
        ws = wb.active
        ws.title = sheet_name

        headers = list(rows[0].keys())
        ws.append(headers)
        for row in rows:
            ws.append([row[h] for h in headers])

        wb.save(xlsx_path)
        return xlsx_path
    except Exception:
        # Fallback: CSV that can be opened directly in Excel.
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        headers = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        return csv_path
