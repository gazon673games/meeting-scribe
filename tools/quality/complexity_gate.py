"""
Cyclomatic complexity gate for the backend codebase.

Thresholds (radon scale: A=1-5, B=6-10, C=11-15, D=16-20, E=21-25, F=26+):
  COMPLEX_MIN      — units at or above this are "complex"   (counted toward the budget)
  CRITICAL_MIN     — units at or above this are "critical"  (zero-tolerance list)
  MAX_COMPLEX_PCT  — max allowed share of complex units vs all units (0–100)
  MAX_CRITICAL     — max allowed count of critical units

Run:
  python tools/quality/complexity_gate.py
  python tools/quality/complexity_gate.py --complex-min 8 --critical-min 15 --max-pct 5 --max-critical 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from radon.complexity import cc_visit, ComplexityVisitor  # noqa: F401
except ImportError:
    sys.exit("radon is not installed — run: pip install radon")

# -- default thresholds --------------------------------------------------------
COMPLEX_MIN = 11     # C and above >> counts toward budget
CRITICAL_MIN = 21    # E and above >> zero-tolerance
MAX_COMPLEX_PCT = 2  # at most 2 % of all units may be "complex"
MAX_CRITICAL = 0     # no critical units allowed

SRC_DIR = Path(__file__).resolve().parents[2] / "backend" / "src"


def collect(src: Path) -> list[dict]:
    records = []
    for path in sorted(src.rglob("*.py")):
        try:
            code = path.read_text(encoding="utf-8", errors="replace")
            blocks = cc_visit(code)
        except Exception as exc:
            print(f"  [skip] {path.relative_to(src)}: {exc}", file=sys.stderr)
            continue
        for block in blocks:
            records.append({
                "file": str(path.relative_to(src)),
                "name": block.fullname,
                "complexity": block.complexity,
                "lineno": block.lineno,
            })
    return records


def grade(cc: int) -> str:
    if cc <= 5:   return "A"
    if cc <= 10:  return "B"
    if cc <= 15:  return "C"
    if cc <= 20:  return "D"
    if cc <= 25:  return "E"
    return "F"


def run(
    *,
    complex_min: int,
    critical_min: int,
    max_complex_pct: float,
    max_critical: int,
) -> bool:
    records = collect(SRC_DIR)
    total = len(records)
    if total == 0:
        print("No Python units found.")
        return True

    complex_units  = [r for r in records if r["complexity"] >= complex_min]
    critical_units = [r for r in records if r["complexity"] >= critical_min]
    complex_pct    = len(complex_units) / total * 100

    # -- summary ---------------------------------------------------------------
    print(f"\n{'-'*60}")
    print(f"  Total units analysed : {total}")
    print(f"  Complex  (cc>={complex_min:2d})      : {len(complex_units):4d}  ({complex_pct:.1f} %)")
    print(f"  Critical (cc>={critical_min:2d})      : {len(critical_units):4d}")
    print(f"{'-'*60}")

    if complex_units:
        print(f"\n  Complex units (cc>={complex_min}):")
        for r in sorted(complex_units, key=lambda x: -x["complexity"]):
            print(f"    [{grade(r['complexity'])}:{r['complexity']:2d}]  {r['file']}:{r['lineno']}  {r['name']}")

    # -- verdict ---------------------------------------------------------------
    failures: list[str] = []
    if complex_pct > max_complex_pct:
        failures.append(
            f"Complex units: {complex_pct:.1f} % > allowed {max_complex_pct} %"
        )
    if len(critical_units) > max_critical:
        failures.append(
            f"Critical units: {len(critical_units)} > allowed {max_critical}"
        )
        for r in sorted(critical_units, key=lambda x: -x["complexity"]):
            failures.append(
                f"  >> [{grade(r['complexity'])}:{r['complexity']}]  {r['file']}:{r['lineno']}  {r['name']}"
            )

    print()
    if failures:
        print("  FAIL")
        for msg in failures:
            print(f"  {msg}")
        print()
        return False

    print(f"  PASS  (complex {complex_pct:.1f} % <= {max_complex_pct} %,"
          f" critical {len(critical_units)} <= {max_critical})")
    print()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Cyclomatic complexity gate")
    parser.add_argument("--complex-min",  type=int,   default=COMPLEX_MIN,    help=f"CC threshold for 'complex'  (default {COMPLEX_MIN})")
    parser.add_argument("--critical-min", type=int,   default=CRITICAL_MIN,   help=f"CC threshold for 'critical' (default {CRITICAL_MIN})")
    parser.add_argument("--max-pct",      type=float, default=MAX_COMPLEX_PCT, help=f"Max %% of complex units      (default {MAX_COMPLEX_PCT})")
    parser.add_argument("--max-critical", type=int,   default=MAX_CRITICAL,   help=f"Max count of critical units  (default {MAX_CRITICAL})")
    args = parser.parse_args()

    ok = run(
        complex_min=args.complex_min,
        critical_min=args.critical_min,
        max_complex_pct=args.max_pct,
        max_critical=args.max_critical,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
