import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.results import RESULTS_DIR, connect

Z = 1.96


def wilson(wins: int, n: int) -> tuple:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def final_table(con, deterministic: int):
    rows = con.execute(
        """
        SELECT b.opponent, b.run_id, SUM(b.won), COUNT(*)
        FROM battles b
        JOIN (SELECT run_id, MAX(train_steps) AS last FROM battles GROUP BY run_id) m
          ON b.run_id = m.run_id AND b.train_steps = m.last
        WHERE b.deterministic = ?
        GROUP BY b.opponent, b.run_id
        """,
        (deterministic,),
    ).fetchall()

    by_opponent = {}
    for opponent, run_id, wins, n in rows:
        by_opponent.setdefault(opponent, []).append((run_id, wins, n))

    print(f"\nfinal checkpoint, deterministic={deterministic}")
    print(f"{'opponent':<24}{'win rate':>10}{'95% CI':>18}{'seeds':>7}{'seed spread':>14}")
    for opponent, entries in sorted(by_opponent.items()):
        per_seed = [w / n for _, w, n in entries]
        wins, n = sum(w for _, w, _ in entries), sum(n for _, _, n in entries)
        p, lo, hi = wilson(wins, n)
        spread = statistics.stdev(per_seed) if len(per_seed) > 1 else float("nan")
        print(f"{opponent:<24}{p:>9.1%}{f'[{lo:.1%}, {hi:.1%}]':>18}{len(per_seed):>7}{spread:>13.1%}")


def curve(con, deterministic: int, x_axis: str, out_path: Path):
    rows = con.execute(
        f"""
        SELECT {x_axis}, opponent, run_id, SUM(won) * 1.0 / COUNT(*), COUNT(*)
        FROM battles WHERE deterministic = ?
        GROUP BY {x_axis}, opponent, run_id
        ORDER BY {x_axis}
        """,
        (deterministic,),
    ).fetchall()

    grouped = {}
    for x, opponent, _, win_rate, n in rows:
        grouped.setdefault((x, opponent), []).append((win_rate, n))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([x_axis, "opponent", "mean_win_rate", "std_across_seeds", "seeds", "battles", "ci_lo", "ci_hi"])
        for (x, opponent), entries in sorted(grouped.items()):
            per_seed = [r for r, _ in entries]
            battles = sum(n for _, n in entries)
            mean = statistics.fmean(per_seed)
            std = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
            _, lo, hi = wilson(round(mean * battles), battles)
            writer.writerow([x, opponent, f"{mean:.4f}", f"{std:.4f}", len(per_seed), battles, f"{lo:.4f}", f"{hi:.4f}"])
    print(f"wrote {out_path}  ({len(grouped)} points)")


def export(con):
    for table in ("runs", "battles"):
        cur = con.execute(f"SELECT * FROM {table}")
        path = RESULTS_DIR / f"{table}.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([d[0] for d in cur.description])
            writer.writerows(cur)
        print(f"wrote {path}")


def main(deterministic: int, x_axis: str):
    con = connect()
    final_table(con, deterministic)
    curve(con, deterministic, x_axis, RESULTS_DIR / f"curve_{x_axis}.csv")
    export(con)
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--x", choices=["train_steps", "train_episodes"], default="train_steps")
    args = parser.parse_args()
    main(args.deterministic, args.x)
