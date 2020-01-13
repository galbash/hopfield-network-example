"""
generates aggregative stats
"""
import os.path
import csv
import collections


def main():
    stats = collections.defaultdict(list)
    for i in range(1, 101):
        print("processing execution:", i)
        with open(os.path.join("batch_results", str(i), "report.csv"), "rt") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats[
                    (row["A"], row["B"], row["C"], row["D"], row["sigma"], row["alpha"])
                ].append(float(row["convergence_rate"]))
    for stat, convs in stats.items():
        print(",".join(stat), sum(convs) / len(convs))


if __name__ == "__main__":
    main()
