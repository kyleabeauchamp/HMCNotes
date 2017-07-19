#!/usr/bin/env python
import pandas as pd
from lb_loader import summarize
import fire


def one(csv_filename, out_filename):
    statedata, summary = summarize(csv_filename)
    summary.to_csv(out_filename)


def many(csv_filenames, out_csv):
    csv_filenames = csv_filenames.split(",")
    x = {filename: pd.read_csv(filename, names=["KEY", "VALUE"], index_col=0).VALUE for filename in csv_filenames}
    x = pd.DataFrame(x)
    x.to_csv(out_csv)


if __name__ == "__main__":
    fire.Fire({"one": one, "many": many})
