import glob
import scipy.stats
import pymbar
import pandas as pd
import numpy as np
pd.set_option('display.width', 1000)
from lb_loader import summarize

filenames = glob.glob("data/switchedaccurateflexiblewater/*.csv")
data = []
for filename in filenames:
    data.append(summarize(filename)[1])

data = pd.DataFrame(data)
data["filename"] = data["filename"].str[35:]
data.sort("filename")
