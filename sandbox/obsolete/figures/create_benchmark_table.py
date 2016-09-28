import pandas as pd

x = pd.read_csv("./tables/raw_performance.csv")
print x.set_index("name")[["ns_per_day"]].to_latex(formatters={"ns_per_day":lambda x: "%.1f" % x})
