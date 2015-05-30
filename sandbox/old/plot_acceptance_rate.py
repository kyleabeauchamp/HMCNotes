import pandas as pd


x = pd.read_csv("./data_flexible_water.csv")

plot(x.steps_per_hmc, x.acceptance, "o")
ylabel("Acceptance Rate")
xlabel("Number of HMC steps")
