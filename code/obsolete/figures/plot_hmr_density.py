import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_palette("bright")
sns.set_style("whitegrid")
sns.set(font_scale=1.2)

x = pd.read_csv("./tables/hmr_density.csv", index_col=0)

plot(x.mass, x.effective_ns_per_day, 'o')
xlabel("Mass [amu]")
ylabel("Effective ns / day")
xlim(0, 5)

plt.savefig("./manuscript/figures/hmr_masses.pdf", bbox_inches="tight")
