import numpy as np
import pandas as pd

O = flatten_array(ref["O"]) #visop['default'].observed
B12 = flatten_array(ref["B-rttov12"]) #rttov122['default'].synthetic
B13 = flatten_array(ref["B-rttov13"]) #rttov132['default'].synthetic
albedo = flatten_array(ref["albedo"])
sza = flatten_array(ref["SZA"])

correction_filter = {
    "sza-correction": lambda x: x<70, 
    "O": lambda x: x<1,
    "default": lambda x: True,
}

mask = O>0

OmB12 = O[mask] - B12[mask]
OmB13 = O[mask] - B13[mask]

mean12 = OmB12.mean()
std12 = OmB12.std()

mean13 = OmB13.mean()
std13 = OmB13.std()

C_a12, C_b12, C_o = cloud_impact_av(0.0, B12[mask], O[mask])
C_a13, C_b13, _ = cloud_impact_av(0.0, B13[mask], O[mask])
settings = "All dataset reflectance predictor (thresh=0.0) at 9, 12, and 15 UTC"

df = pd.DataFrame({
    'V12': OmB12,
    'V13': OmB13,
    "C_o":C_o,
    "C_b12":C_b12,
    "C_b13":C_b13,
    "C_a12":C_a12,
    "C_a13":C_a13,
})

print(df)

# sensitivity to Reflectance

# Number of bins
n_bins = 25

# Calculate bins for clc_low, clc_mid, and clc_frac
bins_O = np.linspace(df['C_o'].min(), df['C_o'].max(), n_bins)
bins_Bv12 = np.linspace(df['C_b12'].min(), df['C_b12'].max(), n_bins)
bins_Bv13 = np.linspace(df['C_b13'].min(), df['C_b13'].max(), n_bins)
bins_sym_Bv12 = np.linspace(df['C_a12'].min(), df['C_a12'].max(), n_bins)
bins_sym_Bv13 = np.linspace(df['C_a13'].min(), df['C_a13'].max(), n_bins)

# Group data and calculate standard deviation and mean
grouped_O_std = df.groupby(pd.cut(df['C_o'], bins=bins_O))[['V12', 'V13']].std()
grouped_O_mean = df.groupby(pd.cut(df['C_o'], bins=bins_O))[['V12', 'V13']].mean()

grouped_Bv12_std = df.groupby(pd.cut(df['C_b12'], bins=bins_Bv12))[['V12']].std()
grouped_Bv12_mean = df.groupby(pd.cut(df['C_b12'], bins=bins_Bv12))[['V12']].mean()
grouped_Bv13_std = df.groupby(pd.cut(df['C_b13'], bins=bins_Bv13))[['V13']].std()
grouped_Bv13_mean = df.groupby(pd.cut(df['C_b13'], bins=bins_Bv13))[['V13']].mean()

grouped_sym_Bv12_std = df.groupby(pd.cut(df['C_a12'], bins=bins_sym_Bv12))[['V12']].std()
grouped_sym_Bv12_mean = df.groupby(pd.cut(df['C_a12'], bins=bins_sym_Bv12))[['V12']].mean()
grouped_sym_Bv13_std = df.groupby(pd.cut(df['C_a13'], bins=bins_sym_Bv13))[['V13']].std()
grouped_sym_Bv13_mean = df.groupby(pd.cut(df['C_a13'], bins=bins_sym_Bv13))[['V13']].mean()

# Get the midpoints of the intervals for the x-axis
x_values_O = np.array([(interval.left + interval.right) / 2 for interval in grouped_O_std.index])
x_values_B12 = np.array([(interval.left + interval.right) / 2 for interval in grouped_Bv12_std.index])
x_values_B13 = np.array([(interval.left + interval.right) / 2 for interval in grouped_Bv13_std.index])
x_values_sym_B12 = np.array([(interval.left + interval.right) / 2 for interval in grouped_sym_Bv12_std.index])
x_values_sym_B13 = np.array([(interval.left + interval.right) / 2 for interval in grouped_sym_Bv13_std.index])

# Create a figure and six subplots (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(9, 7))  # Adjusted figure size
plt.rcParams.update({'font.size': 12})

# --- Plot means in the top row ---
axes[0, 0].plot(x_values_O, np.array(grouped_O_mean['V12']), label='V12', marker='.', color='#f564d4')
axes[0, 0].plot(x_values_O, np.array(grouped_O_mean['V13']), label='V13', marker='.', color='#3ba3ec')
axes[0, 0].set_xlabel('')
axes[0, 0].set_ylabel('Mean(O-B)')
axes[0, 0].set_title('')
axes[0, 0].legend()
axes[0, 0].tick_params(axis='x', labelrotation=45)

axes[0, 1].plot(x_values_B12, np.array(grouped_Bv12_mean['V12']), label='V12', marker='.', color='#f564d4')
axes[0, 1].plot(x_values_B13, np.array(grouped_Bv13_mean['V13']), label='V13', marker='.', color='#3ba3ec')
axes[0, 1].set_xlabel('')
axes[0, 1].set_ylabel('')
axes[0, 1].set_title('')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', labelrotation=45)

axes[0, 2].plot(x_values_sym_B12, np.array(grouped_sym_Bv12_mean['V12']), label='V12', marker='.', color='#f564d4')
axes[0, 2].plot(x_values_sym_B13, np.array(grouped_sym_Bv13_mean['V13']), label='V13', marker='.', color='#3ba3ec')
axes[0, 2].set_xlabel('')
axes[0, 2].set_ylabel('')
axes[0, 2].set_title('')
axes[0, 2].legend()
axes[0, 2].tick_params(axis='x', labelrotation=45)

# --- Plot standard deviations in the bottom row ---
axes[1, 0].plot(x_values_O, np.array(grouped_O_std['V12']), label='V12', marker='.', color='#f564d4')
axes[1, 0].plot(x_values_O, np.array(grouped_O_std['V13']), label='V13', marker='.', color='#3ba3ec')
axes[1, 0].set_xlabel(r'$R_o$')  # Changed x-axis label
axes[1, 0].set_ylabel('Std(O-B)')
axes[1, 0].set_title('')
axes[1, 0].legend()
axes[1, 0].tick_params(axis='x', labelrotation=45)

axes[1, 1].plot(x_values_B12, np.array(grouped_Bv12_std['V12']), label='V12', marker='.', color='#f564d4')
axes[1, 1].plot(x_values_B13, np.array(grouped_Bv13_std['V13']), label='V13', marker='.', color='#3ba3ec')
axes[1, 1].set_xlabel(r'$R_b$')  # Changed x-axis label
axes[1, 1].set_ylabel('')
axes[1, 1].set_title('')
axes[1, 1].legend()
axes[1, 1].tick_params(axis='x', labelrotation=45)

axes[1, 2].plot(x_values_sym_B12, np.array(grouped_sym_Bv12_std['V12']), label='V12', marker='.', color='#f564d4')
axes[1, 2].plot(x_values_sym_B13, np.array(grouped_sym_Bv13_std['V13']), label='V13', marker='.', color='#3ba3ec')
axes[1, 2].set_xlabel(r'($R_o$+$R_b$)/2')  # Changed x-axis label
axes[1, 2].set_ylabel('')
axes[1, 2].set_title('')
axes[1, 2].legend()
axes[1, 2].tick_params(axis='x', labelrotation=45)

# Remove y-axis tick labels for columns 2 and 3
for i in range(1, 3):
    axes[0, i].set_yticklabels([])
    axes[1, i].set_yticklabels([])

# Remove x-axis tick labels for the first row
for i in range(3):
    axes[0, i].set_xticklabels([])


for i in range(3):
    axes[0, i].set_xlim(0, 1)
    axes[1, i].set_xlim(0, 1)
    axes[0, i].set_ylim(-0.3, 0.9)
    axes[1, i].set_ylim(0.0, 0.5)
    axes[0, i].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    axes[1, i].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

plt.subplots_adjust(wspace=0, hspace=0)  # Set horizontal spacing to 0
fig.suptitle(f"settings: {settings}", fontsize=16)
plt.tight_layout()
plt.savefig(f"")
plt.show()

grouped_sym_Bv12_std = grouped_sym_Bv12_std.reset_index()  # If "O" is an index and you want it as a column
grouped_sym_Bv12_std.columns = ["C_a12", "V12"]
grouped_sym_Bv13_std = grouped_sym_Bv13_std.reset_index()  # If "O" is an index and you want it as a column
grouped_sym_Bv13_std.columns = ["C_a13", "V13"]

switch = {"accumulate": False, "clip": False}
if switch["accumulate"]:
    grouped_sym_Bv13_std["V13"] = np.maximum.accumulate(grouped_sym_Bv13_std["V13"])
    grouped_sym_Bv12_std["V12"] = np.maximum.accumulate(grouped_sym_Bv12_std["V12"])

if switch["clip"]:
    grouped_sym_Bv13_std["V13"] = grouped_sym_Bv13_std["V13"].clip(upper=0.15)
    grouped_sym_Bv12_std["V12"] = grouped_sym_Bv12_std["V12"].clip(upper=0.15)

# Create a figure and six subplots (2 rows, 3 columns)
fig, axes = plt.subplots(1, 1, figsize=(6, 5))  # Adjusted figure size
plt.rcParams.update({'font.size': 16})

# --- Plot means in the top row ---


# --- Plot standard deviations in the bottom row ---

axes.plot(x_values_sym_B12, np.array(grouped_sym_Bv12_std['V12']), label='V12', marker='.', color='#f564d4')
axes.plot(x_values_sym_B13, np.array(grouped_sym_Bv13_std['V13']), label='V13', marker='.', color='#3ba3ec')
axes.set_xlabel(r'(max($R_o$-thresh, 0) + max($R_b$-thresh, 0)/2')  # Changed x-axis label
axes.set_ylabel('OmB Stdev')
axes.set_title('')
axes.legend()
axes.tick_params(axis='x', labelrotation=45)

for i in range(3):
    axes.set_xlim(0, 1)
    axes.set_ylim(0.0, 0.5)
    axes.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

plt.subplots_adjust(wspace=0, hspace=0)  # Set horizontal spacing to 0
fig.suptitle(f"settings: {settings}", fontsize=16)
plt.tight_layout()
plt.savefig(f"/home/km4c/to_ucloud/error_vs_predictor_12utc_case_9.png")
plt.show()

# Retrieve the dynamic error

df["Std-V12"] = retrieve_stds_from_binned_Ca(df['C_a12'], grouped_sym_Bv12_std, ["C_a12","V12"], zero_Ca=grouped_sym_Bv13_std["V13"][0])
df["Std-V13"] = retrieve_stds_from_binned_Ca(df['C_a13'], grouped_sym_Bv13_std, ["C_a13","V13"], zero_Ca=grouped_sym_Bv13_std["V13"][0])

df["norm-V12"] = np.array((df["V12"]-mean12)/df["Std-V12"])
df["V12"] = np.array((df["V12"]-mean12)/std12)
df["norm-V13"] = np.array((df["V13"]-mean13)/df["Std-V13"])
df["V13"] = np.array((df["V13"]-mean13)/std13)

print(df)

