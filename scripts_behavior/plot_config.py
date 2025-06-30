import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'

plt.rc('axes', titlesize=16)       # Title size
plt.rc('axes', labelsize=16)       # Axis label size
plt.rc('xtick', labelsize=14)      # X-tick label size
plt.rc('ytick', labelsize=14)      # Y-tick label size
plt.rc('legend', fontsize=12)      # Legend font size
plt.rc('font', size=14)            # General font size

session_colors = {
    'pre': '#2C7BB6',   # Cerulean Blue
    'post': '#D7191C'   # Vermilion
}