import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from lib2Bspec import read_spectrum
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("--B_0", type=float, default=1.0, help="background magnetic field")
ap.add_argument("--flux_min",type=float, default=-0.5)
ap.add_argument("--flux_max", type=float,default=0.51)
ap.add_argument("--flux_step", type=float, default = 0.1)
ap.add_argument("--flux_multiplier", type=float, default = 6.283185)
ap.add_argument("--flux_shift", type=float, default=0.01)
ap.add_argument("--R_0",type=float, default=1.0, help="radius of droplet, can be multiple values")
ap.add_argument("--no_rescale", action="store_true", default=False, help="use this to disable rescaling the axes as the slider moves")

aa = ap.parse_args()

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)  # Make room for the slider

# Initialize with empty plot
line, = ax.plot([], [], 'k_')
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_title('Interactive Data Plot')

# Create slider axis
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, r"$\epsilon$ value for $\Phi=\epsilon h/e$", 
                        aa.flux_min, aa.flux_max, valinit=0., valstep=aa.flux_step)

# Function to load data from file
def load_data(file_index):
    filename = f"data_{file_index}.txt"
    try:
        data = np.loadtxt(filename)
        x = data[:, 0]
        y = data[:, 1]
        return x, y
    except:
        print(f"Could not load file {filename}")
        return np.array([]), np.array([])

# Update function for slider
def update(val):
    flux = val * aa.flux_multiplier + aa.flux_shift
    E,m  = read_spectrum(1.0, flux, aa.R_0)
    line.set_data(m, E)
    if not aa.no_rescale:
        ax.relim()  # Recalculate limits
        ax.autoscale_view()  # Auto-scale
    fig.canvas.draw_idle()
    ax.set_title(f"R_0={aa.R_0}, Flux={flux-aa.flux_shift:.4f}")

# Connect slider to update function
slider.on_changed(update)

# Load and plot initial data
initial_val = 0.
update(initial_val)
ax.relim()  # Recalculate limits
ax.autoscale_view()  # Auto-scale

plt.show()