import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from lib2Bspec import read_spectrum
from argparse import ArgumentParser
import webbrowser

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

m_max = 1000
m_min = 0
E_max = 1000.
E_min = 0.


# Update function for slider
def update(val):
    flux = val * aa.flux_multiplier + aa.flux_shift
    E,m  = read_spectrum(1.0, flux, aa.R_0)
    line.set_data(m, E)
    if not aa.no_rescale:
        ax.relim()  # Recalculate limits
        ax.autoscale_view()  # Auto-scale
    fig.canvas.draw_idle()


# Create a sample interactive plot with slider
fig = go.Figure()

# Set up the parameter space
flux_range = np.arange(aa.flux_min,aa.flux_max,aa.flux_step)
try:
    start_index = np.where(abs(flux_range)<1e-10)[0][0]
except IndexError:
    start_index = 0

# Add traces (one per dataset)
for i,flux_factor in enumerate(flux_range):
    flux = flux_factor * aa.flux_multiplier + aa.flux_shift
    E, m = read_spectrum(1.0,flux,aa.R_0)
    m_max = min(max(m),m_max)
    m_min = min(min(m),m_min)
    E_max = min(max(E)*1.05,E_max)
    E_min = min(min(E),E_min)
    fig.add_trace(
        go.Scatter(
            x=m, y=E,
            mode = "markers",
            marker = dict(
                symbol = "line-ew",
                color  = "black",
                size   = 8,
                line   = dict(width=2)
            ),
            visible=(i==start_index),  # Only first trace visible initially
            name=f"Flux = {flux-aa.flux_shift:.4f}"
        )
    )

# Create slider steps
steps = []

for i,flux_factor in enumerate(flux_range):
    flux = flux_factor * aa.flux_multiplier + aa.flux_shift
    step = dict(
        method="update",
        args=[{"visible": [False] * len(flux_range)},{"title": f"Flux = {flux - aa.flux_shift:.4f}"}],
        label=f"{flux_factor:.2f}"
    )
    step["args"][0]["visible"][i] = True  # Show current trace
    steps.append(step)

# Add slider
sliders = [dict(
    active=start_index,
    bgcolor="steelblue",
    activebgcolor="steelblue",
    currentvalue={"prefix": "Flux value (multiple of h/e): ",
                    "font": {"size":14},
                    },
    steps=steps
)]

steps[0]["args"][0]["visible"][0] = True  # Show first trace

if aa.no_rescale:
    fig.update_layout(
        sliders=sliders,
        title="Flux = 0.0000",
        xaxis_range = [m_min,m_max],
        yaxis_range = [E_min,E_max],
        plot_bgcolor = "lightsteelblue"
    )
else:
    fig.update_layout(
    sliders=sliders,
    title="Flux = 0.0000",
    plot_bgcolor = "lightsteelblue"
    )

# Save as standalone HTML
fig.write_html(f"plots/spectrum_R_{aa.R_0:.2f}.html")

# Open the HTML file
webbrowser.open(f"plots/spectrum_R_{aa.R_0:.2f}.html")


