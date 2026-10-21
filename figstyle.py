"""One style for every paper figure: Computer Modern text and maths (the paper's own font, bundled with matplotlib, so the
result does not depend on a TeX install), TrueType (Type 42) fonts embedded in the PDFs, fixed font sizes in points and figure
widths equal to the sizes at which the paper includes them, so the type is never rescaled in the PDF.

    from figstyle import apply, W, W_HALF, C, ...
    apply()                      # once, before creating figures
    fig = plt.figure(figsize=(W, 2.9))
"""
import logging

import matplotlib

logging.getLogger("fontTools").setLevel(logging.ERROR)  # cmr10 has an empty creation date
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = 6.45  # \textwidth of the paper (16.4 cm) in inches: full-width figures
W_HALF = 3.35  # \includegraphics[width=0.52\textwidth]

# font sizes (pt) as printed
FS_BASE = 9  # axis labels, panel titles, annotations
FS_TICK = 8
FS_LEG = 8
FS_SMALL = 7.5  # labels inside plots

# method colours, shared by all figures
C = dict(qDRIFT="#555555", full="#e08e0b", random="#c0392b", magnitude="#8e6c8a", graph="#1f6fb4", oracle="#2a9d55",
         weights="#7a5195", variances="#d17ba0")


# one dash pattern per method, so that coinciding curves stay visible (the proposed method and full H are solid)
DASH = dict(qDRIFT=(0, (5, 2)), full="-", random=(0, (6, 2, 1.5, 2)), magnitude=(2, (2, 2)), graph="-",
            oracle=(0, (1.2, 1.5)), weights=(0, (4, 1.5, 1, 1.5)), variances=(0, (3, 3)))


def apply():
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,  # cmr10 has no U+2212
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": FS_BASE,
            "axes.labelsize": FS_BASE,
            "axes.titlesize": FS_BASE,
            "axes.titlepad": 5,
            "legend.fontsize": FS_LEG,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            "lines.markeredgewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "0.88",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
