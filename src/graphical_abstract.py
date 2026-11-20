"""Graphical abstract: configuration + divergence speed along the lift."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as cm
from common import C1, C2, C3, INK, INK2, save
from fig1_schematic import panel_a
cm.setup_style()
s3 = json.load(open(f"{cm.RES}/s3_statics.json"))
al = s3["along_lift"]; L = np.array(al["L"]); z = np.array(al["z_payload"])
fig = plt.figure(figsize=(6.6, 2.65))
a0 = fig.add_axes([0.0, 0.0, 0.40, 1.0]); panel_a(a0)
a = fig.add_axes([0.57, 0.17, 0.41, 0.70])
fits = [k for k in al if k.startswith("box") and "x_s" not in k and "Eq." not in k and "|" not in k]
band = np.array([al[k] for k in fits])
a.fill_between(L, band.min(0), band.max(0), color=C2, alpha=0.22, lw=0)
a.plot(L, al["box"], color=C2, lw=1.4, ls="--", label="sharp-edged box, wind 18° off axis (data)")
a.plot(L, al["Munk (BEM box)"], color=C1, lw=1.6, label="potential flow, wind along axis (BEM)")
Uz = 12.0 * (z / cm.z_hub) ** cm.alpha_shear
a.plot(L, Uz, color=INK, lw=0.9, label=r"wind at payload, $U_{hub}$ = 12 m/s")
a.set_xlim(cm.L_mating, 150); a.set_ylim(0, 45)
a.set_xlabel(r"fall length $L$ (m):  mating $\rightarrow$ lift-off")
a.set_ylabel(r"$U_{div}$ (m/s)")
a.legend(fontsize=6.3, loc="upper right")
a.set_title(r"$U_{div}=\sqrt{2Wr_{eq}^2/(\rho A_s L_n c'_{\max} L)}$", fontsize=8)
save(fig, "GraphicalAbstract")
print("ok")
