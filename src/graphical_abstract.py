"""Graphical abstract: configuration + divergence speed along the lift."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as cm
from common import C1, C2, C3, INK, INK2, save
import fig1_schematic as f1s
from fig1_schematic import panel_a
cm.setup_style()
s3 = json.load(open(f"{cm.RES}/s3_statics.json"))
al = s3["along_lift"]; L = np.array(al["L"]); z = np.array(al["z_payload"])
fig = plt.figure(figsize=(6.6, 2.65))
f1s.FS = 0.82
a0 = fig.add_axes([0.0, 0.0, 0.40, 1.0]); panel_a(a0)
a = fig.add_axes([0.57, 0.17, 0.41, 0.70])
fits = [k for k in al if k.startswith("box") and "x_s" not in k and "Eq." not in k and "|" not in k]
band = np.array([al[k] for k in fits])
a.fill_between(L, band.min(0), band.max(0), color=C2, alpha=0.22, lw=0)
a.plot(L, al["box"], color=C2, lw=1.4, ls="--", label="sharp-edged box, wind 18° off axis (data)")
a.plot(L, al["Munk (BEM box)"], color=C1, lw=1.6, label="potential flow, wind along axis (BEM)")
# twin-hook sling connection (a = 0.5 m, b = 2 m, h_s = 8 m): k_c independent of L
cb = json.load(open(f"{cm.RES}/s3b_connection.json"))
kc = cb["twin_hook"]["a=0.5|b=2.0|h=8.0"]["k_c"]
kL = np.array([cm.suspension(x).k_psi for x in L])
a.plot(L, np.array(al["Munk (BEM box)"]) * np.sqrt(kc / (kL + kc)), color=C3, lw=1.3, ls="-.",
       label=r"potential flow, twin-hook slings ($k_c/k_\psi$ = 1 at mating)")
Uz = 12.0 * (z / cm.z_hub) ** cm.alpha_shear
a.plot(L, Uz, color=INK, lw=0.9, label=r"wind at payload, $U_{hub}$ = 12 m/s")
a.set_xlim(cm.L_mating, 150); a.set_ylim(0, 45)
a.set_xlabel(r"fall length $L$ (m):  mating $\rightarrow$ lift-off")
a.set_ylabel(r"$U_{div}$ (m/s)")
a.legend(fontsize=5.8, loc="upper right")
a.text(0.98, 0.04, "bare suspension, no tugger lines;\nopen-loop 1° tolerance at mating: 4.2–5.4 m/s", transform=a.transAxes,
       ha="right", va="bottom", fontsize=6.0, color=INK2)
a.set_title(r"$U_{div}=\sqrt{2Wr_{eq}^2/(\rho A_s L_n c'_{\max} L)}$", fontsize=8)
save(fig, "GraphicalAbstract")
print("ok")
