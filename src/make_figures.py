"""Figures 2-9 of the revised manuscript (single consistent style)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import common as cm
from common import C1, C2, C3, C4, C5, INK, INK2, GRID, panel_label, save

cm.setup_style()
R = lambda n: json.load(open(f"{cm.RES}/{n}.json"))


# ------------------------------------------------------------------ Fig. 2
def fig2():
    s1 = R("s1_suspension")
    fig, ax = plt.subplots(1, 3, figsize=(cm.W2, 2.25))
    # (a) restoring moment
    a = ax[0]
    ps = np.linspace(0, 150, 301)
    a.plot(ps, np.sin(np.radians(ps)), color=C1, lw=1.4, label="closed form (exact)")
    a.plot(ps, np.radians(ps), color=INK2, lw=1.0, ls="--", label=r"linear, $k_\psi\psi$")
    c = s1["moment_cases"][f"axle_L{int(cm.L_mating)}_eta1.0"]
    a.plot(c["psi_deg"], -np.array(c["M_num"]) / c["k_cf"], ls="none", marker="o", mfc="white",
           mec=C2, mew=1.0, ms=4.5, label=r"3D equilibrium, $\eta$=1")
    c = s1["moment_cases"][f"axle_L{int(cm.L_mating)}_eta0.96"]
    a.plot(c["psi_deg"], -np.array(c["M_num"]) / c["k_cf"], ls="none", marker="^", mfc=C3,
           mec="white", mew=0.6, ms=4.5, label=r"3D equilibrium, $\eta$=0.96")
    a.set_xlim(0, 150); a.set_ylim(0, 1.95)
    a.set_xlabel(r"block rotation $\psi$ (deg)")
    a.set_ylabel(r"restoring moment $-M_s/k_\psi$")
    a.set_xticks([0, 30, 60, 90, 120, 150])
    a.legend(loc="upper right", fontsize=6.2)
    panel_label(a, "a")
    # (b) frequencies along the lift
    b = ax[1]
    al = s1["along_lift"]
    L = np.array(al["L"])
    b.axhspan(1 / 12.0, 1 / 5.0, color=GRID, alpha=0.6, lw=0)
    b.text(147, 0.13, "wave band\n(5-12 s)", ha="right", va="center", fontsize=6.5, color=INK2)
    b.plot(L, al["f"], color=C1, lw=1.4, label=r"yaw, $f_\psi$")
    b.plot(L, al["f_pend"], color=C2, lw=1.4, ls="--", label=r"pendulum, $f_p$")
    b.set_yscale("log")
    b.set_xlim(cm.L_mating, 150); b.set_ylim(0.008, 0.3)
    b.set_xlabel(r"fall length $L$ (m)")
    b.set_ylabel("natural frequency (Hz)")
    b.legend(loc="lower left", fontsize=6.5)
    panel_label(b, "b")
    # (c) static twist
    c_ = ax[2]
    c_.plot(L, al["psi0_deg"], color=C1, lw=1.4, label=r"rope-torque twist $\psi_0$")
    c_.axhline(1.0, color=C2, lw=1.0, ls=":", label="mating tolerance")
    c_.set_xlim(cm.L_mating, 150); c_.set_ylim(0, 3.5)
    c_.set_xlabel(r"fall length $L$ (m)")
    c_.set_ylabel(r"$\psi_0$ (deg)")
    c_.legend(loc="upper left", fontsize=6.5)
    panel_label(c_, "c")
    fig.tight_layout(w_pad=1.6)
    save(fig, "Fig2_suspension")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 3
def fig3():
    s2 = R("s2_aero_bem")
    CK = s2["CK"]
    fig, ax = plt.subplots(1, 2, figsize=(cm.W2, 2.45), gridspec_kw=dict(width_ratios=[1.05, 1]))
    a = ax[0]
    rows = [
        ("prolate spheroid (Lamb)", CK["munk_spheroid_own_volume"], None, C1),
        ("superellipsoid, N = 4 (BEM)", CK["bem_superellipsoid_n4"], None, C1),
        ("superellipsoid, N = 8 (BEM)", CK["bem_superellipsoid_n8"], None, C1),
        ("rectangular box (BEM)", CK["bem_box"], None, C1),
        ("sharp-edged box, wind tunnel", CK["box_empirical_mid"],
         (CK["box_empirical_low"], CK["box_empirical_high"]), C2),
        (r"  + suspension axis 1 m downwind", 0.055 + 1.35 / 20, None, C2),
        (r"  + suspension axis 1 m upwind", 0.055 - 1.35 / 20, None, C2),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, rng, col) in zip(y, rows):
        if rng:
            a.plot(rng, [yi, yi], color=col, lw=2.2, solid_capstyle="round", alpha=0.45)
        mk = "o" if col == C1 else "s"
        a.plot([v], [yi], marker=mk, color=col, ms=5.5, mec="white", mew=0.7, ls="none")
        a.text(0.62, yi, f"{v:.3f}", va="center", ha="right", fontsize=6.5, color=INK2)
    a.axvline(0, color=INK2, lw=0.6)
    a.set_yticks(y)
    a.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    a.set_xlim(-0.05, 0.63)
    a.set_xlabel(r"aerodynamic yaw-stiffness coefficient $C_K$")
    a.grid(axis="y", visible=False)
    a.text(0.12, y[1] + 0.0, "potential flow", color=C1, fontsize=7, ha="center", va="center")
    a.text(0.30, y[5] + 0.0, "separated\n(bluff-body) flow", color=C2, fontsize=7, ha="center", va="center")
    a.set_ylim(-0.6, len(rows) - 0.1)
    panel_label(a, "a", x=-0.62)
    # (b) moment curves
    b = ax[1]
    be = np.linspace(0, 90, 361)
    br = np.radians(be)
    r = 0.5
    b.plot(be, 0.5 * CK["bem_box"] * np.sin(2 * br), color=C1, lw=1.4,
           label="potential flow, box (Munk)")
    lo = 0.075 * (1 - r ** 2) * np.sin(2 * br) - 0.04 * r * np.sin(4 * br)
    hi = 0.09 * (1 - r ** 2) * np.sin(2 * br) - 0.03 * r * np.sin(4 * br)
    mid = 0.09 * (1 - r ** 2) * np.sin(2 * br) - 0.04 * r * np.sin(4 * br)
    b.fill_between(be, lo, hi, color=C2, alpha=0.25, lw=0)
    b.plot(be, mid, color=C2, lw=1.4, ls="--", label="sharp-edged box (Greenwell fit)")
    b.axhline(0, color=INK2, lw=0.6)
    b.set_xlim(0, 90); b.set_xticks([0, 15, 30, 45, 60, 75, 90])
    b.set_xlabel(r"incidence $\beta$ (deg)")
    b.set_ylabel(r"$M_a/(q A_s L_n)$")
    b.set_ylim(-0.01, 0.34)
    b.legend(loc="upper center", fontsize=6.3, ncol=1)
    panel_label(b, "b")
    fig.tight_layout(w_pad=1.2)
    save(fig, "Fig3_aero_coefficients")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 4
def fig4():
    s3 = R("s3_statics")
    al = s3["along_lift"]
    L = np.array(al["L"])
    z = np.array(al["z_payload"])
    fig, ax = plt.subplots(1, 2, figsize=(cm.W2, 3.0))
    a = ax[0]
    a.fill_between(L, al["box, low"], al["box, high"], color=C2, alpha=0.22, lw=0)
    a.plot(L, al["box, mid"], color=C2, lw=1.4, ls="--", label="sharp-edged box")
    a.plot(L, al["box, mid, x_s=-1 m"], color=C2, lw=1.0, ls=":", label=r"box, axis 1 m downwind")
    a.plot(L, al["Munk (spheroid)"], color=C3, lw=1.2, ls="-.", label="potential flow, spheroid")
    a.plot(L, al["Munk (BEM box)"], color=C1, lw=1.5, label="potential flow, box")
    # wind at payload height for hub-height winds 10 and 15 m/s
    for Uh, ls in ((10.0, "-"), (15.0, "-")):
        Uz = Uh * (z / cm.z_hub) ** cm.alpha_shear
        a.plot(L, Uz, color=INK, lw=0.9, ls=ls)
        a.text(cm.L_mating + 2, Uz[0] - 0.7, rf"$U_{{hub}}$ = {Uh:.0f} m/s", ha="left", va="top", fontsize=6.3,
               color=INK)
    a.set_xlim(cm.L_mating, 150); a.set_ylim(0, 60)
    a.set_xlabel(r"fall length $L$ (m)  [mating $\rightarrow$ lift-off]")
    a.set_ylabel(r"divergence speed $U_{div}$ (m/s)")
    a.plot([], [], color=INK, lw=0.9, label="mean wind at payload height")
    a.legend(loc="upper center", fontsize=6.0, ncol=2, bbox_to_anchor=(0.5, -0.22))
    sec = a.secondary_xaxis("top", functions=(lambda x: cm.z_payload(x), lambda x: cm.z_payload(x)))
    sec.set_xlabel("payload height above MSL (m)", fontsize=7)
    sec.tick_params(labelsize=6.5)
    panel_label(a, "a", y=1.12)
    # (b) heading dependence
    b = ax[1]
    hd = s3["heading"]
    th = np.array(hd["theta_deg"])
    b.plot(th, np.array(hd["Munk (BEM box)"]) / s3["CK"]["Munk (BEM box)"], color=C1, lw=1.5,
           label="potential flow (Munk)")
    b.plot(th, np.array(hd["box, mid"]) / s3["CK"]["box, mid"], color=C2, lw=1.4, ls="--",
           label="sharp-edged box, centred axis")
    b.plot(th, np.array(hd["box, mid, x_s=+1 m"]) / s3["CK"]["box, mid"], color=C2, lw=1.0, ls=":",
           label=r"box, axis 1 m towards end A ($x_s$ = +1 m)")
    b.axhline(0, color=INK2, lw=0.6)
    b.axvspan(45, 135, color=GRID, alpha=0.5, lw=0)
    b.text(90, 2.2, "wind across the payload:\naerodynamically stabilising", ha="center", fontsize=6.3,
           color=INK2)
    b.set_xlim(0, 180); b.set_xticks([0, 45, 90, 135, 180])
    b.set_ylim(-4.2, 3.2)
    b.set_xlabel(r"mean wind direction relative to payload axis $\theta$ (deg)")
    b.set_ylabel(r"$K_a(\theta)/K_a(0)$")
    b.legend(loc="upper center", fontsize=6.0, ncol=2, bbox_to_anchor=(0.5, -0.22))
    panel_label(b, "b", y=1.12)
    fig.tight_layout(w_pad=1.6)
    save(fig, "Fig4_divergence_map")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 5
def fig5():
    s3 = R("s3_statics")
    u = np.array(s3["u_grid"])
    fig, ax = plt.subplots(1, 2, figsize=(cm.W2, 2.5), sharey=True)
    for a, key, title, lab in ((ax[0], "Munk (BEM box) | L=100", "potential flow (Munk): supercritical", "a"),
                               (ax[1], "box, mid | L=100", "sharp-edged box fit: subcritical", "b")):
        br = s3["branches"][key]
        pts = np.array([(x, p, st) for x, p, st in br["perfect"]], dtype=object)
        uu = np.array([p[0] for p in br["perfect"]], float)
        pp = np.array([p[1] for p in br["perfect"]], float)
        stt = np.array([p[2] for p in br["perfect"]], bool)
        sel = (pp >= -1) & (pp <= 90)
        a.plot(uu[sel & stt], pp[sel & stt], ls="none", marker="o", ms=1.6, color=C1)
        a.plot(uu[sel & ~stt], pp[sel & ~stt], ls="none", marker="o", ms=1.0, color=C1, alpha=0.35)
        # imperfect: all equilibria
        uu2 = np.array([p[0] for p in br["imperfect"]], float)
        pp2 = np.array([p[1] for p in br["imperfect"]], float)
        st2 = np.array([p[2] for p in br["imperfect"]], bool)
        sel2 = (pp2 >= -1) & (pp2 <= 90)
        a.plot(uu2[sel2 & st2], pp2[sel2 & st2], ls="none", marker="s", ms=1.4, color=C2)
        a.plot(uu2[sel2 & ~st2], pp2[sel2 & ~st2], ls="none", marker="s", ms=1.0, color=C2, alpha=0.35)
        amp = np.array([np.nan if v is None else v for v in br["amplification_linear"]], float)
        a.plot(u, amp, color=INK2, lw=0.9, ls="--")
        a.axhline(1.0, color=INK2, lw=0.6, ls=":")
        a.set_xlim(0, 1.6); a.set_ylim(-2, 92)
        a.set_xlabel(r"$U/U_{div}$")
        a.set_title(title, fontsize=7.5, color=INK)
        panel_label(a, lab)
    ax[0].set_ylabel(r"equilibrium yaw $\psi_{eq}$ (deg)")
    ax[0].text(0.22, 30, r"linear amplification" "\n" r"$\psi_0/(1-U^2/U_{div}^2)$", fontsize=6.3,
               color=INK2)
    ax[0].annotate("", xy=(0.93, 12), xytext=(0.55, 29), arrowprops=dict(arrowstyle="->", color=INK2, lw=0.6))
    ax[0].text(1.22, 30, r"$\cos\psi^*=(U_{div}/U)^2$", fontsize=6.5, color=C1)
    ax[1].annotate("fold (jump)", xy=(0.865, 10.3), xytext=(0.35, 40), fontsize=6.5, color=C2,
                   arrowprops=dict(arrowstyle="->", color=C2, lw=0.7))
    handles = [Line2D([], [], color=C1, marker="o", ls="none", ms=4, label=r"no bias ($M_r$ = 0), stable"),
               Line2D([], [], color=C1, marker="o", ls="none", ms=4, alpha=0.35, label="no bias, unstable"),
               Line2D([], [], color=C2, marker="s", ls="none", ms=4, label=r"with rope torque, stable"),
               Line2D([], [], color=C2, marker="s", ls="none", ms=4, alpha=0.35, label="with rope torque, unstable"),
               Line2D([], [], color=INK2, ls=":", lw=0.8, label="mating tolerance (1 deg)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
    save(fig, "Fig5_equilibria")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 6
def fig7():
    v = R("s4_vessel")
    d5 = R("s5_dynamics")
    w = np.array(v["omega"])
    rao = np.array(v["rao_abs"])
    tip = np.array(v["tip_abs"])
    hd = [round(h) for h in v["headings_deg"]]
    fig, ax = plt.subplots(1, 4, figsize=(cm.W2, 2.15))
    j = hd.index(150)
    S = np.array(v["S_wave"])
    for a in ax[:3]:
        a.fill_between(w, 0, S / S.max(), color=GRID, alpha=0.7, lw=0, transform=a.get_xaxis_transform())
    a = ax[0]
    for k_, hh in enumerate((180.0, 150.0, 120.0)):
        jj = hd.index(round(hh))
        a.plot(w, rao[:, jj, 2], color=cm.SERIES[k_], ls=cm.LS[k_], lw=1.2, label=f"{hh:.0f} deg")
    a.set_ylabel("heave RAO (m/m)")
    a.legend(title="wave heading", fontsize=6, title_fontsize=6, loc="upper right")
    panel_label(a, "a")
    a = ax[1]
    for k_, (idx, nm) in enumerate(((3, "roll"), (4, "pitch"), (5, "yaw"))):
        a.plot(w, np.degrees(rao[:, j, idx]), color=cm.SERIES[k_], ls=cm.LS[k_], lw=1.2, label=nm)
    a.set_ylabel("rotation RAO (deg/m)")
    a.legend(fontsize=6, loc="upper right", title="150 deg", title_fontsize=6)
    panel_label(a, "b")
    a = ax[2]
    for k_, hh in enumerate((180.0, 150.0, 120.0)):
        jj = hd.index(round(hh))
        a.plot(w, np.hypot(tip[:, jj, 0], tip[:, jj, 1]), color=cm.SERIES[k_], ls=cm.LS[k_], lw=1.2,
               label=f"{hh:.0f} deg")
    a.set_ylabel("boom-tip horizontal RAO (m/m)")
    panel_label(a, "c")
    for a in ax[:3]:
        a.set_xlim(0.15, 2.0)
        a.set_xlabel(r"$\omega$ (rad/s)")
        a.set_ylim(bottom=0)
    # (d) transfer functions at mating
    a = ax[3]
    tf = d5["transfer"]
    f = np.array(tf["f"])
    a.axvspan(1 / 12, 1 / 5, color=GRID, alpha=0.7, lw=0)
    a.loglog(f, tf["mating|munk|U=12.0|vessel"], color=C1, lw=1.3, label=r"$\psi/\psi_v$, Munk")
    a.loglog(f, tf["mating|box|U=12.0|vessel"], color=C2, lw=1.2, ls="--", label=r"$\psi/\psi_v$, box")
    a.loglog(f, tf["mating|munk|U=12.0|wind"], color=C3, lw=1.2, ls="-.", label=r"$\psi/\theta_w$, Munk")
    a.set_xlim(1e-3, 0.5); a.set_ylim(1e-3, 1e2)
    a.set_xlabel("frequency (Hz)")
    a.set_ylabel("gain (-)")
    a.legend(fontsize=5.8, loc="lower left")
    a.text(0.115, 30, "waves", fontsize=6.3, color=INK2, ha="center")
    panel_label(a, "d")
    fig.tight_layout(w_pad=0.8)
    save(fig, "Fig7_vessel")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 7
def fig6():
    d5 = R("s5_dynamics")
    resp = d5["response"]
    fig, ax = plt.subplots(1, 3, figsize=(cm.W2, 2.45))
    for a, mname, lab, title in ((ax[0], "munk", "a", "potential flow (Munk)"),
                                 (ax[1], "box", "b", "sharp-edged box")):
        for k_, zeta in enumerate(cm.zeta_values):
            r = resp[f"mating|{mname}|TI=0.06|zeta={zeta}"]
            c = np.array(r["curve"])
            a.plot(c[:, 0], c[:, 1], color=cm.SERIES[k_], ls=cm.LS[k_], lw=1.3,
                   label=rf"$\zeta_s$ = {zeta}")
            for p in r["points"]:
                a.errorbar(p["u"], p["std"][0], yerr=p["std"][1], color=cm.SERIES[k_], marker=cm.MK[k_],
                           ms=3.8, mfc="white", mew=0.9, capsize=1.8, lw=0.8, ls="none")
            if k_ == 1:
                a.plot(c[:, 0], c[:, 2], color=INK2, lw=0.9, ls=":", label="quasi-static part")
        r2 = resp[f"long|{mname}|TI=0.06|zeta=0.01"]
        for p in r2["points"]:
            a.plot(p["u"], p["std"][0], marker="x", color=INK, ms=4, ls="none", mew=0.9)
        a.axhline(1.0, color=C2 if mname == "munk" else C1, lw=0.8, ls=":")
        a.set_yscale("log")
        a.set_xlim(0, 1.0); a.set_ylim(0.02, 60)
        a.set_xlabel(r"$U/U_{div}$")
        a.set_title(title, fontsize=7.5)
        panel_label(a, lab)
    ax[0].set_ylabel(r"yaw standard deviation $\sigma_\psi$ (deg)")
    ax[0].legend(loc="lower right", fontsize=5.8)
    hh = [Line2D([], [], marker="o", color=INK2, mfc="white", ls="none", label=rf"time domain, $L$ = {cm.L_mating:.0f} m"),
          Line2D([], [], marker="x", color=INK, ls="none", label=r"time domain, $L$ = 100 m")]
    ax[1].legend(handles=hh, loc="lower right", fontsize=5.8)
    ax[1].annotate("jump to post-critical\nbranch (gust-triggered)", xy=(0.85, 30), xytext=(0.22, 12),
                   fontsize=6.0, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.6))
    # (c) physical wind at mating
    a = ax[2]
    for k_, (mname, nm) in enumerate((("munk", "Munk"), ("box", "box"))):
        for TI, ls in ((0.06, "-"), (0.12, "--")):
            r = resp[f"mating|{mname}|TI={TI}|zeta=0.01"]
            c = np.array(r["curve"])
            U = c[:, 0] * r["U_div"]
            a.plot(U, c[:, 1], color=cm.SERIES[k_], ls=ls, lw=1.3, label=f"{nm}, TI = {TI:.2f}")
    a.axhline(1.0, color=INK2, lw=0.8, ls=":")
    a.text(0.3, 0.85, "mating tolerance", fontsize=6.3, color=INK2, va="top")
    a.set_yscale("log")
    a.set_xlim(0, 20); a.set_ylim(0.02, 60)
    a.set_xlabel(r"mean wind at hub height $U$ (m/s)")
    a.set_title(rf"mating ($L$ = {cm.L_mating:.0f} m), $\zeta_s$ = 0.01", fontsize=7.5)
    a.legend(loc="upper left", fontsize=5.8)
    panel_label(a, "c")
    fig.tight_layout(w_pad=1.0)
    save(fig, "Fig6_turbulent_response")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 8
def fig8():
    s6 = R("s6_control")
    th = s6["threshold"]
    kp = np.array(th["kp"])
    fig, ax = plt.subplots(1, 2, figsize=(cm.W2 * 0.78, 2.75))
    a = ax[0]
    for k_, L in enumerate((int(cm.L_mating), 100)):
        a.plot(kp[1:], np.array(th[f"L={L}|closed_form"])[1:], color=cm.SERIES[k_], lw=1.4,
               label=rf"closed form, $L$ = {L} m")
        for j, (kd, mk) in enumerate(((0.0, "o"), (6e7, "s"))):
            vals = np.array(th[f"L={L}|kd={kd:g}"])[1::3]
            a.plot(kp[1::3], vals, ls="none", marker=mk, ms=3.6, mfc="white", mec=cm.SERIES[k_], mew=0.8,
                   label=(r"eigenvalues, $k_d$ = 0" if kd == 0 else r"eigenvalues, $k_d$ = 6$\times$10$^7$") if k_ == 0 else None)
    a.set_xscale("log")
    a.set_xlim(1e4, 1e7); a.set_ylim(0, 100)
    a.set_xlabel(r"proportional gain $k_p$ (N m/rad)")
    a.set_ylabel(r"closed-loop $U_{div,cl}$ (m/s)")
    a.legend(fontsize=5.8, loc="upper left")
    panel_label(a, "a")
    b = ax[1]
    td = s6["time_domain"]
    Us = [8.0, 12.0, 16.0, 20.0]
    for k_, (mname, nm) in enumerate((("munk", "Munk"), ("box", "box"))):
        for ctrl, ls, mk in ((False, "--", "o"), (True, "-", "s")):
            y = [td[f"{mname}|U={U}|ctrl={ctrl}"]["std"][0] for U in Us]
            e = [td[f"{mname}|U={U}|ctrl={ctrl}"]["std"][1] for U in Us]
            b.errorbar(Us, y, yerr=e, color=cm.SERIES[k_], ls=ls, marker=mk, ms=3.6, lw=1.1, capsize=1.8,
                       mfc="white" if not ctrl else cm.SERIES[k_],
                       label=f"{nm}, {'closed' if ctrl else 'open'} loop")
    b.axhline(1.0, color=INK2, lw=0.8, ls=":")
    b.set_yscale("log")
    b.set_xlim(6, 22); b.set_ylim(0.005, 40)
    b.set_xlabel(r"mean wind $U$ (m/s)")
    b.set_ylabel(r"$\sigma_\psi$ (deg)")
    b.legend(fontsize=5.8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.24))
    panel_label(b, "b")
    fig.tight_layout(w_pad=1.2)
    save(fig, "Fig8_control")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 9
def fig9():
    s7 = R("s7_verify_6dof")
    s2 = R("s2_aero_bem")
    fig, ax = plt.subplots(1, 3, figsize=(cm.W2, 2.3), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    a = ax[0]
    t = np.array(s7["turbulent"]["t"])
    a.plot(t, s7["turbulent"]["psi_6dof"], color=C1, lw=1.6, label="6-DOF rigid-body plant")
    a.plot(t, s7["turbulent"]["psi_red"], color=C2, lw=0.9, ls="--", label="reduced yaw model")
    a.set_xlim(0, 1500)
    a.set_xlabel("time (s)")
    a.set_ylabel(r"$\psi$ (deg)")
    a.legend(fontsize=6.0, loc="upper left", ncol=1)
    a.set_ylim(-6, 22)
    panel_label(a, "a", x=-0.1)
    b = ax[1]
    sp = s2["spheroid"]
    nf = [r["nf"] for r in sp]
    b.plot(nf, [r["k1"] for r in sp], marker="o", color=C1, ms=3.8, mfc="white", label=r"$k_1$ (BEM)")
    b.plot(nf, [r["k2"] for r in sp], marker="s", color=C2, ms=3.8, mfc="white", label=r"$k_2$ (BEM)")
    b.axhline(s2["lamb"]["k1"], color=C1, lw=0.8, ls=":")
    b.axhline(s2["lamb"]["k2"], color=C2, lw=0.8, ls=":")
    b.text(330, s2["lamb"]["k2"] - 0.045, "dotted: Lamb (exact)", fontsize=6.0, ha="left", color=INK2)
    b.set_xscale("log")
    b.set_xlabel("panels")
    b.set_ylabel("added-mass coefficient")
    b.set_ylim(0.15, 0.8)
    b.legend(fontsize=6.0, loc="center right")
    panel_label(b, "b")
    c = ax[2]
    bx = s2["box"]
    c.plot([r["nf"] for r in bx], [r["CK"] for r in bx], marker="o", color=C1, ms=3.8, mfc="white",
           label="box (BEM)")
    c.axhline(s2["box_CK_extrap"], color=C1, lw=0.8, ls=":")
    c.text(260, s2["box_CK_extrap"] - 0.006, "extrapolated", fontsize=6.2, color=INK2, va="top")
    c.set_xscale("log")
    c.set_xlabel("panels")
    c.set_ylabel(r"potential-flow $C_K$")
    c.set_ylim(0.48, 0.52)
    panel_label(c, "c")
    fig.tight_layout(w_pad=2.0)
    save(fig, "Fig9_verification")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["2", "3", "4", "5", "6", "7", "8", "9"]
    for w_ in which:
        globals()[f"fig{w_}"]()
        print("fig", w_, "done")
