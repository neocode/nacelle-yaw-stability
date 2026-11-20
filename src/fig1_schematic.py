"""Fig. 1: physical configuration (a) and reeving plan with the symbols of the model (b)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle, FancyArrowPatch, Arc
import common as cm
from common import C1, C2, C3, C4, C5, INK, INK2, GRID, save

cm.setup_style()


def arrow(ax, p0, p1, color=INK, lw=0.8, style="->", ms=7, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, **kw))


def dim(ax, p0, p1, text, off=(0, 0), color=INK2, fs=6.5, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<->", mutation_scale=6, color=color, lw=0.6))
    m = ((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1])
    ax.text(*m, text, fontsize=fs, color=color, ha="center", va="center", **kw)


def panel_a(ax):
    ax.set_xlim(-67, 122); ax.set_ylim(-22, 200)
    ax.set_aspect("equal"); ax.axis("off")
    x = np.linspace(-66, 122, 400)
    ax.fill_between(x, -22, 1.2 * np.sin(x / 4.0), color="#d9e8f7", lw=0)
    ax.plot(x, 1.2 * np.sin(x / 4.0), color=C1, lw=0.6)
    # vessel hull and crane
    ax.add_patch(Polygon([(-62, 9), (-59, -9), (15, -9), (18, 9)], closed=True, fc="#c9c8c3", ec=INK2, lw=0.7))
    ax.text(-22, -7.5, r"crane vessel ($\xi_1 \ldots \xi_6$)", fontsize=6.0, color=INK,
            ha="center", va="bottom")
    ax.add_patch(Rectangle((-19, 9), 7, 14, fc="#b5b4ae", ec=INK2, lw=0.6))
    tip = np.array([62.0, 184.0])
    base = np.array([-15.5, 23.0])
    ax.plot([base[0], tip[0] - 1], [base[1], tip[1] - 2], color=INK2, lw=3.2, solid_capstyle="round")
    ax.plot([base[0], tip[0] - 1], [base[1], tip[1] - 2], color="#e8e7e2", lw=1.6, solid_capstyle="round")
    ax.text(2, 70, "crane boom", rotation=64, fontsize=6.5, color=INK2, ha="center")
    # upper block, falls, lower block
    ax.add_patch(Rectangle((tip[0] - 4, tip[1] - 5), 8, 5, fc=C4, ec=INK, lw=0.6))
    ax.text(tip[0] + 6, tip[1] - 2.5, "upper (boom-tip) block", fontsize=6.5, color=INK, va="center")
    Lm = 20.0   # drawn length (figure not to scale)
    blk = np.array([tip[0], tip[1] - 5 - Lm])
    for dx in np.linspace(-3.2, 3.2, 6):
        ax.plot([tip[0] + dx, blk[0] + dx], [tip[1] - 5, blk[1] + 4], color=C1, lw=0.7)
    ax.text(tip[0] + 11, tip[1] - 15, r"falls (rope parts), $n$ = 12", fontsize=6.5, color=C1, va="center")
    dim(ax, (tip[0] + 5.5, tip[1] - 5), (tip[0] + 5.5, blk[1] + 4), "", color=INK)
    ax.text(tip[0] + 7, tip[1] - 8.0, r"$L$", fontsize=7, color=INK, va="center")
    ax.add_patch(Rectangle((blk[0] - 4, blk[1]), 8, 4, fc=C4, ec=INK, lw=0.6))
    ax.text(blk[0] + 6, blk[1] + 1.5, "lower (hook) block", fontsize=6.5, color=INK, va="center")
    yoke_y = blk[1] - 8
    ax.plot([blk[0], blk[0]], [blk[1], yoke_y], color=INK, lw=0.9)
    ax.add_patch(Rectangle((blk[0] - 9, yoke_y - 1.2), 18, 1.2, fc=INK2, ec=INK, lw=0.5))
    nac_top = yoke_y - 7
    ax.plot([blk[0] - 8, blk[0] - 8], [yoke_y - 1.2, nac_top], color=INK, lw=0.7)
    ax.plot([blk[0] + 8, blk[0] + 8], [yoke_y - 1.2, nac_top], color=INK, lw=0.7)
    ax.text(blk[0] + 11, yoke_y - 0.5, "yoke and slings", fontsize=6.5, color=INK, va="center")
    ax.add_patch(Rectangle((blk[0] - 10, nac_top - 10), 20, 10, fc="#f2c7ae", ec=INK, lw=0.7))
    ax.add_patch(Polygon([(blk[0] + 10, nac_top - 1.5), (blk[0] + 15, nac_top - 5), (blk[0] + 10, nac_top - 8.5)],
                         closed=True, fc="#f2c7ae", ec=INK, lw=0.7))
    ax.plot(blk[0], nac_top - 5, marker="+", color=INK, ms=5)
    ax.text(blk[0] + 17, nac_top - 5, "nacelle (payload)", fontsize=6.5, color=INK, va="center")
    ax.text(blk[0] + 17, nac_top - 14, rf"mating: $L$ ≈ {cm.L_mating:.0f} m, $|\psi|$ ≤ 1°", fontsize=6.3, color=INK,
            va="center")
    # tower
    ax.add_patch(Polygon([(blk[0] - 3, nac_top - 10.5), (blk[0] + 3, nac_top - 10.5), (blk[0] + 5, 0),
                          (blk[0] - 5, 0)], closed=True, fc="#eeeeec", ec=INK2, lw=0.6))
    ax.text(blk[0] + 8, 70, "wind-turbine tower\n(fixed foundation)", fontsize=6.5, color=INK2, va="center")
    # tugger line from the boom to the rear of the nacelle
    f = 0.80
    bp = base + f * (tip - np.array([1, 2]) - base)
    ax.plot([bp[0], blk[0] - 9.5], [bp[1], nac_top - 4], color=C2, lw=0.9, ls=(0, (4, 2)))
    ax.text(bp[0] - 3, bp[1] + 3.5, "tugger line", fontsize=6.5, color=C2, ha="right")
    # payload at lift-off
    ax.add_patch(Rectangle((-50, 9), 20, 10, fc="none", ec=INK2, lw=0.6, ls="--"))
    ax.text(-40, 21.5, "lift-off\n($L$ ≈ 150 m)", fontsize=6.0, color=INK2, ha="center", va="bottom")
    # vessel motions
    arrow(ax, (6, 11), (6, 21), color=INK2, lw=0.6)
    arrow(ax, (1, 11), (11, 11), color=INK2, lw=0.6)
    # wind profile
    zz = np.linspace(8, 175, 30)
    uu = 12 * (zz / 150.0) ** 0.14
    ax.plot(-65 + uu, zz, color=C3, lw=1.0)
    for zi in (45, 95, 145):
        ui = 12 * (zi / 150.0) ** 0.14
        arrow(ax, (-65, zi), (-65 + ui, zi), color=C3, lw=0.8)
    ax.text(-65, 181, r"wind $U(z)+u(t)$, $v(t)$", fontsize=6.5, color=C3)
    ax.text(105, -15, "MSL", fontsize=6.0, color=C1)


def panel_b(ax):
    ax.set_xlim(-2.7, 2.9); ax.set_ylim(-2.75, 2.75)
    ax.set_aspect("equal"); ax.axis("off")
    D, p, ns = cm.D_sheave, cm.p_sheave * 2.2, 6
    y = (np.arange(ns) - 2.5) * p
    pts = np.array([(sx * D / 2, yy) for yy in y for sx in (1, -1)])
    psi = np.radians(22)
    R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
    box = np.array([[-1.0, -2.0], [1.0, -2.0], [1.0, 2.0], [-1.0, 2.0]]) * 1.02
    nb = (R @ box.T).T
    ax.add_patch(Polygon(nb, closed=True, fc="#fbe7db", ec=INK2, lw=0.6, zorder=0))
    ax.plot([0, 0], [-2.35, 2.35], color=INK2, lw=0.6, ls=":")
    ax.plot(*(R @ np.array([[0, 0], [-2.35, 2.35]])), color=INK, lw=0.6, ls=":")
    for yy in y:
        c = R @ np.array([0, yy]); e = R @ np.array([D / 2, 0])
        ax.plot([c[0] - e[0], c[0] + e[0]], [c[1] - e[1], c[1] + e[1]], color=C4, lw=2.0, alpha=0.6)
    rb = (R @ pts.T).T
    for a_, b_ in zip(pts, rb):
        ax.plot([a_[0], b_[0]], [a_[1], b_[1]], color=C2, lw=0.6)
    ax.plot(pts[:, 0], pts[:, 1], ls="none", marker="o", mfc="white", mec=C1, ms=4.0, mew=0.8)
    ax.plot(rb[:, 0], rb[:, 1], ls="none", marker="o", color=C1, ms=3.6)
    i = 10
    ax.plot([0, pts[i, 0]], [0, pts[i, 1]], color=INK, lw=0.8)
    ax.plot([0, rb[i, 0]], [0, rb[i, 1]], color=INK, lw=0.8)
    ax.text(pts[i, 0] + 0.12, pts[i, 1] - 0.05, r"$\mathbf{a}_i$", fontsize=7.5, color=INK, va="top")
    ax.text(rb[i, 0] - 0.12, rb[i, 1] + 0.12, r"$\mathbf{R}\mathbf{b}_i$", fontsize=7.5, color=INK, ha="right")
    ax.add_patch(Arc((0, 0), 3.3, 3.3, theta1=90, theta2=90 + np.degrees(psi), color=INK, lw=0.8))
    ax.text(-0.42, 1.75, r"$\psi$", fontsize=8, color=INK)
    ax.text(-0.12, 2.47, "fixed axle (upper block)", fontsize=5.8, color=INK2, ha="right", va="bottom")
    dim(ax, (-D / 2, -2.55), (D / 2, -2.55), "", fs=6.3)
    ax.text(0, -2.68, r"$D$", fontsize=6.8, color=INK2, ha="center", va="top")
    arrow(ax, (-2.55, -2.55), (-2.0, -2.55), color=INK, lw=0.7)
    arrow(ax, (-2.55, -2.55), (-2.55, -2.0), color=INK, lw=0.7)
    ax.text(-1.97, -2.62, r"$x$", fontsize=7); ax.text(-2.63, -1.93, r"$y$", fontsize=7)
    th = np.radians(-30)
    w0 = np.array([2.7, -0.2]); w1 = w0 + 0.8 * np.array([-np.cos(th), -np.sin(th)])
    arrow(ax, tuple(w0), tuple(w1), color=C3, lw=1.2, ms=9)
    ax.text(2.35, -0.55, "mean wind\n" r"(direction $\theta$;" "\n" r"incidence $\beta=\psi-\theta$)",
            fontsize=5.8, color=C3, ha="center", va="top")
    ax.plot([-2.55], [2.3], marker="o", mfc="white", mec=C1, ms=4, ls="none")
    ax.text(-2.42, 2.3, r"upper fall ends $\mathbf{a}_i$", fontsize=6.0, va="center")
    ax.plot([-2.55], [2.08], marker="o", color=C1, ms=3.6, ls="none")
    ax.text(-2.42, 2.08, r"lower fall ends $\mathbf{R}\mathbf{b}_i$", fontsize=6.0, va="center")
    ax.plot([-2.6, -2.48], [1.86, 1.86], color=C2, lw=0.8)
    ax.text(-2.42, 1.86, "fall offset (chord)", fontsize=6.0, va="center")
    ax.text(1.25, 2.3, "plan view; payload outline\nshaded; sheave pitch\nexaggerated for clarity",
            fontsize=5.8, color=INK2, va="top")


def main():
    fig = plt.figure(figsize=(cm.W2, 3.55))
    ax1 = fig.add_axes([0.0, 0.0, 0.56, 1.0])
    ax2 = fig.add_axes([0.575, 0.04, 0.42, 0.92])
    panel_a(ax1)
    panel_b(ax2)
    ax1.text(0.01, 0.985, "(a)", transform=ax1.transAxes, fontsize=8, fontweight="bold", va="top")
    ax2.text(0.0, 1.03, "(b)", transform=ax2.transAxes, fontsize=8, fontweight="bold", va="top")
    save(fig, "Fig1_configuration")
    print("ok")


if __name__ == "__main__":
    main()
