"""
make_figures.py — regenerates Figures 2-4 of the paper (seed 42).
Figure 1 is fig1_device_3d.html (browser-based).

Usage:
    python make_figures.py [outdir]   (default: current directory)

Outputs:
    fig2_forcebalance.{pdf,png}
    fig3_distributions.{pdf,png}
    fig4_bhagat.{pdf,png}

Physics (paper section in parentheses):
    * inertial-lift profile, Eq. 19 (Sec. 4.1)
    * centrifugal buoyancy F_B = (rho_p - rho_f) V U^2/R, Eq. 11 (Sec. 2.7)
    * overdamped Langevin, Euler-Maruyama, 4000 steps (Sec. 4.1)
    * Dean drag = 0 per streamline-following regime (Sec. 3.5)

Requirements: numpy matplotlib scipy
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# Global style
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          8,
    'axes.labelsize':     9,
    'axes.linewidth':     1.2,  # thickened for print
    'xtick.labelsize':    7.5,
    'ytick.labelsize':    7.5,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'xtick.major.width':  1.2,
    'ytick.major.width':  1.2,
    'lines.linewidth':    1.3,
    'legend.frameon':     False,
    'legend.fontsize':    7,
    'figure.dpi':         300,
    'savefig.dpi':        600,
    'savefig.bbox':       'tight',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'pdf.fonttype':       42,
})

C_PTFE = '#ff7a1a'  # orange, Fig 1 callout D/F
C_PVDF = '#18c84a'  # green, Fig 1 callout E/G
C_TEAL = '#00e8c0'  # teal, Fig 1 callout A
C_REF   = '#bbbbbb'
_WB     = dict(facecolor='white', edgecolor='none', alpha=1.0, pad=2.5)    # fully opaque white bbox

# Physical constants (design point)
RHO_F    = 1025.0
ETA      = 1.07e-3
W        = 500e-6
H        = 200e-6
D_H      = 2*W*H / (W+H)
R_IN     = 3.0e-3
R_OUT    = 7.5e-3
R_C      = 5.25e-3
R_MID    = (R_IN + R_OUT) / 2
N_TURNS  = 5
L_PATH   = 2*np.pi*R_MID*N_TURNS
U_MEAN   = 0.55
RE_C     = RHO_F*U_MEAN*D_H / ETA
A_P      = 8e-6
V_P      = np.pi/6 * A_P**3
RHO_PTFE = 2200.0
RHO_PVDF = 1780.0
K_B      = 1.380649e-23
T_KELV   = 298.15
SEED     = 42
FOCUS_THR = 0.07

# Bhagat (2008)
B_W    = 100e-6
B_H    = 50e-6
B_DH   = 2*B_W*B_H / (B_W+B_H)
B_U    = 0.53
B_RC   = 4.0e-3
B_RHOF = 998.0
B_ETA  = 1.00e-3
B_REC  = B_RHOF*B_U*B_DH / B_ETA
A_LARGE = 7.32e-6
RHO_PS  = 1050.0

OOKAWARA_C   = 1.8e-4
OOKAWARA_EXP = 1.63

# Physics
def _lift_coeff(y_hat, y_eq=0.60, C1=2.0):
    y_hat = np.clip(y_hat, -0.90, 0.90)
    num   = y_eq - np.abs(y_hat)
    den   = (1.0 - np.abs(y_hat))**2 + 0.01
    return C1 * np.sign(y_hat) * num / den


def F_lift(y, h, a, U, D_h, rho_f):
    return _lift_coeff(y / (h/2)) * rho_f * U**2 * a**4 / D_h**2


def F_buoyancy(rho_p, rho_f, Vp, U, R_c):
    return (rho_p - rho_f) * Vp * U**2 / R_c


def net_force(y, a, h, D_h, rho_f, rho_p, Vp, U, eta, Re_c, R_c):
    return F_lift(y, h, a, U, D_h, rho_f) + F_buoyancy(rho_p, rho_f, Vp, U, R_c)


def _spiral_R(s):
    b    = (R_OUT - R_IN) / (2*np.pi*N_TURNS)
    s    = np.clip(s, 0.0, L_PATH)
    disc = np.maximum(R_OUT**2 - 2*b*s, 0.0)
    theta = (R_OUT - np.sqrt(disc)) / b
    return np.clip(R_OUT - b*theta, R_IN, R_OUT)


def _bm_sigma(a, eta, dt):
    drag = 6*np.pi*eta*(a/2)
    return np.sqrt(2*K_B*T_KELV*drag*dt) / drag


def run_ensemble(rho_p, n=600, n_steps=4000,
                 y0_lo=None, y0_hi=None, seed=SEED,
                 h=H, D_h=D_H, rho_f=RHO_F, eta=ETA,
                 Re_c=RE_C, U=U_MEAN, a=A_P, L=L_PATH):
    if y0_lo is None:
        y0_lo = -h/2 * 0.85
    if y0_hi is None:
        y0_hi = -h/2 * 0.15
    rng  = np.random.default_rng(seed)
    y    = rng.uniform(y0_lo, y0_hi, n)
    ds   = L / n_steps
    dt   = ds / U
    sig  = _bm_sigma(a, eta, dt)
    drag = 6*np.pi*eta*(a/2)
    wall = h/2 * 0.90
    Vp   = np.pi/6 * a**3
    R_arr = np.array([_spiral_R(s)
                      for s in np.linspace(0, L, n_steps, endpoint=False)])
    for k in range(n_steps):
        F  = net_force(y, a, h, D_h, rho_f, rho_p, Vp, U, eta, Re_c, R_arr[k])
        y += (F/drag)*dt + sig*rng.standard_normal(n)
        y  = np.clip(y, -wall, wall)
    return y


def _stable_zeros(y_arr, F_arr):
    out = []
    for i in range(len(F_arr)-1):
        if F_arr[i]*F_arr[i+1] < 0:
            y_eq = y_arr[i] - F_arr[i]*(y_arr[i+1]-y_arr[i])/(F_arr[i+1]-F_arr[i])
            if (F_arr[i+1]-F_arr[i])/(y_arr[i+1]-y_arr[i]) < 0:
                out.append(y_eq)
    return out


# Figure 2 — Force balance: F_B shifts inner equilibria by Δy* ≈ 16 µm,
# producing a clear bifurcation without tight inertial focusing.
def fig2_forcebalance(outdir):
    print('[fig2]  Computing force-balance curves at Rc=5.25 mm...')

    y_scan = np.linspace(-H/2*0.97, H/2*0.97, 4000)
    y_um   = y_scan * 1e6

    FB_ptfe = F_buoyancy(RHO_PTFE, RHO_F, V_P, U_MEAN, R_C) * 1e12  # pN
    FB_pvdf = F_buoyancy(RHO_PVDF, RHO_F, V_P, U_MEAN, R_C) * 1e12

    FL_pN   = np.array([F_lift(y, H, A_P, U_MEAN, D_H, RHO_F)
                        for y in y_scan]) * 1e12
    FN_ptfe = FL_pN + FB_ptfe
    FN_pvdf = FL_pN + FB_pvdf

    eq_ptfe = _stable_zeros(y_um, FN_ptfe)
    eq_pvdf = _stable_zeros(y_um, FN_pvdf)
    print(f'[fig2]  PTFE eq (um): {[f"{e:.1f}" for e in eq_ptfe]}')
    print(f'[fig2]  PVDF eq (um): {[f"{e:.1f}" for e in eq_pvdf]}')

    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.9))

    panels = [
        {'xlim': (-76, -18), 'label': '(a)', 'title': 'inner focusing region'},
        {'xlim': ( 18,  82), 'label': '(b)', 'title': 'outer focusing region'},
    ]

    for ax, pan in zip(axes, panels):
        xlo, xhi = pan['xlim']
        mask = (y_um >= xlo) & (y_um <= xhi)

        ax.axhline(0, color=C_REF, lw=0.6, zorder=1)
        ax.plot(y_um[mask], FN_ptfe[mask], color=C_PTFE, lw=1.5, zorder=3)
        ax.plot(y_um[mask], FN_pvdf[mask], color=C_PVDF, lw=1.5, ls='--', zorder=3)

        # Equilibrium markers at F = 0
        for eq in eq_ptfe:
            if xlo <= eq <= xhi:
                ax.plot(eq, 0, 'o', color=C_PTFE, ms=5.5, zorder=6)
        for eq in eq_pvdf:
            if xlo <= eq <= xhi:
                ax.plot(eq, 0, 's', color=C_PVDF, ms=5, mfc='white', mew=1.4, zorder=6)

        # Δy* annotation in inner panel only
        if xlo < 0:
            inner_ptfe = [e for e in eq_ptfe if xlo <= e <= xhi]
            inner_pvdf = [e for e in eq_pvdf if xlo <= e <= xhi]
            if inner_ptfe and inner_pvdf:
                ep = max(inner_ptfe)
                ev = min(inner_pvdf)
                dy = abs(ep - ev)
                f_range = max(FN_ptfe[mask].max(), FN_pvdf[mask].max())
                mid = (ep + ev) / 2
                ax.annotate('', xy=(ep, 0), xytext=(ev, 0),
                            arrowprops=dict(arrowstyle='|-|',
                                            color='#444444', lw=1.0,
                                            mutation_scale=8))
                ax.text(mid, f_range * 0.08,
                        r'$\Delta y^*$' + f' = {dy:.0f} µm',
                        ha='center', va='bottom', fontsize=7, color='#333333',
                        bbox=_WB, zorder=7)

        # Species labels at max curve separation
        y_gap   = FN_ptfe[mask] - FN_pvdf[mask]
        idx_max = np.argmax(y_gap[y_gap > 0]) if (y_gap > 0).any() else 0
        y_lbl   = y_um[mask][idx_max]
        # Clamp to a reasonable range so label doesn't fall off the edge
        y_lbl   = np.clip(y_lbl, xlo + 0.15*(xhi-xlo), xhi - 0.15*(xhi-xlo))

        fl_at = np.interp(y_lbl, y_um[mask], FN_ptfe[mask])
        vl_at = np.interp(y_lbl, y_um[mask], FN_pvdf[mask])

        ax.text(y_lbl, fl_at, 'PTFE', ha='center', va='bottom', fontsize=7.5,
                color=C_PTFE, fontweight='bold', bbox=_WB, zorder=8)
        ax.text(y_lbl, vl_at, 'PVDF', ha='center', va='top', fontsize=7.5,
                color=C_PVDF, fontweight='bold', bbox=_WB, zorder=8)

        ax.set_xlabel('Lateral position  $y$  (µm)', labelpad=6)
        ax.set_xlim(xlo, xhi)
        ax.set_title(f'{pan["label"]}  {pan["title"]}',
                     fontsize=8, pad=6, loc='left')

    axes[0].set_ylabel('Net lateral force  (pN)', labelpad=6)
    axes[1].set_ylabel('Net lateral force  (pN)', labelpad=6)

    fig.tight_layout(w_pad=1.4)
    _save(fig, outdir, 'fig2_forcebalance')
    print('[fig2]  Done.')


# Figure 3 — Exit distributions: sheath-flow inlet achieves 100%/~99% purity,
# 13 µm inter-species gap. KDE curves only.
def fig3_distributions(outdir):
    print('[fig3]  Running focused-inlet ensembles (n=5000 per species)...')

    y_ptfe = run_ensemble(RHO_PTFE, n=5000) * 1e6
    y_pvdf = run_ensemble(RHO_PVDF, n=5000) * 1e6

    y_bif    = 0.5 * (y_ptfe.mean() + y_pvdf.mean())
    gap      = abs(y_ptfe.mean() - y_pvdf.mean())
    pur_ptfe = 100 * np.mean(y_ptfe > y_bif)
    pur_pvdf = 100 * np.mean(y_pvdf < y_bif)

    print(f'[fig3]  PTFE mean={y_ptfe.mean():.1f} um   PVDF mean={y_pvdf.mean():.1f} um')
    print(f'[fig3]  Gap={gap:.1f} um   Bif={y_bif:.1f} um')
    print(f'[fig3]  Purity: PTFE={pur_ptfe:.1f}%  PVDF={pur_pvdf:.1f}%')

    all_y = np.concatenate([y_ptfe, y_pvdf])
    x_lo  = all_y.min() - 4
    x_hi  = all_y.max() + 4
    x     = np.linspace(x_lo, x_hi, 800)

    kde_ptfe = gaussian_kde(y_ptfe, bw_method='scott')(x)
    kde_pvdf = gaussian_kde(y_pvdf, bw_method='scott')(x)
    y_top    = max(kde_ptfe.max(), kde_pvdf.max())

    fig, ax = plt.subplots(figsize=(3.54, 2.90))

    # Outlet shading, behind everything
    ax.axvspan(x_lo, y_bif, alpha=0.07, color=C_PVDF, zorder=0, lw=0)
    ax.axvspan(y_bif, x_hi, alpha=0.07, color=C_PTFE, zorder=0, lw=0)

    # KDE curves
    ax.plot(x, kde_pvdf, color=C_PVDF, lw=1.6, ls='--', zorder=3)
    ax.plot(x, kde_ptfe, color=C_PTFE, lw=1.6, zorder=3)

    # Outlet labels pinned to axes corners
    ax.text(0.02, 1.10, 'PVDF Outlet',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=7.5, fontweight='bold', color=C_PVDF)
    ax.text(0.02, 1.02, f'({pur_pvdf:.0f}% purity)',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=6.5, color=C_PVDF)

    ax.text(0.98, 1.10, 'PTFE Outlet',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7.5, fontweight='bold', color=C_PTFE)
    ax.text(0.98, 1.02, f'({pur_ptfe:.0f}% purity)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=6.5, color=C_PTFE)

    # Gap dimension line
    x_ptfe_pk = x[np.argmax(kde_ptfe)]   
    x_pvdf_pk = x[np.argmax(kde_pvdf)]   
    y_ptfe_pk = kde_ptfe.max()
    y_pvdf_pk = kde_pvdf.max()
    y_line    = y_top * 1.12             

    ax.annotate('', xy=(x_ptfe_pk, y_line), xytext=(x_pvdf_pk, y_line),
                arrowprops=dict(arrowstyle='|-|', color='#444444',
                                lw=1.0, mutation_scale=5))

    ax.plot([x_ptfe_pk, x_ptfe_pk], [y_ptfe_pk, y_line],
            color=C_PTFE, lw=0.9, ls='-', zorder=4)

    ax.plot([x_pvdf_pk, x_pvdf_pk], [y_pvdf_pk, y_line],
            color=C_PVDF, lw=0.9, ls='--', zorder=4)

    ax.text((x_ptfe_pk + x_pvdf_pk) / 2, y_line + y_top * 0.09,
            r'$\Delta y$' + f' = {gap:.0f} µm',
            ha='center', va='bottom', fontsize=7, color='#333333', zorder=7)

    # Legend
    ax.legend(
        handles=[
            Line2D([0], [0], color=C_PTFE, lw=1.5, ls='-',  label='PTFE'),
            Line2D([0], [0], color=C_PVDF, lw=1.5, ls='--', label='PVDF'),
        ],
        loc='upper right',
        fontsize=6.5, handlelength=1.8,
        frameon=True, framealpha=0.9, edgecolor='#dddddd')

    ax.set_xlabel('Cross-channel exit position  $y$  (µm)', labelpad=6)
    ax.set_ylabel('Probability density  (µm$^{-1}$)', labelpad=6)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-y_top * 0.04, y_top * 1.55)

    fig.tight_layout()
    fig.subplots_adjust(top=0.78)   
    _save(fig, outdir, 'fig3_distributions')
    print('[fig3]  Done.')
    

# Figure 4 — Bhagat (2008) validation: model places 7.32 µm PS bead equilibrium
# within 2.5% of channel width of the reported 10-15 µm band.
# Y-axis clipped to show the zero-crossing region clearly.
def fig4_bhagat(outdir):
    print('[fig4]  Computing Bhagat force profile...')

    y_scan  = np.linspace(-B_H/2*0.97, B_H/2*0.97, 4000)
    F_arr   = np.array([F_lift(y, B_H, A_LARGE, B_U, B_DH, B_RHOF)
                        for y in y_scan]) * 1e12   # pN

    y_wall = (y_scan + B_H/2) * 1e6   # µm from inner wall

    crossings = _stable_zeros(y_wall, F_arr)
    pos_cross = [c for c in crossings if c > 0]
    y_pred    = min(pos_cross) if pos_cross else np.nan
    deviation = abs(y_pred - 12.5) / (B_W*1e6) * 100

    print(f'[fig4]  y* = {y_pred:.2f} um   deviation = {deviation:.2f}%')

    mask   = (y_wall >= 0) & (y_wall <= 22)
    F_clip = 550.0   # pN, clips near-wall region to show crossing clearly

    fig, ax = plt.subplots(figsize=(3.54, 2.90))

    ax.set_title('Bhagat (2008) validation, 7.32 µm polystyrene beads',
                 fontsize=7.5, pad=6, loc='left')

    # Zero line
    ax.axhline(0, color=C_REF, lw=0.6, zorder=1)

    # Reported band, subtle grey, behind curve
    ax.axvspan(10, 15, color='#e8e8e8', zorder=1, lw=0,
                label='reported band (10–15 µm)')

    # Force curve, clipped at F_clip for readability
    F_plot = np.clip(F_arr[mask], -200, F_clip)
    ax.plot(y_wall[mask], F_plot, color=C_TEAL, lw=1.5, zorder=3,
            label='model force  ($a$ = 7.32 µm PS)')

    # Predicted equilibrium marker
    ax.plot(y_pred, 0, 'o', color=C_TEAL, ms=6, zorder=5)

    # 1. Top-right text box: y* prediction
    ax.text(1.0, 0.99,
            f'$y^* = {y_pred:.1f}$ µm  (predicted)\nDeviation: {deviation:.1f}% of channel width',
            transform=ax.transAxes, ha='right', va='top', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.40', fc='white',
                      ec='#cccccc', lw=0.6))

    # 2. Legend pinned below text box
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.76), 
              fontsize=6.5, handlelength=1.6,
              frameon=True, framealpha=0.9,
              edgecolor='#cccccc', borderpad=0.6)

    ax.set_xlabel('Distance from inner wall  (µm)', labelpad=6)
    ax.set_ylabel('Net lateral force  (pN)', labelpad=6)
    ax.set_xlim(0, 22)
    ax.set_ylim(-200, F_clip + 20)

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    _save(fig, outdir, 'fig4_bhagat')
    print('[fig4]  Done.')

# Utilities
def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'{name}.{ext}'), dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'        Saved  {name}.pdf  +  {name}.png')

if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    print('=' * 56)
    print('  Paper figures 2-4  (Figure 1 = fig1_device_3d.html)')
    print('=' * 56)
    
    fig2_forcebalance(outdir)
    print()
    fig3_distributions(outdir)
    print()
    fig4_bhagat(outdir)
    print()
    
    print('All done.')