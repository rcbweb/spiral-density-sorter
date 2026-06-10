"""verify_numbers.py - recomputes EVERY number quoted in the paper from
make_figures.py's own constants and functions. Deterministic (seed 42).
Run:  python verify_numbers.py"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figures as mf

um = 1e6; pN = 1e12
print("="*72)
print("A. DIMENSIONLESS GROUPS / OPERATING POINT")
Dh = mf.D_H
Re = mf.RE_C
print(f"D_h        = {Dh*um:.2f} um")
print(f"Re_c       = {Re:.2f}")
print(f"a/D_h      = {mf.A_P/Dh:.4f}")
print(f"Re_p       = {Re*(mf.A_P/Dh)**2:.4f}")
De_out = Re*np.sqrt(Dh/(2*mf.R_OUT)); De_in = Re*np.sqrt(Dh/(2*mf.R_IN))
De_mid = Re*np.sqrt(Dh/(2*mf.R_C))
print(f"De_out     = {De_out:.2f}   De_in = {De_in:.2f}   De(Rc=5.25mm) = {De_mid:.2f}")
print(f"L_path     = {mf.L_PATH*1e3:.2f} mm   tau = {mf.L_PATH/mf.U_MEAN:.4f} s")
Q = mf.U_MEAN*mf.W*mf.H
print(f"Q          = {Q*1e6*60:.3f} mL/min")

print("="*72)
print("B. FORCES AT DESIGN POINT (Rc = 5.25 mm)")
FLscale = mf.RHO_F*mf.U_MEAN**2*mf.A_P**4/Dh**2
print(f"F_L scale (f_L=1)   = {FLscale*pN:.2f} pN ;  f_L=0.5 -> {0.5*FLscale*pN:.2f} pN")
ylin = np.linspace(-mf.H/2*0.97, mf.H/2*0.97, 200001)
fL = mf._lift_coeff(ylin/(mf.H/2))
bulk = np.abs(ylin) < 0.60*mf.H/2
print(f"|f_L| max in bulk (|y|<60um) = {np.abs(fL[bulk]).max():.3f} -> "
      f"{np.abs(fL[bulk]).max()*FLscale*pN:.1f} pN")
FB_P = mf.F_buoyancy(mf.RHO_PTFE, mf.RHO_F, mf.V_P, mf.U_MEAN, mf.R_C)
FB_V = mf.F_buoyancy(mf.RHO_PVDF, mf.RHO_F, mf.V_P, mf.U_MEAN, mf.R_C)
dFB = FB_P-FB_V
print(f"F_B PTFE = {FB_P*pN:.2f} pN   F_B PVDF = {FB_V*pN:.2f} pN   dF_B = {dFB*pN:.3f} pN")
U_De = mf.OOKAWARA_C*De_mid**mf.OOKAWARA_EXP
F_D  = 3*np.pi*mf.ETA*mf.A_P*U_De
print(f"U_De(De={De_mid:.2f}) = {U_De*1e3:.3f} mm/s   F_D = {F_D*pN:.0f} pN = {F_D*1e9:.3f} nN")
print(f"dF_B/F_D = {dFB/F_D*100:.3f}%   dF_B/F_L(f=0.5) = {dFB/(0.5*FLscale):.2f}   "
      f"dF_B/F_L(bulk max) = {dFB/(np.abs(fL[bulk]).max()*FLscale):.2f}")
print(f"F_B,PTFE/F_B,PVDF = {FB_P/FB_V:.3f}")

print("="*72)
print("C. EQUILIBRIA, STIFFNESS, RELAXATION (net F = F_L + F_B)")
drag = 6*np.pi*mf.ETA*(mf.A_P/2)
kT = mf.K_B*mf.T_KELV
for name, rho in [("PTFE", mf.RHO_PTFE), ("PVDF", mf.RHO_PVDF)]:
    F = np.array([mf.net_force(y, mf.A_P, mf.H, Dh, mf.RHO_F, rho, mf.V_P,
                               mf.U_MEAN, mf.ETA, Re, mf.R_C) for y in ylin])
    eqs = mf._stable_zeros(ylin*um, F)
    for e in eqs:
        i = np.argmin(np.abs(ylin*um - e))
        k = -(F[i+1]-F[i-1])/(ylin[i+1]-ylin[i-1])   # N/m
        tau_r = drag/k; sig_th = np.sqrt(kT/k)
        print(f"  {name}: y* = {e:+7.2f} um  k = {k:.2e} N/m  "
              f"tau_relax = {tau_r:.3f} s  sigma_th = {sig_th*um:.3f} um")
# gaps
FP = np.array([mf.net_force(y, mf.A_P, mf.H, Dh, mf.RHO_F, mf.RHO_PTFE, mf.V_P,
                            mf.U_MEAN, mf.ETA, Re, mf.R_C) for y in ylin])
FV = np.array([mf.net_force(y, mf.A_P, mf.H, Dh, mf.RHO_F, mf.RHO_PVDF, mf.V_P,
                            mf.U_MEAN, mf.ETA, Re, mf.R_C) for y in ylin])
eqP = mf._stable_zeros(ylin*um, FP); eqV = mf._stable_zeros(ylin*um, FV)
inP, outP = min(eqP), max(eqP); inV, outV = min(eqV), max(eqV)
print(f"  inner gap = {abs(inP-inV):.2f} um   outer gap = {abs(outP-outV):.2f} um")

print("="*72)
print("D. FOCUSED-INLET ENSEMBLES (n=5000/species, seed 42, y0 in [-85,-15] um)")
yP = mf.run_ensemble(mf.RHO_PTFE, n=5000)*um
yV = mf.run_ensemble(mf.RHO_PVDF, n=5000)*um
bif = 0.5*(yP.mean()+yV.mean())
print(f"PTFE exit: mean {yP.mean():+.2f}  std {yP.std():.2f}  median {np.median(yP):+.2f}  "
      f"range [{yP.min():+.1f},{yP.max():+.1f}]")
print(f"PVDF exit: mean {yV.mean():+.2f}  std {yV.std():.2f}  median {np.median(yV):+.2f}  "
      f"range [{yV.min():+.1f},{yV.max():+.1f}]")
print(f"gap = {abs(yP.mean()-yV.mean()):.2f} um   splitter y_s = {bif:+.2f} um")
nP_out = int((yP > bif).sum()); nV_in = int((yV < bif).sum())
nP_in = 5000-nP_out; nV_out = 5000-nV_in
print(f"routing: PTFE->outer {nP_out}/5000 = {nP_out/50:.2f}%   "
      f"PVDF->inner {nV_in}/5000 = {nV_in/50:.2f}%")
purO = 100*nP_out/(nP_out+nV_out); purI = 100*nV_in/(nV_in+nP_in)
print(f"outer outlet purity (PTFE) = {purO:.2f}%   recovery = {nP_out/5000:.4f}")
print(f"inner outlet purity (PVDF) = {purI:.2f}%   recovery = {nV_in/5000:.4f}")
print("-- timestep check (n_steps=8000) --")
yP8 = mf.run_ensemble(mf.RHO_PTFE, n=5000, n_steps=8000)*um
yV8 = mf.run_ensemble(mf.RHO_PVDF, n=5000, n_steps=8000)*um
print(f"PTFE mean {yP8.mean():+.2f} (shift {abs(yP8.mean()-yP.mean()):.3f} um)   "
      f"PVDF mean {yV8.mean():+.2f} (shift {abs(yV8.mean()-yV.mean()):.3f} um)")

print("="*72)
print("E. UNIFORM-INLET ENSEMBLES (n=5000/species, y0 in [-85,+85] um), same splitter")
lo, hi = -mf.H/2*0.85, mf.H/2*0.85
uP = mf.run_ensemble(mf.RHO_PTFE, n=5000, y0_lo=lo, y0_hi=hi)*um
uV = mf.run_ensemble(mf.RHO_PVDF, n=5000, y0_lo=lo, y0_hi=hi)*um
for nm, y in [("PTFE", uP), ("PVDF", uV)]:
    fin = (y < 0).mean()
    print(f"{nm}: inner-branch fraction (y<0) = {fin*100:.1f}%  "
          f"inner mean {y[y<0].mean():+.2f}  outer mean {y[y>0].mean():+.2f}")
nP_out = int((uP > bif).sum()); nV_in = int((uV < bif).sum())
nP_in = 5000-nP_out; nV_out = 5000-nV_in
purO = 100*nP_out/(nP_out+nV_out); purI = 100*nV_in/(nV_in+nP_in)
print(f"splitter at {bif:+.2f} um:")
print(f"  outer outlet purity (PTFE) = {purO:.2f}%   PTFE recovery = {nP_out/5000:.4f}")
print(f"  inner outlet purity (PVDF) = {purI:.2f}%   PVDF recovery = {nV_in/5000:.4f}")

print("="*72)
print("F. BHAGAT (2008) VALIDATION")
B_Dh = mf.B_DH
print(f"B: D_h = {B_Dh*um:.2f} um   Re_c = {mf.B_REC:.1f}   "
      f"a/Dh(7.32) = {mf.A_LARGE/B_Dh:.4f}   a/Dh(1.9) = {1.9e-6/B_Dh:.4f}")
ysc = np.linspace(-mf.B_H/2*0.97, mf.B_H/2*0.97, 400001)
Fb = np.array([mf.F_lift(y, mf.B_H, mf.A_LARGE, mf.B_U, B_Dh, mf.B_RHOF) for y in ysc])
ywall = (ysc+mf.B_H/2)*um
cr = mf._stable_zeros(ywall, Fb)
ypred = min([c for c in cr if c > 0])
print(f"y* = {ypred:.3f} um from wall;  band midpoint 12.5 um")
print(f"deviation = {abs(ypred-12.5):.2f} um = {abs(ypred-12.5)/(mf.B_W*um)*100:.2f}% of width "
      f"(= {abs(ypred-12.5)/(mf.B_H*um)*100:.2f}% of height)")

print("="*72)
print("G. PRESSURE DROP")
alpha = mf.H/mf.W
C_AR = 96*(1-1.3553*alpha+1.9467*alpha**2-1.7012*alpha**3+0.9564*alpha**4-0.2537*alpha**5)
print(f"Shah-London C(alpha={alpha}) = {C_AR:.2f}; square C = 56.91")
def dp(C):
    f = C/Re
    return f*(mf.L_PATH/Dh)*mf.RHO_F*mf.U_MEAN**2/2
print(f"dP square = {dp(56.91)/1e3:.2f} kPa   dP AR = {dp(C_AR)/1e3:.2f} kPa")
def ito(De): return 21.5*De/(1.56+np.log10(De))**5.73
print(f"Ito ratio: De_out {ito(De_out):.4f}  De_mid {ito(De_mid):.4f}  De_in {ito(De_in):.4f}")
print(f"dP Ito-corrected (conservative, De_in factor) = {dp(C_AR)*ito(De_in)/1e3:.2f} kPa")
print(f"dP at 2U (laminar, ~linear in U)              = {2*dp(C_AR)*ito(De_in)/1e3:.1f} kPa")
print(f"Re at 2U = {2*Re:.0f}   De range at 2U = {2*De_out:.0f}-{2*De_in:.0f}")

print("="*72)
print("H. BROWNIAN FLOOR (a = 8 um)")
D = kT/drag
sig = np.sqrt(2*D*mf.L_PATH/mf.U_MEAN)
v = dFB/drag; drift = v*mf.L_PATH/mf.U_MEAN
print(f"D = {D:.3e} m2/s   sqrt(2 D tau) = {sig*um:.3f} um")
print(f"v_drift = {v*um:.1f} um/s   drift = {drift*um:.1f} um   ratio = {drift/sig:.0f}")
print(f"a_min = 8 um * ratio^(-0.4) = {8*(drift/sig)**-0.4:.2f} um")
