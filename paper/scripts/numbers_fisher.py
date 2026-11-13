"""Generate results/numbers_fisher.tex from the Fisher-theory checks (analyses/fisher/out/*.json). Run from paper/. Do not edit the .tex by hand."""
import json, os, numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analyses", "fisher", "out")
J = lambda f: json.load(open(f"{D}/{f}"))
c2, c4, c7, c8, c9 = J("c2.json"), J("c4.json"), J("c7.json"), J("c8.json"), J("c9.json")
c13 = J("c13.json")
c2b = J("c2b_lower.json")
R = {}
# C2: local radius at the vacuum = largest eps at which both detections are within 2% of the leading term (c_local_vac)
cv = c2["c_local_vac"]
ok = [rq[0] for rq, rw in zip(cv["het"], cv["hom"]) if abs(rq[1] - 1) < 0.02 and abs(rw[1] - 1) < 0.02]
R["loc_vac_eps"] = f"10^{{{int(round(np.log10(max(ok))))}}}"
R["loc_nb1_ratio"] = f"{c2['c_local']['1.0']['het'][1][1]:.2f}"
# C2e/C2d: distant states, ratio xi_W/xi_Q for even vs odd cats (|alpha| = 1..2.5) and for orthogonally squeezed states
ec = c2["e_cats"]
R["cat_ratio_a1"], R["cat_ratio_a25"] = f"{ec['1.0']['ratio']:.1f}", f"{ec['2.5']['ratio']:.0f}"
pref = [v["ratio_over_e_a2"] for v in ec.values()]
R["cat_pref"] = f"{min(pref):.2f}\\mbox{{--}}{max(pref):.2f}"
xp = c2["d_squeezed_random"]          # random recorded phase, the convention of F_W in the paper
R["sq_ratio_r1"], R["sq_ratio_r2"], R["sq_ratio_r3"] = (f"{xp[r]['ratio']:.2f}" for r in ("1.0", "2.0", "3.0"))
# C2b: heterodyne local advantage unbounded: min F_W/F_Q at nbar=1, d=4..10
R["minWQ_nb1_d10"] = f"{c2b['nb1.0_d10']:.2f}"
# C4: typical efficiencies (medians over random directions), range over nbar in {0.5,1,2,5} and d in {4,6,8}
med = [v[k] for key, v in c4.items() if key.startswith("thermal_") for k in ("etaQ_median", "etaW_median")]
R["eta_typ"] = f"{min(med):.2f}\\mbox{{--}}{max(med):.2f}"
# C9: sharpness of the heterodyne sector bound for the edge direction (a^dag)^3 rho0 + h.c. at nbar=1 (largest support d=30)
r = c9["nb1.0_l3"]["rows"][-1]
R["sharp_het_nb1_l3"], R["sharp_hom_nb1_l3"] = f"{r[1]:.2f}", f"{r[2]:.2f}"
# C7: noise channels in the mode picture, largest relative deviation of direct integrals from Eq. (noisemodes)
def relerr(prefix):
    e = [abs(v[x + "_direct"] - v.get(x + "_formula", v.get(x + "_sectors"))) / abs(v.get(x + "_formula", v.get(x + "_sectors")))
         for k, v in c7.items() if k.startswith(prefix) for x in ("FW", "FQ")]
    return f"10^{{{int(np.ceil(np.log10(max(e))))}}}"
R["noise_err_ld"] = max(relerr("loss"), relerr("deph"), key=lambda s: int(s[4:-1]))
R["noise_err_gauss"] = relerr("gauss")
# C13: thermalization toward rho0, largest relative deviation of the Fisher information from the mode-by-mode decay
R["therm_err"] = f"10^{{{int(np.ceil(np.log10(c13['max_rel_dev'])))}}}"
R["therm_eig"] = f"10^{{{int(np.ceil(np.log10(c13['max_eigen_rel'])))}}}"
# C8: dephasing, base set
e, a = c8["eqn"], c8["ahm"]
R["deph_eqn_clean"], R["deph_eqn_deph"] = f"{e['median_ratio_clean']:.2f}", f"{e['median_ratio_deph']:.2f}"
R["deph_ahm_clean"], R["deph_ahm_deph"] = f"{a['median_ratio_clean']:.2f}", f"{a['median_ratio_deph']:.2f}"
R["deph_eqn_fixed_clean"], R["deph_eqn_fixed_deph"] = f"{e['fixed_pair_median_clean']:.2f}", f"{e['fixed_pair_median_deph']:.2f}"
R["deph_eqn_changed"] = f"{100*e['nearest_family_changed']:.0f}\\%"
R["deph_ahm_changed"] = f"{100*a['nearest_family_changed']:.0f}\\%"
R["deph_rho_eqn"], R["deph_rho_ahm"] = f"{e['sector']['spearman_fhigh_vs_dlogratio']:.2f}", f"{a['sector']['spearman_fhigh_vs_dlogratio']:.2f}"
R["deph_p_eqn"] = f"{e['sector']['p']:.0e}".replace("e-0", "\\times10^{-") + "}"
R["deph_p_ahm"] = f"{a['sector']['p']:.0e}".replace("e-0", "\\times10^{-") + "}"
R["fhigh_eqn"], R["fhigh_ahm"] = f"{e['sector']['median_fhigh']:.2f}", f"{a['sector']['median_fhigh']:.2f}"
# C14: chi^2 divergences from the thermal state (any distance), displacement of any size, mixtures with distant states
c14 = J("c14.json")
def p10(x): return f"10^{{{int(np.ceil(np.log10(x)))}}}"
R["chi2_err"], R["disp_err"] = p10(c14["C_chi2_max_rel"]), p10(c14["B_max_rel"])
R["nmin_het"], R["nmin_hom"], R["nmin_any"] = (str(c14["N_example"][k]) for k in ("Q", "W", "SLD"))
R["path_clear"], R["path_same_half"], R["path_same_one"] = str(c14["C_clear"]), str(c14["C_same_to_0.5"]), str(c14["C_same_to_1"])
assert c14["C_inequalities_ok"] and c14["A"]["upper_ok"] and c14["A"]["weak_ok"] and abs(c14["A"]["lower_c"] - 0.375) < 1e-6
# C15: squeezed thermal reference state (scope of the theory)
c15 = J("c15.json")
for r, tag in (("0.2", "02"), ("0.5", "05")):
    R[f"sqref_Q_r{tag}"], R[f"sqref_W_r{tag}"] = f"{c15[r]['maxeffQ']:.2f}", f"{c15[r]['maxeffW']:.2f}"
    R[f"sqref_off_r{tag}"] = f"{max(c15[r]['offQ'], c15[r]['offW']):.2f}"
# C16, C17: order-n blocks against direct integrals near a squeezed thermal state (r = 0.3); bounds of Theorems 2, 3 and Corollary 2
c16, c17 = J("c16.json"), J("c17.json")
R["block_err"] = p10(max(c16["T1_het_r0.3"][1], c16["T2_hom_r0.3"][1], c16["T3_hom_weighted"][1]))
R["bound_exc"] = p10(max(c17[k] for k in ("A_max_excess", "B_max_excess", "C_max_excess", "E_hom_excess")))
assert c17["D_max_dev"] < 1e-12 and c17["E_het_dev"] < 1e-12 and abs(c17["F_order2_eff"] - 1) < 1e-12 and c17["B_tight_cases"] > 0
out = ["% generated by scripts/numbers_fisher.py -- do not edit by hand"]
out += [f"\\expandafter\\def\\csname res@{k}\\endcsname{{{v}}}" for k, v in R.items()]
open("results/numbers_fisher.tex", "w").write("\n".join(out) + "\n"); print("\n".join(out))
