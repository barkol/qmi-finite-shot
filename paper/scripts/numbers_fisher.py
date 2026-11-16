"""Generate results/numbers_fisher.tex from the Fisher-theory checks (analyses/fisher/out/*.json). Run from paper/. Do not edit the .tex by hand."""
import json, os, numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analyses", "fisher", "out")
J = lambda f: json.load(open(f"{D}/{f}"))
c2, c4, c7, c8, c9 = J("c2.json"), J("c4.json"), J("c7.json"), J("c8.json"), J("c9.json")
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
# C16, C17: order-n blocks against direct integrals near a squeezed thermal state (r = 0.3); bounds of Theorems 2, 3 and Corollary 1
c16, c17 = J("c16.json"), J("c17.json")
R["block_err"] = p10(max(c16["T1_het_r0.3"][1], c16["T2_hom_r0.3"][1], c16["T3_hom_weighted"][1]))
R["bound_exc"] = p10(max(c17[k] for k in ("A_max_excess", "B_max_excess", "C_max_excess", "E_hom_excess")))
assert c17["D_max_dev"] < 1e-12 and c17["E_het_dev"] < 1e-12 and abs(c17["F_order2_eff"] - 1) < 1e-12 and c17["B_tight_cases"] > 0
# C18: prediction of the theory of Sec. II for the pairs of the dephasing figure (Sec. III A)
c18 = J("c18.json")
for f in ("eqn", "ahm"):
    R[f"pred_rho_{f}"] = f"{c18[f]['spearman']:.2f}"; R[f"predratio_rho_{f}"] = f"{c18[f]['spearman_ratio']:.2f}"
    R[f"pred_med_{f}"] = f"{c18[f]['median_pred']:+.2f}".replace("-", "-"); R[f"predobs_med_{f}"] = f"{c18[f]['median_obs']:+.2f}"
    assert c18[f]["spearman_p"] < 1e-10
# C19: origin of the offset of the prediction (reference state) and its range of validity
c19 = J("c19.json"); SHIFT_MAX = 0.15
R["range_thr"] = f"{SHIFT_MAX:.2f}"
for f in ("eqn", "ahm"):
    sm = c19["summary"][f]; R[f"ownref_rho_{f}"] = f"{sm['spearman_own']:.2f}"; R[f"ownref_med_{f}"] = f"{sm['median_own']:+.2f}"
    R[f"dQ_xi_{f}"], R[f"dQ_th_{f}"] = f"{sm['median_dln_Qxi']:.2f}", f"{sm['median_dln_QFth']:.2f}"
    R[f"dW_xi_{f}"], R[f"dW_th_{f}"] = f"{sm['median_dln_Wxi']:.2f}", f"{sm['median_dln_WFth']:.2f}"
    pr = c19["pairs"][f]; g = lambda key, i: np.array([x[key][i] for x in pr])
    d = lambda det, i: np.log(g(det + "d", i) / g(det + "c", i)); obs, th = d("W", 0) - d("Q", 0), d("W", 1) - d("Q", 1)
    assert np.abs(th - np.array(c18[f]["pred"])).max() < 0.05 and np.abs(obs - np.array(c18[f]["obs"])).max() < 1e-3      # same pairs as C18 (pixel grid vs fine grid)
    m = np.array([x["shiftQ"] for x in pr]) <= SHIFT_MAX
    from scipy.stats import spearmanr as _sp
    R[f"range_n_{f}"] = str(int(m.sum())); R[f"range_rho_{f}"] = f"{_sp(obs[m], np.array(c18[f]['pred'])[m]).correlation:.2f}"
    R[f"range_off_{f}"] = f"{np.median((obs - np.array(c18[f]['pred']))[m]):+.2f}"; R[f"out_off_{f}"] = f"{np.median((obs - np.array(c18[f]['pred']))[~m]):+.2f}"
# multimode theorems and the two-mode squeezed vacuum (C20: symmetric powers and Gaussian formulas; C21: two-mode Fock basis)
c20, c21 = J("c20.json"), J("c21.json")
assert c20["A_rank_ok"] and c20["B_mult_ok"] and c20["B_order1_gap"] < 1e-12 and c20["F_order1_min_gap"] > 0
R["mm_J_err"] = p10(c20["A_J_dev"]); R["mm_cases"] = str(c20["B_cases"]); R["mm_bound_exc"] = p10(c20["B_max_excess"])
R["tmsv_spec_err"] = p10(max(c20["C_dev"], c20["D_max_dev"], c20["D_form_dev"])); R["tmsv_dir_err"] = p10(c20["E_max_dev"])
_dev = 0.0
for _k, _v in c21.items():
    if _k == "params": continue
    for _a, _b in (("het_direct", "het_pred"), ("hom_uniform_direct", "hom_uniform_pred"), ("hom_setting_direct", "hom_setting_pred"),
                   ("het_direct", "pred_both"), ("hom_uniform_direct", "pred_both"), ("F_SLD", "F_SLD_pred"),
                   ("F_het_direct", "F_het_pred"), ("F_hom_uniform_direct", "F_hom_uniform_pred")):
        if _a in _v and _b in _v and not isinstance(_v[_a], str): _dev = max(_dev, abs(_v[_a] / _v[_b] - 1))
assert abs(c21["params"]["r"] - 0.5) < 1e-12 and len(c21) == 14
R["tmsv_fock_err"] = p10(_dev)
for _tag, _r in (("a", "r0.5"), ("b", "r1.0")):
    _e = c20["E_directions"][_r]
    R[f"tmsv_r_{_tag}"] = f"{_e['r_formula']:.3f}"; R[f"tmsv_ph_{_tag}"] = f"{_e['phase_formula']:.3f}"; R[f"tmsv_kerr_{_tag}"] = f"{_e['kerr_formula']:.3f}"
_f = c20["F_cases"]["r0.5_nb0.1"]
R["tmst_disp_het"] = f"{_f['disp_het']:.3f}"; R["tmst_disp_hom"] = f"{_f['disp_hom']:.3f}"
R["tmst_thr_a"] = f"{c20['F_threshold']['r0.5']:.2f}"; R["tmst_thr_b"] = f"{c20['F_threshold']['r1']:.2f}"; R["tmst_err"] = p10(c20["F_max_dev"])
out = ["% generated by scripts/numbers_fisher.py -- do not edit by hand"]
out += [f"\\expandafter\\def\\csname res@{k}\\endcsname{{{v}}}" for k, v in R.items()]
open("results/numbers_fisher.tex", "w").write("\n".join(out) + "\n"); print("\n".join(out))
