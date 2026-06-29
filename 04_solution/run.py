"""STAGE 04 - SOLUTION
Runs the USMH-1 solver for the calibrated case and writes every engineering
output as CSV (profiles, slug characteristics, transient slugging, hydrate
assessment, flow-regime map, MEG sweep and KPI summary)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import style, common
import usmh_solver as U

cfg = U.CaseConfig()
DS = common.DIRS["solution"]
DD = common.DIRS["data"]

# ---- core march ------------------------------------------------------------
res = U.march(cfg)
hy = U.hydrate_assessment(cfg, res)
sp = U.slug_profile(cfg, res)
n = len(res["s"])

# 1) full spatial profiles ---------------------------------------------------
prof = pd.DataFrame({
    "s_m": res["s"], "elevation_m": res["elev"], "inclination_deg": res["theta"],
    "P_bar": res["P"]/1e5, "T_C": res["T"]-273.15, "Tamb_C": res["Tamb"]-273.15,
    "Teq_hydrate_C": hy["Teq_eff"]-273.15, "subcooling_C": hy["subcool"],
    "hydrate_risk": hy["risk"], "Vsl_ms": res["Vsl"], "Vsg_ms": res["Vsg"],
    "Vm_ms": res["Vm"], "liquid_holdup_HL": res["HL"], "GVF_alpha": res["GVF"],
    "rho_liquid": res["rho_l"], "rho_gas": res["rho_g"], "rho_mix": res["rho_m"],
    "mu_liquid_cP": res["mu_l"]*1e3, "mu_gas_cP": res["mu_g"]*1e3,
    "Reynolds": res["Re"], "friction_factor": res["f"],
    "dPfric_Pa_m": res["dP_fric"], "dPgrav_Pa_m": res["dP_grav"],
    "Rs_scf_stb": res["Rs"], "Bo_rb_stb": res["Bo"], "Zfactor": res["Z"],
    "flow_regime": sp["regime"]})
common.save_table(prof, "profiles_full.csv", "solution", "Full spatial profiles",
                  "Cell-by-cell engineering output along the tieback: pressure, "
                  "temperature, hydrate equilibrium and sub-cooling, holdup, "
                  "velocities, fluid properties and flow regime.")

# 2) slug characteristics ----------------------------------------------------
slug = pd.DataFrame({"s_m": res["s"], "regime": sp["regime"],
                     "slug_freq_Hz": sp["fs"], "transl_vel_ms": sp["Vt"],
                     "slug_holdup_HLs": sp["HLs"], "unit_cell_len_m": sp["Lu"],
                     "slug_len_m": sp["Ls"], "film_len_m": sp["Lf"]})
common.save_table(slug, "slug_characteristics.csv", "solution",
                  "Hydrodynamic slug characteristics",
                  "Slug frequency, translational velocity, slug-body holdup and slug / "
                  "unit-cell lengths along the pipeline where free gas is present.")

# 3) transient terrain-slugging cycle (at the sag) ---------------------------
ss = U.severe_slugging_cycle(cfg, res, t_end=7200.0, dt=1.0)
sst = pd.DataFrame({"time_s": ss["t"], "time_min": ss["t"]/60.0,
                    "Psag_bar": ss["P_base"]/1e5, "liquid_level_m": ss["z"],
                    "trap_holdup": ss["HL_riser"], "Qliq_out_kgs": ss["Wl_out"],
                    "stage": ss["phase"]})
common.save_table(sst[::10], "terrain_slugging_timeseries.csv", "solution",
                  "Terrain-slugging transient (sag)",
                  "Reduced-order terrain/severe-slugging limit cycle at the dominant sag: "
                  "trap pressure, liquid level and the outlet liquid surge versus time.")

# 3b) terrain-slugging susceptibility versus turndown ------------------------
rows = []
for frac in [1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2]:
    c = U.CaseConfig(qo_stb_d=cfg.qo_stb_d*frac)
    r = U.march(c)
    if not (np.all(np.isfinite(r["P"])) and r["P"][-1] > 2e5):
        continue
    sc = U.severe_slugging_number(c, r)
    rows.append({"turndown_pct": 100*frac, "qo_stb_d": cfg.qo_stb_d*frac,
                 "Boe_number_PI": sc["PI_ss"], "susceptible": sc["susceptible"]})
td = pd.DataFrame(rows)
common.save_table(td, "terrain_slug_susceptibility.csv", "solution",
                  "Terrain-slugging susceptibility vs turndown",
                  "Boe terrain-slugging number versus throughput; the line is "
                  "hydrodynamically slugging at design rate and enters the severe "
                  "terrain-slugging regime (PI<1) at deep turndown.")

# 3c) shutdown cool-down (severe-hydrate design driver) ----------------------
cd = U.shutdown_cooldown(cfg, res, t_hours=24.0, n_t=120)
np.savez(os.path.join(DD, "cooldown.npz"), t=cd["t"], Tfield=cd["Tfield"],
         subcool=cd["subcool"], tth_h=cd["tth_h"], tau=cd["tau"],
         s=res["s"], Psettle=cd["Psettle"])
cdt = pd.DataFrame({"s_km": res["s"]/1000.0, "tau_h": cd["tau"]/3600.0,
                    "time_to_hydrate_h": cd["tth_h"]})
common.save_table(cdt[::4], "shutdown_cooldown.csv", "solution",
                  "Shutdown cool-down / time-to-hydrate",
                  "Per-section thermal time constant and time-to-hydrate after an "
                  "unplanned shutdown (settle-out pressure) - the hydrate design driver.")

# 3d) 3-D resolved field volume ---------------------------------------------
vol = U.build_3d_volume(cfg, res, n_axial=60, n_r=22, n_phi=40)
np.savez(os.path.join(DD, "volume3d.npz"), **{k: vol[k] for k in
         ["X", "Y", "Z", "W", "phase", "T", "P", "route", "s_idx"]})

# 4) hydrodynamic slug train (space-time) -----------------------------------
ht = U.hydrodynamic_slug_train(cfg, res)
np.savez(os.path.join(DD, "slug_train.npz"), **{k: ht[k] for k in
         ["x", "t", "HL", "P_base", "period", "Vt", "fs", "Ls", "Lu", "HLs"]})

# 5) flow-regime map ---------------------------------------------------------
# representative gas-zone conditions for the map
ig = np.where(res["Vsg"] > 0.05)[0]
imap = ig[len(ig)//2] if len(ig) else n-5
fm = U.flow_regime_map(cfg, res["P"][imap], res["T"][imap], np.deg2rad(res["theta"][imap]))
np.savez(os.path.join(DD, "regime_map.npz"), Vsg=fm["Vsg"], Vsl=fm["Vsl"],
         Z=fm["Z"], labels=np.array(fm["labels"], dtype=object),
         path_Vsg=res["Vsg"], path_Vsl=res["Vsl"])

# 6) MEG inhibitor sweep -----------------------------------------------------
rows = []
for meg in np.arange(0, 56, 5):
    c = U.CaseConfig(MEG_wt=meg)
    r = U.march(c); h = U.hydrate_assessment(c, r)
    rows.append({"MEG_wt_pct": meg, "Teq_depression_C": h["dT_inhib"],
                 "hydrate_zone_frac": float((h["subcool"] > 0).mean()),
                 "max_subcooling_C": float(np.nanmax(h["subcool"]))})
meg = pd.DataFrame(rows)
common.save_table(meg, "meg_sweep.csv", "solution", "MEG inhibition sweep",
                  "Hydrate-temperature depression, in-envelope length fraction and "
                  "maximum sub-cooling versus MEG dosage (Hammerschmidt).")

# 7) terrain-slugging susceptibility & KPI summary --------------------------
sscrit = U.severe_slugging_number(cfg, res)
slug_mask = np.array([("Slug" in r or "intermittent" in r) for r in sp["regime"]])
hyd_mask = hy["subcool"] > 0
kpi = pd.DataFrame({
    "KPI": ["Inlet pressure", "Outlet (arrival) pressure", "Total pressure drop",
            "Inlet temperature", "Minimum temperature", "Outlet temperature",
            "Inlet GVF", "Outlet GVF", "Mean liquid holdup",
            "Slug-flow length (% of line)",
            "Mean slug frequency", "Mean slug length", "Slug transl. velocity",
            "Dominant sag location", "Sag uphill leg height",
            "Terrain-slug period", "Terrain-slug pressure swing",
            "Terrain-slug liquid surge", "Boe number @ design",
            "Hydrate-risk zone length", "Max hydrate sub-cooling (flowing)",
            "Hydrate zone (% of line, flowing)", "Shutdown no-touch time",
            "Recommended MEG dose"],
    "Value": [f"{cfg.P_in/1e5:.0f} bar", f"{res['P'][-1]/1e5:.1f} bar",
              f"{(cfg.P_in-res['P'][-1])/1e5:.1f} bar",
              f"{cfg.T_in-273.15:.0f} degC", f"{res['T'].min()-273.15:.1f} degC",
              f"{res['T'][-1]-273.15:.1f} degC",
              f"{res['GVF'][0]:.2f}", f"{res['GVF'][-1]:.2f}", f"{res['HL'].mean():.2f}",
              f"{100*slug_mask.mean():.0f} %",
              f"{np.nanmean(sp['fs'][slug_mask]):.3f} Hz" if slug_mask.any() else "n/a",
              f"{np.nanmean(sp['Ls'][slug_mask]):.1f} m" if slug_mask.any() else "n/a",
              f"{np.nanmean(sp['Vt'][slug_mask]):.2f} m/s" if slug_mask.any() else "n/a",
              f"{sscrit['sag_s']/1000:.1f} km", f"{sscrit['H_leg']:.0f} m",
              f"{ss['period_est']/60:.1f} min",
              f"{(ss['P_base'].max()-ss['P_base'].min())/1e5:.1f} bar",
              f"{ss['Wl_out'].max()/max(ss['Wl'],1e-9):.0f} x mean",
              f"{sscrit['PI_ss']:.1f} ({'susceptible' if sscrit['susceptible'] else 'stable, slugs at turndown'})",
              f"{(res['s'][hyd_mask].max()-res['s'][hyd_mask].min())/1000:.1f} km" if hyd_mask.any() else "0",
              f"{np.nanmax(hy['subcool']):.1f} degC", f"{100*hyd_mask.mean():.0f} %",
              f"{cd['no_touch_time_h']:.1f} h" if cd['no_touch_time_h']==cd['no_touch_time_h'] else ">24 h",
              "~30 wt% (flowing) / ~50 wt% (shut-in)"]})
common.save_table(kpi, "kpi_summary.csv", "solution", "Key performance indicators",
                  "Headline predicted results for the calibrated base case.")

# ---- save full result bundle for postprocessing ---------------------------
bundle = {k: res[k] for k in res if isinstance(res[k], np.ndarray)}
for k in ["Teq_eff", "subcool", "risk", "Teq"]:
    bundle["hy_"+k] = hy[k]
for k in ["fs", "Vt", "HLs", "Lu", "Ls", "Lf"]:
    bundle["slug_"+k] = sp[k]
np.savez(os.path.join(DD, "results.npz"), **bundle)
np.savez(os.path.join(DD, "severe_slug.npz"),
         t=ss["t"], P_base=ss["P_base"], z=ss["z"], HL_riser=ss["HL_riser"],
         Wl_out=ss["Wl_out"], phase=ss["phase"], period=ss["period_est"],
         t_build=ss["t_build"], t_blow=ss["t_blow"], Wl=ss["Wl"], Wg=ss["Wg"])
np.save(os.path.join(DD, "regime_labels.npy"), sp["regime"])

print("Stage 04 solution complete. P_out=%.1f bar, max subcool=%.1f C, SS period=%.1f min"
      % (res["P"][-1]/1e5, np.nanmax(hy["subcool"]), ss["period_est"]/60))
