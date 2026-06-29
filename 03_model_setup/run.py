"""STAGE 03 - MODEL SETUP
Defines fluids, operating point, boundary conditions, calibration constants and
numerical settings; generates the PVT property curves used by the solver."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import style, common
import usmh_solver as U

style.apply_style()
cfg = U.CaseConfig()
D = common.DIRS["setup"]

# ---------- input tables ----------------------------------------------------
fluids = pd.DataFrame({
    "Property": ["Oil API gravity", "Oil specific gravity", "Gas specific gravity (air=1)",
                 "Producing GOR", "Water cut", "Produced-water density",
                 "Water viscosity", "Gas-oil interfacial tension",
                 "Gas-water interfacial tension", "Liquid heat capacity",
                 "Gas heat capacity"],
    "Symbol": ["API", "gamma_o", "gamma_g", "GOR", "WC", "rho_w", "mu_w",
               "sigma_go", "sigma_gw", "cp_l", "cp_g"],
    "Value": [cfg.API, round(cfg.gamma_o, 4), cfg.gas_grav, cfg.GOR, cfg.water_cut,
              cfg.rho_w, cfg.mu_w, cfg.sigma_go, cfg.sigma_gw, 2100, 2400],
    "Unit": ["degAPI", "-", "-", "scf/stb", "fraction", "kg/m3", "Pa.s",
             "N/m", "N/m", "J/kg/K", "J/kg/K"]})
common.save_table(fluids, "fluid_properties.csv", "setup", "Fluid properties (live crude)",
                  "Reservoir-fluid and produced-water properties used to initialise the "
                  "PVT and multiphase closure relations.")

oper = pd.DataFrame({
    "Quantity": ["Oil rate (stock tank)", "Liquid rate (in-situ, inlet)",
                 "Inlet (wellhead/manifold) pressure", "Inlet temperature",
                 "Seabed (ambient) temperature", "Pipeline OHTC", "Mass flow rate"],
    "Symbol": ["qo", "ql", "P_in", "T_in", "T_seabed", "U_pipe", "mdot"],
    "Value": [cfg.qo_stb_d, round(U.local_state(cfg,cfg.P_in,cfg.T_in,0)['Vsl']*cfg.area,4),
              cfg.P_in/1e5, cfg.T_in-273.15, cfg.T_seabed-273.15,
              cfg.U_pipe, round(U.march(cfg)["mdot"], 1)],
    "Unit": ["stb/d", "m3/s", "bar", "degC", "degC", "W/m2/K", "kg/s"]})
common.save_table(oper, "operating_conditions.csv", "setup", "Operating conditions",
                  "Production and thermal operating point of the subsea pipeline.")

bcs = pd.DataFrame({
    "Boundary": ["Inlet (wellhead/manifold)", "Inlet (wellhead/manifold)",
                 "Inlet (wellhead/manifold)", "Pipe wall", "Outlet (host)"],
    "Type": ["Dirichlet pressure", "Dirichlet temperature", "Mass-flow inlet",
             "Convective (Robin)", "Pressure outlet"],
    "Specification": [f"P = {cfg.P_in/1e5:.0f} bar", f"T = {cfg.T_in-273.15:.0f} degC",
                      "fixed liquid + gas mass flow",
                      f"U = {cfg.U_pipe} W/m2/K, T_amb = {cfg.T_seabed-273.15:.0f} degC (seabed)",
                      "arrival/separator pressure (free)"]})
common.save_table(bcs, "boundary_conditions.csv", "setup", "Boundary conditions",
                  "Inlet, wall and outlet boundary conditions for the coupled "
                  "momentum-energy march.")

calib = pd.DataFrame({
    "Constant": ["C_fr", "C_sl", "C_hy"],
    "Meaning": ["Flow-regime transition factor", "Slug-frequency factor",
                "Hydrate equilibrium-temperature offset"],
    "Calibrated value": [cfg.C_fr, cfg.C_sl, cfg.C_hy],
    "Unit": ["-", "-", "degC"],
    "Calibration source": ["Taitel-Dukler (1976) map", "Gregory & Scott (1969) data",
                           "Deaton & Frost (1946) methane data"]})
common.save_table(calib, "calibration_constants.csv", "setup", "Calibration constants",
                  "The three dimensionless / offset constants that calibrate the "
                  "universal solver to the case.")

num = pd.DataFrame({
    "Setting": ["Spatial discretisation", "Cells", "ODE integrator",
                "Relative tolerance", "Absolute tolerance", "Transient time step",
                "PVT model", "Holdup model", "Friction model", "Hydrate model",
                "Inhibitor model", "Slug model", "Severe-slugging model"],
    "Value": ["Finite-volume, graded", cfg.n_cells, "RK45 (adaptive)",
              "1e-6", "1e-2", "0.5 s", "Standing + Beggs-Brill Z",
              "Bendiksen (1984) drift-flux", "Haaland (1983)",
              "Motiee (1991), calibrated", "Hammerschmidt (1934)",
              "Gregory-Scott + Bendiksen", "Taitel (1986)-type cycle"]})
common.save_table(num, "numerical_settings.csv", "setup", "Numerical settings",
                  "Discretisation, solver tolerances and sub-model selection.")

# ---------- PVT curves ------------------------------------------------------
P = np.linspace(5e5, 230e5, 80)
T = cfg.T_in
Rs = np.array([min(U.standing_rs(p, T, cfg.API, cfg.gas_grav), cfg.GOR) for p in P])
Bo = np.array([U.standing_bo(rs, T, cfg.API, cfg.gas_grav) for rs, p in zip(Rs, P)])
mu_o = np.array([U.beggs_robinson_mu_o(p, T, cfg.API, rs)*1e3 for p, rs in zip(P, Rs)])
Z = np.array([U.z_factor(p, T, cfg.gas_grav) for p in P])
rho_g = np.array([U.gas_density(p, T, cfg.gas_grav) for p in P])
mu_g = np.array([U.gas_viscosity(p, T, cfg.gas_grav)*1e3 for p in P])

pvt = pd.DataFrame({"P_bar": P/1e5, "Rs_scf_stb": Rs, "Bo_rb_stb": Bo,
                    "mu_oil_cP": mu_o, "Z_factor": Z, "rho_gas_kgm3": rho_g,
                    "mu_gas_cP": mu_g})
common.save_table(pvt, "pvt_table.csv", "setup", "PVT property table",
                  "Live-crude PVT properties versus pressure at inlet temperature "
                  f"({T-273.15:.0f} degC): solution GOR, FVF, viscosities, gas Z and density.")

def two_panel(x, ys, labels, ylabels, title, fname, caption):
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, y, lab, yl, col in zip(axs, ys, labels, ylabels, style.PALETTE):
        ax.plot(x, y, color=col, lw=2.6)
        ax.set_xlabel("Pressure (bar)"); ax.set_ylabel(yl); ax.set_title(lab)
    fig.suptitle(title, color=style.TITLE, fontweight="bold")
    common.register_figure(style.finish(fig, os.path.join(D, fname)),
                           "setup", title, caption)

two_panel(P/1e5, [Rs, Bo], ["Solution gas-oil ratio", "Oil formation volume factor"],
          ["$R_s$ (scf/stb)", "$B_o$ (rb/stb)"],
          "PVT - solution gas and oil FVF (Standing)", "fig_pvt_rs_bo.png",
          "Solution gas-oil ratio and oil formation volume factor versus pressure "
          "(Standing correlation); the bubble point is near the inlet pressure.")
two_panel(P/1e5, [mu_o, mu_g], ["Live-oil viscosity", "Gas viscosity"],
          ["$\\mu_o$ (cP)", "$\\mu_g$ (cP)"],
          "PVT - phase viscosities", "fig_pvt_viscosity.png",
          "Live-oil (Beggs-Robinson) and gas (Lee et al.) viscosities versus pressure.")
two_panel(P/1e5, [Z, rho_g], ["Gas compressibility factor", "Gas density"],
          ["Z (-)", "$\\rho_g$ (kg/m³)"],
          "PVT - gas Z-factor and density", "fig_pvt_z_rhog.png",
          "Real-gas compressibility factor (Beggs-Brill) and density versus pressure.")

print("Stage 03 model setup complete.")
