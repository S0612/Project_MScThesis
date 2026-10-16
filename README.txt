# Survey Expectations and Finite-Sample Bias in a Macro-Finance Gaussian Affine Term Structure Model

MATLAB code for my master's thesis in Economics at Aarhus University, 2026.

The thesis studies whether finite sample bias in a Gaussian affine term structure model (GATSM) can be reduced by adding survey expectations of future interest rates to the estimation.

We find a small improvement in overall bias generally. 

## Contents
- `2F_codes/` two factor latent GATSM
- `4F_codes/` four factor macro finance GATSM, with and without surveys
- `Thesis.pdf` and `Thesis_Appendices.pdf`

## Reproducing the results
CMA-ES is a randomised optimiser and was not seeded internally, so results may not replicate exactly on a first run. The stored workspaces (`.mat` files) contain the results used in the thesis and can be used for verification.

Requires MATLAB. The parallel version of CMA-ES uses `parfor` and benefits from the Parallel Computing Toolbox. Data are included as Excel and CSV files. [Data sources]

## 2F_codes: two factor latent GATSM
- `StartingParams2F_US.m` computes starting values.
- `GATSM_US_2F_V2.m` main estimation script. Uses Kalman filtering, with the likelihood maximised by `fminsearch` (Nelder-Mead) and by CMA-ES (`cmaes_dsgeDisplay.m`, and `cmaes_dsgeDisplay_mp_v2.m`, the `parfor` version). The CMA-ES code is by Andreasen (2010).
- `MonteCarloGATSM_US_2F.m` Monte Carlo simulations using the CMA-ES estimates as the data generating process. Estimation uses `fminsearch` started at the true values, to save computing time.
- `GATSM_US_2F_CampbellShiller.m` Campbell-Shiller regressions on data sampled from the estimated model, compared with actual data.
- `CMAES_VS_NM_Plots.m` plots comparing the two estimators.

Stored workspaces:
- `GATSM_2F_Estimates_US_NelderMead.mat` estimates using `fminsearch`
- `GATSM_2F_Estimates_US_CMAES.mat` estimates using CMA-ES
- `MonteCarlo_GATSM_US_2F.mat` Monte Carlo results (R = 1000)
- `CS_Simulation_Results.mat` Campbell-Shiller regressions on actual data and on yields sampled from the CMA-ES estimates

## 4F_codes: four factor macro finance GATSM

**Estimation**
- `GATSM_US_4F_MacroFinance_Quarterly_v23.m` main script, estimates the model without surveys.
- `GATSM_US_4F_MacroFinance_Quarterly_Surveys_Proper_KO_V6.m` adds one quarter ahead surveys.
- `GATSM_US_4F_MacroFinance_Quarterly_Surveys_Proper_KO_Q4_V1.m` adds four quarter ahead surveys.
- Results: `GATSM_4F_MacroFinance_Results_V23.mat`, `GATSM_4F_MacroFinance_Results_KO_proper_V6.mat`, `GATSM_4F_MacroFinance_Results_Surveys_Q4.mat`

**Monte Carlo**
- `MonteCarloGATSM_KO_ThreeEstimators.m` simulations comparing the no survey (NS), 1Q and 4Q estimators. Results: `MonteCarlo_KO_3EST_T216.mat`, `MonteCarlo_KO_3EST_T100.mat`
- `AnalyseKO_ThreeEstimators_V1.m` creates LaTeX tables from these simulations.
- `MonteCarloGATSM_KO_NoisySurveys.m` robustness test with artificial noise added to the surveys. Results: `MonteCarlo_KO_NoisySurveys_kappa05_T216.mat`, `..._kappa05_T100.mat`, `..._kappa10_T216.mat`, `..._kappa10_T100.mat`

**Further analysis**
- `FEVD.m` forecast error variance decomposition. Result: `FEVD_results.mat`
- `ChiSquareTest.m` the informal and formal tests described in Hamilton and Wu (2012).
- `ComputeMarketPriceOfRisk.m` market price of risk for the 1Q and NS estimators.
- `CampbellShiller_4F.m` Campbell-Shiller regressions on model implied yields and actual data. Result: `CS_4F.mat`
- `IRF` and `IRF2` produce the impulse responses used in the thesis.

**Survey calibration**
- `CalibrateSurvey1Q.m` and `CalibrateSurvey4Q.m` calibrate the one and four quarter ahead survey forecasts.
- `CalibrateSurvey_Combined.m` produces the figures.
