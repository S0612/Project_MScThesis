# Four-factor macro-finance Gaussian affine term structure model with survey expectations

MATLAB code for my master's thesis *Survey Expectations and Finite-Sample Bias in a Macro-Finance
Gaussian Affine Term Structure Model* (Aarhus University, 2026). The thesis and its appendices are in
this repository as `Thesis.pdf` and `Thesis_Appendices.pdf`.

The thesis asks whether adding Survey of Professional Forecasters (SPF) expectations to the
estimation of the physical (P) dynamics reduces the finite-sample downward bias in P persistence
(the Kendall-Pope bias), and what that does to the decomposition of yields into expected short rates
and term premia. The model has four factors: two observed macro factors (inflation and
unemployment) and two latent yield factors, as in Ang and Piazzesi (2003). The model is estimated
with the Hamilton and Wu (2012) minimum chi-square estimator (MCSE), with survey information added
as in Kim and Orphanides (2012) through a Theil-Goldberger mixed estimator.

A two-factor yields-only baseline is in `2F_codes`. A simplified Python version of that baseline is
in a separate repository: https://github.com/S0612/Project_Gatsm2F_python. The Python version does not include the four-factor model, the
survey expectations or the MCSE.

## The model

Time is quarterly. The factor vector stacks the observed macro block and the latent block:

$$X_t = \begin{pmatrix} M_t \\ L_t \end{pmatrix}, \qquad M_t = \begin{pmatrix} \pi_t \\ u_t \end{pmatrix}, \quad L_t = \begin{pmatrix} L_{1t} \\ L_{2t} \end{pmatrix}$$

where $\pi_t$ is GDP-deflator inflation and $u_t$ is unemployment.

```
short rate     r_t = delta0 + delta1' X_t
P dynamics     X_t = mu + Phi^P X_{t-1} + Sigma eps_t,           eps_t ~ N(0, I_4)
Q dynamics     X_t = mu^Q + Phi^Q X_{t-1} + Sigma eps^Q_t,       eps^Q_t ~ N(0, I_4)
```

Both `Phi^P` and `Phi^Q` are block diagonal, `blockdiag(Phi_mm, Phi_ll)`, exactly as in Ang and
Piazzesi (2003): macro and latent factors are independent under both measures. <!-- CHECK: state here whether Sigma is block diagonal and how Sigma_mm is parameterised (e.g. lower-triangular Cholesky). -->

**Bond pricing.** With one-period discounting at `r_t`, log bond prices are affine,
`p_t(n) = A_n + B_n' X_t`, with `A_0 = 0`, `B_0 = 0` and

```
A_n  = A_{n-1} + B_{n-1}' mu^Q + 0.5 B_{n-1}' Sigma Sigma' B_{n-1} - delta0
B_n' = B_{n-1}' Phi^Q - delta1'
y_t(n) = -p_t(n) / n
```

**Price of risk.** The difference between the P and Q parameters is the market price of risk. With
the convention `eps_t = eps^Q_t - lambda_t`, `lambda_t = Sigma^{-1} [(mu - mu^Q) + (Phi^P - Phi^Q) X_{t-1}]`,
which is essentially affine (Duffee, 2002). This P-Q wedge is the structural source of term premia.

**Observation equation.** Six yields are used. Following Hamilton and Wu (2012) they are split into
`Y1` (as many yields as latent factors, so two) priced without error, and `Y2` (the other four)
priced with measurement error. The latent factors are obtained by inverting the `Y1` pricing
equations. <!-- CHECK: which two maturities are Y1, and the six maturities used (I assumed 1, 2, 5, 7, 10, 15 years as in the 2F code). -->

**Identification.** The observed macro block needs no normalisation. The latent `Q` block is
normalised in equal-diagonal Jordan form (`lev`, `lb_off`, `lc_off` in the code) with the
restriction `b <= c`. This covers real distinct, complex conjugate and repeated eigenvalues, and
together with the restriction achieves global identification of the latent block.
<!-- CHECK: write out the 2x2 matrix exactly as coded. If it is [lev b; c lev], the eigenvalues are lev +/- sqrt(bc). -->

**Yield decomposition.** The expected short rate for maturity `n` is the average expected short rate
under P over the life of the bond, and the term premium is the fitted yield minus that:

```
y_t(n) = (1/n) sum_{i=0}^{n-1} E^P_t[r_{t+i}]  +  TP_t(n)
E^P_t[r_{t+i}] = delta0 + delta1' ( sum_{j=0}^{i-1} (Phi^P)^j mu + (Phi^P)^i X_t )
```

The latent factors in this decomposition come from a Kalman filter run once after estimation (see
below).

## Estimation

The estimator is not maximum likelihood, which is the main difference from the two-factor code.

1. **Reduced form.** The unrestricted parameters `pi_hat` are estimated by OLS: a VAR(1) for the
   observable state (the macro factors and `Y1`) and a regression of `Y2` on that state.
2. **Survey augmentation.** For the macro block, the OLS estimates are combined with SPF
   expectations by a Theil-Goldberger (1961) mixed estimator, in the spirit of Kim and Orphanides
   (2012). The survey forecasts are treated as noisy observations of the model-implied P expectations
   of the macro factors. <!-- CHECK: write the stochastic restriction and the horizon mapping exactly as coded, including how the survey noise variance is set. -->
3. **Minimum chi-square.** The structural parameters `theta` minimise
   `(pi_hat - g(theta))' R_hat (pi_hat - g(theta))`, where `g` maps structural parameters to the
   reduced-form parameters and `R_hat` is the weighting matrix built from the information matrix.
   The minimisation uses CMA-ES (`cmaes_dsgeDisplay.m`).
4. **Over-identification test.** The valid statistic re-minimises the criterion with the
   efficient weighting matrix (`lsqnonlin` from the Optimization Toolbox, 40 multi-start
   perturbations). It is asymptotically chi-square with 13 degrees of freedom.
5. **Standard errors.** FIML standard errors: the numerical Hessian (central finite differences) of
   the Kalman filter log likelihood at the MCSE point estimates. Hamilton and Wu (2012) show this is
   asymptotically equivalent to the MCSE sandwich formula under correct specification.
6. **Factor extraction.** The Kalman filter is run once at the final estimates, only to produce the
   filtered latent states for the decomposition and the impulse responses, and a scalar log
   likelihood as a diagnostic. It is not the estimation engine here, unlike in the two-factor model.

## Quick start

MATLAB with the Optimization Toolbox is required. Run from `4F_codes`:

```
GATSM_US_4F_MacroFinance_Quarterly_v23                       % V23: no-survey baseline
GATSM_US_4F_MacroFinance_Quarterly_Surveys_Proper_KO_V6      % V6: survey-augmented, 1Q SPF
ComputeFIMLStandardErrors_V2                                 % standard errors for the saved estimates
MonteCarloGATSM_KO_ThreeEstimators                           % finite-sample Monte Carlo
```
<!-- CHECK: script vs function, run order, input/output .mat file names, and which folder each file is in. -->

## Data

Raw data are monthly, January 1972 to December 2025 (648 months). The model is estimated on
quarterly data, 1972Q1 to 2025Q4 (216 quarters). <!-- CHECK: how monthly is aggregated to quarterly (quarter-end or average), and how a quarterly GDP deflator appears in a monthly macro file. -->

| File | Content |
|---|---|
| `US_monthly_yields_Jan1972_Dec2025.csv` | 648 rows, 15 maturities, yields in percent, no header |
| `US_monthly_yields_Jan1972_Dec2025_maturities.csv` | One row with the 15 maturities in years |
| `US_macro_inflation_unemployment_MF.csv` | 648 rows, 2 columns: inflation and unemployment (column order: CHECK) |
| `US_SurveyData.xlsx` | SPF forecasts: `DPGDP3`, `UNEMP3` (one quarter ahead) and `DPGDP6`, `UNEMP6` (four quarters ahead) |

The main survey model, V6, uses the one-quarter-ahead forecasts. All yield data is from the zero-coupon Svensson yields by Gurkaynak, Sack, and Wright (2006).

## Files

| File | Content |
|---|---|
| `GATSM_US_4F_MacroFinance_Quarterly_Surveys_Proper_KO_V6.m` | Main survey-augmented model (V6) |
| `GATSM_US_4F_MacroFinance_Quarterly_v23.m` | No-survey baseline (V23) |
| `GATSM_US_2F_V2.m` | Two-factor baseline estimated by Kalman filter maximum likelihood |
| `ComputeFIMLStandardErrors.m`, `ComputeFIMLStandardErrors_V2.m` | FIML Hessian standard errors |
| `MonteCarloGATSM_KO_ThreeEstimators.m` | Monte Carlo comparing three estimators |
| `MonteCarloGATSM_KO_NoisySurveys.m` | Monte Carlo with noisy synthetic surveys |
| `CalibrateSurvey1Q.m` | Calibrates the synthetic one-quarter survey |

## Monte Carlo

The data generating process is the V6 estimate. Each replication simulates a sample from the model,
initialised with a burn-in of 100 quarters, builds synthetic survey data calibrated by
`CalibrateSurvey1Q.m`, and re-estimates. Bias is the mean of the estimates minus the true value.

Because V6 is both the data generating process and the specification being evaluated, and the
synthetic survey persistence is calibrated from the same SPF sample as the estimated macro
persistence, the design has a degree of circularity. The Monte Carlo therefore shows what the
estimator does if V6 were true, not that V6 is true.

## Results

Full tables and figures are in `Thesis.pdf`. The headline findings:

- Survey augmentation raises the estimated P persistence of the macro factors and with it the
  term premia. V6 gives term premia of about 1.55 percentage points at the 15-year maturity,
  where the no-survey baseline V23 gives term premia close to zero. <!-- CHECK: peak, or sample average? -->
- This is consistent with the bias-correction mechanism. OLS estimates of the macro AR dynamics are
  biased downward in persistence, roughly `-(1 + 3 phi) / T` for an AR(1) with an estimated
  intercept (Kendall, 1954) and by the matrix formula of Pope (1990) for a VAR. The macro OLS
  estimates feed `Phi^P_mm` directly, so the bias passes one for one into expected short-rate paths,
  which pushes term premia towards zero.
- The model is formally rejected: the properly weighted over-identification statistic is
  chi-square(13) = 87.18, p of about 5e-13.
- The correction can overshoot. In the Monte Carlo the unemployment persistence under survey
  augmentation is about 0.9955 against a true value of about 0.8945.

## Reading the output

- `chi2_stored` is the identity-weighted criterion at the optimum (about 0.0863). It has no
  chi-square distribution and must not be used for inference. Use the re-minimised statistic
  described above.
- Fitted yields at the `Y1` maturities are close to observed yields by construction, because the
  latent factors are inverted from them. Pricing errors at those maturities are not evidence of fit.
  Only the `Y2` yields give a genuine test of the model.
- Results are saved as `.mat` files in MATLAB v7.3 (HDF5) format. When they are read in Python with
  `h5py`, dimensions come out transposed. Every 2D array, including square matrices, needs `.T`
  before it is used in matrix multiplication.

## Limitations

- The over-identification test rejects the model. The standard errors, the equivalence of MCSE and
  FIML, and the Monte Carlo are all conditional on the specification being correct, so they should
  be read as conditional on V6 and not as evidence that V6 is correct.
- Macro and latent factors are independent under both P and Q by construction. There is no
  feedback from yield factors to macro variables.
- The unemployment persistence correction overshoots, so the term premium increase may be partly an
  over-correction rather than pure bias removal.
- The factors are close to a unit root, which is where finite-sample bias is largest and where
  asymptotic standard errors are least reliable.
- The latent factors in the decomposition are filtered, not smoothed.
- Results depend on the survey horizon (one quarter in V6), the survey noise assumption, and the
  model restrictions (block-diagonal dynamics, equal-diagonal normalisation).
- The synthetic surveys generated are not truly model-consistent.

## References

- Ang, A. and Piazzesi, M. (2003). A no-arbitrage vector autoregression of term structure dynamics
  with macroeconomic and latent variables. *Journal of Monetary Economics*.
- Bauer, M., Rudebusch, G. and Wu, J. C. (2012). Correcting estimation bias in dynamic term
  structure models. *Journal of Business and Economic Statistics*.
- Duffee, G. (2002). Term premia and interest rate forecasts in affine models. *Journal of Finance*.
- Hamilton, J. and Wu, J. C. (2012). Identification and estimation of Gaussian affine term
  structure models. *Journal of Econometrics*.
- Gürkaynak, Refet S., Brian Sack, and Jonathan H. Wright. "The US Treasury yield curve: 1961 to the present." Journal of monetary Economics 54.8 (2007): 2291-2304.
- Joslin, S., Priebsch, M. and Singleton, K. (2014). Risk premiums in dynamic term structure models
  with unspanned macro risks. *Journal of Finance*.
- Kendall, M. (1954). Note on bias in the estimation of autocorrelation. *Biometrika*.
- Kim, D. and Orphanides, A. (2012). Term structure estimation with survey data on interest rate
  forecasts. *Journal of Financial and Quantitative Analysis*.
- Pope, A. (1990). Biases of estimators in multivariate non-Gaussian autoregressions. *Journal of
  Time Series Analysis*.
- Theil, H. and Goldberger, A. (1961). On pure and mixed statistical estimation in economics.
  *International Economic Review*.
