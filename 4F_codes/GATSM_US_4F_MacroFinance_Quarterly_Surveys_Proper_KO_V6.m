% This script incorporates surveys into the four-factor GATSM
% ignore the naming it is not what kim and orphanides do in fact.
%
% Files needed in dir
%   US_monthly_yields_Jan1972_Dec2025.csv
%   US_monthly_yields_Jan1972_Dec2025_maturities.csv
%   US_macro_inflation_unemployment_MF.csv


%% User Settings

% Maturities to use in estimation (years).
% Expressed in quarters; must match values in the maturities CSV
% when converted from years (multiply years by 4).
matSelect = [1 2 5 7 10 15] * 4;     % Maturities in quarters

% Yields priced exactly (used to invert latent factors); must equal nxL = 2.
% Remaining maturities are priced with measurement error.
% We choose the 1-year and 10-year to span the curve.
exactYears = [1 10];                  % years; must be a subset of matSelect/4
errorYears = setdiff(matSelect/4, exactYears);

% Factor dimensions
nxM = 2;    % number of observable macro factors (inflation, unemployment)
nxL = 2;    % number of latent yield factors
nx  = nxM + nxL;   % total state dimension (= 4)

% Annualisation factor: quarterly data -> annual yields
ann = 4;

w_svy = 1.0;


%% Load data

% 2a: Yield data (monthly CSV -> quarterly sub-sample)
yields_raw = csvread('Data/US_monthly_yields_Jan1972_Dec2025.csv');
mats_years = csvread('Data/US_monthly_yields_Jan1972_Dec2025_maturities.csv');

% Validate that every requested maturity exists in the CSV
assert(all(ismember(matSelect/ann, mats_years)), ...
    ['One or more maturities in matSelect are not available in the CSV. ' ...
     'Available maturities (years): ' num2str(mats_years)]);

% Sub-sample monthly -> quarterly: rows 3, 6, 9, ... (Mar, Jun, Sep, Dec)
% Starting Jan 1972, the first quarterly obs is March 1972 = row 3.
T_monthly = size(yields_raw, 1);
qIdx      = 3 : 3 : T_monthly;     % row indices of quarterly observations
T         = length(qIdx);           % number of quarterly obs

numObs = length(matSelect);         % total number of yield maturities used
data   = zeros(numObs, T);          % numObs x T  (annualised decimal)
for i = 1:numObs
    col       = find(mats_years == matSelect(i)/ann);
    data(i,:) = yields_raw(qIdx, col)' / 100;
end

% 2b:
% Y1 has nxL rows — one per latent factor (used to invert x_latent).
% Y2 has Ne = numObs - nxL rows — priced with common measurement error stdY.
idxExact  = ismember(matSelect/ann, exactYears);   % logical, length numObs
idxError  = ~idxExact;
Ne        = sum(idxError);

matSelect_Y1 = matSelect(idxExact);   % nxL-element vector (quarters)
matSelect_Y2 = matSelect(idxError);   % Ne-element vector (quarters)

Y1 = data(idxExact, :);   % nxL x T  yields priced without error
Y2 = data(idxError, :);   % Ne  x T  yields priced with measurement error

% 2c: Macro data (quarterly, aligned to yield sub-sample)
% The macro CSV has valid values every 3rd row (same cadence as qIdx).
macro_raw = csvread('Data/US_macro_inflation_unemployment_MF.csv');  % T_monthly x 2
macro_Q   = macro_raw(qIdx, :);     % T x 2  [inflation, unemployment]

% Fill any residual NaNs by linear interpolation
for j = 1:2
    col    = macro_Q(:,j);
    nanIdx = isnan(col);
    if any(nanIdx)
        col(nanIdx) = interp1(find(~nanIdx), col(~nanIdx), ...
                              find(nanIdx), 'linear', 'extrap');
        macro_Q(:,j) = col;
    end
end

% Normalise to zero mean and unit variance
macro_mean = mean(macro_Q);
macro_std  = std(macro_Q);
xMacro     = ((macro_Q - macro_mean) ./ macro_std)';   % nxM x T

if w_svy > 0
    fprintf('  Loading KO survey data (w_svy=%.2f)...\n', w_svy);
    T_infl_raw  = readtable('US_SurveyData.xlsx', 'Sheet', 'Inflation',    ...
                             'ReadRowNames', true);
    T_unemp_raw = readtable('US_SurveyData.xlsx', 'Sheet', 'Unemployment', ...
                             'ReadRowNames', true);

    surv_infl_raw  = double(T_infl_raw{:,  'dpgdp3'});   % T x 1  (% ann., 1Q-ahead)
    surv_unemp_raw = double(T_unemp_raw{:, 'UNEMP3'});   % T x 1  (%, 1Q-ahead)

    % Normalise each survey series by its own mean and std
    svy_mean_infl  = mean(surv_infl_raw,  'omitnan');
    svy_std_infl   = std( surv_infl_raw,  'omitnan');
    svy_mean_unemp = mean(surv_unemp_raw, 'omitnan');
    svy_std_unemp  = std( surv_unemp_raw, 'omitnan');

    xSvy_infl  = (surv_infl_raw  - svy_mean_infl)  / svy_std_infl;   % T x 1
    xSvy_unemp = (surv_unemp_raw - svy_mean_unemp) / svy_std_unemp;  % T x 1

    % Stack: xSurvey is nxM x T  (rows = [infl; unemp])
    xSurvey = [xSvy_infl'; xSvy_unemp'];

    fprintf('  Survey normalised: infl(mean=%.4f,std=%.4f), unemp(mean=%.4f,std=%.4f)\n', ...
        mean(xSvy_infl), std(xSvy_infl), mean(xSvy_unemp), std(xSvy_unemp));
else
    xSurvey = [];   % no survey data; Block m OLS uses actual only
    fprintf('  w_svy=0: survey augmentation disabled (standard V23).\n');
end
dateStart   = datenum(1972, 3, 1);
setup_dates = dateStart + (0:T-1)' * 91;   % approximate quarterly spacing
dateVec     = datetime(1972, 3, 1) + calmonths(3*(0:T-1));

fprintf('  Quarterly obs: T = %d\n', T);
fprintf('  Maturities (years): %s\n', num2str(matSelect/ann));
fprintf('  Priced exactly (Y1): %s yr\n', num2str(exactYears));
fprintf('  Priced with error (Y2): %s yr\n', num2str(errorYears));

%% First-stage OLS  (reduced-form VAR)

T1 = T - 1;   % usable observations after one lag

ym_act  = xMacro(:, 2:end)';                    % T1 x nxM  (actual at t+1)
xm_act  = [ones(T1,1), xMacro(:,1:end-1)'];    % T1 x (1+nxM)

if w_svy > 0 && ~isempty(xSurvey)
    % Survey dependent variable: xSurvey(:,t) = E^svy_{t-1}[x_{macro,t}]
    % We need survey at t matching actual at t (both are t+1 in the OLS).
    % xSurvey(:,t) = dpgdp3/UNEMP3 at t = 1Q-ahead forecast made at t for t+1.
    % For OLS rows t=1..T1:
    %   actual dep  = xMacro(:,t+1)    indexed as xMacro(:,2:end) -> ym_act
    %   survey dep  = xSurvey(:,t)     indexed as xSurvey(:,1:end-1) -> ym_svy
    %   regressor   = xMacro(:,t)      indexed as xMacro(:,1:end-1) -> xm_act/xm_svy
    % Note: dpgdp3_t forecasts x_m(t+1) using info at t, with no look-ahead.
    % For h=1 surveys: dpgdp3_t forecasts x_m(t+1), so for OLS row t
    % we need xSurvey(:,t), i.e. indices 1..T-1 (not 2..T).
    ym_svy = xSurvey(:, 1:end-1)';             % T1 x nxM  (survey at t, forecasting t+1)
    xm_svy = [ones(T1,1), xMacro(:,1:end-1)']; % T1 x (1+nxM)  (same RHS as actual)

    % Stack with weight w_svy
    ym = [ym_act;        w_svy * ym_svy];
    xm = [xm_act;        w_svy * xm_svy];
else
    ym = ym_act;
    xm = xm_act;
end

paramM       = xm \ ym;
Am_star      = paramM(1, :)';                        % nxM x 1
phiP_starmm  = paramM(2:end, :)';                    % nxM x nxM

% Residuals and Omega_starm computed from ACTUAL data only (structural noise)
umt_star     = ym_act - xm_act * paramM;             % T1 x nxM
Omega_starm  = (umt_star' * umt_star) / T1;          % nxM x nxM

fprintf('  Block m: %d actual + %d survey obs (w_svy=%.2f).\n', ...
    T1, T1 * double(w_svy>0), w_svy);
fprintf('  phiP_starmm:\n');  disp(phiP_starmm);

% Block 1: Y1 on lagged Y1 and contemporaneous macro
y1           = Y1(:, 2:end)';                                      % T1 x nxL
x1           = [ones(T1,1), Y1(:,1:end-1)', xMacro(:,2:end)'];   % T1 x (1+nxL+nxM)
param1         = x1 \ y1;
A1_star        = param1(1, :)';                     % nxL x 1  (annual)
phiP_star11    = param1(2:1+nxL, :)';              % nxL x nxL  (lagged Y1)
psiP_star1m    = param1(2+nxL:end, :)';            % nxL x nxM  (contemp. macro, annual)
u1t_star       = y1 - x1 * param1;                 % T1 x nxL
Omega_star1    = (u1t_star' * u1t_star) / T1;      % nxL x nxL  (annual^2)

% Block 2: Y2 on contemporaneous macro and Y1
y2           = Y2(:, 2:end)';                                      % T1 x Ne
x2           = [ones(T1,1), xMacro(:,2:end)', Y1(:,2:end)'];     % T1 x (1+nxM+nxL)
param2       = x2 \ y2;
A2_star      = param2(1, :)';                       % Ne x 1  (annual)
phiP_star2m  = param2(2:1+nxM, :)';                % Ne x nxM
phiP_star21  = param2(2+nxM:end, :)';              % Ne x nxL
u2t_star     = y2 - x2 * param2;                   % T1 x Ne

fprintf('  OLS complete.\n');

%% Recover sigma and phiP analytically
sigma_macro = chol(Omega_starm, 'lower');     % nxM x nxM  lower triangular
sigma       = blkdiag(sigma_macro, eye(nxL)); % nx  x nx

B1l_chol    = chol(Omega_star1, 'lower');     % nxL x nxL  (Cholesky of Y1 resid var)
phiP_mm     = phiP_starmm;                    % nxM x nxM  (from macro OLS)
phiP_ll     = B1l_chol \ phiP_star11 * B1l_chol;  % nxL x nxL  (similarity)
phiP        = blkdiag(phiP_mm, phiP_ll);      % nx  x nx   (block diagonal)

eigsP = abs(eig(phiP));
fprintf('  max|eig(phiP)| = %.4f ', max(eigsP));
if max(eigsP) >= 1
    fprintf('  WARNING: P-dynamics non-stationary.\n');
else
    fprintf('  (stationary)\n');
end
fprintf('  sigma_macro =\n');  disp(sigma_macro);

%% Solve for (Lambda_mm, Lambda_ll, beta) numerically

% OLS targets (all in annual decimal units)
B1m_OLS  = psiP_star1m;                       % nxL x nxM
B2m_OLS  = phiP_star2m + phiP_star21 * B1m_OLS; % Ne  x nxM
B1B1_OLS = Omega_star1;                        % nxL x nxL
B2B1_OLS = phiP_star21 * Omega_star1;          % Ne  x nxL

% Compact maturity struct
mature.exact = matSelect_Y1;
mature.error = matSelect_Y2;

% ---- CMA-ES options (shared by both subproblems) ------------------------
cmaOpts.Display       = 'off';
cmaOpts.Plotting      = 'off';
cmaOpts.Saving        = 0;
cmaOpts.VerboseModulo = 0;
cmaOpts.TolFun        = 1e-14;
cmaOpts.TolX          = 1e-14;
cmaOpts.MaxFunEvals   = 1e6;
cmaOpts.StopOnWarnings = 'no';

% Subproblem A: macro block (Lambda_mm, beta_m)
% Parameter vector: p = [Lambda_mm(:); beta_m]   (nxM^2 + nxM elements)
% SSR = ||local_macroBlock_residuals(p)||^2
%
% SIGN: beta_m[k] sign equals the column-sum sign of B1m_OLS.
% SCALE: |beta_m[k]| ~ max(|B1m_OLS[:,k]|) / (ann * 0.86).

beta_m_sign  = sign(sum(B1m_OLS, 1));                         % 1 x nxM
beta_m_scale = max(abs(B1m_OLS), [], 1)' / (ann * 0.86);     % nxM x 1

% Lower/upper bounds for beta_m, derived from the sign and expected scale.
% For a positive-sign factor: lb = +scale/10, ub = +5*scale  (positive range)
% For a negative-sign factor: lb = -5*scale,  ub = -scale/10 (negative range)
% The floor of scale/10 (away from zero) prevents CMA-ES from collapsing a
% factor to zero. min/max ensure lb < ub regardless of sign.
lb_A = [-5*ones(nxM^2,1); min(beta_m_sign'.*beta_m_scale/10, beta_m_sign'.*beta_m_scale*5)];
ub_A = [ 5*ones(nxM^2,1); max(beta_m_sign'.*beta_m_scale/10, beta_m_sign'.*beta_m_scale*5)];

x0_A   = [zeros(nxM^2,1); beta_m_sign'.*beta_m_scale];
sig0_A = [0.5*ones(nxM^2,1); beta_m_scale];

cmaOpts.LBounds = lb_A;
cmaOpts.UBounds = ub_A;
cmaOpts.PopSize = 4 + floor(3*log(length(x0_A)));

objA = @(p) sum(local_macroBlock_residuals(p, phiP_mm, sigma_macro, ...
                                            B1m_OLS, B2m_OLS, mature, nxM, ann).^2);
[xBestA, fBestA] = cmaes_dsgeDisplay(objA, x0_A, 1, sig0_A, cmaOpts);

Lambda_mm = reshape(xBestA(1:nxM^2), nxM, nxM);
beta_m    = xBestA(nxM^2+1 : end);
phiQ_mm   = phiP_mm - sigma_macro * Lambda_mm;
fprintf('  Macro CMA-ES SSR: %.2e\n', fBestA);
fprintf('  eig(phiQ_mm): %s\n', num2str(abs(eig(phiQ_mm))'));
fprintf('  beta_m: %s\n', num2str(beta_m'));

% Subproblem B: latent block (Lambda_ll_Jordan, beta_l)
% Parameter vector: p = [lev; lb_off; lc_off; beta_l]   (3 + nxL elements)
% Lambda_ll = [[lev, lb_off], [lc_off, lev]]   equal-diagonal Jordan form
% phiQ_ll   = phiP_ll - Lambda_ll
%
% IDENTIFICATION (H-W Propositions 1-2):
%   (a) H-W Proposition 2: for Nl=2, the Jordan parameterisation has
%       two observationally equivalent solutions related by transposing
%       phiQ_ll. Resolved by the canonical condition lb <= lc (i.e. the
%       (1,2) off-diagonal of Lambda_ll <= its (2,1) counterpart).
%   (b) Sign normalisation (H-W Proposition 1): beta_l[k] > 0 for all
%       latent factors. Flipping sign of beta_l[k] and of row/col k of
%       phiQ_ll leaves B loadings unchanged — this is the remaining
%       rotation ambiguity for each factor independently. Resolved by
%       enforcing strict positivity via CMA-ES lower bounds.
%
% SCALE: |beta_l[k]| ~ diag(B1l_chol)[k] / (ann * 0.86)

beta_l_scale = diag(B1l_chol) / (ann * 0.86);   % nxL x 1

% Strictly positive lower bound for beta_l (H-W sign normalisation).
% BETA_L FLOOR: 0.0001 on BOTH entries.
% This blocks the degenerate near-zero basin (beta_l~0) that CMA-ES
% The Jordan swap can permute the two entries, so both must be floored to be safe.
lb_B = [-5; -5; -5; 0.0001; 0.0001];
ub_B = [ 5;  5;  5; beta_l_scale(1)*5;  beta_l_scale(2)*5];

% THREE STARTS for robustness, where we keep the lowest SSR solution.
% Subproblem B inputs (B1B1_OLS, B2B1_OLS, phiP_ll) are identical to NS
% regardless of w_svy, so the V23 solution is always a valid starting point.
%
%   Start 1: canonical form    (b<c satisfied)
%   Start 2: pre-swap form     (b>c; same solution, different orientation)
%   Start 3: scale-based default (broad search fallback)
x0_B_s1   = [-0.0689; 0.1351; 0.1520; 0.00226; 0.00062];
sig0_B_s1 = [0.03;    0.03;   0.03;   0.00050; 0.00015];

x0_B_s2   = [-0.0689; 0.1714; 0.1157; 0.00232; 0.00038];
sig0_B_s2 = [0.03;    0.03;   0.03;   0.00050; 0.00010];

x0_B_s3   = [0; 0; 0; beta_l_scale];
sig0_B_s3 = [0.3; 0.1; 0.1; beta_l_scale];

cmaOpts.LBounds = lb_B;
cmaOpts.UBounds = ub_B;
cmaOpts.PopSize = 4 + floor(3*log(length(x0_B_s1)));

objB = @(p) sum(local_latentBlock_residuals(p, phiP_ll, B1B1_OLS, B2B1_OLS, ...
                                             mature, nxL, ann).^2);

[xB_s1, fB_s1] = cmaes_dsgeDisplay(objB, x0_B_s1, 1, sig0_B_s1, cmaOpts);
[xB_s2, fB_s2] = cmaes_dsgeDisplay(objB, x0_B_s2, 1, sig0_B_s2, cmaOpts);
[xB_s3, fB_s3] = cmaes_dsgeDisplay(objB, x0_B_s3, 1, sig0_B_s3, cmaOpts);

fprintf('  Latent SSR: s1=%.2e  s2=%.2e  s3=%.2e\n', fB_s1, fB_s2, fB_s3);

[fBestB, iBest] = min([fB_s1, fB_s2, fB_s3]);
xBestBs = {xB_s1, xB_s2, xB_s3};
xBestB  = xBestBs{iBest};
fprintf('  Latent best: start %d (SSR=%.2e)\n', iBest, fBestB);

lev    = xBestB(1);
lb     = xBestB(2);
lc     = xBestB(3);
beta_l = xBestB(4:5);

Lambda_ll = [lev, lb; lc, lev];
phiQ_ll   = phiP_ll - Lambda_ll;

% Enforce H-W Proposition 2 canonical form: phiQ_ll[1,2] <= phiQ_ll[2,1]
%
% the canonical condition is on phiQ_ll itself, NOT on the
% Lambda_ll entries lb and lc. Since phiQ_ll = phiP_ll - Lambda_ll, and
% phiP_ll has non-zero off-diagonals, lb <= lc does NOT imply
% phiQ_ll[1,2] <= phiQ_ll[2,1] in general.
%
% The two equivalent solutions are phiQ_ll and phiQ_ll' (transpose),
% with beta_l permuted accordingly. We keep the representative
% with phiQ_ll[1,2] <= phiQ_ll[2,1].
if phiQ_ll(1,2) > phiQ_ll(2,1)
    phiQ_ll = phiQ_ll';          % transpose gives the other equivalent form
    beta_l  = beta_l([2,1]);     % permute beta_l to match
    % Recompute Lambda_ll from the canonical phiQ_ll
    Lambda_ll = phiP_ll - phiQ_ll;
end
fprintf('  Latent CMA-ES SSR: %.2e\n', fBestB);
fprintf('  eig(phiQ_ll): %s\n', num2str(abs(eig(phiQ_ll))'));
fprintf('  beta_l: %s  (both > 0: %d)\n', num2str(beta_l'), all(beta_l > 0));
fprintf('  Jordan b<=c: b=%.4f, c=%.4f  (satisfied: %d)\n', ...
        phiQ_ll(1,2), phiQ_ll(2,1), phiQ_ll(1,2) <= phiQ_ll(2,1) + 1e-10);

% Assemble full model matrices (block-diagonal: Q-independence)
phiQ = blkdiag(phiQ_mm, phiQ_ll);
beta = [beta_m; beta_l];

eigsQ = abs(eig(phiQ));
fprintf('  eig(phiQ) (moduli): %s\n', num2str(sort(eigsQ,'descend')'));
fprintf('  beta = %s\n', num2str(beta'));

%% Compute B loadings for downstream steps

[B1m_ann, ~, B1l_ann, B2l_ann] = local_computeB_ann(phiQ, beta, mature, nxM, nxL, ann);

%% Solve for (muQ, alpha) numerically

% Solve:  A2_star - A2_ann(alpha,muQ) + B2l_ann / B1l_ann * A1_ann = 0
% for x = [alpha; muQ_macro]   (muQ_latent = 0 by normalisation)

muQ = zeros(nx, 1);   % muQ_latent = 0 by normalisation; muQ_macro solved below

% Parameter vector: x = [alpha; muQ_macro]   (1 + nxM elements)
%
% ALPHA BOUNDS: narrow to economically plausible range.
%   H-W delta0 (alpha) = -0.0082 quarterly which is around r0 = -3.3% ann?
%   The unconditional short rate is E[r] = ann*(alpha + beta'*(I-phiP)^{-1}*muP).
%   With E[y_1yr] ≈ 4.86%, alpha should be in roughly -0.02 to +0.025 quarterly.
%   We use [-0.03, +0.03] to give CMA-ES adequate room without allowing
%   alpha to go to -0.05  which results in degenerate solution that occurs with wider bounds.
%
% UNCONDITIONAL MEAN SOFT CONSTRAINT:
%   The A-intercept moment conditions alone do not uniquely pin down the
%   (alpha, muQ) decomposition: many combinations give the same A2_star fit
%   but produce wildly different unconditional yield levels (we observed
%   alpha = -0.05, E[r] = -7.3% when the data mean is +4.9%).
%   We add a soft penalty that anchors the unconditional short rate:
%     penalty = w * (ann*(alpha + beta'*(I-phiP)^{-1}*muP) - mean_y1yr)^2
%   where w is chosen so the penalty is comparable to the A2_star residuals.
%   This does NOT change what is being estimated (it is not a hard constraint)
%   but prevents CMA-ES from settling on degenerate (alpha, muQ) pairs.

mean_y1yr  = mean(data(1,:));              % mean 1yr yield (annual decimal)

x0_7   = [mean_y1yr/ann; zeros(nxM,1)];   % alpha start: mean short rate / ann
sig0_7 = [0.005; 0.3*ones(nxM,1)];

% TIGHTENED muQ BOUNDS (from Survey experience):
%   muQ = c^Q: the constant market price of risk intercept.

%   Allowing ±50 lets Step 7 find
%   large-muQ solutions that imply huge muP_l.
%   Bounding at ±3 prevents this while giving 30× the HW magnitude.
lb_7   = [-0.03; -3*ones(nxM,1)];
ub_7   = [ 0.03;  3*ones(nxM,1)];

cmaOpts7          = cmaOpts;
cmaOpts7.LBounds  = lb_7;
cmaOpts7.UBounds  = ub_7;
cmaOpts7.PopSize  = 4 + floor(3*log(length(x0_7)));
cmaOpts7.TolFun   = 1e-10;
cmaOpts7.TolX     = 1e-10;

% we have three soft constraints
%
% Constraint 1 — alpha anchor (existing):
%   Penalise alpha deviating from mean_y1yr/ann.
%   Prevents alpha drifting to boundaries while muQ compensates.
%
% Constraint 2 — muQ shrinkage (from Survey V7):
%   Penalise large |muQ_macro|.  muQ enters A_ann through the recursion;
%   large muQ -> large A_ann -> large muP_l -> large E[x_latent] -> level wrong.
%   Even with tight bounds, shrinkage prevents muQ from sitting at ±3.
%
% Constraint 3 — FULL UNCONDITIONAL YIELD LEVEL:
%   Embed the Step 8 muP_l recovery inside the objective so that
%   E[r] = ann*(alpha + beta'*(I-phiP)^{-1}*muP) can be directly penalised.
%   This breaks the circular dependency between Step 7 and Step 8 by
%   computing the implied muP_l forward from any candidate (alpha, muQ).
%   We anchor at the 10yr yield mean (data mean = 5.97%) because the 1yr
%   is already anchored by Constraint 1, and the 10yr level captures the
%   long-end slope problem most directly.
%
%   Implementation: for each candidate x=[alpha;muQ], compute A1_ann(x),
%   recover muP_l analytically, then compute E[r] and E[y_10yr] and penalise.
%
%   A2 residuals scaled by 1e6  -> w_mean and w_muQ in same units
%   Yield level in decimal       -> w_level scales to match 1e6-scaled residuals

w_mean  = 1e6;    % alpha anchor weight
w_muQ   = 1e4;    % muQ shrinkage weight
w_level = 5e5;    % level anchor weight (0.01% deviation = 1e-4 decimal * 5e5 = 50 cost)

IminusPhiP_ll = eye(nxL) - phiP_ll;   % precompute for speed

% muP_m = Am_star (AP2003 independence: macro intercept = Block m OLS intercept).
% Defined here so it is available inside the step7_objective closure;
% Step 8 will also assign it explicitly for clarity.
muP_m = Am_star;

obj7 = @(x) step7_objective(x, mature, phiQ, beta, sigma, ...
                              A2_star, A1_star, B1l_ann, B2l_ann, B1l_chol, ...
                              phiP_ll, IminusPhiP_ll, muP_m, ...
                              nxM, nxL, ann, mean_y1yr, ...
                              w_mean, w_muQ, w_level);

[xBest7, fBest7] = cmaes_dsgeDisplay(obj7, x0_7, 0.5, sig0_7, cmaOpts7);

alpha      = xBest7(1);
muQ(1:nxM) = xBest7(2:end);

fprintf('  CMA-ES SSR: %.2e\n', fBest7);
fprintf('  alpha = %.6f  (quarterly; r0 = %.4f%%)\n', alpha, alpha*ann*100);
fprintf('  muQ_macro = %s\n', num2str(muQ(1:nxM)'));

%% Recover muP analytically

% With the AP2003 independence assumption (phiP_ml = phiP_lm = 0):
%
%   muP_m = Am_star   (macro intercept from block m OLS; note xMacro has
%                      zero mean by construction so Am_star ≈ 0)
%   muP_l = B1l_ann \ (A1_star - A1_ann + B1l_ann * phiP_ll / B1l_ann * A1_ann)
%   muP   = [muP_m; muP_l]

[A1_ann, ~] = local_computeA_ann(alpha, muQ, phiQ, beta, sigma, mature, nxM, nxL, ann);

% muP_m = Am_star (independence: zero cross-block correction)
% muP_l uses B1l_chol (same rotation as phiP_ll, so frames are consistent)
muP_m = Am_star;
muP_l = B1l_chol \ (A1_star - A1_ann + B1l_chol * phiP_ll / B1l_chol * A1_ann);
muP   = [muP_m; muP_l];

fprintf('  muP = %s\n', num2str(muP'));

%% Compute bond pricing loadings (g0, gx, A, B)

% Bond pricing recursion (quarterly periods)
%
%   k=1:  A(1,1) = -alpha,    B(:,1) = -beta
%   k>1:  A(1,k) = -alpha + A(1,k-1) + B(:,k-1)'*muQ
%                  + 0.5*B(:,k-1)'*sigma2*B(:,k-1)
%         B(:,k) = -beta + phiQ'*B(:,k-1)
%
% Here A and B store raw (per-quarter, unnormalised) recursion coefficients.
% Annualised yield loadings at maturity n quarters:
%   g0(n) = ann * (-A(1,n) / n)
%   gx(n) = ann * (-B(:,n)' / n)

maxMat = max(matSelect);
A      = zeros(1, maxMat);    % 1 x maxMat  (matches 2F convention)
B      = zeros(nx, maxMat);   % nx x maxMat

sigma2 = sigma * sigma';      % nx x nx  innovation covariance

for k = 1:maxMat
    if k == 1
        A(1,k) = -alpha;
        B(:,k) = -beta;
    else
        A(1,k) = -alpha + A(1,k-1) + B(:,k-1)'*muQ + ...
                  0.5*B(:,k-1)'*sigma2*B(:,k-1);
        B(:,k) = -beta + phiQ'*B(:,k-1);
    end
end

% Annualised yield loadings at the selected maturities
% Matches 2F convention:  g0 (numObs x 1),  gx (numObs x nx)
g0 = ann * (-A(1, matSelect)' ./ matSelect');                    % numObs x 1
gx = ann * (-B(:, matSelect)' ./ repmat(matSelect', 1, nx));     % numObs x nx

% Short-rate loadings (annualised) — used in yield decomposition
r0 = ann * alpha;
rx = ann * beta';   % 1 x nx

% Pack model struct
model.g0        = g0;
model.gx        = gx;
model.muP       = muP;
model.phiP      = phiP;
model.alpha     = alpha;
model.beta      = beta;
model.phiQ      = phiQ;
model.muQ       = muQ;
model.sigma     = sigma;
model.A         = A;
model.B         = B;
model.r0        = r0;
model.rx        = rx;
model.matSelect = matSelect;
model.nxM       = nxM;
model.nxL       = nxL;


%% Kalman filter

% Measurement equation:  data_t = g0 + gx * x_t + stdY * v_t
%   Y1 rows: measurement error = 0  (priced exactly)
%   Y2 rows: common scalar measurement error stdY
%
% Transition:  x_{t+1} = muP + phiP * x_t + sigma * u_{t+1}


stdY = mean(std(u2t_star));   % scalar common measurement error std

% Measurement noise covariance Rv (numObs x numObs):
%   zero for Y1 rows, stdY^2 * I for Y2 rows
Rv = zeros(numObs, numObs);
Rv(idxError, idxError) = eye(Ne) * stdY^2;

outKF = local_KalmanFilter(data, g0, gx, Rv, muP, phiP, sigma2);
outKF.model = model;

fprintf('  Average log-likelihood: %.6f\n', outKF.sumLogL / T);

%% Display parameters

fprintf('\n=============================================================\n');
fprintf(' Parameter Summary\n');
fprintf('=============================================================\n');
fprintf('\n--- Short rate ---\n');
fprintf('  alpha = %.6f  (quarterly; r0 = %.4f%%)\n', alpha, r0*100);
fprintf('  beta  = %s\n', num2str(beta'));
fprintf('\n--- Q-dynamics diagonal ---\n');
fprintf('  diag(phiQ) = %s\n', num2str(diag(phiQ)'));
fprintf('\n--- P-dynamics diagonal ---\n');
fprintf('  diag(phiP) = %s\n', num2str(diag(phiP)'));
fprintf('\n--- muP ---\n');
fprintf('  muP = %s\n', num2str(muP'));
fprintf('\n--- muQ ---\n');
fprintf('  muQ = %s\n', num2str(muQ'));
fprintf('\n--- sigma ---\n');  disp(sigma);
fprintf('--- stdY = %.6f  (annualised: %.2f bps)\n', stdY, stdY*ann*10000);

%% Pricing errors
yHat        = outKF.g0 + outKF.gx * outKF.xHat;   % numObs x T
merrors     = outKF.data - yHat;                   % numObs x T  (decimal)
merrors_bps = merrors * 10000;                      % basis points

matYears_display = matSelect / ann;

fprintf('\n%-8s  %10s  %10s  %10s  %10s\n', ...
    'Mat(yr)', 'Mean(bps)', 'Std(bps)', 'RMSE(bps)', 'MaxAbs(bps)');
fprintf('%s\n', repmat('-', 1, 54));
for i = 1:numObs
    e = merrors_bps(i, :);
    fprintf('%-8.1f  %10.3f  %10.3f  %10.3f  %10.3f\n', ...
        matYears_display(i), mean(e), std(e), sqrt(mean(e.^2)), max(abs(e)));
end
fprintf('%-8s  %10.3f  %10.3f  %10.3f  %10.3f\n', 'Overall', ...
    mean(merrors_bps(:)), std(merrors_bps(:)), ...
    sqrt(mean(merrors_bps(:).^2)), max(abs(merrors_bps(:))));

%% Yield fit plot
colours = lines(numObs);
nCols   = 3;
nRows   = ceil(numObs / nCols);

figure('Name', 'Yield fit: actual vs model-implied', ...
       'Units', 'normalized', 'Position', [0.05 0.05 0.90 0.85]);
for i = 1:numObs
    subplot(nRows, nCols, i);
    actual = outKF.data(i, :) * 100;
    fitted = yHat(i, :)       * 100;
    plot(dateVec, actual, 'Color', [0.6 0.6 0.6], 'LineWidth', 2.0); hold on;
    plot(dateVec, fitted, 'Color', colours(i, :),  'LineWidth', 0.5);
    hold off;
    title(sprintf('%g-year yield', matYears_display(i)), 'FontSize', 9);
    ylabel('Yield (%)', 'FontSize', 8);
    xlabel('');
    xlim([dateVec(1) dateVec(end)]);
    legend({'Actual', 'Model'}, 'Location', 'best', 'FontSize', 7);
    grid on; box on;
    set(gca, 'FontSize', 8);
end
sgtitle('US Yield Curve — Actual vs Model-Implied Yields', ...
        'FontSize', 11, 'FontWeight', 'bold');
exportgraphics(gcf, 'Graphs/YieldFitPlot_Survey_1Q.pdf');

figure('Name', 'Pricing errors (basis points)', ...
       'Units', 'normalized', 'Position', [0.05 0.05 0.90 0.85]);
for i = 1:numObs
    subplot(nRows, nCols, i);
    plot(dateVec, merrors_bps(i, :), 'Color', colours(i, :), 'LineWidth', 1.2);
    hold on;
    yline(0, 'k--', 'LineWidth', 0.8);
    hold off;
    title(sprintf('%g-year: RMSE = %.2f bps', matYears_display(i), ...
          sqrt(mean(merrors_bps(i,:).^2))), 'FontSize', 9);
    ylabel('Error (bps)', 'FontSize', 8);
    xlim([dateVec(1) dateVec(end)]);
    grid on; box on;
    set(gca, 'FontSize', 8);
end
sgtitle('US Yield Curve — Pricing Errors (Actual minus Model-Implied)', ...
        'FontSize', 11, 'FontWeight', 'bold');
exportgraphics(gcf, 'Graphs/PricingErrorPlot_Survey_1Q.pdf');

%% Factor plot
figure('Name', 'State factors', 'Units', 'normalized', ...
       'Position', [0.05 0.10 0.88 0.70]);
factorTitles  = {'Inflation factor', 'Unemployment factor', ...
                 'Latent factor 1',  'Latent factor 2'};
factorColours = {'b','r','k','m'};
for i = 1:nx
    subplot(2, 2, i);
    plot(dateVec, outKF.xHat(i,:), factorColours{i}, 'LineWidth', 1.2);
    title(factorTitles{i}, 'FontSize', 9);
    ylabel('Std. units', 'FontSize', 8);
    xlim([dateVec(1) dateVec(end)]);
    grid on; box on;
    set(gca, 'FontSize', 8);
end
sgtitle('4-Factor MF-GATSM: State Factors', 'FontSize', 11, 'FontWeight', 'bold');
exportgraphics(gcf, 'Graphs/FactorsPlot_Survey_1Q.pdf');

%% LaTeX parameter table

texFilename = 'Tables/GATSM_4F_MacroFinance_Survey_params_1Q.tex';
fid = fopen(texFilename, 'w');
fprintf(fid, '\\begin{table}[htbp]\n\\centering\n');
fprintf(fid, '\\caption{Estimated parameters -- 4-factor Macro-Finance GATSM}\n');
fprintf(fid, '\\label{tab:gatsm4f_estimates}\n');
fprintf(fid, '\\begin{tabular}{lc}\n\\toprule\n');
fprintf(fid, ' & MCSE \\\\\n & (1) \\\\\n\\midrule\n');
factorLabels = {'Infl.','Unempl.','Latent 1','Latent 2'};
for i = 1:nx
    fprintf(fid, '$\\phi^Q_{%d%d}$ (%s) & %.6f \\\\\n', i,i,factorLabels{i},phiQ(i,i));
end
fprintf(fid, '$\\alpha$ & %.6f \\\\\n', alpha);
for i = 1:nx
    fprintf(fid, '$\\beta_%d$ & %.6f \\\\\n', i, beta(i));
end
for i = 1:nx
    fprintf(fid, '$\\mu^P_%d$ & %.6f \\\\\n', i, muP(i));
end
for i = 1:nx
    fprintf(fid, '$\\Phi^P_{%d%d}$ & %.6f \\\\\n', i,i,phiP(i,i));
end
for i = 1:nx
    fprintf(fid, '$\\Sigma_{%d%d}$ & %.6f \\\\\n', i,i,sigma(i,i));
end
fprintf(fid, '$\\sigma_v$ & %.6f \\\\\n', stdY);
fprintf(fid, '\\midrule\n');
fprintf(fid, '$\\frac{\\log \\mathcal{L}}{T}$ & %.6f \\\\\n', outKF.sumLogL/T);
fprintf(fid, '\\bottomrule\n\\end{tabular}\n\\end{table}\n');
fclose(fid);
fprintf('\nLaTeX table written to: %s\n', texFilename);

%% Save Data
%save('GATSM_4F_MacroFinance_Results_KO_proper_V6.mat', ...
 %    'model','outKF','phiQ','phiP','muQ','muP','alpha','beta', ...
  %   'sigma','stdY','g0','gx','A','B','r0','rx', ...
   %  'data','matSelect','numObs','T','dateVec', ...
    % 'xMacro','macro_mean','macro_std', ...
     %'w_svy','-v7.3');


%% Yield curve decomposition  (matches local_yieldCurveDecom)

resDecom = local_yieldCurveDecom(outKF, model);

matYearsDecom = resDecom.matSelect / ann;
ny            = length(resDecom.matSelect);

fprintf('\n=== Yield Curve Decomposition: time-averaged values (%%)\n');
fprintf('%-10s  %12s  %12s  %12s\n', ...
    'Mat(yr)', 'FittedYield', 'ExpShortRate', 'TermPremium');
fprintf('%s\n', repmat('-', 1, 52));
for i = 1:ny
    fprintf('%-10.1f  %12.4f  %12.4f  %12.4f\n', ...
        matYearsDecom(i), ...
        mean(resDecom.yHat(:,i))       * 100, ...
        mean(resDecom.rExp(:,i))       * 100, ...
        mean(resDecom.termPremia(:,i)) * 100);
end

figure('Name', 'Term Premia', 'Units', 'normalized', ...
       'Position', [0.05 0.05 0.90 0.85]);
nCols3 = 3;  nRows3 = ceil(ny / nCols3);  colours3 = lines(ny);
for i = 1:ny
    subplot(nRows3, nCols3, i);
    plot(dateVec, resDecom.termPremia(:,i)*100, 'Color',colours3(i,:),'LineWidth',1.2);
    hold on; yline(0,'k--','LineWidth',0.8); hold off;
    title(sprintf('%g-year term premium', matYearsDecom(i)),'FontSize',12);
    ylabel('Term premium (%)','FontSize',12);
    xlim([dateVec(1) dateVec(end)]); grid on; box on; set(gca,'FontSize',12);
end
sgtitle('US Yield Curve — Term Premia','FontSize',11,'FontWeight','bold');
exportgraphics(gcf,'Graphs/TermPremiaPlot_Survey_1Q.pdf');

selMats = [2 5 10];
selIdx  = arrayfun(@(m) find(resDecom.matSelect==m*ann,1,'first'), ...
                   selMats(ismember(selMats*ann,resDecom.matSelect)));
if ~isempty(selIdx)
    figure('Name','Yield decomposition: selected maturities', ...
           'Units','normalized','Position',[0.05 0.05 0.90 0.75]);
    for k = 1:length(selIdx)
        i = selIdx(k);
        subplot(1,length(selIdx),k);
        plot(dateVec,resDecom.yHat(:,i)*100,'k','LineWidth',1.4); hold on;
        plot(dateVec,resDecom.rExp(:,i)*100,'b--','LineWidth',1.2);
        plot(dateVec,resDecom.termPremia(:,i)*100,'r:','LineWidth',1.2);
        hold off;
        title(sprintf('%g-year yield',matYearsDecom(i)),'FontSize',12);
        ylabel('Rate (%)','FontSize',12);
        xlim([dateVec(1) dateVec(end)]);
        legend({'Fitted yield','Exp. short rate','Term premium'}, ...
               'Location','best','FontSize',12);
        grid on; box on; set(gca,'FontSize',12);
    end
    sgtitle('US Yield Decomposition','FontSize',14,'FontWeight','bold');
    exportgraphics(gcf,'Graphs/YieldDecompositionPlot_Survey_1Q.pdf');
end

%% Campbell-Schiller regression  (matches 2F Step 9 exactly)

m_CS = ann;   % CS step size = 1 year = 4 quarters

yields_raw_full = csvread('US_monthly_yields_Jan1972_Dec2025.csv');
mats_full_yrs   = csvread('US_monthly_yields_Jan1972_Dec2025_maturities.csv');
mats_full_qtrs  = mats_full_yrs * ann;
annualMats      = m_CS : m_CS : max(mats_full_qtrs);
annualMats      = annualMats(ismember(annualMats, mats_full_qtrs));
T_cs            = length(qIdx);
dataCS_actual   = zeros(T_cs, length(annualMats));
for i = 1:length(annualMats)
    col = find(mats_full_yrs == annualMats(i)/ann);
    dataCS_actual(:,i) = yields_raw_full(qIdx, col) / 100;
end
dataCS_model = zeros(T_cs, length(annualMats));
for t = 1:T_cs
    dataCS_model(t,:) = interp1(resDecom.matSelect, resDecom.yHat(t,:), ...
                                annualMats, 'linear', 'extrap');
end

resCS_actual = local_campbellSchiller(dataCS_actual, m_CS);
resCS_model  = local_campbellSchiller(dataCS_model,  m_CS);

mats_CS_yr = annualMats / ann;
fprintf('\n=== Campbell-Schiller Regression: beta coefficients ===\n');
fprintf('  (EH implies beta = 1; beta < 1 means time-varying term premia)\n\n');
fprintf('%-10s  %10s  %10s  %10s  %10s  %10s  %10s\n', ...
    'Mat(yr)','Act.beta','Act.SE','Act.t','Mod.beta','Mod.SE','Mod.t');
fprintf('%s\n', repmat('-',1,72));
for i = 2:length(annualMats)
    fprintf('%-10.0f  %10.4f  %10.4f  %10.3f  %10.4f  %10.4f  %10.3f\n', ...
        mats_CS_yr(i), ...
        resCS_actual.CSbetta(2,i), resCS_actual.CSbetta_se(2,i), ...
        resCS_actual.CSBetta_tstat(2,i), ...
        resCS_model.CSbetta(2,i),  resCS_model.CSbetta_se(2,i), ...
        resCS_model.CSBetta_tstat(2,i));
end

figure('Name','Campbell-Schiller betas','Units','normalized', ...
       'Position',[0.10 0.15 0.75 0.55]);
plotMats = mats_CS_yr(2:end);
beta_act = resCS_actual.CSbetta(2,2:end);
beta_mod = resCS_model.CSbetta(2,2:end);
ci95_act = resCS_actual.CSBetta_CI95(:,2:end);
ci95_mod = resCS_model.CSBetta_CI95(:,2:end);
fill([plotMats,fliplr(plotMats)],[ci95_act(1,:),fliplr(ci95_act(2,:))], ...
     [0.7 0.7 0.7],'FaceAlpha',0.4,'EdgeColor','none'); hold on;
fill([plotMats,fliplr(plotMats)],[ci95_mod(1,:),fliplr(ci95_mod(2,:))], ...
     [0.6 0.8 1.0],'FaceAlpha',0.4,'EdgeColor','none');
plot(plotMats,beta_act,'k-o','LineWidth',1.6,'MarkerSize',5,'DisplayName','Actual yields');
plot(plotMats,beta_mod,'b--s','LineWidth',1.4,'MarkerSize',5,'DisplayName','Model-implied yields');
yline(1,'r--','LineWidth',1.2,'DisplayName','EH benchmark (\beta=1)');
yline(0,'k:','LineWidth',0.8,'HandleVisibility','off');
hold off;
xlabel('Maturity (years)','FontSize',10); ylabel('\beta coefficient','FontSize',10);
title('Campbell-Schiller Regression — US Yields','FontSize',11,'FontWeight','bold');
legend('95% CI (actual)','95% CI (model)','Actual yields', ...
       'Model-implied','EH benchmark','Location','best','FontSize',8);
grid on; box on; set(gca,'FontSize',9);
xlim([plotMats(1)-0.2 plotMats(end)+0.2]);
exportgraphics(gcf,'Graphs/CS_Survey_1Q.pdf');

%% Local Functions

function F = local_macroBlock_residuals(para, phiP_mm, sigma_macro, ...
                                         B1m_OLS, B2m_OLS, mature, nxM, ann)
% Residuals for the macro sub-problem in MCSE Step 2.
% para = [Lambda_mm(:); beta_m]  (nxM^2 + nxM elements)
% phiQ_mm = phiP_mm - sigma_macro * Lambda_mm
Lambda_mm = reshape(para(1:nxM^2), nxM, nxM);
beta_m    = para(nxM^2+1 : end);

phiQ_mm = phiP_mm - sigma_macro * Lambda_mm;
% No stationarity constraint on phiQ_mm. The Q-measure dynamics
% can legitimately have |eig(phiQ_mm)| > 1 
% The B-loading recursion stays finite for
% finite maturities regardless of Q-measure stationarity.

% B macro columns only depend on phiQ_mm and beta_m (Q-independence)
mats = [mature.exact, mature.error];
B_mac = zeros(length(mats), nxM);
phiQt = phiQ_mm';
for im = 1:length(mats)
    n = mats(im); S = zeros(nxM,1); P = eye(nxM);
    for j = 0:n-1; S = S + P*beta_m; P = P*phiQt; end
    B_mac(im,:) = (ann * S / n)';
end

% if B loadings diverged (phiQ_mm very explosive), return large penalty
if ~all(isfinite(B_mac(:)))
    nEq = numel(B1m_OLS) + numel(B2m_OLS);
    F = 1e6 * ones(nEq, 1);
    return;
end
nE    = length(mature.exact);
B1m_s = B_mac(1:nE, :);
B2m_s = B_mac(nE+1:end, :);

F = [(B1m_OLS - B1m_s) * 1e5; (B2m_OLS - B2m_s) * 1e5];
F = F(:);
end

function F = local_latentBlock_residuals(para, phiP_ll, B1B1_OLS, B2B1_OLS, ...
                                          mature, nxL, ann)
% Residuals for the latent sub-problem in MCSE Step 2.
% para = [lev; lb; lc; beta_l]  (3 + nxL elements)
% Lambda_ll = [[lev, lb], [lc, lev]]   (equal-diagonal Jordan form)
% phiQ_ll   = phiP_ll - Lambda_ll
lev = para(1); lb = para(2); lc = para(3);
beta_l = para(4:end);

Lambda_ll = [lev, lb; lc, lev];
phiQ_ll   = phiP_ll - Lambda_ll;

% HYBRID CONSTRAINT
%   Latent Q-block: enforce stationarity |eig(phiQ_ll)| < 0.9999.

%   Without this constraint the landscape has degenerate local minima where
%   phiQ_ll is explosive and beta_l[2] collapses to zero, making the second
%   latent factor unidentified. The stationarity penalty prevents this.
if any(abs(eig(phiQ_ll)) >= 0.9999)
    nEq = nxL*(nxL+1)/2 + numel(B2B1_OLS);
    F = 1e8 * ones(nEq, 1);
    return;
end

mats = [mature.exact, mature.error];
B_lat = zeros(length(mats), nxL);
phiQt = phiQ_ll';
for im = 1:length(mats)
    n = mats(im); S = zeros(nxL,1); P = eye(nxL);
    for j = 0:n-1; S = S + P*beta_l; P = P*phiQt; end
    B_lat(im,:) = (ann * S / n)';
end

% if B loadings diverged (phiQ_ll very explosive) return large penalty
if ~all(isfinite(B_lat(:)))
    nEq = nxL*(nxL+1)/2 + numel(B2B1_OLS);
    F = 1e8 * ones(nEq, 1);
    return;
end
nE    = length(mature.exact);
B1l_s = B_lat(1:nE, :);
B2l_s = B_lat(nE+1:end, :);

% Symmetric part of B1l*B1l'
M1 = B1B1_OLS - B1l_s * B1l_s';
r1 = M1([1,2,4]) * 1e5;    % [M(1,1) M(2,1) M(2,2)] = upper-left, lower-left, lower-right

M2 = B2B1_OLS - B2l_s * B1l_s';
r2 = M2(:) * 1e5;

F = [r1(:); r2(:)];
end

function [B1m, B2m, B1l, B2l] = local_computeB_ann(phiQ, beta, mature, nxM, nxL, ann)
% Compute ANNUALISED B yield loadings for the MCSE moment conditions.
%
% Annual yield loading:  gx(n) = ann * (1/n)*sum_{j=0}^{n-1}(phiQ')^j * beta
%
% This returns B in annual decimal units, matching the OLS targets
% psiP_star1m and Omega_star1 which come from regressions on annual yields.
%
% Returns B partitioned into macro (m) and latent (l) column blocks,
% for maturities in mature.exact (Y1) and mature.error (Y2).

nx   = nxM + nxL;
mats = [mature.exact, mature.error];

B_all  = zeros(length(mats), nx);
phiQt  = phiQ';
for im = 1:length(mats)
    n   = mats(im);
    S   = zeros(nx, 1);
    pow = eye(nx);
    for j = 0:n-1
        S   = S + pow * beta;
        pow = pow * phiQt;
    end
    % Multiply by ann to convert per-quarter to annual units
    B_all(im,:) = (ann * S / n)';
end

nE = length(mature.exact);
B1 = B_all(1:nE, :);
B2 = B_all(nE+1:end, :);

B1m = B1(:, 1:nxM);
B1l = B1(:, nxM+1:end);
B2m = B2(:, 1:nxM);
B2l = B2(:, nxM+1:end);
end

function [A1, A2] = local_computeA_ann(alpha, muQ, phiQ, beta, sigma, ...
                                        mature, nxM, nxL, ann)
% Compute ANNUALISED A yield intercepts.
%
% Annual intercept: g0(n) = ann * (-A_bar(n)/n)

sigma2 = sigma * sigma';
nMax   = max([mature.exact, mature.error]);

% Per-quarter recursion (same sign convention as main script Step 9)
A_raw    = zeros(1, nMax);
A_bar    = -alpha;
B_bar    = -beta;
A_raw(1) = -A_bar;    % = alpha

for k = 2:nMax
    A_bar    = -alpha + A_bar + B_bar'*muQ + 0.5*(B_bar'*sigma2*B_bar);
    B_bar    = phiQ' * B_bar - beta;
    A_raw(k) = -A_bar / k;
end

% Annualise
A_ann = ann * A_raw;

A1 = A_ann(mature.exact)';   % nxL x 1
A2 = A_ann(mature.error)';   % Ne  x 1
end

function F = local_alphaMuQ_residuals(x, mature, phiQ, beta, sigma, ...
                                       A2_star, B1l_ann, B2l_ann, nxM, nxL, ann)
% Residuals of MCSE Step 4:  A2_star - A2_ann + B2l_ann/B1l_ann * A1_ann = 0
% All quantities in ANNUAL decimal units.
nx    = nxM + nxL;
alpha = x(1);
muQ   = zeros(nx, 1);
muQ(1:nxM) = x(2:end);
try
    [A1_ann, A2_ann] = local_computeA_ann(alpha, muQ, phiQ, beta, sigma, ...
                                           mature, nxM, nxL, ann);
    F = (A2_star - A2_ann + B2l_ann / B1l_ann * A1_ann) * 1e6;
catch
    F = 1e10 * ones(length(mature.error), 1);
end
end

function out = local_KalmanFilter(y, g0, gx, Rv, h0, hx, Reps)
% Standard Kalman filter
% Measurement:  y_t     = g0 + gx*x_t     + Sv*v_t
% Transition:   x_{t+1} = h0 + hx*x_t + Seps*w_{t+1}

[ny, T] = size(y);
nx      = size(hx, 1);

x0   = (eye(nx) - hx) \ h0;
vecP = (eye(nx^2) - kron(hx,hx)) \ reshape(Reps, nx^2, 1);
P0   = reshape(vecP, nx, nx);

xHat      = zeros(nx, T);
xBar      = zeros(nx, T);
PHat      = zeros(nx, nx, T);
SxHat     = zeros(nx, nx, T);
PBar      = zeros(nx, nx, T);
VaryBar   = zeros(ny, ny, T);
yBar      = zeros(ny, T);
logL      = zeros(T, 1);
K         = zeros(nx, ny, T);
predError = zeros(ny, T);

for t = 1:T
    if t == 1
        xBar(:,t)   = h0 + hx*x0;
        PBar(:,:,t) = hx*P0*hx' + Reps;
    else
        xBar(:,t)   = h0 + hx*xHat(:,t-1);
        PBar(:,:,t) = hx*PHat(:,:,t-1)*hx' + Reps;
    end
    yBar(:,t)      = g0 + gx*xBar(:,t);
    VaryBar(:,:,t) = gx*PBar(:,:,t)*gx' + Rv;

    selectY    = ~isnan(y(:,t));
    ny_t       = sum(selectY);
    invVaryBar = VaryBar(selectY, selectY, t) \ eye(ny_t, ny_t);

    K(:, selectY, t)  = PBar(:,:,t) * gx(selectY,:)' * invVaryBar;
    predError(:,t)    = y(:,t) - yBar(:,t);
    xHat(:,t)         = xBar(:,t) + K(:,selectY,t) * predError(selectY,t);
    PHat(:,:,t)       = PBar(:,:,t) - ...
                        K(:,selectY,t)*VaryBar(selectY,selectY,t)*K(:,selectY,t)';

    % Ensure PHat stays symmetric and numerically positive definite.
    % Symmetrise first to eliminate floating-point asymmetry.
    PHat(:,:,t) = (PHat(:,:,t) + PHat(:,:,t)') / 2;
    % chol with two outputs returns a *partial* factor (p x p) when the
    % matrix is not PD, which would cause a size mismatch on assignment.
    [Ltmp, info] = chol(PHat(:,:,t), 'lower');
    if info == 0
        SxHat(:,:,t) = Ltmp;
    else
        % PHat is not PD: add a ridge until it is
        ridge = 1e-12;
        for iTry = 1:20
            [Ltmp, info] = chol(PHat(:,:,t) + ridge*eye(nx), 'lower');
            if info == 0
                SxHat(:,:,t) = Ltmp;
                break;
            end
            ridge = ridge * 100;
        end
    end

    logL(t,1) = -(ny_t/2)*log(2*pi) ...
                - 0.5*log(det(VaryBar(selectY,selectY,t))) ...
                - 0.5*predError(selectY,t)'*invVaryBar*predError(selectY,t);
end
sumLogL = sum(logL, 1);

out.data      = y;
out.xHat      = xHat;
out.PHat      = PHat;
out.Sx        = SxHat;
out.logL      = logL;
out.sumLogL   = sumLogL;
out.Kgain     = K;
out.predError = predError;
out.VaryBar   = VaryBar;
out.xBar      = xBar;
out.PBar      = PBar;
out.g0        = g0;
out.gx        = gx;
out.h0        = h0;
out.hx        = hx;
out.Rv        = Rv;
out.Reps      = Reps;
out.yBar      = yBar;
end

function res = local_yieldCurveDecom(outKF, model)
% Decomposes model-implied yields into expected short rates and term premia.

xhat   = outKF.xHat;
[nx,T] = size(xhat);
muP    = model.muP;
phiP   = model.phiP;
maxMat = max(model.matSelect);

xExp = nan(nx, maxMat, T);
for t = 1:T
    for i = 1:maxMat
        if i == 1
            xExp(:,1,t) = xhat(:,t);
        else
            xExp(:,i,t) = muP + phiP * xExp(:,i-1,t);
        end
    end
end

rExp = nan(maxMat, T);
for t = 1:T
    for i = 1:maxMat
        rExp(i,t) = model.r0 + model.rx * xExp(:,i,t);
    end
end

yHat    = outKF.g0 + outKF.gx * outKF.xHat;
ny      = length(model.matSelect);
rExpAvg = nan(ny, T);
for t = 1:T
    for i = 1:ny
        rExpAvg(i,t) = mean(rExp(1:model.matSelect(i), t));
    end
end

res.yHat       = yHat';
res.rExp       = rExpAvg';
res.termPremia = res.yHat - res.rExp;
res.matSelect  = model.matSelect;
end

function res = local_campbellSchiller(Data, m)

[T, n]         = size(Data);
matSelect_CS   = m : m : n*m;
shortRate      = Data(1:T-m, 1);
numYields      = n;

CSBetta        = NaN(2, numYields);
CSBetta_tstat  = NaN(2, numYields);
CSBetta_se     = NaN(2, numYields);
Ydata          = NaN(T-m, numYields);
Xdata          = NaN(T-m, numYields);
R2             = NaN(1, numYields);

for i = 1 : numYields - 1
    k  = matSelect_CS(i+1);
    Y  = Data(1+m:T, i) - Data(1:T-m, i+1);
    x  = (Data(1:T-m, i+1) - shortRate);
    X  = [ones(T-m, 1),  x * m / (k - m)];
    resOLS               = local_nwest(Y, X, m+1);
    CSBetta(:, i+1)      = resOLS.beta;
    CSBetta_se(:, i+1)   = resOLS.beta ./ resOLS.tstat;
    CSBetta_tstat(:,i+1) = resOLS.tstat;
    Ydata(:, i+1)        = Y;
    Xdata(:, i+1)        = x * m / (k - m);
    R2(:, i+1)           = resOLS.rsqr;
end

res.CSbetta        = CSBetta;
res.maturities     = matSelect_CS;
res.CSbetta_se     = CSBetta_se;
res.CSBetta_tstat  = CSBetta_tstat;
res.CSBetta_CI95   = [CSBetta(2,:) - 1.96  * CSBetta_se(2,:); ...
                      CSBetta(2,:) + 1.96  * CSBetta_se(2,:)];
res.CSBetta_CI99   = [CSBetta(2,:) - 2.575 * CSBetta_se(2,:); ...
                      CSBetta(2,:) + 2.575 * CSBetta_se(2,:)];
res.Ydata          = Ydata;
res.Xdata          = Xdata;
res.R2             = R2;
end

function res = local_nwest(y, X, lag)
% Newey-West OLS

[T, k] = size(X);
beta   = (X'*X) \ (X'*y);
e      = y - X*beta;

S = (e .* X)' * (e .* X);
for l = 1 : lag
    w   = 1 - l / (lag + 1);
    Xl  = X(l+1:T, :);
    el  = e(l+1:T);
    X0  = X(1:T-l, :);
    e0  = e(1:T-l);
    Gl  = (e0 .* X0)' * (el .* Xl);
    S   = S + w * (Gl + Gl');
end

XpX_inv = (X'*X) \ eye(k);
V       = T * XpX_inv * S * XpX_inv;
se      = sqrt(diag(V) / T);

res.beta  = beta;
res.tstat = beta ./ se;
res.se    = se;
res.rsqr  = 1 - var(e) / var(y);
end

% Extended Step 7 objective with three soft constraints:
%   1: Standard A2_star moment conditions (A-intercept fit)
%   2: Alpha anchor: alpha ≈ mean_y1yr/ann
%   3: muQ shrinkage: |muQ| small
%   4: FULL UNCONDITIONAL LEVEL: embed Step 8 muP_l recovery to penalise
%       deviations of the model-implied 10yr yield from its sample mean.
%
% The level constraint breaks the circular dependency between Steps 7 and 8
% by computing the implied muP_l forward from any candidate (alpha, muQ),
% then evaluating E[y_10yr] = g0_10yr + gx_10yr * E[x].

function val = step7_objective(x, mature, phiQ, beta, sigma, ...
                                A2_star, A1_star, B1l_ann, B2l_ann, B1l_chol, ...
                                phiP_ll, IminusPhiP_ll, muP_m, ...
                                nxM, nxL, ann, mean_y1yr, ...
                                w_mean, w_muQ, w_level)

alpha_c  = x(1);
muQ_c    = [x(2:end); zeros(nxL,1)];   % nxM macro + nxL=0 latent muQ

% Standard A2_star residuals
r_A2 = local_alphaMuQ_residuals(x, mature, phiQ, beta, sigma, ...
                                  A2_star, B1l_ann, B2l_ann, nxM, nxL, ann);
val = sum(r_A2.^2);

% Constraint 1: alpha anchor
val = val + w_mean^2 * (alpha_c - mean_y1yr/ann)^2;

% onstraint 2: muQ shrinkage
val = val + w_muQ^2 * sum(x(2:end).^2);

% Constraint 3: full unconditional 10yr level
% Recover muP_l implied by this (alpha, muQ) candidate (mirrors Step 8)
try
    [A1_ann_c, ~] = local_computeA_ann(alpha_c, muQ_c, phiQ, beta, sigma, ...
                                        mature, nxM, nxL, ann);
    muP_l_c = B1l_chol \ (A1_star - A1_ann_c + ...
                           B1l_chol * phiP_ll / B1l_chol * A1_ann_c);
    muP_c   = [muP_m; muP_l_c];

    % Unconditional factor means E[x] = (I-phiP)^{-1} * muP
    % Use block structure: E[xm] = (I-phiP_mm)^{-1}*muP_m ≈ 0
    %                      E[xl] = (I-phiP_ll)^{-1}*muP_l
    Ex_l = IminusPhiP_ll \ muP_l_c;    % nxL x 1

    % Implied 10yr yield: g0(10yr) + gx(10yr,:) * E[x]
    % gx row index for 10yr: matSelect = [4,8,20,28,40,60]; 10yr=40Q is index 5
    gx_10yr = gx(5,:);   % 1 x nx  (annualised)
    g0_10yr = gx_10yr * zeros(nxM+nxL,1);  % placeholder; actual g0 not yet computed

    % Approximate E[y_10yr] using only the factor mean contribution:
    % gx(10yr) = ann * (-B(:,40)') / 40  from the phiQ/beta recursion.
    % We penalise gx_latent * E[xl] — the latent drift contribution to
    % the 10yr yield. If this is large, the level will be off.
    % Compute B(:,40) directly (40 quarters = 10yr):
    n10 = 40;
    B_k = -beta;
    for kk = 2:n10
        B_k = -beta + phiQ' * B_k;
    end
    gx_10yr_lat = ann * (-B_k(nxM+1:end)' / n10);   % 1 x nxL

    latent_drift_10yr = gx_10yr_lat * Ex_l;   % scalar (annualised decimal)
    val = val + w_level^2 * latent_drift_10yr^2;
catch
    val = val + 1e20;   % penalise if recovery fails
end
end