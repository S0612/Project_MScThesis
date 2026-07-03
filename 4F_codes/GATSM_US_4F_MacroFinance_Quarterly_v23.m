% This is the main estimation script for the NS estimator.
%
% Files needed in dir
%   US_monthly_yields_Jan1972_Dec2025.csv
%   US_monthly_yields_Jan1972_Dec2025_maturities.csv
%   US_macro_inflation_unemployment_MF.csv

close all; clear; clc;

%% User Settins

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

%% Load data

% ---- 2a: Yield data (monthly CSV -> quarterly sub-sample) ---------------
yields_raw = csvread('Data/US_monthly_yields_Jan1972_Dec2025.csv');
mats_years = csvread('Data/US_monthly_yields_Jan1972_Dec2025_maturities.csv');

% Validate that every requested maturity exists in the CSV

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

% 2b: Split data into Y1 (priced exactly) and Y2 (priced with error)
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
% data is actually already normalised so this line here is redundant. 
xMacro     = ((macro_Q - macro_mean) ./ macro_std)';   % nxM x T

% 2d: Date vector (quarterly)
dateStart   = datenum(1972, 3, 1);
setup_dates = dateStart + (0:T-1)' * 91;
dateVec     = datetime(1972, 3, 1) + calmonths(3*(0:T-1));

fprintf('  Quarterly obs: T = %d\n', T);
fprintf('  Maturities (years): %s\n', num2str(matSelect/ann));
fprintf('  Priced exactly (Y1): %s yr\n', num2str(exactYears));
fprintf('  Priced with error (Y2): %s yr\n', num2str(errorYears));

%% First-stage OLS  (reduced-form VAR)

T1 = T - 1;   % usable observations after one lag

% Block m: macro AR(1), no Y1 lags
ym           = xMacro(:, 2:end)';                    % T1 x nxM
xm           = [ones(T1,1), xMacro(:,1:end-1)'];    % T1 x (1+nxM)
paramM       = xm \ ym;
Am_star      = paramM(1, :)';                        % nxM x 1
phiP_starmm  = paramM(2:end, :)';                    % nxM x nxM
umt_star     = ym - xm * paramM;                     % T1 x nxM
Omega_starm  = (umt_star' * umt_star) / T1;          % nxM x nxM

% Block 1: Y1 on lagged Y1 and contemporaneous macro (no lagged macro)
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

% CMA-ES options, shared by both subproblems
% We set them quite tight
cmaOpts.Display       = 'off';
cmaOpts.Plotting      = 'off';
cmaOpts.Saving        = 0;
cmaOpts.VerboseModulo = 0;
cmaOpts.TolFun        = 1e-14;
cmaOpts.TolX          = 1e-14;
cmaOpts.MaxFunEvals   = 1e6;
cmaOpts.StopOnWarnings = 'no';

% Subproblem A: macro block (Lambda_mm, beta_m)

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

beta_l_scale = diag(B1l_chol) / (ann * 0.86);   % nxL x 1

% Strictly positive lower bound for beta_l (H-W sign normalisation).
% The floor of scale/10 mirrors the beta_m treatment, which prevents beta_l[k]
% collapsing to zero (which makes that latent factor unidentified and
% causes alpha to hit its boundary to compensate in Step 7).
lb_B = [-5; -5; -5; beta_l_scale(1)/10; beta_l_scale(2)/10];
ub_B = [ 5;  5;  5; beta_l_scale(1)*5;  beta_l_scale(2)*5];

x0_B   = [0; 0; 0; beta_l_scale];
sig0_B = [0.3; 0.1; 0.1; beta_l_scale];

cmaOpts.LBounds = lb_B;
cmaOpts.UBounds = ub_B;
cmaOpts.PopSize = 4 + floor(3*log(length(x0_B)));
objB = @(p) sum(local_latentBlock_residuals(p, phiP_ll, B1B1_OLS, B2B1_OLS, ...
                                             mature, nxL, ann).^2);
[xBestB, fBestB] = cmaes_dsgeDisplay(objB, x0_B, 1, sig0_B, cmaOpts);

lev    = xBestB(1);
lb     = xBestB(2);
lc     = xBestB(3);
beta_l = xBestB(4:5);

Lambda_ll = [lev, lb; lc, lev];
phiQ_ll   = phiP_ll - Lambda_ll;

if phiQ_ll(1,2) > phiQ_ll(2,1)
    phiQ_ll = phiQ_ll';          % transpose gives the other equivalent form
    beta_l  = beta_l([2,1]);     % permute beta_l to match
    % Recompute Lambda_ll from the canonical phiQ_ll for consistency
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

muQ = zeros(nx, 1);   % muQ_latent = 0 by normalisation; muQ_macro solved below

mean_y1yr  = mean(data(1,:));              % mean 1yr yield (annual decimal)

x0_7   = [mean_y1yr/ann; zeros(nxM,1)];   % alpha start: mean short rate / ann
sig0_7 = [0.005; 0.3*ones(nxM,1)];
lb_7   = [-0.03; -50*ones(nxM,1)];
ub_7   = [ 0.03;  50*ones(nxM,1)];

cmaOpts7          = cmaOpts;
cmaOpts7.LBounds  = lb_7;
cmaOpts7.UBounds  = ub_7;
cmaOpts7.PopSize  = 4 + floor(3*log(length(x0_7)));
cmaOpts7.TolFun   = 1e-10;
cmaOpts7.TolX     = 1e-10;

% Soft-constraint weight, scale so penalty is in the same order as A2 residuals
% A2 residuals are scaled by 1e6; penalty in same units
w_mean = 1e6;

% E[r] = ann*(alpha + beta' * (I-phiP)^{-1} * muP)
% muP is not yet known at Step 7, but we know Am_star ≈ muP_m,
% and muP_l is determined by A1_star and phiP_ll.
% Approximation, use alpha alone to anchor the mean short-rate level.
% alpha_quarterly should be near mean_y1yr/ann when macro factors have
% zero unconditional mean (which they do, by construction of xMacro).
obj7 = @(x) sum(local_alphaMuQ_residuals(x, mature, phiQ, beta, sigma, ...
                                          A2_star, B1l_ann, B2l_ann, nxM, nxL, ann).^2) + ...
             w_mean^2 * (x(1) - mean_y1yr/ann)^2;
[xBest7, fBest7] = cmaes_dsgeDisplay(obj7, x0_7, 0.5, sig0_7, cmaOpts7);

alpha      = xBest7(1);
muQ(1:nxM) = xBest7(2:end);

fprintf('  CMA-ES SSR: %.2e\n', fBest7);
fprintf('  alpha = %.6f  (quarterly; r0 = %.4f%%)\n', alpha, alpha*ann*100);
fprintf('  muQ_macro = %s\n', num2str(muQ(1:nxM)'));

%% Recover muP analytically 

[A1_ann, ~] = local_computeA_ann(alpha, muQ, phiQ, beta, sigma, mature, nxM, nxL, ann);

% muP_m = Am_star, independence: zero cross-block correction
% muP_l uses B1l_chol, same rotation as phiP_ll
muP_m = Am_star;
muP_l = B1l_chol \ (A1_star - A1_ann + B1l_chol * phiP_ll / B1l_chol * A1_ann);
muP   = [muP_m; muP_l];

fprintf('  muP = %s\n', num2str(muP'));

%% STEP 9: Compute bond pricing loadings (g0, gx, A, B)

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

% Pack model struct — field names match 2F local_solveATSM output exactly
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

%% STEP 10: Kalman filter


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


%% Local Functions

% -------------------------------------------------------------------------
function F = local_macroBlock_residuals(para, phiP_mm, sigma_macro, ...
                                         B1m_OLS, B2m_OLS, mature, nxM, ann)
% Residuals for the macro sub-problem in MCSE Step 2.
% para = [Lambda_mm(:); beta_m]  (nxM^2 + nxM elements)
% phiQ_mm = phiP_mm - sigma_macro * Lambda_mm
Lambda_mm = reshape(para(1:nxM^2), nxM, nxM);
beta_m    = para(nxM^2+1 : end);

phiQ_mm = phiP_mm - sigma_macro * Lambda_mm;
% NOTE: No stationarity constraint on phiQ_mm. The Q-measure dynamics
% can legitimately have |eig(phiQ_mm)| > 1 (H-W Table 6 has two
% Q-eigenvalues at 1.025). The B-loading recursion stays finite for
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

% Guard: if B loadings diverged (phiQ_mm very explosive), return large penalty
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

% -------------------------------------------------------------------------
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

% HYBRID CONSTRAINT (matches H-W Table 6 structure):
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

% Guard: if B loadings diverged (phiQ_ll very explosive), return large penalty
if ~all(isfinite(B_lat(:)))
    nEq = nxL*(nxL+1)/2 + numel(B2B1_OLS);
    F = 1e8 * ones(nEq, 1);
    return;
end
nE    = length(mature.exact);
B1l_s = B_lat(1:nE, :);
B2l_s = B_lat(nE+1:end, :);

% Symmetric part of B1l*B1l' (3 unique elements for nxL=2)
M1 = B1B1_OLS - B1l_s * B1l_s';
r1 = M1([1,2,4]) * 1e5;    % [M(1,1) M(2,1) M(2,2)] = upper-left, lower-left, lower-right

M2 = B2B1_OLS - B2l_s * B1l_s';
r2 = M2(:) * 1e5;

F = [r1(:); r2(:)];
end

% -------------------------------------------------------------------------
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

% -------------------------------------------------------------------------
function [A1, A2] = local_computeA_ann(alpha, muQ, phiQ, beta, sigma, ...
                                        mature, nxM, nxL, ann)
% Compute ANNUALISED A yield intercepts.
%
% Annual intercept: g0(n) = ann * (-A_bar(n)/n)
%
% Uses the same per-quarter bond pricing recursion as the main script,
% then multiplies by ann to return annual decimal values.

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

% -------------------------------------------------------------------------
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

% -------------------------------------------------------------------------
function out = local_KalmanFilter(y, g0, gx, Rv, h0, hx, Reps)
% Standard Kalman filter — copied verbatim from GATSM_US_2F_V2.m.
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
    % Instead we test with a try/catch and regularise if needed.
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

% -------------------------------------------------------------------------
function res = local_yieldCurveDecom(outKF, model)
% Decomposes model-implied yields into expected short rates and term premia.
% Mirrors local_yieldCurveDecom from GATSM_US_2F_V2.m exactly.

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

% -------------------------------------------------------------------------
function res = local_campbellSchiller(Data, m)
% Campbell-Schiller (1991) regression — copied verbatim from
% GATSM_US_2F_V2.m (local_campbellSchiller).

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

% -------------------------------------------------------------------------
function res = local_nwest(y, X, lag)
% Newey-West OLS — copied verbatim from GATSM_US_2F_V2.m (local_nwest).

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