% This script computes CS regressions using sampled data from the
% two-factor estimates and compares it to CS regs. on actual data.
%
% CS regression (for each long maturity k, short maturity m):
%   y_{t+m}^{(k-m)} - y_t^{(k)} = alpha + beta*m/(k-m)*(y_t^{(k)} - y_t^{(m)}) + eps
% 
% m = 1 year maturity.
%
%
% Needs in directory:
%   GATSM_2F_Estimates_US_CMAES.mat     - MLE output of the 2-factor GATSM
%   US_monthly_yields_Jan1972_Dec2025.csv - T x 15 annual yields (%), 1..15y
%

clear; close all; clc;

%% User settings

targetMats_years = [2 5 7 10 15];     % target long maturities (years)
shortRateMat_yr  = 1;                 % short-rate maturity (years)
m_months         = 12 * shortRateMat_yr;   % CS step size in months
Nsim             = 100000;            % number of Monte Carlo simulations
rngSeed          = 1;                 % random number seed for reproducibility
includeMeasErr   = true;              % add stdY measurement noise to simulated yields
printEvery       = 10000;             % progress print frequency


%% Load GATSM estimates

S = load('GATSM_2F_Estimates_US_CMAES.mat');

% The model object (built inside local_solveATSM in GATSM_US_2F_V2.m) lives
% in outML.outKF.model and contains all objects required for simulation.
model = S.outML.outKF.model;

muP   = model.muP;         % nx x 1
phiP  = model.phiP;        % nx x nx
sigma = model.sigma;       % nx x nx lower-triangular Cholesky factor of
                           %   innovation covariance, i.e. var(eps_t)=sigma*sigma'
A     = model.A;           % 1   x maxMat (maxMat = 180 months)
B     = model.B;           % nx  x maxMat

% Measurement-error std (one common value across all yields), as in the
% Kalman filter measurement equation y_t = g0 + gx*x_t + stdY * v_t
stdY  = S.outML.paramsOpt.stdY;

nx = size(phiP, 1);
fprintf('Loaded GATSM estimates: nx = %d latent factors, measurement std = %.6f\n', nx, stdY);

%% STEP 3: Construct yield loadings on the FULL annual maturity grid

% In the GATSM the yield-to-maturity (annualised, decimal) is
%   y_t^{(k)} = -12/k * A(k) - 12/k * B(:,k)' * x_t
% with k measured in months, hence g0 = -12*A/k, gx = -12*B'/k.

annualMats_months = 12 : 12 : 180;    % 15 maturities: 12, 24, ..., 180 months
K = length(annualMats_months);

g0_full = 12 * (-A(annualMats_months)' ./ annualMats_months');        % K x 1
gx_full = 12 * (-B(:, annualMats_months)' ./ ...
                repmat(annualMats_months', 1, nx));                   % K x nx

% Sanity check: g0_full entries at matSelect should match model.g0
assert(max(abs(g0_full(ismember(annualMats_months, model.matSelect)) - model.g0)) < 1e-12, ...
       'g0 reconstruction does not match model.g0');

% Target long maturities (index into the annual grid)
idxShort  = find(annualMats_months == m_months);                      % 1
idxLong   = arrayfun(@(y) find(annualMats_months == 12*y), targetMats_years);
assert(idxShort == 1, 'Short rate must be the first column of the annual grid.');

%% Load actual U.S. yield data

yields_raw = csvread('US_monthly_yields_Jan1972_Dec2025.csv');   % T x 15 (%)
T          = size(yields_raw, 1);
dataActual = yields_raw / 100;                                   % decimal

assert(size(dataActual, 2) == K, ...
    'CSV has %d columns but expected %d annual maturities.', size(dataActual,2), K);
fprintf('Loaded actual data: T = %d monthly observations, %d maturities.\n', T, K);

%% Campbell-Schiller regression on ACTUAL data

% Uses Newey-West HAC standard errors with bandwidth m+1 (standard choice
% given the m-period forecasting horizon and overlapping observations).
resCS_actual = csRegression(dataActual, m_months, idxLong, idxShort, true);

fprintf('\n=== Campbell-Schiller regressions - ACTUAL U.S. data ===\n');
fprintf('     (EH implies beta = 1; Newey-West SEs, lag = %d)\n\n', m_months+1);
fprintf('%-8s  %10s  %10s  %10s  %10s\n', 'Mat(yr)', 'beta', 'NW-SE', 't-stat', 'R^2');
fprintf('%s\n', repmat('-', 1, 54));
for i = 1:length(targetMats_years)
    fprintf('%-8d  %10.4f  %10.4f  %10.3f  %10.4f\n', ...
        targetMats_years(i), ...
        resCS_actual.beta(i), resCS_actual.se(i), ...
        resCS_actual.tstat(i), resCS_actual.R2(i));
end

%% Monte Carlo section

% For each of Nsim replications we:
%   draw x_0 from the stationary distribution of x under P
%   simulate x_{t+1} = muP + phiP * x_t + sigma * eps_{t+1}, t = 1..T
%   compute yields y_t = g0_full + gx_full * x_t (+ stdY*v_t if on)
%   run the CS regression at each target maturity and store the slope

fprintf('\n=== Simulating %d Campbell-Schiller regressions ...\n', Nsim);

% Stationary mean of x under P: (I-phiP)^{-1} * muP
x_uncMean = (eye(nx) - phiP) \ muP;

% Stationary covariance of x: vec(P0) = (I - phiP kron phiP)^{-1} * vec(sigma*sigma')
Reps      = sigma * sigma';
vecP0     = (eye(nx^2) - kron(phiP, phiP)) \ Reps(:);
P0        = reshape(vecP0, nx, nx);
P0        = (P0 + P0') / 2;                     % symmetrise (num. noise)
L0        = chol(P0, 'lower');                  % for drawing x_0

rng(rngSeed, 'twister');

% Pre-allocate storage. We store only the slope coefficients (Nsim x numLong)
numLong     = length(targetMats_years);
beta_sim    = zeros(Nsim, numLong);
R2_sim      = zeros(Nsim, numLong);

% Pre-compute regression bookkeeping that is identical across simulations:
% For each target long maturity we need column indices and the m/(k-m) scaling.
T_eff      = T - m_months;
kIdxVec    = idxLong(:)';                      % 1 x numLong : col index for y_t^{(k)}
kmIdxVec   = kIdxVec - 1;                      % 1 x numLong : col index for y_t^{(k-m)}
kMonthsVec = 12 * kIdxVec;                     % 1 x numLong
scaleVec   = m_months ./ (kMonthsVec - m_months);  % 1 x numLong : m/(k-m)

tStart = tic;
for s = 1:Nsim
    % Draw initial factor state from stationary distribution
    x0 = x_uncMean + L0 * randn(nx, 1);

    % Simulate factor path: x_{t+1} = muP + phiP * x_t + sigma * eps_{t+1}
    Xpath  = zeros(nx, T);
    x      = x0;
    shocks = sigma * randn(nx, T);              % nx x T pre-drawn innovations
    for t = 1:T
        x          = muP + phiP * x + shocks(:, t);
        Xpath(:,t) = x;
    end

    % Convert to yields on the full annual grid: T x K
    Ysim = (g0_full + gx_full * Xpath)';

    % Add Kalman measurement noise if requested
    if includeMeasErr
        Ysim = Ysim + stdY * randn(T, K);
    end

    % Fast CS regressions (bivariate OLS slope = cov(X,Y) / var(X))
    % For each target long maturity k:
    %   Y = y_{t+m}^{(k-m)} - y_t^{(k)}                      (T_eff x 1)
    %   X = m/(k-m) * (y_t^{(k)} - y_t^{(m)})                (T_eff x 1)
    % beta = cov(X,Y) / var(X); R2 = 1 - var(Y-bX-a)/var(Y)
    shortCol_t  = Ysim(1:T_eff,        idxShort);                     % T_eff x 1
    yLong_t     = Ysim(1:T_eff,        kIdxVec);                      % T_eff x numLong
    yShort_tp1  = Ysim(m_months+1:T,   kmIdxVec);                     % T_eff x numLong
    Ymat        = yShort_tp1 - yLong_t;                               % T_eff x numLong
    Xmat        = (yLong_t - shortCol_t) .* scaleVec;                 % broadcasted

    % De-mean columnwise
    Ybar = mean(Ymat, 1);  Xbar = mean(Xmat, 1);
    Yd   = Ymat - Ybar;    Xd   = Xmat - Xbar;

    % cov and var per column (sum of products divided by (T_eff-1))
    covXY = sum(Xd .* Yd, 1);                                         % 1 x numLong
    varX  = sum(Xd .* Xd, 1);                                         % 1 x numLong
    varY  = sum(Yd .* Yd, 1);                                         % 1 x numLong
    b     = covXY ./ varX;
    % R^2 = (cov(X,Y))^2 / (var(X) * var(Y))
    R2    = (covXY .* covXY) ./ (varX .* varY);

    beta_sim(s, :) = b;
    R2_sim(s, :)   = R2;

    if mod(s, printEvery) == 0
        fprintf('  completed %6d / %d sims (%5.1f%%, %.1fs elapsed)\n', ...
                s, Nsim, 100*s/Nsim, toc(tStart));
    end
end
fprintf('Simulation finished: total elapsed %.1f s.\n\n', toc(tStart));

% Struct simulation results
resCS_sim.beta         = beta_sim;
resCS_sim.R2           = R2_sim;
resCS_sim.targetMats   = targetMats_years;
resCS_sim.shortRateMat = shortRateMat_yr;
resCS_sim.Nsim         = Nsim;
resCS_sim.T            = T;
resCS_sim.includeMeasErr = includeMeasErr;

%% Summarise the simulated distribution

mean_sim = mean(beta_sim, 1);
med_sim  = median(beta_sim, 1);
std_sim  = std(beta_sim, 0, 1);
q_sim    = quantile(beta_sim, [0.025 0.05 0.50 0.95 0.975], 1);

% Share of simulations in which the simulated beta is < the actual beta
% (a common Monte-Carlo p-value style measure of how extreme the actual
% sample is relative to the model-implied sampling distribution).
pLE = mean(beta_sim <= resCS_actual.beta(:)', 1);
% Two-sided bootstrap-style p-value: 2*min(pLE, 1-pLE)
pTwo = 2 * min(pLE, 1 - pLE);

fprintf('=== Simulated CS beta distribution (N = %d draws) vs actual ===\n\n', Nsim);
fprintf(['%-8s  %10s   %10s %10s %10s   %10s %10s %10s   %10s\n'], ...
    'Mat(yr)', 'Actual', 'Sim mean', 'Sim med', 'Sim std', ...
    'Sim 2.5%', 'Sim 50%', 'Sim 97.5%', 'p(sim<=act)');
fprintf('%s\n', repmat('-', 1, 108));
for i = 1:numLong
    fprintf('%-8d  %10.4f   %10.4f %10.4f %10.4f   %10.4f %10.4f %10.4f   %10.4f\n', ...
        targetMats_years(i), resCS_actual.beta(i), ...
        mean_sim(i), med_sim(i), std_sim(i), ...
        q_sim(1,i), q_sim(3,i), q_sim(5,i), ...
        pLE(i));
end

fprintf('Two-sided simulation-based p-values (actual vs model distribution):\n');
for i = 1:numLong
    fprintf('  %2d-year: p = %.4f\n', targetMats_years(i), pTwo(i));
end

% =========================================================================
%% Figure - simulated distribution + actual + EH benchmark
% =========================================================================
figure('Name', 'GATSM-simulated Campbell-Schiller betas', ...
       'Units', 'normalized', 'Position', [0.05 0.10 0.90 0.55]);

for i = 1:numLong
    subplot(1, numLong, i);
    histogram(beta_sim(:,i), 80, 'Normalization', 'pdf', ...
              'FaceColor', [0.25 0.45 0.80], 'EdgeColor', 'none'); hold on;
    yl = ylim;
    % Actual beta (red solid)
    plot([resCS_actual.beta(i) resCS_actual.beta(i)], yl, 'r-', 'LineWidth', 1.8);
    % EH benchmark beta = 1 (black dashed)
    plot([1 1], yl, 'k--', 'LineWidth', 1.2);
    % Simulated mean (blue dashed)
    plot([mean_sim(i) mean_sim(i)], yl, 'b--', 'LineWidth', 1.2);
    hold off;
    title(sprintf('%d-year', targetMats_years(i)));
    xlabel('CS slope \beta'); ylabel('density');
    if i == 1
        legend({'Simulated', 'Actual', 'EH (\beta = 1)', 'Sim mean'}, ...
               'Location', 'best', 'FontSize', 7);
    end
    grid on; box on;
end
sgtitle(sprintf(['Distribution of Campbell-Schiller slope coefficients\n' ...
                 '2-factor GATSM, %d simulations, T = %d months'], Nsim, T), ...
        'FontWeight', 'bold');

%% To Latex

if ~exist('resCS_sim', 'var') || ~exist('resCS_actual', 'var')
    load('CS_Simulation_Results.mat', ...
         'resCS_actual', 'resCS_sim', 'targetMats_years');
end
 
beta_sim   = resCS_sim.beta;                 % Nsim x numLong
beta_act   = resCS_actual.beta(:)';          % 1 x numLong (row)
 
% Summary statistics
mean_sim = mean(beta_sim, 1);                % 1 x numLong
med_sim  = median(beta_sim, 1);
std_sim  = std(beta_sim, 0, 1);
CI95     = prctile(beta_sim, [2.5 97.5], 1); % 2 x numLong  (lo; hi)
 
% Write LaTeX table
texFile = 'CS_Table.tex';
fid     = fopen(texFile, 'w');
 
fprintf(fid, '\\begin{table}[htbp]\n\\centering\n');
fprintf(fid, ['\\caption{Campbell-Schiller regressions: actual U.S.\\ data vs.\\ ' ...
              '%d simulations from the 2-factor GATSM}\n'], size(beta_sim,1));
fprintf(fid, '\\label{tab:cs_results}\n');
fprintf(fid, '\\begin{tabular}{cccccc}\n\\toprule\n');
fprintf(fid, ['Mat (yr) & Actual $\\beta$ & Sim mean & Sim median ' ...
              '& Sim std & Sim 95\\%% CI \\\\\n\\midrule\n']);
 
for i = 1:length(targetMats_years)
    fprintf(fid, '%d & %7.4f & %7.4f & %7.4f & %7.4f & $[%6.4f,\\; %6.4f]$ \\\\\n', ...
        targetMats_years(i), beta_act(i), ...
        mean_sim(i), med_sim(i), std_sim(i), ...
        CI95(1,i), CI95(2,i));
end
 
fprintf(fid, '\\bottomrule\n\\end{tabular}\n');
fprintf(fid, ['\\begin{flushleft}\\footnotesize Notes: Campbell-Schiller slope ' ...
              'coefficients $\\beta$ from the regression ' ...
              '$y_{t+m}^{(k-m)}-y_t^{(k)}=\\alpha+\\beta\\,\\tfrac{m}{k-m}' ...
              '(y_t^{(k)}-y_t^{(m)})+\\varepsilon_{t+1}$ with $m=12$ months. ' ...
              'The 95\\%% CI reports the 2.5 and 97.5 percentiles of the ' ...
              'simulated distribution.\\end{flushleft}\n']);
fprintf(fid, '\\end{table}\n');
fclose(fid);
 
% Print so we can see
type(texFile);
fprintf('\nLaTeX table written to %s\n', texFile);

%% Save results

%save('CS_Simulation_Results.mat', ...
%     'resCS_actual', 'resCS_sim', ...
%     'targetMats_years', 'shortRateMat_yr', 'Nsim', 'T', ...
%     'mean_sim', 'med_sim', 'std_sim', 'q_sim', 'pLE', 'pTwo');
%fprintf('\nResults saved to CS_Simulation_Results.mat\n');

%% LOCAL FUNCTIONS
function res = csRegression(Data, m, idxLong, idxShort, doNeweyWest)
% Run Campbell-Shiller regressions for the maturities indexed by idxLong.
%
% Regression (for each long maturity k, short maturity m, both in months):
%   y_{t+m}^{(k-m)} - y_t^{(k)} = a + beta * m/(k-m) * (y_t^{(k)} - y_t^{(m)}) + eps
%
% INPUT:
%   Data        : T x K matrix of annual yields (columns at 12,24,...,K*12 months)
%   m           : CS step size in months (= maturity of the short rate)
%   idxLong     : 1 x nLong vector of column indices (into Data) giving the
%                 long maturities k; implied k = 12*idxLong months.
%   idxShort    : scalar column index of the short rate (should = 1 for y_1y).
%   doNeweyWest : if true, compute Newey-West HAC standard errors (lag = m+1).
%
% OUTPUT struct res:
%   res.beta   - nLong x 1 slope coefficients
%   res.alpha  - nLong x 1 intercepts
%   res.R2     - nLong x 1 OLS R-squared
%   res.se     - nLong x 1 standard errors of beta (NaN if doNeweyWest=false)
%   res.tstat  - nLong x 1 t-statistics of beta (NaN if doNeweyWest=false)

[T, ~]  = size(Data);
T_eff   = T - m;
shrt    = Data(1:T_eff, idxShort);           % y_t^{(m)}
nLong   = length(idxLong);
beta    = zeros(nLong, 1);
alpha_  = zeros(nLong, 1);
R2v     = zeros(nLong, 1);
sev     = nan(nLong, 1);
tstatv  = nan(nLong, 1);

for j = 1:nLong
    kIdx       = idxLong(j);                 % col index of y_t^{(k)}
    kmIdx      = kIdx - 1;                   % col index of y_t^{(k-m)} on annual grid
    k_months   = 12 * kIdx;                  % k in months
    km_months  = 12 * kmIdx;                 % k-m in months

    y_long_t   = Data(1:T_eff,  kIdx);       % y_t^{(k)}
    y_shrt_tp1 = Data(m+1:T,    kmIdx);      % y_{t+m}^{(k-m)}

    Y  = y_shrt_tp1 - y_long_t;
    xr = (y_long_t - shrt) * (m / km_months);
    X  = [ones(T_eff, 1), xr];

    b   = (X' * X) \ (X' * Y);
    e   = Y - X * b;
    alpha_(j) = b(1);
    beta(j)   = b(2);
    R2v(j)    = 1 - var(e) / var(Y);

    if doNeweyWest
        lag = m + 1;
        % Newey-West HAC sandwich (Bartlett kernel)
        S = (e .* X)' * (e .* X);
        for l = 1:lag
            w   = 1 - l/(lag+1);
            Xl  = X(l+1:T_eff, :);
            el  = e(l+1:T_eff);
            X0  = X(1:T_eff-l, :);
            e0  = e(1:T_eff-l);
            Gl  = (e0 .* X0)' * (el .* Xl);
            S   = S + w * (Gl + Gl');
        end
        XpXi = (X' * X) \ eye(2);
        V    = T_eff * XpXi * S * XpXi;
        seB  = sqrt(diag(V) / T_eff);
        sev(j)    = seB(2);
        tstatv(j) = beta(j) / seB(2);
    end
end

res.beta  = beta;
res.alpha = alpha_;
res.R2    = R2v;
res.se    = sev;
res.tstat = tstatv;
end