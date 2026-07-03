% This script computes campbell shiller regressions using the implied-yield from the four-factor
% models and compares it to actual data. Not simulations as in the
% two-factor case.
%
% Needs in dir
%   GATSM_4F_MacroFinance_Results_V23.mat         (V23 no-survey estimator)
%   GATSM_4F_MacroFinance_Results_KO_proper_V6.mat (V6  survey estimator)
%   US_monthly_yields_Jan1972_Dec2025.csv
%   US_monthly_yields_Jan1972_Dec2025_maturities.csv


clear; close all; clc;

%% User settings

% Annualisation factor (quarterly model: 4 quarters per year)
ann = 4;

% CS step size: 1 year = 4 quarters
m_CS = ann;

% Maturity grid: 1, 2, ..., 15 years expressed in quarters
mats_qtrs_full = (1:15) * ann;      % [4 8 12 ... 60] quarters


%% Load actual U.S. yield data  (monthly -> quarterly subsample)

yields_raw_monthly = csvread('Data/US_monthly_yields_Jan1972_Dec2025.csv');
mats_years_csv     = csvread('Data/US_monthly_yields_Jan1972_Dec2025_maturities.csv');

% Quarterly subsample: rows 3, 6, 9, ... starting from Jan 1972
%  (first quarterly obs = March 1972 = row 3)
T_monthly = size(yields_raw_monthly, 1);
qIdx      = 3 : 3 : T_monthly;
T_qtrs    = length(qIdx);

% Build T_qtrs x 15 matrix of annualised decimal yields at 1y,2y,...,15y
mats_qtrs_csv = mats_years_csv * ann;   % maturities in quarters
n_mats        = length(mats_qtrs_full);
dataActual    = zeros(T_qtrs, n_mats);
for i = 1 : n_mats
    col              = find(mats_years_csv == mats_qtrs_full(i)/ann);
    dataActual(:, i) = yields_raw_monthly(qIdx, col) / 100;  % decimal
end

fprintf('  Quarterly obs: T = %d  |  Maturities: %d (1y .. 15y)\n', T_qtrs, n_mats);

%% Load NS and 1Q model results

fprintf('Loading V23 (no-survey) estimates...\n');
S23 = load('GATSM_4F_MacroFinance_Results_V23.mat');
fprintf('Loading V6  (survey) estimates...\n');
S6  = load('GATSM_4F_MacroFinance_Results_KO_proper_V6.mat');

% Both .mat files store model-implied fitted yields via outKF and model.
% resDecom is NOT stored in the .mat; we recompute it from outKF + model.
resDecom23 = local_yieldCurveDecom(S23.outKF, S23.model);
resDecom6  = local_yieldCurveDecom(S6.outKF,  S6.model);

%% Build model-implied CS yield matrices

% The CS regression requires yields on the full 1y..15y annual grid.
% resDecom.yHat is T x ny at the maturities in model.matSelect (quarters).
% We interpolate model-fitted yields onto the full annual quarter grid.

% NS model-implied yields interpolated onto 1y..15y quarterly grid
dataModel23 = zeros(T_qtrs, n_mats);
for t = 1 : T_qtrs
    dataModel23(t, :) = interp1(resDecom23.matSelect, resDecom23.yHat(t, :), ...
                                mats_qtrs_full, 'linear', 'extrap');
end

% 1Q model-implied yields interpolated onto 1y..15y quarterly grid
dataModel6 = zeros(T_qtrs, n_mats);
for t = 1 : T_qtrs
    dataModel6(t, :) = interp1(resDecom6.matSelect, resDecom6.yHat(t, :), ...
                               mats_qtrs_full, 'linear', 'extrap');
end

fprintf('Model yield matrices built (%d x %d).\n', T_qtrs, n_mats);


%% Run Campbell-Shiller regressions (Newey-West HAC, lag = m+1)

fprintf('Running CS regressions...\n');

resCS_actual  = local_campbellSchiller(dataActual,  m_CS);
resCS_model23 = local_campbellSchiller(dataModel23, m_CS);
resCS_model6  = local_campbellSchiller(dataModel6,  m_CS);


%% Print results

mats_yr = mats_qtrs_full / ann;   % maturities in years for display

fprintf('\n');
fprintf('=================================================================\n');
fprintf('  Campbell-Shiller Regression: beta coefficients\n');
fprintf('  (EH: beta = 1; typically beta << 1 in U.S. data)\n');
fprintf('  m = %d quarters (1-year step)\n', m_CS);
fprintf('=================================================================\n');
fprintf('%-8s  %10s  %10s  %10s  %10s  %10s  %10s\n', ...
    'Mat(yr)', 'Act.beta', 'Act.SE', ...
    'V23.beta', 'V23.SE', 'V6.beta', 'V6.SE');
fprintf('%s\n', repmat('-', 1, 76));

for i = 2 : n_mats   % skip 1y (short-rate column has no LHS)
    fprintf('%-8.0f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f\n', ...
        mats_yr(i), ...
        resCS_actual.CSbetta(2,i),  resCS_actual.CSbetta_se(2,i), ...
        resCS_model23.CSbetta(2,i), resCS_model23.CSbetta_se(2,i), ...
        resCS_model6.CSbetta(2,i),  resCS_model6.CSbetta_se(2,i));
end

%% Figure with 95% CI bands

plotMats    = mats_yr(2:end);          % 2y 15y

% lope coefficients
beta_act  = resCS_actual.CSbetta(2,  2:end);
beta_v23  = resCS_model23.CSbetta(2, 2:end);
beta_v6   = resCS_model6.CSbetta(2,  2:end);

% 95% Newey-West confidence intervals
ci_act    = resCS_actual.CSBetta_CI95(:,  2:end);   % 2 x nPlot
ci_v23    = resCS_model23.CSBetta_CI95(:, 2:end);
ci_v6     = resCS_model6.CSBetta_CI95(:,  2:end);

% Colour palette (colourblind-safe)
col_act = [0.15 0.15 0.15];   % near-black
col_v23 = [0.00 0.45 0.70];   % blue 
col_v6  = [0.80 0.15 0.15];   % red

% Figure layout
figure('Name', 'Campbell-Schiller: NS and 1Q', ...
       'Units', 'centimeters', 'Position', [2 2 18 11]);

% CI shading and drawing in muted colours behind the lines
fill([plotMats, fliplr(plotMats)], ...
     [ci_act(1,:), fliplr(ci_act(2,:))], ...
     col_act, 'FaceAlpha', 0.10, 'EdgeColor', 'none'); hold on;

fill([plotMats, fliplr(plotMats)], ...
     [ci_v23(1,:), fliplr(ci_v23(2,:))], ...
     col_v23, 'FaceAlpha', 0.12, 'EdgeColor', 'none');

fill([plotMats, fliplr(plotMats)], ...
     [ci_v6(1,:), fliplr(ci_v6(2,:))], ...
     col_v6, 'FaceAlpha', 0.12, 'EdgeColor', 'none');

% EH benchmark
yline(1, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.0, ...
      'DisplayName', 'EH benchmark (\beta = 1)');
yline(0, ':', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.7, ...
      'HandleVisibility', 'off');

% Model & actual lines
plot(plotMats, beta_v23, '-o', ...
     'Color', col_v23, 'LineWidth', 1.6, 'MarkerSize', 5, ...
     'MarkerFaceColor', col_v23, ...
     'DisplayName', 'V23 — No-survey (model-implied)');

plot(plotMats, beta_v6, '-square', ...
     'Color', col_v6, 'LineWidth', 1.6, 'MarkerSize', 5, ...
     'MarkerFaceColor', col_v6, ...
     'DisplayName', 'V6 — KO survey (model-implied)');

plot(plotMats, beta_act, '-x', ...
     'Color', col_act, 'LineWidth', 1.8, 'MarkerSize', 5, ...
     'MarkerFaceColor', col_act, ...
     'DisplayName', 'Actual U.S. yields');

hold off;

% Labels & formatting
xlabel('Maturity (years)', 'FontSize', 11);
ylabel('\beta coefficient', 'FontSize', 11);
title({'Campbell-Shiller Regressions', ...
       'NS, 1Q, and actual data'}, ...
      'FontSize', 11, 'FontWeight', 'bold');

legend({'95% CI — Actual', '95% CI — NS', '95% CI — 1Q', ...
        'EH benchmark (\beta = 1)', ...
        'NS', '1Q', 'Actual'}, ...
       'Location', 'southwest', 'FontSize', 10, 'Box', 'on');

xlim([plotMats(1) - 0.3, plotMats(end) + 0.3]);
grid on; box on;
set(gca, 'FontSize', 10, 'TickDir', 'out');

exportgraphics(gcf, 'Graphs/CS_Compare_V23_V6.pdf', 'ContentType', 'vector');
fprintf('\nFigure saved: Graphs/CS_Compare_V23_V6.pdf\n');

%% Save results

save('CS_4F.mat', ...
     'resCS_actual', 'resCS_model23', 'resCS_model6', ...
     'plotMats', 'beta_act', 'beta_v23', 'beta_v6', ...
     'ci_act', 'ci_v23', 'ci_v6', ...
     'mats_yr', 'T_qtrs', 'm_CS');
fprintf('Results saved');

%% Local Functions

% -------------------------------------------------------------------------
function res = local_campbellSchiller(Data, m)
% Campbell-Shiller (1991) regression.
%
% Regression for each long maturity k (column i+1 in Data) vs short
% rate (column 1, maturity m):
%
%   y_{t+m}^{(k-m)} - y_t^{(k)} = alpha + beta * [m/(k-m)] * (y_t^{(k)} - y_t^{(m)}) + eps
%
% Under the pure EH, beta = 1. Newey-West HAC SEs with lag = m+1.
%
% INPUT:
%   Data   : T x n matrix of annualised decimal yields on an equally-
%            spaced maturity grid (step m quarters), column 1 = m-qtr yield.
%   m      : step size in quarters (= CS holding period / short maturity)
%
% OUTPUT struct res:
%   .CSbetta        (2 x n)  OLS coefficients [alpha; beta] for each maturity
%   .CSbetta_se     (2 x n)  Newey-West standard errors
%   .CSBetta_tstat  (2 x n)  t-statistics
%   .CSBetta_CI95   (2 x n)  95% confidence interval [lo; hi] on beta
%   .CSBetta_CI99   (2 x n)  99% confidence interval on beta
%   .maturities     (1 x n)  maturity in quarters for each column
%   .R2             (1 x n)  OLS R-squared
%   .Ydata          (T-m x n) LHS of the regression for each maturity
%   .Xdata          (T-m x n) RHS (scaled yield spread) for each maturity

[T, n]         = size(Data);
matSelect_CS   = m : m : n * m;        % maturities in quarters: m, 2m, ..., nm
shortRate      = Data(1:T-m, 1);       % y_t^{(m)}, short-rate column

CSBetta        = NaN(2, n);
CSBetta_se     = NaN(2, n);
CSBetta_tstat  = NaN(2, n);
Ydata          = NaN(T-m, n);
Xdata          = NaN(T-m, n);
R2             = NaN(1, n);

% Loop over long maturities (column i+1 corresponds to k = (i+1)*m quarters)
for i = 1 : n-1
    k      = matSelect_CS(i+1);        % long maturity in quarters
    km     = k - m;                    % long maturity minus step

    % LHS: change in yield at (k-m) from t to t+m
    %   y_{t+m}^{(k-m)} - y_t^{(k)}
    Y = Data(1+m : T, i) - Data(1 : T-m, i+1);

    % RHS: scaled yield spread (normalised as in Campbell-Shiller 1991)
    %   [m/(k-m)] * (y_t^{(k)} - y_t^{(m)})
    x = (Data(1:T-m, i+1) - shortRate) * (m / km);
    X = [ones(T-m, 1), x];

    % Newey-West OLS
    resOLS = local_nwest(Y, X, m+1);

    CSBetta(:,     i+1) = resOLS.beta;
    CSBetta_se(:,  i+1) = resOLS.se;
    CSBetta_tstat(:,i+1)= resOLS.tstat;
    Ydata(:, i+1)       = Y;
    Xdata(:, i+1)       = x;
    R2(:,    i+1)       = resOLS.rsqr;
end

res.CSbetta        = CSBetta;
res.CSbetta_se     = CSBetta_se;
res.CSBetta_tstat  = CSBetta_tstat;
res.CSBetta_CI95   = [CSBetta(2,:) - 1.960 * CSBetta_se(2,:); ...
                      CSBetta(2,:) + 1.960 * CSBetta_se(2,:)];
res.CSBetta_CI99   = [CSBetta(2,:) - 2.575 * CSBetta_se(2,:); ...
                      CSBetta(2,:) + 2.575 * CSBetta_se(2,:)];
res.maturities     = matSelect_CS;
res.Ydata          = Ydata;
res.Xdata          = Xdata;
res.R2             = R2;
end

% -------------------------------------------------------------------------
function res = local_nwest(y, X, lag)
% Newey-West (1987) HAC OLS — Bartlett kernel, bandwidth = lag.
%
% Standard sandwich form:
%   V = T * (X'X)^{-1} * S * (X'X)^{-1}
% where S is the HAC-corrected score covariance.

[T, k] = size(X);
beta   = (X' * X) \ (X' * y);
e      = y - X * beta;

S = (e .* X)' * (e .* X);          % heteroskedasticity component (l=0)
for l = 1 : lag
    w   = 1 - l / (lag + 1);       % Bartlett weight
    Xl  = X(l+1:T, :);
    el  = e(l+1:T);
    X0  = X(1:T-l, :);
    e0  = e(1:T-l);
    Gl  = (e0 .* X0)' * (el .* Xl);
    S   = S + w * (Gl + Gl');       % symmetrise autocovariance contribution
end

XpX_inv = (X' * X) \ eye(k);
V       = T * XpX_inv * S * XpX_inv;
se      = sqrt(diag(V) / T);

res.beta  = beta;
res.se    = se;
res.tstat = beta ./ se;
res.rsqr  = 1 - var(e) / var(y);
end

% -------------------------------------------------------------------------
function res = local_yieldCurveDecom(outKF, model)
% Decomposes model-implied yields into expected short rates and term premia.
% Mirrors local_yieldCurveDecom from GATSM_US_2F_V2.m and V23/V6 exactly.
%
% Returns:
%   res.yHat       (T x ny) fitted yields (annualised decimal)
%   res.rExp       (T x ny) average expected short rate over horizon n
%   res.termPremia (T x ny) = yHat - rExp
%   res.matSelect  (1 x ny) maturities in quarters

xhat   = outKF.xHat;           % nx x T
[nx, T] = size(xhat);
muP    = model.muP;             % nx x 1
phiP   = model.phiP;            % nx x nx
maxMat = max(model.matSelect);  % in quarters

% Iterate E_t[x_{t+i}] = muP + phiP * E_t[x_{t+i-1}]
xExp = nan(nx, maxMat, T);
for t = 1 : T
    for i = 1 : maxMat
        if i == 1
            xExp(:, 1, t) = xhat(:, t);
        else
            xExp(:, i, t) = muP + phiP * xExp(:, i-1, t);
        end
    end
end

% Expected short rate at each horizon: r_t = r0 + rx * x_t (annualised)
rExp_full = nan(maxMat, T);
for t = 1 : T
    for i = 1 : maxMat
        rExp_full(i, t) = model.r0 + model.rx * xExp(:, i, t);
    end
end

% Fitted yields and average expected short rates at selected maturities
yHat    = outKF.g0 + outKF.gx * outKF.xHat;  % ny x T
ny      = length(model.matSelect);
rExpAvg = nan(ny, T);
for t = 1 : T
    for i = 1 : ny
        rExpAvg(i, t) = mean(rExp_full(1:model.matSelect(i), t));
    end
end

res.yHat       = yHat';                        % T x ny
res.rExp       = rExpAvg';                     % T x ny
res.termPremia = res.yHat - res.rExp;          % T x ny
res.matSelect  = model.matSelect;              % 1 x ny (quarters)
end