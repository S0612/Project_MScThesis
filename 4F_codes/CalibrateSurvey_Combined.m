% This script produces two combined diagnostic figures that compare the 1Q-ahead and
% 4Q-ahead SPF survey calibrations side by side, organised by macro variable:
%
%   Figure 1  —  Inflation:    1Q-ahead (dpgdp3) | 4Q-ahead (dpgdp6)
%   Figure 2  —  Unemployment: 1Q-ahead (UNEMP3) | 4Q-ahead (UNEMP6)
%
% required files:
%   US_SurveyData.xlsx
%   US_macro_inflation_unemployment_MF.csv

clear; clc;

%% Load SPF survey data


T_inf = readtable('US_SurveyData.xlsx', 'Sheet', 'Inflation');
T_une = readtable('US_SurveyData.xlsx', 'Sheet', 'Unemployment');

% 1Q-ahead forecasts (h=1)
svy_inf_1Q_raw = T_inf.dpgdp3;   % forecast made at t for quarter t+1
svy_une_1Q_raw = T_une.UNEMP3;

% 4Q-ahead forecasts (h=4)
svy_inf_4Q_raw = T_inf.dpgdp6;
% Replace isolated NaN in dpgdp6 (1974Q3) with dpgdp5
nan_inf4 = isnan(svy_inf_4Q_raw);
if any(nan_inf4)
    svy_inf_4Q_raw(nan_inf4) = T_inf.dpgdp5(nan_inf4);
    fprintf('  Replaced %d NaN(s) in dpgdp6 with dpgdp5.\n', sum(nan_inf4));
end

svy_une_4Q_raw = T_une.UNEMP6;
% Replace isolated NaN in UNEMP6 with UNEMP5
nan_une4 = isnan(svy_une_4Q_raw);
if any(nan_une4)
    svy_une_4Q_raw(nan_une4) = T_une.UNEMP5(nan_une4);
    fprintf('  Replaced %d NaN(s) in UNEMP6 with UNEMP5.\n', sum(nan_une4));
end

T_svy = length(svy_inf_1Q_raw);
fprintf('  Survey observations: T=%d\n\n', T_svy);

%% Normalise each survey series by its own mean and std

% 1Q inflation
mu_inf_1Q  = nanmean(svy_inf_1Q_raw);   sd_inf_1Q  = nanstd(svy_inf_1Q_raw);
svy_inf_1Q = (svy_inf_1Q_raw - mu_inf_1Q) / sd_inf_1Q;

% 1Q unemployment
mu_une_1Q  = nanmean(svy_une_1Q_raw);   sd_une_1Q  = nanstd(svy_une_1Q_raw);
svy_une_1Q = (svy_une_1Q_raw - mu_une_1Q) / sd_une_1Q;

% 4Q inflation
mu_inf_4Q  = mean(svy_inf_4Q_raw);      sd_inf_4Q  = std(svy_inf_4Q_raw);
svy_inf_4Q = (svy_inf_4Q_raw - mu_inf_4Q) / sd_inf_4Q;

% 4Q unemployment
mu_une_4Q  = mean(svy_une_4Q_raw);      sd_une_4Q  = std(svy_une_4Q_raw);
svy_une_4Q = (svy_une_4Q_raw - mu_une_4Q) / sd_une_4Q;

%% Load macro actuals — quarterly subsample

fprintf('Loading macro data...\n');
macro_raw = readmatrix('US_macro_inflation_unemployment_MF.csv');

% End-of-quarter rows: every 3rd row starting from row 3
qi     = 3:3:size(macro_raw,1);
xMacro = macro_raw(qi,:)';   % 2 x T  (row 1 = inflation, row 2 = unemployment)
T_mac  = size(xMacro,2);

fprintf('  Quarterly macro observations: T=%d\n', T_mac);
assert(T_mac == T_svy, ...
    sprintf('Length mismatch: macro T=%d, survey T=%d', T_mac, T_svy));

%% OLS calibration — helper function defined at end of file
%    Returns: mu, phi, sigma, R2, and the cleaned (x_, s_) pair for plotting.

% Q inflation
[mu_inf1, phi_inf1, sig_inf1, R2_inf1, x_inf1, s_inf1] = ...
    ols_calib(xMacro(1,:), svy_inf_1Q);

% 4Q inflation
[mu_inf4, phi_inf4, sig_inf4, R2_inf4, x_inf4, s_inf4] = ...
    ols_calib(xMacro(1,:), svy_inf_4Q);

% 1Q unemployment
[mu_une1, phi_une1, sig_une1, R2_une1, x_une1, s_une1] = ...
    ols_calib(xMacro(2,:), svy_une_1Q);

% 4Q unemployment
[mu_une4, phi_une4, sig_une4, R2_une4, x_une4, s_une4] = ...
    ols_calib(xMacro(2,:), svy_une_4Q);

% Print summary
fprintf('\n%-25s  %8s  %8s  %8s  %6s\n', ...
    'Series', 'mu', 'phi', 'sigma', 'R^2');
fprintf('%s\n', repmat('-',1,62));
fprintf('%-25s  %8.4f  %8.4f  %8.4f  %6.4f\n', ...
    'Inflation  1Q (dpgdp3)',  mu_inf1, phi_inf1, sig_inf1, R2_inf1);
fprintf('%-25s  %8.4f  %8.4f  %8.4f  %6.4f\n', ...
    'Inflation  4Q (dpgdp6)',  mu_inf4, phi_inf4, sig_inf4, R2_inf4);
fprintf('%-25s  %8.4f  %8.4f  %8.4f  %6.4f\n', ...
    'Unemployment 1Q (UNEMP3)',mu_une1, phi_une1, sig_une1, R2_une1);
fprintf('%-25s  %8.4f  %8.4f  %8.4f  %6.4f\n', ...
    'Unemployment 4Q (UNEMP6)',mu_une4, phi_une4, sig_une4, R2_une4);

%% Figure 1 — Inflation: 1Q-ahead vs 4Q-ahead

fig1 = figure('Name','Survey Calibration — Inflation','NumberTitle','off', ...
              'Units','inches','Position',[1 1 12 5]);

% Left panel: 1Q-ahead
ax1 = subplot(1,2,1);
scatter(x_inf1, s_inf1, 14, [0.2 0.4 0.8], 'filled', 'MarkerFaceAlpha', 0.55);
hold on;
xv1 = linspace(min(x_inf1), max(x_inf1), 200)';
plot(xv1, mu_inf1 + phi_inf1*xv1, 'r-', 'LineWidth', 2);
xlabel('Normalised inflation factor  x_{m,norm}(t)', 'FontSize', 11);
ylabel('Normalised SPF forecast  s_{norm}(t+1)', 'FontSize', 11);
title(sprintf('1Q-ahead  (dpgdp3)\n\\phi = %.4f,  R^2 = %.3f,  \\sigma = %.4f', ...
    phi_inf1, R2_inf1, sig_inf1), 'FontSize', 11);
legend({'Data', 'OLS fit'}, 'Location', 'northwest', 'FontSize', 10);
grid on;  box on;

% Right panel: 4Q-ahead
ax2 = subplot(1,2,2);
scatter(x_inf4, s_inf4, 14, [0.2 0.4 0.8], 'filled', 'MarkerFaceAlpha', 0.55);
hold on;
xv4 = linspace(min(x_inf4), max(x_inf4), 200)';
plot(xv4, mu_inf4 + phi_inf4*xv4, 'r-', 'LineWidth', 2);
xlabel('Normalised Inflation Factor  x_{m,norm}(t)', 'FontSize', 11);
ylabel('Normalised SPF forecast  s_{norm}(t+1|t)', 'FontSize', 11);
title(sprintf('4Q-ahead  (dpgdp6)\n\\phi = %.4f,  R^2 = %.3f,  \\sigma = %.4f', ...
    phi_inf4, R2_inf4, sig_inf4), 'FontSize', 11);
legend({'Data', 'OLS fit'}, 'Location', 'northwest', 'FontSize', 10);
grid on;  box on;

% Align y-axis limits across panels for direct visual comparison
y_inf_lim = [ min([ax1.YLim(1), ax2.YLim(1)]), ...
              max([ax1.YLim(2), ax2.YLim(2)]) ];
ax1.YLim = y_inf_lim;
ax2.YLim = y_inf_lim;

sgtitle({'Survey Calibration: Inflation', ...
         's_{norm}(t+1|t) = \mu + \phi \cdot x_{m,norm}(t) + \eta_t'}, ...
    'FontSize', 13, 'FontWeight', 'bold');

exportgraphics(fig1, 'Graphs/CalibrationSurvey_Inflation_1Q_vs_4Q.pdf', ...
    'ContentType', 'vector');
fprintf('\nFigure 1 saved: Graphs/CalibrationSurvey_Inflation_1Q_vs_4Q.pdf\n');

%% Figure 2 — Unemployment: 1Q-ahead vs 4Q-ahead

fig2 = figure('Name','Survey Calibration — Unemployment','NumberTitle','off', ...
              'Units','inches','Position',[1 1 12 5]);

% Left panel: 1Q-ahead
ax3 = subplot(1,2,1);
scatter(x_une1, s_une1, 14, [0.15 0.55 0.35], 'filled', 'MarkerFaceAlpha', 0.55);
hold on;
xu1 = linspace(min(x_une1), max(x_une1), 200)';
plot(xu1, mu_une1 + phi_une1*xu1, 'r-', 'LineWidth', 2);
xlabel('Normalised unemployment factor  x_{m,norm}(t)', 'FontSize', 11);
ylabel('Normalised SPF forecast  s_{norm}(t+1|t)', 'FontSize', 11);
title(sprintf('1Q-ahead  (UNEMP3)\n\\phi = %.4f,  R^2 = %.3f,  \\sigma = %.4f', ...
    phi_une1, R2_une1, sig_une1), 'FontSize', 11);
legend({'Data', 'OLS fit'}, 'Location', 'northwest', 'FontSize', 10);
grid on;  box on;

% Right panel: 4Q-ahead
ax4 = subplot(1,2,2);
scatter(x_une4, s_une4, 14, [0.15 0.55 0.35], 'filled', 'MarkerFaceAlpha', 0.45);
hold on;
xu4 = linspace(min(x_une4), max(x_une4), 200)';
plot(xu4, mu_une4 + phi_une4*xu4, 'r-', 'LineWidth', 2);
xlabel('Normalised unemployment factor  x_{m,norm}(t)', 'FontSize', 11);
ylabel('Normalised SPF forecast  s_{norm}(t+1)', 'FontSize', 11);
title(sprintf('4Q-ahead  (UNEMP6)\n\\phi = %.4f,  R^2 = %.3f,  \\sigma = %.4f', ...
    phi_une4, R2_une4, sig_une4), 'FontSize', 11);
legend({'Data', 'OLS fit'}, 'Location', 'northwest', 'FontSize', 10);
grid on;  box on;

% Align y-axis limits across panels
y_une_lim = [ min([ax3.YLim(1), ax4.YLim(1)]), ...
              max([ax3.YLim(2), ax4.YLim(2)]) ];
ax3.YLim = y_une_lim;
ax4.YLim = y_une_lim;

sgtitle({'Survey Calibration: Unemployment', ...
         's_{norm}(t+1|t) = \mu + \phi \cdot x_{m,norm}(t) + \eta_t'}, ...
    'FontSize', 13, 'FontWeight', 'bold');

exportgraphics(fig2, 'Graphs/CalibrationSurvey_Unemployment_1Q_vs_4Q.pdf', ...
    'ContentType', 'vector');
fprintf('Figure 2 saved: Graphs/CalibrationSurvey_Unemployment_1Q_vs_4Q.pdf\n\n');

fprintf('Done.\n');

%% Local Functions
%
%   Regresses s_norm(t+1) on x_m_norm(t) for t = 1..T-1.
%   Removes NaN observations before estimation.
%
%   Inputs:
%     x_row  : 1 x T or T x 1  normalised macro factor (row of xMacro)
%     s_col  : T x 1 or 1 x T  normalised survey series
%
%   Outputs:
%     mu, phi  : OLS intercept and slope
%     sigma    : residual std  (ddof = n-2)
%     R2       : coefficient of determination
%     x_out    : n_valid x 1  macro values used in regression (for scatter)
%     s_out    : n_valid x 1  survey values used in regression (for scatter)

function [mu, phi, sigma, R2, x_out, s_out] = ols_calib(x_row, s_col)

    % Align as column vectors and form (t+1, t) pairs
    s_next = s_col(:);    s_next = s_next(2:end);
    x_curr = x_row(:);   x_curr = x_curr(1:end-1);

    % Drop NaN observations
    ok    = ~(isnan(s_next) | isnan(x_curr));
    s_out = s_next(ok);
    x_out = x_curr(ok);
    n     = sum(ok);

    % OLS: s = [1, x] * [mu; phi]
    X   = [ones(n,1), x_out];
    b   = X \ s_out;
    mu  = b(1);
    phi = b(2);

    % Residuals, sigma (ddof = n-2), R^2
    resid = s_out - X*b;
    sigma = sqrt(sum(resid.^2) / (n-2));
    R2    = 1 - sum(resid.^2) / sum((s_out - mean(s_out)).^2);

end