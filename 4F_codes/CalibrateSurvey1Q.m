% Calibrates the survey simulation parameters for the 1Q MC study
% (MonteCarloGATSM_KO_TwoEstimators_V6.m).

clear; clc;

%% Load SPF survey data — 1Q-ahead forecasts

T_inf = readtable('US_SurveyData.xlsx', 'Sheet', 'Inflation');
T_une = readtable('US_SurveyData.xlsx', 'Sheet', 'Unemployment');

svy_inf_raw = T_inf.dpgdp3;   % h=1: forecast made at t for quarter t+1  (% ann.)
svy_une_raw = T_une.UNEMP3;   % h=1: forecast made at t for quarter t+1  (%)
T_svy = length(svy_inf_raw);

fprintf('  Survey column (inflation):    dpgdp3  (h=1, 1Q-ahead)\n');
fprintf('  Survey column (unemployment): UNEMP3  (h=1, 1Q-ahead)\n');
fprintf('  Observations: T=%d\n', T_svy);
fprintf('  Inflation  raw: mean=%.6f  std=%.6f\n', ...
    nanmean(svy_inf_raw), nanstd(svy_inf_raw));
fprintf('  Unemployment raw: mean=%.6f  std=%.6f\n', ...
    nanmean(svy_une_raw), nanstd(svy_une_raw));

%% Normalise surveys by their own mean and std

mu_raw_inf = nanmean(svy_inf_raw);   std_raw_inf = nanstd(svy_inf_raw);
mu_raw_une = nanmean(svy_une_raw);   std_raw_une = nanstd(svy_une_raw);

svy_inf = (svy_inf_raw - mu_raw_inf) / std_raw_inf;
svy_une = (svy_une_raw - mu_raw_une) / std_raw_une;

fprintf('\nAfter normalisation by own mean/std:\n');
fprintf('  Inflation:    mean=%.6f  std=%.6f\n', ...
    nanmean(svy_inf), nanstd(svy_inf));
fprintf('  Unemployment: mean=%.6f  std=%.6f\n', ...
    nanmean(svy_une), nanstd(svy_une));

%% Load macro actuals (quarterly, already normalised)

macro_raw = readmatrix('US_macro_inflation_unemployment_MF.csv');

qi     = 3:3:size(macro_raw,1);   % end-of-quarter rows (every 3rd)
xMacro = macro_raw(qi,:)';         % 2 x T  (row1=inflation, row2=unemployment)
T_mac  = size(xMacro,2);

fprintf('  Quarterly observations: T=%d\n', T_mac);
fprintf('  Inflation (normalised):    mean=%.6f  std=%.6f\n', ...
    mean(xMacro(1,:)), std(xMacro(1,:)));
fprintf('  Unemployment (normalised): mean=%.6f  std=%.6f\n', ...
    mean(xMacro(2,:)), std(xMacro(2,:)));

assert(T_mac == T_svy, ...
    sprintf('Length mismatch: macro T=%d, survey T=%d', T_mac, T_svy));

%% OLS calibration
%
%   Regression: s_norm(t+1) = mu + phi * x_m_norm(t) + eta_t
%
%   s_norm(t+1): dpgdp3/UNEMP3 at time t+1 (1Q-ahead forecast made at t+1
%                for quarter t+2 — but what enters Block m OLS is the survey
%                at t forecasting t+1, so we regress s(t+1) on x(t))
%
%   In MATLAB: s_next = survey(2:end), x_curr = xMacro(1:end-1)
%   This gives: s(t+1) ~ x_m(t)  for t = 1..T-1

survey_names = {'Inflation', 'Unemployment'};
survey_data  = {svy_inf, svy_une};
macro_rows   = {xMacro(1,:), xMacro(2,:)};
spf_labels   = {'dpgdp3 (1Q-ahead inflation, h=1)', ...
                 'UNEMP3 (1Q-ahead unemployment, h=1)'};

svy_mu  = zeros(2,1);
svy_phi = zeros(2,1);
svy_sig = zeros(2,1);
svy_R2  = zeros(2,1);

for i = 1:2

    % s(t+1) on x_m(t): force column vectors
    s_next = survey_data{i}(2:end);    s_next = s_next(:);   % (T-1) x 1
    x_curr = macro_rows{i}(1:end-1);  x_curr = x_curr(:);   % (T-1) x 1

    ok = ~(isnan(s_next) | isnan(x_curr));
    s_ = s_next(ok);
    x_ = x_curr(ok);
    n  = sum(ok);

    % OLS
    X       = [ones(n,1), x_];
    b       = X \ s_;
    mu_hat  = b(1);
    phi_hat = b(2);

    resid     = s_ - X*b;
    sigma_hat = sqrt(sum(resid.^2) / (n-2));   % ddof = n-2
    ss_res    = sum(resid.^2);
    ss_tot    = sum((s_ - mean(s_)).^2);
    R2        = 1 - ss_res/ss_tot;

    % Standard errors
    s2      = ss_res / (n-2);
    se_b    = sqrt(s2 * diag(inv(X'*X)));
    t_phi   = phi_hat / se_b(2);

    % Autocorrelations
    s_all = survey_data{i}(:);
    x_all = macro_rows{i}(:);
    ac_svy = corr(s_all(1:end-1), s_all(2:end));
    ac_act = corr(x_all(1:end-1), x_all(2:end));

    svy_mu(i)  = mu_hat;
    svy_phi(i) = phi_hat;
    svy_sig(i) = sigma_hat;
    svy_R2(i)  = R2;

    fprintf('\n%s (%s)\n', upper(survey_names{i}), spf_labels{i});
    fprintf('  Raw mean: %.6f   Raw std: %.6f\n', ...
        mu_raw_inf*(i==1) + mu_raw_une*(i==2), ...
        std_raw_inf*(i==1) + std_raw_une*(i==2));
    fprintf('  Regression s_norm(t+1) ~ x_norm(t)  (n=%d):\n', n);
    fprintf('    mu  = %.10f  (SE=%.6f, t=%.2f)\n', ...
        mu_hat, se_b(1), mu_hat/se_b(1));
    fprintf('    phi = %.10f  (SE=%.6f, t=%.2f)\n', ...
        phi_hat, se_b(2), t_phi);
    fprintf('  sigma_svy = %.10f\n', sigma_hat);
    fprintf('  R^2       = %.4f\n', R2);
    fprintf('  AC(survey)=%.6f   AC(actual)=%.6f\n', ac_svy, ac_act);

end

%% Summary for MC script

fprintf('svy_mu  = [%.10f; %.10f];\n', svy_mu(1),  svy_mu(2));
fprintf('svy_phi = [%.10f; %.10f];\n', svy_phi(1), svy_phi(2));
fprintf('svy_sig = [%.10f; %.10f];\n', svy_sig(1), svy_sig(2));
fprintf('\n');
fprintf('%% Expected values (h=1, dpgdp3/UNEMP3):\n');
fprintf('%%   svy_mu  ~ [ 0.0003; -0.0035]\n');
fprintf('%%   svy_phi ~ [ 0.8812;  0.9955]  (unemp near-perfect predictor)\n');
fprintf('%%   svy_sig ~ [ 0.4749;  0.1161]  (unemp very tight, R^2=0.987)\n');

%% Diagnostic plot

figure('Name','Survey Calibration V6 (h=1)','NumberTitle','off');

for i = 1:2
    s_next = survey_data{i}(2:end);    s_next = s_next(:);
    x_curr = macro_rows{i}(1:end-1);  x_curr = x_curr(:);
    ok     = ~(isnan(s_next) | isnan(x_curr));

    subplot(2,2,(i-1)*2+1);
    scatter(x_curr(ok), s_next(ok), 10, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    xv   = linspace(min(x_curr(ok)), max(x_curr(ok)), 100)';
    yfit = svy_mu(i) + svy_phi(i)*xv;
    plot(xv, yfit, 'r-', 'LineWidth', 1.5);
    xlabel('x\_m\_norm(t)');  ylabel('s\_norm(t+1)');
    title(sprintf('%s: phi=%.4f, R^2=%.3f', survey_names{i}, svy_phi(i), svy_R2(i)));
    grid on;

    subplot(2,2,(i-1)*2+2);
    s_   = s_next(ok);
    x_   = x_curr(ok);
    resid = s_ - svy_mu(i) - svy_phi(i)*x_;
    histogram(resid, 30, 'Normalization', 'pdf', 'FaceColor', [0.5,0.7,1]);
    hold on;
    xr = linspace(min(resid), max(resid), 200);
    plot(xr, normpdf(xr, 0, svy_sig(i)), 'r-', 'LineWidth', 1.5);
    xlabel('Residual');  ylabel('Density');
    title(sprintf('%s residuals: sigma=%.4f', survey_names{i}, svy_sig(i)));
    grid on;
end

sgtitle('Survey Calibration 1Q (h=1): s\_norm\_(t+1|t) = mu + phi*x\_m(t) + eta');

if ~exist('Graphs','dir'), mkdir('Graphs'); end
exportgraphics(gcf, 'Graphs/CalibrationSurveyPlot1Q.pdf');

%% 7. Save results

calib = struct();
calib.svy_mu       = svy_mu;
calib.svy_phi      = svy_phi;
calib.svy_sig      = svy_sig;
calib.svy_R2       = svy_R2;
calib.mu_raw_inf   = mu_raw_inf;
calib.std_raw_inf  = std_raw_inf;
calib.mu_raw_une   = mu_raw_une;
calib.std_raw_une  = std_raw_une;
calib.survey_names = survey_names;
calib.spf_cols     = {'dpgdp3','UNEMP3'};
calib.horizon      = 1;   % h=1 quarters ahead

save('SurveyCalibration1Q.mat', '-struct', 'calib');
fprintf('\nResults saved to SurveyCalibration_V6.mat\n');
fprintf('Diagnostic plot saved to Graphs/CalibrationSurveyPlot_V6.pdf\n');