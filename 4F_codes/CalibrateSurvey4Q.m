% Calibrates the survey simulation parameters used in the
% Monte Carlo study (MonteCarloGATSM_KO_TwoEstimators.m)

clear; clc;

%% Load SPF survey data

T_inf  = readtable('US_SurveyData.xlsx', 'Sheet', 'Inflation');
T_une  = readtable('US_SurveyData.xlsx', 'Sheet', 'Unemployment');

svy_inf_raw = T_inf.dpgdp6;   % 4Q-ahead inflation forecast (% ann.), T x 1
% Handle 1 NaN in dpgdp6 (1974Q3) by substituting dpgdp5
nan_inf = isnan(svy_inf_raw);
if any(nan_inf)
    svy_inf_raw(nan_inf) = T_inf.dpgdp5(nan_inf);
    fprintf('  Replaced %d NaN(s) in dpgdp6 with dpgdp5.\n', sum(nan_inf));
end
svy_une_raw = T_une.UNEMP6;   % 4Q-ahead unemployment forecast (%),   T x 1
% Handle 1 NaN in UNEMP6 by substituting UNEMP5
nan_une = isnan(svy_une_raw);
if any(nan_une)
    svy_une_raw(nan_une) = T_une.UNEMP5(nan_une);
    fprintf('  Replaced %d NaN(s) in UNEMP6 with UNEMP5.\n', sum(nan_une));
end
T_svy = length(svy_inf_raw);

fprintf('  Observations: T=%d\n', T_svy);
fprintf('  Inflation  raw: mean=%.6f  std=%.6f\n', ...
    mean(svy_inf_raw), std(svy_inf_raw));
fprintf('  Unemployment raw: mean=%.6f  std=%.6f\n', ...
    mean(svy_une_raw), std(svy_une_raw));

%% Normalise surveys by their own mean and std

mu_raw_inf  = mean(svy_inf_raw);   std_raw_inf  = std(svy_inf_raw);
mu_raw_une  = mean(svy_une_raw);   std_raw_une  = std(svy_une_raw);

svy_inf = (svy_inf_raw - mu_raw_inf) / std_raw_inf;
svy_une = (svy_une_raw - mu_raw_une) / std_raw_une;

fprintf('\nAfter normalisation by own mean/std:\n');
fprintf('  Inflation:    mean=%.6f  std=%.6f\n', mean(svy_inf), std(svy_inf));
fprintf('  Unemployment: mean=%.6f  std=%.6f\n', mean(svy_une), std(svy_une));

%% Load macro actuals (quarterly, already normalised)

macro_raw = readmatrix('US_macro_inflation_unemployment_MF.csv');

% Subsample to quarterly frequency: end-of-quarter rows (every 3rd row,
% starting from row 3, i.e. MATLAB indices 3,6,9,...)
qi     = 3:3:size(macro_raw,1);
xMacro = macro_raw(qi,:)';   % 2 x T  (row 1 = inflation, row 2 = unemployment)
T_mac  = size(xMacro,2);

fprintf('  Quarterly observations: T=%d\n', T_mac);
fprintf('  Inflation (normalised):    mean=%.6f  std=%.6f\n', ...
    mean(xMacro(1,:)), std(xMacro(1,:)));
fprintf('  Unemployment (normalised): mean=%.6f  std=%.6f\n', ...
    mean(xMacro(2,:)), std(xMacro(2,:)));

% Verify alignment
assert(T_mac == T_svy, ...
    sprintf('Length mismatch: macro T=%d, survey T=%d', T_mac, T_svy));

%% OLS calibration for each macro factor
%
%    Regression: s_norm(t+1) = mu + phi * x_m_norm(t) + eta_t
%
%    s_norm(t+1): survey forecast available at t, pertaining to t+1
%    x_m_norm(t): current normalised macro state at t
%
%    t = 1..T-1  (lose one observation for the lag)

survey_names = {'inflation', 'unemployment'};
survey_data  = {svy_inf, svy_une};
macro_rows   = {xMacro(1,:), xMacro(2,:)};
spf_labels   = {'dpgdp6 (4Q-ahead inflation)', 'UNEMP6 (4Q-ahead unemployment)'};

% storage
svy_mu  = zeros(2,1);
svy_phi = zeros(2,1);
svy_sig = zeros(2,1);
svy_R2  = zeros(2,1);

for i = 1:2

    % Force column vectors regardless of readtable / matrix orientation
    s_next = survey_data{i}(2:end);    s_next = s_next(:);   % (T-1) x 1
    x_curr = macro_rows{i}(1:end-1);  x_curr = x_curr(:);   % (T-1) x 1

    % Remove NaN rows
    ok     = ~(isnan(s_next) | isnan(x_curr));
    s_     = s_next(ok);
    x_     = x_curr(ok);
    n      = sum(ok);

    % OLS: s = X*b + e,  X = [1, x_m]
    X      = [ones(n,1), x_];
    b      = X \ s_;
    mu_hat = b(1);
    phi_hat= b(2);

    % Residuals and sigma
    resid     = s_ - X*b;
    sigma_hat = std(resid, 0);   % std with ddof=n-1 (MATLAB default)
    % Use ddof=n-2 (OLS convention: 2 parameters estimated)
    sigma_hat = sqrt(sum(resid.^2) / (n-2));

    ss_res    = sum(resid.^2);
    ss_tot    = sum((s_ - mean(s_)).^2);
    R2        = 1 - ss_res/ss_tot;

    % Standard errors
    s2       = ss_res / (n-2);
    XtX_inv  = inv(X'*X);
    se_b     = sqrt(s2 * diag(XtX_inv));
    t_mu     = mu_hat  / se_b(1);
    t_phi    = phi_hat / se_b(2);

    % Autocorrelations (force column vectors)
    s_all  = survey_data{i}(:);
    x_all  = macro_rows{i}(:);
    ac_svy = corr(s_all(1:end-1), s_all(2:end));
    ac_act = corr(x_all(1:end-1), x_all(2:end));

    % Store
    svy_mu(i)  = mu_hat;
    svy_phi(i) = phi_hat;
    svy_sig(i) = sigma_hat;
    svy_R2(i)  = R2;

    fprintf('\n%s (%s)\n', upper(survey_names{i}), spf_labels{i});
    fprintf('  Raw mean: %.6f   Raw std: %.6f\n', ...
        mu_raw_inf*(i==1) + mu_raw_une*(i==2), ...
        std_raw_inf*(i==1) + std_raw_une*(i==2));
    fprintf('  Regression (n=%d):\n', n);
    fprintf('    s_norm(t+1) = %.8f + %.8f * x_norm(t) + eta\n', mu_hat, phi_hat);
    fprintf('    SE:           (%.6f)   (%.6f)\n', se_b(1), se_b(2));
    fprintf('    t-stat:       [%.2f]        [%.2f]\n', t_mu, t_phi);
    fprintf('  sigma_svy (residual std, ddof=n-2): %.8f\n', sigma_hat);
    fprintf('  R^2 = %.4f\n', R2);
    fprintf('  Autocorrelation: AC(survey)=%.6f   AC(actual)=%.6f\n', ...
        ac_svy, ac_act);

end

%% Summary: parameters for MC script

fprintf('svy_mu  = [%.10f; %.10f];\n', svy_mu(1),  svy_mu(2));
fprintf('svy_phi = [%.10f; %.10f];\n', svy_phi(1), svy_phi(2));
fprintf('svy_sig = [%.10f; %.10f];\n', svy_sig(1), svy_sig(2));
fprintf('\n');
fprintf('%% Expected 4Q-ahead values:\n');
fprintf('%%   svy_mu  ~ [-0.0009; -0.0020]  (near zero after normalisation)\n');
fprintf('%%   svy_phi ~ [ 0.7879;  0.9578]  (dpgdp6/UNEMP6)\n');
fprintf('%%   svy_sig ~ [ 0.6184;  0.2938]  (inflation noisier, unemp tighter)\n');

%% Diagnostic plot

figure('Name','Survey Calibration Diagnostic','NumberTitle','off');

for i = 1:2
    s_next = survey_data{i}(2:end);  s_next = s_next(:);
    x_curr = macro_rows{i}(1:end-1); x_curr = x_curr(:);
    ok     = ~(isnan(s_next) | isnan(x_curr));

    subplot(2,2,(i-1)*2+1);
    scatter(x_curr(ok), s_next(ok), 10, 'b', 'filled', 'MarkerFaceAlpha', 0.4);
    hold on;
    xline_vec = linspace(min(x_curr(ok)), max(x_curr(ok)), 100)';
    yfit = svy_mu(i) + svy_phi(i)*xline_vec;
    plot(xline_vec, yfit, 'r-', 'LineWidth', 1.5);
    xlabel('x\_m\_norm(t)'); ylabel('s\_norm(t+1)');
    title(sprintf('%s: phi=%.4f, R^2=%.3f', survey_names{i}, svy_phi(i), svy_R2(i)));
    grid on;

    subplot(2,2,(i-1)*2+2);
    % Residual histogram
    s_     = s_next(ok);
    x_     = x_curr(ok);
    resid  = s_ - svy_mu(i) - svy_phi(i)*x_;
    histogram(resid, 30, 'Normalization', 'pdf', 'FaceColor', [0.5,0.7,1]);
    hold on;
    xr = linspace(min(resid), max(resid), 200);
    plot(xr, normpdf(xr, 0, svy_sig(i)), 'r-', 'LineWidth', 1.5);
    xlabel('Residual'); ylabel('Density');
    title(sprintf('%s residuals: sigma=%.4f', survey_names{i}, svy_sig(i)));
    grid on;
end

sgtitle('Survey Model Calibration (4Q-ahead): s\_norm(t+1) = mu + phi*x\_m(t) + eta');
exportgraphics(gcf,'Graphs/CalibrationSurveyPlot4Q.pdf');

%% Save results

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

save('SurveyCalibration4Q.mat', '-struct', 'calib');
fprintf('\nResults saved to SurveyCalibration4Q.mat\n');
fprintf('Diagnostic plot generated.\n');