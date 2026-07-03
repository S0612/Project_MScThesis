% Monte Carlo Simulations using the estimate from GATSM_US_2F_V2.

clear; clc; close all;

%%  USER SETTINGS

R = 1000;            % number of Monte Carlo replications
T_sim = 648;        % sample length (same as original: Jan 1972 – Dec 2025)
K = 2;              % number of latent factors

% Maturities (in months) — matches matSelect in GATSM_US_2factor.m exactly:
%   [1 2 5 7 10 15] years -> [12 24 60 84 120 180] months
maturities = [1 2 5 7 10 15] * 12;
N = length(maturities);   % = 6

% delta1 normalisation (same as estimation)
% delta1 is beta in our notation. delta is the old notation from Ang & Piazzesi.
delta1 = ones(K, 1);

%%  True param vector

load('GATSM_2F_Estimates_US_CMAES.mat', 'outML');

model_true   = outML.outKF.model;           % struct built by local_solveATSM

delta0_v2    = model_true.alpha;            % monthly short-rate intercept : alpha
muP_v2       = model_true.muP;              % K x 1, monthly P-mean
phiP_v2      = model_true.phiP;             % K x K, monthly P-AR matrix
sigma_v2     = model_true.sigma;            % K x K, lower-triangular Cholesky
phiQ_diag_v2 = diag(model_true.phiQ);      % K x 1 diagonal only (off-diags are
                                            %   Jordan corrections, not free params)
stdY_v2      = outML.paramsOpt.stdY / 12;  % annualised -> monthly units (divide by 12)

% Ensure phiQ diagonal is sorted descending
phiQ_diag_v2 = sort(phiQ_diag_v2, 'descend');

% Pack into the Monte Carlo theta vector
theta_true = pack_params(delta0_v2, muP_v2, phiP_v2, sigma_v2, ...
                         phiQ_diag_v2, stdY_v2, K);

%%  Unpack 'true' parameters

[delta0_true, mu_true, Phi_true, Sigma_true, phiQ_true, sigma_e_true] = ...
    unpack_params(theta_true, K);

PhiQ_true = diag(phiQ_true);
Q_cov_true = Sigma_true * Sigma_true';

% Measurement loadings under the true DGP
[a_true, H_true] = measurement_loadings(PhiQ_true, Sigma_true, delta0_true, delta1, maturities);

% Unconditional mean and variance of the state (if stationary)
max_eig_true = max(abs(eig(Phi_true)));
if max_eig_true < 1
    X_uncond = (eye(K) - Phi_true) \ mu_true;
    P_uncond = dlyap_solve(Phi_true, Q_cov_true);
else
    X_uncond = mu_true;
    P_uncond = 10 * Q_cov_true;
end

n_params = length(theta_true);

fprintf('=== Monte Carlo Simulation for GATSM Finite-Sample Bias ===\n');
fprintf('Replications: %d | T = %d | K = %d | N = %d maturities\n', R, T_sim, K, N);

%%  Setup storage matrices


theta_hat_all = NaN(R, n_params);   % estimated theta for each replication
nll_all       = NaN(R, 1);          % neg-loglik at optimum
converged     = false(R, 1);        % convergence flag
elapsed       = NaN(R, 1);          % wall-clock time per replication

%%  This here is the main Monte Carlo (parfor-enabled)
% It uses parfor so it requires the matlab paralel computing toolboxing
% addon

% Pre-generate one independent RNG seed per replication for reproducibility. Each worker will initialise its own RNG with its seed
rng(456, 'twister');
seeds = randi(2^31 - 1, R, 1);

% Nelder-Mead options
opts_nm = optimset('Display', 'off', ...
                   'MaxIter', 10000, ...
                   'MaxFunEvals', 100000, ...
                   'TolFun', 1e-6, ...
                   'TolX', 1e-6);

% Pre-compute Cholesky of unconditional covariance (constant across reps)
chol_P = chol(P_uncond, 'lower');

% Open parallel pooling
pool = gcp('nocreate');
if isempty(pool)
    try
        parpool;
    catch
        fprintf('Parallel pool not available — running sequential for-loop.\n');
    end
end

fprintf('Starting %d replications ...\n\n', R);
timer_total = tic;

%% MAIN LOOP
parfor r = 1:R
    t_start = tic;

    % Reproducible RNG for this replication
    rng_local = RandStream('twister', 'Seed', seeds(r));

    %% 1: Simulate data from the true DGP ---
    
    X_sim = zeros(T_sim, K);
    X_sim(1, :) = (X_uncond + chol_P * randn(rng_local, K, 1))';

    for t = 2:T_sim
        X_sim(t, :) = (mu_true + Phi_true * X_sim(t-1, :)' ...
                       + Sigma_true * randn(rng_local, K, 1))';
    end

    yields_sim = zeros(T_sim, N);
    for t = 1:T_sim
        yields_sim(t, :) = (a_true + H_true * X_sim(t, :)' ...
                            + sigma_e_true * randn(rng_local, N, 1))';
    end

    %% 2: Build PCA-based starting values ---
    
    yields_dm = yields_sim - mean(yields_sim, 1);
    [~, score, ~] = pca(yields_dm);
    X_pca = score(:, 1:K);

    X_lag = X_pca(1:end-1, :);
    X_cur = X_pca(2:end, :);
    beta_var = [ones(T_sim-1, 1), X_lag] \ X_cur;
    mu0   = beta_var(1, :)';
    Phi0  = beta_var(2:end, :)';
    resid = X_cur - [ones(T_sim-1, 1), X_lag] * beta_var;
    Sigma_resid = cov(resid);
    Sigma0 = chol(Sigma_resid, 'lower');

    eig_Phi = sort(real(eig(Phi0)), 'descend');
    phiQ_diag0 = min(max(eig_Phi, 0.5), 0.999);

    delta0_init = mean(yields_sim(:, 1));
    sigma_e_init = 0.001;

    theta0_pca = pack_params(delta0_init, mu0, Phi0, Sigma0, phiQ_diag0, sigma_e_init, K);

    %% 3: Nelder-Mead starting values.
    % We use the DGP and the PCA-based approach as the starting values for
    % each rep.
    % Each rep then goes with the best of the two
    
    obj = @(theta) neg_log_likelihood(theta, yields_sim, maturities, delta1, K);

    best_nll_r   = Inf;
    best_theta_r = theta_true;

    % Start 1: true parameter vector
    try
        [theta_nm1, nll_nm1] = fminsearch(obj, theta_true, opts_nm);
        if nll_nm1 < best_nll_r
            best_nll_r   = nll_nm1;
            best_theta_r = theta_nm1;
        end
    catch
    end

    % Start 2: PCA-based initialisation
    try
        [theta_nm2, nll_nm2] = fminsearch(obj, theta0_pca, opts_nm);
        if nll_nm2 < best_nll_r
            best_nll_r   = nll_nm2;
            best_theta_r = theta_nm2;
        end
    catch
    end

    %% 4: Store results in storage matrices
    
    theta_hat_all(r, :) = best_theta_r';
    nll_all(r)          = best_nll_r;
    converged(r)        = isfinite(best_nll_r);
    elapsed(r)          = toc(t_start);

    % Progress print
    fprintf('Replication %4d/%d done  |  nll = %10.2f  |  time = %.1f s\n', ...
        r, R, best_nll_r, elapsed(r));
end

% timer for how long all replications took
total_time = toc(timer_total);
fprintf('\n=== Monte Carlo Complete ===\n');
fprintf('Converged: %d / %d\n', sum(converged), R);
fprintf('Mean time per replication: %.1f s\n', mean(elapsed));
fprintf('Total wall-clock time: %.1f minutes (%.2f hours)\n\n', ...
    total_time / 60, total_time / 3600);

%%  Analyse bias

% Use only converged replications
idx_ok = converged;
theta_ok = theta_hat_all(idx_ok, :);
R_ok = sum(idx_ok);

% Prints how many replications converged 
fprintf('%d converged replications', R_ok);

% Unpack true parameters into a labelled structure for comparison
[d0_t, mu_t, Phi_t, Sig_t, phiQ_t, se_t] = unpack_params(theta_true, K);

% Unpack each replication's estimates
delta0_hat_all = NaN(R_ok, 1);
mu_hat_all     = NaN(R_ok, K);
Phi_hat_all    = NaN(R_ok, K^2);
Sigma_hat_all  = NaN(R_ok, K*(K+1)/2);
phiQ_hat_all   = NaN(R_ok, K);
sigma_e_hat_all = NaN(R_ok, 1);

for r = 1:R_ok
    [d0, mu_r, Phi_r, Sig_r, phiQ_r, se_r] = unpack_params(theta_ok(r, :)', K);
    delta0_hat_all(r) = d0;
    mu_hat_all(r, :)  = mu_r';
    Phi_hat_all(r, :) = Phi_r(:)';
    
    % Extract lower-triangular entries
    eidx = 0;
    sig_entries = zeros(1, K*(K+1)/2);
    for col = 1:K
        for row = col:K
            eidx = eidx + 1;
            sig_entries(eidx) = Sig_r(row, col);
        end
    end
    Sigma_hat_all(r, :) = sig_entries;
    phiQ_hat_all(r, :)  = phiQ_r';
    sigma_e_hat_all(r)  = se_r;
end

% True values in matching format
Sig_true_entries = zeros(1, K*(K+1)/2);
eidx = 0;
for col = 1:K
    for row = col:K
        eidx = eidx + 1;
        Sig_true_entries(eidx) = Sig_t(row, col);
    end
end

%% Bias table

param_names = {};
true_vals   = [];
mean_vals   = [];
median_vals = [];
std_vals    = [];

% computes mean and bias
param_names{end+1} = 'delta0';
true_vals(end+1) = d0_t;
mean_vals(end+1) = mean(delta0_hat_all);
median_vals(end+1) = median(delta0_hat_all);
std_vals(end+1) = std(delta0_hat_all);

% mu
for i = 1:K
    param_names{end+1} = sprintf('mu_%d', i);
    true_vals(end+1) = mu_t(i);
    mean_vals(end+1) = mean(mu_hat_all(:, i));
    median_vals(end+1) = median(mu_hat_all(:, i));
    std_vals(end+1) = std(mu_hat_all(:, i));
end

% Phi
for j = 1:K
    for i = 1:K
        param_names{end+1} = sprintf('Phi_%d%d', i, j);
        true_vals(end+1) = Phi_t(i, j);
        mean_vals(end+1) = mean(Phi_hat_all(:, (j-1)*K + i));
        median_vals(end+1) = median(Phi_hat_all(:, (j-1)*K + i));
        std_vals(end+1) = std(Phi_hat_all(:, (j-1)*K + i));
    end
end

% Sigma entries
sigma_labels = {};
eidx = 0;
for col = 1:K
    for row = col:K
        eidx = eidx + 1;
        sigma_labels{eidx} = sprintf('Sigma_%d%d', row, col);
    end
end

for i = 1:length(sigma_labels)
    param_names{end+1} = sigma_labels{i};
    true_vals(end+1) = Sig_true_entries(i);
    mean_vals(end+1) = mean(Sigma_hat_all(:, i));
    median_vals(end+1) = median(Sigma_hat_all(:, i));
    std_vals(end+1) = std(Sigma_hat_all(:, i));
end

% phiQ
for i = 1:K
    param_names{end+1} = sprintf('phiQ_%d', i);
    true_vals(end+1) = phiQ_t(i);
    mean_vals(end+1) = mean(phiQ_hat_all(:, i));
    median_vals(end+1) = median(phiQ_hat_all(:, i));
    std_vals(end+1) = std(phiQ_hat_all(:, i));
end

% sigma_e
param_names{end+1} = 'sigma_e';
true_vals(end+1) = se_t;
mean_vals(end+1) = mean(sigma_e_hat_all);
median_vals(end+1) = median(sigma_e_hat_all);
std_vals(end+1) = std(sigma_e_hat_all);

% Display table
bias_vals = mean_vals - true_vals;
rel_bias  = 100 * bias_vals ./ max(abs(true_vals), 1e-8);

fprintf('\n%-12s  %10s  %10s  %10s  %10s  %10s  %10s\n', ...
    'Parameter', 'True', 'Mean', 'Median', 'Std', 'Bias', 'Bias(%)');
fprintf('%s\n', repmat('-', 1, 78));
for i = 1:length(param_names)
    fprintf('%-12s  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %9.2f%%\n', ...
        param_names{i}, true_vals(i), mean_vals(i), median_vals(i), ...
        std_vals(i), bias_vals(i), rel_bias(i));
end

%% Saving the workspace

%save("MonteCarlo_GATSM_US_2F") ;

%% Latex Table

fid = fopen('MC_bias_table.tex', 'w');
fprintf(fid, '\\begin{table}[htbp]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Monte Carlo Finite-Sample Bias ($R=%d$, $T=%d$)}\n', R, T_sim);
fprintf(fid, '\\label{tab:mc_bias}\n');
fprintf(fid, '\\begin{tabular}{l|ccccc}\n');
fprintf(fid, '\\hline\\hline\n');
fprintf(fid, 'Parameter & True & Mean & Median & Std & Bias \\\\\n');
fprintf(fid, '\\hline\n');

for i = 1:length(param_names)
    % Make parameter names LaTeX-friendly
    pname = param_names{i};
    pname = strrep(pname, '_', '\_');
    fprintf(fid, '%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\\n', ...
        pname, true_vals(i), mean_vals(i), median_vals(i), std_vals(i), bias_vals(i));
end

fprintf(fid, '\\hline\\hline\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\end{table}\n');
fclose(fid);
fprintf('\nLaTeX bias table saved to MC_bias_table.tex\n');


%%  Histograms

% --- Histograms of key parameters ---
figure('Name', 'MC: PhiQ Diagonals', 'Position', [100 100 900 400]);
for i = 1:K
    subplot(1, K, i);
    histogram(phiQ_hat_all(:, i), 30, 'Normalization', 'pdf', 'FaceAlpha', 0.6);
    hold on;
    xline(phiQ_t(i), 'r-', 'LineWidth', 2);
    xline(mean(phiQ_hat_all(:, i)), 'b--', 'LineWidth', 1.5);
    title(sprintf('\\lambda_{%d}^Q', i));
    legend('Estimates', 'True', 'Mean est.', 'Location', 'best');
    grid on;
end
sgtitle('Monte Carlo Distribution of \Phi^Q Eigenvalues');

figure('Name', 'MC: delta0 and sigma_e', 'Position', [100 100 800 350]);
subplot(1, 2, 1);
histogram(delta0_hat_all, 30, 'Normalization', 'pdf', 'FaceAlpha', 0.6);
hold on;
xline(d0_t, 'r-', 'LineWidth', 2);
xline(mean(delta0_hat_all), 'b--', 'LineWidth', 1.5);
title('\delta_0 (\alpha)');
legend('Estimates', 'True', 'Mean est.', 'Location', 'best');
grid on;

subplot(1, 2, 2);
histogram(sigma_e_hat_all, 30, 'Normalization', 'pdf', 'FaceAlpha', 0.6);
hold on;
xline(se_t, 'r-', 'LineWidth', 2);
xline(mean(sigma_e_hat_all), 'b--', 'LineWidth', 1.5);
title('\sigma_e');
legend('Estimates', 'True', 'Mean est.', 'Location', 'best');
grid on;
sgtitle('Monte Carlo Distribution of \delta_0 and \sigma_e');

figure('Name', 'MC: Phi diagonal', 'Position', [100 100 900 400]);
for i = 1:K
    subplot(1, K, i);
    phi_diag_idx = (i-1)*K + i;  % diagonal element index in column-major
    histogram(Phi_hat_all(:, phi_diag_idx), 30, 'Normalization', 'pdf', 'FaceAlpha', 0.6);
    hold on;
    xline(Phi_t(i, i), 'r-', 'LineWidth', 2);
    xline(mean(Phi_hat_all(:, phi_diag_idx)), 'b--', 'LineWidth', 1.5);
    title(sprintf('\\Phi_{%d%d}^P', i, i));
    legend('Estimates', 'True', 'Mean est.', 'Location', 'best');
    grid on;
end
sgtitle('Monte Carlo Distribution of \Phi^P Diagonal Elements');

%%  LOCAL FUNCTIONS

%--------------------------------------------------------------------------
% PACK PARAMETERS
%--------------------------------------------------------------------------
function theta = pack_params(delta0, mu, Phi, Sigma_lower, phiQ_diag, sigma_e, K)
    sigma_entries = zeros(K*(K+1)/2, 1);
    idx = 0;
    for col = 1:K
        for row = col:K
            idx = idx + 1;
            if row == col
                sigma_entries(idx) = log(Sigma_lower(row, col));
            else
                sigma_entries(idx) = Sigma_lower(row, col);
            end
        end
    end
    phiQ_unc = log(phiQ_diag ./ (1 - phiQ_diag));
    theta = [delta0; mu(:); Phi(:); sigma_entries; phiQ_unc(:); log(sigma_e)];
end

%--------------------------------------------------------------------------
% UNPACK PARAMETERS
%--------------------------------------------------------------------------
function [delta0, mu, Phi, Sigma_mat, phiQ_diag, sigma_e] = unpack_params(theta, K)
    n_sigma = K*(K+1)/2;
    idx = 1;
    delta0 = theta(idx);                                 idx = idx + 1;
    mu     = theta(idx:idx+K-1);                         idx = idx + K;
    Phi    = reshape(theta(idx:idx+K^2-1), K, K);        idx = idx + K^2;
    sigma_entries = theta(idx:idx+n_sigma-1);             idx = idx + n_sigma;
    phiQ_unc = theta(idx:idx+K-1);                       idx = idx + K;
    sigma_e  = exp(theta(idx));
    
    Sigma_mat = zeros(K, K);
    eidx = 0;
    for col = 1:K
        for row = col:K
            eidx = eidx + 1;
            if row == col
                Sigma_mat(row, col) = exp(sigma_entries(eidx));
            else
                Sigma_mat(row, col) = sigma_entries(eidx);
            end
        end
    end
    phiQ_diag = 1 ./ (1 + exp(-phiQ_unc));
    phiQ_diag = sort(phiQ_diag, 'descend');
end

% BOND PRICING RECURSIONS
function [A, B] = bond_recursions(PhiQ, Sigma_mat, delta0, delta1, n_max)
    K = length(delta1);
    A = zeros(n_max, 1);
    B = zeros(K, n_max);
    SigSig = Sigma_mat * Sigma_mat';
    B(:, 1) = -delta1;
    A(1)    = -delta0;
    for n = 2:n_max
        B(:, n) = PhiQ' * B(:, n-1) - delta1;
        A(n)    = A(n-1) + 0.5 * B(:, n-1)' * SigSig * B(:, n-1) - delta0;
    end
end

% MEASUREMENT LOADINGS
function [a, H] = measurement_loadings(PhiQ, Sigma_mat, delta0, delta1, maturities)
    N = length(maturities);
    K = length(delta1);
    n_max = max(maturities);
    [A_all, B_all] = bond_recursions(PhiQ, Sigma_mat, delta0, delta1, n_max);
    a = zeros(N, 1);
    H = zeros(N, K);
    for j = 1:N
        n = maturities(j);
        a(j)    = -A_all(n) / n;
        H(j, :) = -B_all(:, n)' / n;
    end
end

% KALMAN FILTER
function [loglik, X_filt, y_filt] = kalman_filter_full(theta, yields, maturities, delta1, K)
    [T, N] = size(yields);
    [delta0, mu, Phi, Sigma_mat, phiQ_diag, sigma_e] = unpack_params(theta, K);
    PhiQ = diag(phiQ_diag);
    [a, H] = measurement_loadings(PhiQ, Sigma_mat, delta0, delta1, maturities);
    R     = sigma_e^2 * eye(N);
    Q_cov = Sigma_mat * Sigma_mat';
    
    max_eig = max(abs(eig(Phi)));
    if max_eig < 0.999
        X_tt = (eye(K) - Phi) \ mu;
        P_tt = dlyap_solve(Phi, Q_cov);
    else
        X_tt = mu;
        P_tt = 10 * Q_cov;
    end
    
    loglik = 0;
    X_filt = zeros(T, K);
    y_filt = zeros(T, N);
    
    for t = 1:T
        X_pred = mu + Phi * X_tt;
        P_pred = Phi * P_tt * Phi' + Q_cov;
        P_pred = 0.5 * (P_pred + P_pred');
        
        y_pred = a + H * X_pred;
        v_t    = yields(t, :)' - y_pred;
        F_t    = H * P_pred * H' + R;
        F_t    = 0.5 * (F_t + F_t');
        
        [L_F, flag] = chol(F_t, 'lower');
        if flag ~= 0
            loglik = -1e10;
            return;
        end
        log_det_F = 2 * sum(log(diag(L_F)));
        F_inv_v   = L_F' \ (L_F \ v_t);
        loglik    = loglik - 0.5 * (N * log(2*pi) + log_det_F + v_t' * F_inv_v);
        
        K_gain = (P_pred * H') / F_t;
        X_tt   = X_pred + K_gain * v_t;
        P_tt   = (eye(K) - K_gain * H) * P_pred;
        P_tt   = 0.5 * (P_tt + P_tt');
        
        X_filt(t, :) = X_tt';
        y_filt(t, :) = (a + H * X_tt)';
    end
end

% DISCRETE LYAPUNOV SOLVER
function P = dlyap_solve(Phi, Q)
    K = size(Phi, 1);
    P_vec = (eye(K^2) - kron(Phi, Phi)) \ Q(:);
    P = reshape(P_vec, K, K);
    P = 0.5 * (P + P');
end

% NEGATIVE LOG-LIKELIHOOD
function nll = neg_log_likelihood(theta, yields, maturities, delta1, K)
    loglik = kalman_filter_full(theta, yields, maturities, delta1, K);
    nll = -loglik;
    if ~isfinite(nll)
        nll = 1e10;
    end
end