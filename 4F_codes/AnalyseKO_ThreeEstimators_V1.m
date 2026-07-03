% This script produces monte carlo tables from the output of MonteCarloGATSM_KO_ThreeEstimators.m
%
% Requires: MonteCarlo_KO_3EST_T216.mat and MonteCarlo_KO_3EST_T100.mat

clear; clc;
ann    = 4;
T_list = [216, 100];

%% Load results

results(1).T=[]; results(1).A=[]; results(1).B1=[]; results(1).B2=[];
results(1).tp=[]; results(1).R=[];
results(2) = results(1);

for iT = 1:length(T_list)
    T_sim = T_list(iT);
    fname = sprintf('MonteCarlo_KO_3EST_T%d.mat', T_sim);
    if ~exist(fname,'file')
        fprintf('File not found: %s\n', fname); continue;
    end
    d = load(fname);
    results(iT).T  = T_sim;
    results(iT).A  = d.A;
    results(iT).B1 = d.B1;
    results(iT).B2 = d.B2;
    results(iT).tp = d.true_params;
    results(iT).R  = d.R;
    fprintf('Loaded %s  (R=%d, convA=%d, convB1=%d, convB2=%d)\n', ...
        fname, d.R, sum(d.A.converged), sum(d.B1.converged), sum(d.B2.converged));
end

if isempty(results(1).T), error('No result files found.'); end

%% Parameter list and section breaks
params = {
    '$\Phi^{\mathbb{P}}_{mm,11}$', 'phiP_mm_hat', 1, @(tp) tp.phiP(1,1);
    '$\Phi^{\mathbb{P}}_{mm,21}$', 'phiP_mm_hat', 2, @(tp) tp.phiP(2,1);
    '$\Phi^{\mathbb{P}}_{mm,12}$', 'phiP_mm_hat', 3, @(tp) tp.phiP(1,2);
    '$\Phi^{\mathbb{P}}_{mm,22}$', 'phiP_mm_hat', 4, @(tp) tp.phiP(2,2);
    '$\alpha$',                    'alpha_hat',   1, @(tp) tp.alpha;
    '$\mu^{\mathbb{Q}}_1$',        'muQ_m_hat',   1, @(tp) tp.muQ(1);
    '$\mu^{\mathbb{Q}}_2$',        'muQ_m_hat',   2, @(tp) tp.muQ(2);
    '$\Sigma_{11}$',               'sigma_mm_hat',1, @(tp) tp.sigma(1,1);
    '$\Sigma_{21}$',               'sigma_mm_hat',2, @(tp) tp.sigma(2,1);
    '$\Sigma_{22}$',               'sigma_mm_hat',3, @(tp) tp.sigma(2,2);
    '$\sigma_e$ (bps)',            'stdY_hat',    1, @(tp) tp.stdY * ann * 1e4;
};
nP = size(params,1);

sections = {
    '$\mathbb{P}$-dynamics: Macro AR ($\Phi^{\mathbb{P}}_{mm}$)', [1,4];
    'Short-rate intercept',                                        [5,5];
    'Risk-neutral intercept ($\mu^{\mathbb{Q}}$)',                 [6,7];
    'Volatility: Cholesky ($\Sigma_{mm}$)',                        [8,10];
    'Measurement error',                                           [11,11];
};

%% Write LaTeX table

fid = fopen('Tables/KO_ThreeEst_BiasTable.tex','w');

fprintf(fid, '%% Requires: \\usepackage{booktabs,threeparttable}\n\n');
fprintf(fid, '\\begin{table}[htbp]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\begin{threeparttable}\n');
fprintf(fid, '\\caption{Finite-Sample Properties: Standard vs Survey-Augmented MCSE, Horizon Comparison --- 4-Factor MF-GATSM (V6 DGP, $R=%d$ replications)}\n', results(1).R);
fprintf(fid, '\\label{tab:three_est_bias}\n');
fprintf(fid, '\\small\\setlength{\\tabcolsep}{3pt}\n');
fprintf(fid, '\\begin{tabular}{l r rrrr@{\\hspace{6pt}} rrrr@{\\hspace{6pt}} rrrr}\n');
fprintf(fid, '\\toprule\n');
fprintf(fid, ' & & \\multicolumn{4}{c}{\\textbf{Estimator A}} & \\multicolumn{4}{c}{\\textbf{Estimator B1 (h=1)}} & \\multicolumn{4}{c}{\\textbf{Estimator B2 (h=4)}} \\\\\n');
fprintf(fid, '\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}\n');
fprintf(fid, 'Parameter & True & Mean & Bias & SD & Mean & Bias & SD & Mean & Bias & SD \\\\\n');

for iT = 1:length(T_list)
    T_sim = T_list(iT);
    if isempty(results(iT).T), continue; end
    tp = results(iT).tp;

    fprintf(fid, '\\midrule\n');
    fprintf(fid, '\\multicolumn{11}{l}{\\textbf{$T=%d$ quarters}} \\\\\n', T_sim);
    fprintf(fid, '\\midrule\n');

    sec_idx = 1;
    for ip = 1:nP
        lbl   = params{ip,1};
        fld   = params{ip,2};
        col   = params{ip,3};
        tv_fn = params{ip,4};
        tv    = tv_fn(tp);

        % Section header
        while sec_idx <= size(sections,1) && ip == sections{sec_idx,2}(1)
            fprintf(fid, '\\multicolumn{11}{l}{\\textit{%s}} \\\\\n', sections{sec_idx,1});
            sec_idx = sec_idx + 1;
        end

        % Extract estimates
        hA  = get_hat(results(iT).A,  fld, col, ann);
        hB1 = get_hat(results(iT).B1, fld, col, ann);
        hB2 = get_hat(results(iT).B2, fld, col, ann);

        [mnA, ~, sdA, bA]    = compute_stats(hA,  tv);
        [mnB1,~, sdB1,bB1]   = compute_stats(hB1, tv);
        [mnB2,~, sdB2,bB2]   = compute_stats(hB2, tv);

        fprintf(fid, '%s & $%.5f$', lbl, tv);
        fprintf(fid, ' & $%.5f$ & $%+.5f$ & $%.5f$', mnA,  bA,  sdA);
        fprintf(fid, ' & $%.5f$ & $%+.5f$ & $%.5f$', mnB1, bB1, sdB1);
        fprintf(fid, ' & $%.5f$ & $%+.5f$ & $%.5f$', mnB2, bB2, sdB2);
        fprintf(fid, ' \\\\\n');
    end
    fprintf(fid, '\\addlinespace[2pt]\n');
end

fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\begin{tablenotes}\\small\n');
fprintf(fid, '\\item \\textit{Notes:} Single DGP: V6 parameters (1Q-ahead survey-augmented MCSE). ');
fprintf(fid, 'Estimator~A $=$ standard MCSE ($w_{\\mathrm{svy}}=0$). ');
fprintf(fid, 'Estimator~B1 $=$ survey-augmented, h$=1$ (dpgdp3/UNEMP3; $\\phi_{\\mathrm{infl}}=0.881$, $\\phi_{\\mathrm{unemp}}=0.995$). ');
fprintf(fid, 'Estimator~B2 $=$ survey-augmented, h$=4$ (dpgdp6/UNEMP6; $\\phi_{\\mathrm{infl}}=0.788$, $\\phi_{\\mathrm{unemp}}=0.958$). ');
fprintf(fid, 'Bias $= \\bar{\\hat{\\theta}} - \\theta_0$. ');
fprintf(fid, 'SD $=$ standard deviation of estimates across replications.\n');
fprintf(fid, '\\end{tablenotes}\n');
fprintf(fid, '\\end{threeparttable}\n');
fprintf(fid, '\\end{table}\n');
fclose(fid);
fprintf('\nLaTeX table written to KO_ThreeEst_BiasTable.tex\n');

%% Console summary

for iT = 1:length(T_list)
    T_sim = T_list(iT);
    if isempty(results(iT).T), continue; end
    tp = results(iT).tp;
    fprintf('\n=== T=%d ===\n', T_sim);
    fprintf('%-28s %8s | %8s %7s | %8s %7s | %8s %7s\n', ...
        'Parameter','True', ...
        'BiasA','SD_A', ...
        'BiasB1','SD_B1', ...
        'BiasB2','SD_B2');
    fprintf('%s\n', repmat('-',1,90));
    for ip = 1:nP
        lbl   = params{ip,1};
        fld   = params{ip,2};
        col   = params{ip,3};
        tv_fn = params{ip,4};
        tv    = tv_fn(tp);
        hA  = get_hat(results(iT).A,  fld, col, ann);
        hB1 = get_hat(results(iT).B1, fld, col, ann);
        hB2 = get_hat(results(iT).B2, fld, col, ann);
        [~,~,sdA, bA]   = compute_stats(hA,  tv);
        [~,~,sdB1,bB1]  = compute_stats(hB1, tv);
        [~,~,sdB2,bB2]  = compute_stats(hB2, tv);
        fprintf('%-28s %8.5f | %+8.5f %7.5f | %+8.5f %7.5f | %+8.5f %7.5f\n', ...
            lbl, tv, bA, sdA, bB1, sdB1, bB2, sdB2);
    end
end

%% Local Functions

function [mn, rmse, sd, bias] = compute_stats(hat, true_val)
    hat  = hat(:);
    mn   = mean(hat);
    bias = mn - true_val;
    sd   = std(hat, 0);
    rmse = sqrt(mean((hat - true_val).^2));
end

function hat = get_hat(est, fld, col, ann)
    conv = logical(est.converged);
    if strcmp(fld,'alpha_hat') || strcmp(fld,'stdY_hat')
        hat = est.(fld)(conv);
    else
        hat = est.(fld)(conv, col);
    end
    if strcmp(fld,'stdY_hat')
        hat = hat * ann * 1e4;
    end
end