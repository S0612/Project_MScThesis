% This script computes the FEVD using the appendix of Ang and Piazzesi
%
%
% Sune Grøn Pedersen, 2026

clear; clc; close all;

%% SECTION 1: Settings

ann       = 4;
matSelect = [1 2 5 7 10 15] * ann;
mats_yr   = matSelect / ann;
numObs    = length(matSelect);
H         = 40;
nxM       = 2;
nxL       = 2;
nx        = nxM + nxL;

shock_labels = {'Inflation','Unemployment','Latent 1','Latent 2'};
tbl_horizons = [1 4 8 20 40];
tbl_hlabels  = {'1Q','1Y','2Y','5Y','10Y'};

%% SECTION 2: Load estimates

fprintf('Loading estimates...\n');
r6  = load('GATSM_4F_MacroFinance_Results_KO_proper_V6.mat');
r23 = load('GATSM_4F_MacroFinance_Results_V23.mat');

phiP_v6  = r6.phiP;    sigma_v6  = r6.sigma;   gx_v6  = r6.gx;
phiP_v23 = r23.phiP;   sigma_v23 = r23.sigma;  gx_v23 = r23.gx;

fprintf('  V6  phiP_mm diagonal: [%.4f, %.4f]\n', phiP_v6(1,1),  phiP_v6(2,2));
fprintf('  V23 phiP_mm diagonal: [%.4f, %.4f]\n', phiP_v23(1,1), phiP_v23(2,2));

%% SECTION 3: Diagnostics

fprintf('\n--- DIAGNOSTIC: phiP stationarity ---\n');
eig_v6  = max(abs(eig(phiP_v6)));
eig_v23 = max(abs(eig(phiP_v23)));
fprintf('  V6  max|eig(phiP)| = %.6f  %s\n', eig_v6,  iif(eig_v6  < 1, 'PASS', 'FAIL'));
fprintf('  V23 max|eig(phiP)| = %.6f  %s\n', eig_v23, iif(eig_v23 < 1, 'PASS', 'FAIL'));

fprintf('\n--- DIAGNOSTIC: sigma structure ---\n');
fprintf('  V6  lower-triangular: %s\n', ...
    iif(norm(triu(sigma_v6,1),'fro') < 1e-12, 'PASS', 'FAIL'));
fprintf('  V23 lower-triangular: %s\n', ...
    iif(norm(triu(sigma_v23,1),'fro') < 1e-12, 'PASS', 'FAIL'));
fprintf('  V6  sigma_mm diagonal: [%.4f, %.4f]  (estimated from VAR residuals)\n', ...
    sigma_v6(1,1),  sigma_v6(2,2));
fprintf('  V23 sigma_mm diagonal: [%.4f, %.4f]\n', sigma_v23(1,1), sigma_v23(2,2));
fprintf('  V6  sigma_ll diagonal: [%.4f, %.4f]  (identity — identification convention)\n', ...
    sigma_v6(3,3),  sigma_v6(4,4));
fprintf('  Scale ratio (latent/macro): ~%.2fx in std, ~%.2fx in variance\n', ...
    sigma_v6(3,3)/sigma_v6(1,1), (sigma_v6(3,3)/sigma_v6(1,1))^2);

fprintf('\n--- DIAGNOSTIC: gx loadings (ann. decimal per z-score unit) ---\n');
fprintf('  %-14s', '');
fprintf('  %4dy', mats_yr); fprintf('\n');
for k = 1:nx
    fprintf('  %-14s', shock_labels{k});
    fprintf('  %+.4f', gx_v6(:,k)');
    fprintf('\n');
end

fprintf('\n--- DIAGNOSTIC: h=0 IRF (bps) — one std-dev shock ---\n');
fprintf('  (Macro = one std-dev of z-score factor; Latent = one identification unit)\n');
fprintf('  %-14s', '');
fprintf('  %4dy', mats_yr); fprintf('\n');
for k = 1:nx
    fprintf('  %-14s', shock_labels{k});
    fprintf('  %+6.2f', (gx_v6 * sigma_v6(:,k))' * 10000);
    fprintf('\n');
end

fprintf('\n--- DIAGNOSTIC: implied quarterly yield volatility (ann. %%) ---\n');
Q_v6  = sigma_v6  * sigma_v6';
Q_v23 = sigma_v23 * sigma_v23';
fprintf('  %-10s', 'V6');
for j = 1:numObs
    fprintf('  %5.2f', sqrt(gx_v6(j,:) * Q_v6 * gx_v6(j,:)') * 100);
end
fprintf('\n  %-10s', 'V23');
for j = 1:numObs
    fprintf('  %5.2f', sqrt(gx_v23(j,:) * Q_v23 * gx_v23(j,:)') * 100);
end
fprintf('\n  (maturities: ');
fprintf(' %4dy', mats_yr); fprintf(')\n\n');

%% SECTION 4: FEVD computation

fprintf('Computing FEVD...\n');

function FEVD = compute_FEVD(phiP, sigma, gx, H, numObs, nx)
    % Implements A&P Appendix D, equations D.1-D.4.
    %
    % At each horizon h, builds Psi_{h-1} = gx * phiP^{h-1} * sigma [numObs x nx].
    % Accumulates cumSS(j,k) = sum_{i=0}^{h-1} Psi_{jk,i}^2  (eq. D.3).
    % Returns FEVD(j,k,h) = cumSS(j,k) / sum_k cumSS(j,k)     (eq. D.4).
    %
    % sigma is passed without modification. The Cholesky identification
    % is already embedded: sigma_mm from estimated VAR residual covariance,
    % sigma_ll = eye(nxL) by identification convention.

    cumSS   = zeros(numObs, nx);
    FEVD    = zeros(numObs, nx, H);
    phiP_pw = eye(nx);

    for h = 1:H
        Psi_h = gx * phiP_pw * sigma;          % [numObs x nx]
        cumSS = cumSS + Psi_h .^ 2;
        MSE   = sum(cumSS, 2);                  % [numObs x 1]
        for k = 1:nx
            FEVD(:, k, h) = cumSS(:, k) ./ MSE;
        end
        phiP_pw = phiP_pw * phiP;
    end
end

FEVD_v6  = compute_FEVD(phiP_v6,  sigma_v6,  gx_v6,  H, numObs, nx);
FEVD_v23 = compute_FEVD(phiP_v23, sigma_v23, gx_v23, H, numObs, nx);

% Sanity checks — at every horizon, not just the last
fprintf('  Checking row sums and non-negativity at all horizons...\n');
tol = 1e-10;
for h = 1:H
    assert(max(abs(sum(FEVD_v6(:,:,h),  2) - 1)) < tol, 'V6  row sum != 1 at h=%d', h);
    assert(max(abs(sum(FEVD_v23(:,:,h), 2) - 1)) < tol, 'V23 row sum != 1 at h=%d', h);
end
assert(min(FEVD_v6(:))  >= 0, 'V6  FEVD contains negative values');
assert(min(FEVD_v23(:)) >= 0, 'V23 FEVD contains negative values');
fprintf('  All sanity checks passed.\n\n');

% Macro group total [numObs x H]
macro_v6  = squeeze(sum(FEVD_v6(:,  1:nxM, :), 2));
macro_v23 = squeeze(sum(FEVD_v23(:, 1:nxM, :), 2));

%% SECTION 5: Console tables

function print_table(FEVD, macro_mat, mats_yr, tbl_horizons, shock_labels, nx, header)
    fprintf('\n%s\n', header);
    fprintf('  %-14s', 'Factor \ h');
    for h = tbl_horizons, fprintf('  %7s', sprintf('h=%d', h)); end
    fprintf('\n  %s\n', repmat('-', 1, 14 + 9*length(tbl_horizons)));
    for j = 1:length(mats_yr)
        fprintf('  --- %dy yield ---\n', mats_yr(j));
        for k = 1:nx
            fprintf('  %-14s', shock_labels{k});
            for h = tbl_horizons
                fprintf('  %6.1f%%', FEVD(j,k,h)*100);
            end
            fprintf('\n');
        end
        fprintf('  %-14s', 'MACRO TOTAL');
        for h = tbl_horizons
            fprintf('  %6.1f%%', macro_mat(j,h)*100);
        end
        fprintf('\n');
    end
end

print_table(FEVD_v6,  macro_v6,  mats_yr, tbl_horizons, shock_labels, nx, ...
    '=== FEVD — V6 (Surveys) ===');
print_table(FEVD_v23, macro_v23, mats_yr, tbl_horizons, shock_labels, nx, ...
    '=== FEVD — V23 (No Surveys) ===');

% V23 - V6 difference table
fprintf('\n=== Survey Bias: V23 - V6 macro share (percentage points) ===\n');
fprintf('  %-8s', 'Mat \ h');
for h = tbl_horizons, fprintf('  %7s', sprintf('h=%d', h)); end
fprintf('\n  %s\n', repmat('-', 1, 8 + 9*length(tbl_horizons)));
for j = 1:length(mats_yr)
    fprintf('  %-8s', sprintf('%dy', mats_yr(j)));
    for h = tbl_horizons
        fprintf('  %+6.1fpp', (macro_v23(j,h) - macro_v6(j,h))*100);
    end
    fprintf('\n');
end

%% SECTION 6: LaTeX tables

fprintf('Writing LaTeX tables...\n');

function write_latex_table(fname, FEVD, macro_mat, mats_yr, ...
                            tbl_horizons, tbl_hlabels, shock_labels, ...
                            nx, nxM, caption_str, label_str)
    fid = fopen(fname, 'w');
    Nh  = length(tbl_horizons);
    Nm  = length(mats_yr);

    fprintf(fid, '%% Auto-generated by FEVD.m\n');
    fprintf(fid, '\\begin{table}[htbp]\n');
    fprintf(fid, '\\centering\n');
    fprintf(fid, '\\caption{%s}\n', caption_str);
    fprintf(fid, '\\label{%s}\n', label_str);
    fprintf(fid, '\\small\n');
    fprintf(fid, '\\begin{tabular}{@{}l%s@{}}\n', repmat('r', 1, Nh));
    fprintf(fid, '\\toprule\n');

    % Header
    fprintf(fid, '\\textit{Maturity}');
    for hi = 1:Nh
        fprintf(fid, ' & \\textit{%s}', tbl_hlabels{hi});
    end
    fprintf(fid, ' \\\\\n\\midrule\n');

    % One panel per factor
    for k = 1:nx
        fprintf(fid, '\\multicolumn{%d}{@{}l}{\\textit{%s shock}} \\\\\n', ...
                Nh+1, shock_labels{k});
        for j = 1:Nm
            fprintf(fid, '%dy', mats_yr(j));
            for hi = 1:Nh
                fprintf(fid, ' & %.1f', FEVD(j, k, tbl_horizons(hi))*100);
            end
            fprintf(fid, ' \\\\\n');
        end
        % After the last macro factor: insert Macro Total subtotal
        if k == nxM
            fprintf(fid, '\\addlinespace[2pt]\n');
            fprintf(fid, '\\multicolumn{%d}{@{}l}{\\textit{Macro total (INF + UNEMP)}} \\\\\n', Nh+1);
            for j = 1:Nm
                fprintf(fid, '%dy', mats_yr(j));
                for hi = 1:Nh
                    fprintf(fid, ' & \\textbf{%.1f}', macro_mat(j, tbl_horizons(hi))*100);
                end
                fprintf(fid, ' \\\\\n');
            end
            fprintf(fid, '\\addlinespace[4pt]\n');
        elseif k < nx
            fprintf(fid, '\\addlinespace[4pt]\n');
        end
    end

    fprintf(fid, '\\bottomrule\n');
    fprintf(fid, '\\end{tabular}\n');
    fprintf(fid, ['\\vspace{4pt}\n\\begin{minipage}{\\linewidth}\n' ...
        '{\\footnotesize \\textit{Notes:} Entries are percentages. ' ...
        'Each entry $\\Omega_{jk}(h)$ reports the fraction of the ' ...
        '$h$-step ahead forecast error variance of yield $j$ ' ...
        'attributable to structural shock $k$, computed following ' ...
        'Ang and Piazzesi (2003, Appendix~D): ' ...
        '$\\Omega_{jk}(h)=\\bigl(\\sum_{i=0}^{h-1}\\Psi_{jk,i}^2\\bigr)' ...
        '/\\mathrm{MSE}_j(h)$, $\\Psi_i=B_n^{\\prime}\\Phi^i\\Sigma$. ' ...
        'Macro shocks are identified by Cholesky decomposition of the ' ...
        'VAR innovation covariance (inflation ordered first). ' ...
        'Latent shocks are normalised to unit variance by the ' ...
        'identification convention $\\Sigma_{LL}=I$. ' ...
        'Macro total rows in bold.}\n\\end{minipage}\n']);
    fprintf(fid, '\\end{table}\n');
    fclose(fid);
end

write_latex_table('Tables/FEVD_table_V6.tex',  FEVD_v6,  macro_v6,  mats_yr, ...
    tbl_horizons, tbl_hlabels, shock_labels, nx, nxM, ...
    'Forecast Error Variance Decomposition --- V6 (Survey-Augmented)', ...
    'tab:fevd_v6');

write_latex_table('Tables/FEVD_table_V23.tex', FEVD_v23, macro_v23, mats_yr, ...
    tbl_horizons, tbl_hlabels, shock_labels, nx, nxM, ...
    'Forecast Error Variance Decomposition --- V23 (No Surveys)', ...
    'tab:fevd_v23');

fprintf('  Written: FEVD_table_V6.tex\n');
fprintf('  Written: FEVD_table_V23.tex\n\n');

%% SECTION 7: Save

save('FEVD_results.mat', ...
    'FEVD_v6',      'FEVD_v23',     ...
    'macro_v6',     'macro_v23',    ...
    'mats_yr',      'matSelect',    ...
    'H',            'nx',           ...
    'nxM',          'nxL',          ...
    'shock_labels', 'tbl_horizons', ...
    'tbl_hlabels');

fprintf('Results saved to FEVD_results.mat\n');
fprintf('LaTeX tables: FEVD_table_V6.tex, FEVD_table_V23.tex\n');
fprintf('=================================================================\n');
fprintf(' FEVD.m complete.\n');
fprintf('=================================================================\n');

%% Local Functions

function out = iif(cond, a, b)
    if cond,  out = a;  else,  out = b;  end
end