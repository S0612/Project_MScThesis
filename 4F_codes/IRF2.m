% This script computes the second batch of IRFs
% files needed in dir
%   GATSM_4F_MacroFinance_Results_KO_proper_V6.mat   (survey-augmented)
%   GATSM_4F_MacroFinance_Results_V23.mat            (no surveys)

clear; clc; close all;

%% User Settings

ann       = 4;                           % quarters per year
H         = 20;                          % horizon in quarters (5 years)
hyr       = (1:H) / ann;                 % horizon axis in years
bps       = 10000;                       % basis-point conversion
nxM       = 2;                           % number of macro factors
nxL       = 2;                           % number of latent factors
nx        = nxM + nxL;                   % total state dimension

% Maturities used in estimation (quarters)
matSelect = [4 8 20 28 40 60];           % 1, 2, 5, 7, 10, 15 years
mats_yr   = matSelect / ann;

% Select 2yr and 10yr for slope (indices in matSelect)
idx_2yr  = find(mats_yr == 2);
idx_10yr = find(mats_yr == 10);

% Labels
shock_name  = {'Inflation shock', 'Unemployment shock'};
factor_name = {'Inflation', 'Unemployment', 'Latent 1', 'Latent 2'};
macro_name  = {'Inflation', 'Unemployment'};
latent_name = {'Latent 1', 'Latent 2'};

% Colours
col_v6      = [0.12 0.47 0.71];   % blue  — survey-augmented (1Q)
col_v23     = [0.80 0.15 0.15];   % red   — no surveys (NS)
col_P       = [0.13 0.55 0.13];   % green — P-measure
col_Q       = [0.75 0.40 0.10];   % orange — Q-measure
col_fill_v6  = [0.65 0.82 0.93];
col_fill_v23 = [0.96 0.70 0.70];
lw = 2.2;

%% Load estimates

r6  = load('GATSM_4F_MacroFinance_Results_KO_proper_V6.mat');
r23 = load('GATSM_4F_MacroFinance_Results_V23.mat');

% survey-augmented
phiP_v6  = r6.phiP;
phiQ_v6  = r6.phiQ;
beta_v6  = r6.beta(:);
sigma_v6 = r6.sigma;
alpha_v6 = r6.alpha;
muQ_v6   = r6.muQ(:);
gx_v6    = r6.gx;          % numObs x nx  (annualised decimal)

% no surveys
phiP_v23  = r23.phiP;
phiQ_v23  = r23.phiQ;
beta_v23  = r23.beta(:);
sigma_v23 = r23.sigma;
alpha_v23 = r23.alpha;
muQ_v23   = r23.muQ(:);
gx_v23    = r23.gx;

fprintf('  V6  phiP_mm diag: [%.4f, %.4f]\n', phiP_v6(1,1),  phiP_v6(2,2));
fprintf('  V23 phiP_mm diag: [%.4f, %.4f]\n', phiP_v23(1,1), phiP_v23(2,2));
fprintf('  V6  phiP_ll diag: [%.4f, %.4f]\n', phiP_v6(3,3),  phiP_v6(4,4));
fprintf('  V6  phiQ_mm diag: [%.4f, %.4f]\n', phiQ_v6(1,1),  phiQ_v6(2,2));
fprintf('\n');

%% Core IRF computation

% Macro shocks only: yield IRFs
[IRF_y_v6,  IRF_esr_v6,  IRF_tp_v6 ] = ...
    irf_yields(gx_v6,  phiP_v6,  beta_v6,  sigma_v6,  matSelect, ann, H, bps, nxM);
[IRF_y_v23, IRF_esr_v23, IRF_tp_v23] = ...
    irf_yields(gx_v23, phiP_v23, beta_v23, sigma_v23, matSelect, ann, H, bps, nxM);

% Short-rate IRF under P and Q
% IRF_rP(h,k) = ann * beta' * phiP^(h-1) * sigma(:,k)  * bps
% IRF_rQ(h,k) = ann * beta' * phiQ^(h-1) * sigma(:,k)  * bps

IRF_rP_v6  = zeros(H, nxM);
IRF_rQ_v6  = zeros(H, nxM);
IRF_rP_v23 = zeros(H, nxM);
IRF_rQ_v23 = zeros(H, nxM);

rx_v6  = ann * beta_v6';    % 1 x nx  annualised short-rate loadings
rx_v23 = ann * beta_v23';

phiP_h_v6  = eye(nx);   phiQ_h_v6  = eye(nx);
phiP_h_v23 = eye(nx);   phiQ_h_v23 = eye(nx);

for h = 1:H
    for k = 1:nxM
        IRF_rP_v6(h,k)  = rx_v6  * phiP_h_v6  * sigma_v6(:,k)  * bps;
        IRF_rQ_v6(h,k)  = rx_v6  * phiQ_h_v6  * sigma_v6(:,k)  * bps;
        IRF_rP_v23(h,k) = rx_v23 * phiP_h_v23 * sigma_v23(:,k) * bps;
        IRF_rQ_v23(h,k) = rx_v23 * phiQ_h_v23 * sigma_v23(:,k) * bps;
    end
    phiP_h_v6  = phiP_h_v6  * phiP_v6;
    phiQ_h_v6  = phiQ_h_v6  * phiQ_v6;
    phiP_h_v23 = phiP_h_v23 * phiP_v23;
    phiQ_h_v23 = phiQ_h_v23 * phiQ_v23;
end

% Block sub-IRFs of state vector
% Macro block: shocks k=1,2; responses of factors 1,2
% Latent block: shocks k=3,4; responses of factors 3,4
% Only within-block responses are non-zero (block-diagonal phiP, sigma).
%
% IRF_xMacro(i,h,k) = e_i' * phiP^(h-1) * sigma(:,k),  i,k in {1,2}
% IRF_xLatent(i,h,k)= e_i' * phiP^(h-1) * sigma(:,k),  i,k in {3,4}

IRF_xMacro_v6  = zeros(nxM, H, nxM);   % [response factor, horizon, shock]
IRF_xLatent_v6 = zeros(nxL, H, nxL);
IRF_xMacro_v23 = zeros(nxM, H, nxM);

phiP_h_v6  = eye(nx);
phiP_h_v23 = eye(nx);

for h = 1:H
    for k = 1:nxM
        shock_k = sigma_v6(:, k);
        for i = 1:nxM
            IRF_xMacro_v6(i, h, k) = phiP_h_v6(i,:) * shock_k;
        end
        shock_k23 = sigma_v23(:, k);
        for i = 1:nxM
            IRF_xMacro_v23(i, h, k) = phiP_h_v23(i,:) * shock_k23;
        end
    end
    for k = 1:nxL
        shock_k = sigma_v6(:, nxM + k);   % latent shock column
        for i = 1:nxL
            IRF_xLatent_v6(i, h, k) = phiP_h_v6(nxM+i,:) * shock_k;
        end
    end
    phiP_h_v6  = phiP_h_v6  * phiP_v6;
    phiP_h_v23 = phiP_h_v23 * phiP_v23;
end

fprintf('  IRFs computed.\n\n');

%% Fig 1: Slope IRF  (10yr - 2yr term spread)

% Layout (2x3):
%   Row 1 — Inflation shock
%   Row 2 — Unemployment shock
%   Col 1 — Total slope response (V6 vs V23)
%   Col 2 — ESR slope component
%   Col 3 — TP slope component

if ~exist('Graphs', 'dir'), mkdir('Graphs'); end

% Compute slope IRFs
slope_y_v6    = squeeze(IRF_y_v6(idx_10yr,:,:)   - IRF_y_v6(idx_2yr,:,:));
slope_esr_v6  = squeeze(IRF_esr_v6(idx_10yr,:,:) - IRF_esr_v6(idx_2yr,:,:));
slope_tp_v6   = squeeze(IRF_tp_v6(idx_10yr,:,:)  - IRF_tp_v6(idx_2yr,:,:));

slope_y_v23   = squeeze(IRF_y_v23(idx_10yr,:,:)   - IRF_y_v23(idx_2yr,:,:));
slope_esr_v23 = squeeze(IRF_esr_v23(idx_10yr,:,:) - IRF_esr_v23(idx_2yr,:,:));
slope_tp_v23  = squeeze(IRF_tp_v23(idx_10yr,:,:)  - IRF_tp_v23(idx_2yr,:,:));

% For H=20, the above squeeze gives (H x nxM) = (20 x 2)
% slope_y_v6(:,1) = inflation shock slope response, etc.

fig1 = figure('Name', 'Fig1: Slope IRF', ...
    'Units', 'normalized', 'Position', [0.05 0.10 0.90 0.75]);

col_titles = {'Total slope response', 'ESR slope component', 'TP slope component'};
data_v6_cols  = {slope_y_v6,  slope_esr_v6,  slope_tp_v6};
data_v23_cols = {slope_y_v23, slope_esr_v23, slope_tp_v23};

for ks = 1:2   % shock
    for ic = 1:3   % column (Total / ESR / TP)
        sp = (ks-1)*3 + ic;
        subplot(2, 3, sp);
        hold on;

        d6  = data_v6_cols{ic}(:, ks)';
        d23 = data_v23_cols{ic}(:, ks)';

        fill_between(hyr, d6, d23, [0.80 0.88 0.96], 0.40);
        yline(0, 'k-', 'LineWidth', 0.6, 'Alpha', 0.5);

        p1 = plot(hyr, d6,  '-',  'Color', col_v6,  'LineWidth', lw, 'DisplayName', '1Q (surveys)');
        p2 = plot(hyr, d23, '--', 'Color', col_v23, 'LineWidth', lw, 'DisplayName', 'NS (no surveys)');

        hold off;
        title(sprintf('%s\n%s', shock_name{ks}, col_titles{ic}), 'FontSize', 9);
        xlabel('Horizon (years)', 'FontSize', 8);
        ylabel('Response (bps)', 'FontSize', 8);
        xlim([0 H/ann]);
        grid on; box on; set(gca, 'FontSize', 8);

        if sp == 1
            legend([p1 p2], 'Location', 'best', 'FontSize', 9);
        end
    end
end

sgtitle({'Slope IRF: Response of 10yr minus 2yr Term Spread to Macro Shocks'}, ...
    'FontSize', 11, 'FontWeight', 'bold');

exportgraphics(fig1, 'Graphs/IRF_Slope.pdf', 'ContentType', 'vector');
fprintf('  Saved: Graphs/IRF_Slope.pdf\n');

%% Fig 2: P-measure vs Q-measure short-rate IRF
%
% Layout (2x2):
%   [1,1] Inflation shock    — V6 (1Q surveys)
%   [1,2] Inflation shock    — V23 (no surveys)
%   [2,1] Unemployment shock — V6
%   [2,2] Unemployment shock — V23
%
% Each panel plots:
%   Solid green  = short-rate response under P (physical measure)
%   Dashed orange = short-rate response under Q (risk-neutral measure)
%   Shaded gap   = instantaneous risk premium response

fig2 = figure('Name', 'Fig2: P vs Q short-rate IRF', ...
    'Units', 'normalized', 'Position', [0.05 0.10 0.80 0.70]);

ver_label = {'1Q', 'NS'};
rP_all    = {IRF_rP_v6,  IRF_rP_v23};
rQ_all    = {IRF_rQ_v6,  IRF_rQ_v23};

for ks = 1:2       % shock (row)
    for iv = 1:2   % estimator (column)
        sp = (ks-1)*2 + iv;
        subplot(2, 2, sp);
        hold on;

        rP = rP_all{iv}(:, ks)';   % 1 x H
        rQ = rQ_all{iv}(:, ks)';

        % Shade the gap (risk premium response)
        fill_between(hyr, rP, rQ, [0.85 0.92 0.80], 0.45);
        yline(0, 'k-', 'LineWidth', 0.6, 'Alpha', 0.5);

        p1 = plot(hyr, rP, '-',  'Color', col_P, 'LineWidth', lw, ...
            'DisplayName', 'Under P');
        p2 = plot(hyr, rQ, '--', 'Color', col_Q, 'LineWidth', lw, ...
            'DisplayName', 'Under Q');

        hold off;
        title(sprintf('%s — %s', shock_name{ks}, ver_label{iv}), 'FontSize', 11);
        xlabel('Horizon (years)', 'FontSize', 11);
        ylabel('Short-rate response (bps)', 'FontSize', 11);
        xlim([0 H/ann]);
        grid on; box on; set(gca, 'FontSize', 11);

        if sp == 1
            legend([p1 p2], 'Location', 'best', 'FontSize', 11);
            text(0.97, 0.06, 'Shaded = risk premium response', ...
                'Units', 'normalized', 'FontSize', 7.5, ...
                'HorizontalAlignment', 'right', 'Color', [0.4 0.4 0.4]);
        end
    end
end

sgtitle({'P and Q Short-Rate IRF'}, ...
    'FontSize', 11, 'FontWeight', 'bold');

exportgraphics(fig2, 'Graphs/IRF_PvsQ_ShortRate.pdf', 'ContentType', 'vector');
fprintf('  Saved: Graphs/IRF_PvsQ_ShortRate.pdf\n');

%% Fig 3: Block sub-IRFs of the state vector
%
% Panel A — Macro block (2x2)
%   How each macro factor responds to each macro shock.
%   Rows = responding factor (inflation, unemployment)
%   Cols = shock (inflation, unemployment)
%   V6 vs V23 shown (survey augmentation changes phiP_mm and hence
%   the within-block dynamics).
%
% Panel B — Latent block (2x2)
%   How each latent factor responds to each latent shock.
%   Only V6 shown (latent block uses true phiP_ll in both estimators).

fig3 = figure('Name', 'Fig3: Block Sub-IRFs', ...
    'Units', 'normalized', 'Position', [0.05 0.05 0.92 0.85]);

% Panel A: Macro block (upper 2x2)
for ki = 1:nxM     % response factor (row)
    for kj = 1:nxM   % shock (column)
        sp = (ki-1)*2 + kj;   % subplots 1-4
        subplot(4, 2, sp);
        hold on;

        r6_path  = squeeze(IRF_xMacro_v6(ki, :, kj));    % 1 x H
        r23_path = squeeze(IRF_xMacro_v23(ki, :, kj));

        fill_between(hyr, r6_path, r23_path, [0.78 0.85 0.95], 0.40);
        yline(0, 'k-', 'LineWidth', 0.5, 'Alpha', 0.5);

        p1 = plot(hyr, r6_path,  '-',  'Color', col_v6,  'LineWidth', lw, ...
            'DisplayName', '1Q (surveys)');
        p2 = plot(hyr, r23_path, '--', 'Color', col_v23, 'LineWidth', lw, ...
            'DisplayName', 'NS (no surveys)');

        hold off;
        title(sprintf('Macro: %s \\rightarrow %s', ...
            macro_name{kj}, macro_name{ki}), 'FontSize', 11);
        xlabel('Horizon (years)', 'FontSize', 11);
        ylabel('Response (s.d.)', 'FontSize', 11);
        xlim([0 H/ann]);
        grid on; box on; set(gca, 'FontSize', 11);

        if sp == 1
            legend([p1 p2], 'Location', 'best', 'FontSize', 11);
        end
    end
end

% Panel B: Latent block (lower 2x2)
for ki = 1:nxL     % response factor
    for kj = 1:nxL   % shock
        sp = 4 + (ki-1)*2 + kj;   % subplots 5-8
        subplot(4, 2, sp);
        hold on;

        r_path = squeeze(IRF_xLatent_v6(ki, :, kj));

        yline(0, 'k-', 'LineWidth', 0.5, 'Alpha', 0.5);
        p1 = plot(hyr, r_path, '-', 'Color', col_v6, 'LineWidth', lw, ...
            'DisplayName', '1Q (surveys)');

        hold off;
        title(sprintf('Latent: %s \\rightarrow %s', ...
            latent_name{kj}, latent_name{ki}), 'FontSize', 9);
        xlabel('Horizon (years)', 'FontSize', 11);
        ylabel('Response (s.d.)', 'FontSize', 11);
        xlim([0 H/ann]);
        grid on; box on; set(gca, 'FontSize', 11);
    end
end

sgtitle({'State Vector Block Sub-IRFs', ...
    'Note: Cross-block responses are zero by the independence assumption'}, ...
    'FontSize', 12, 'FontWeight', 'bold');

exportgraphics(fig3, 'Graphs/IRF_BlockSubIRF.pdf', 'ContentType', 'vector');
fprintf('  Saved: Graphs/IRF_BlockSubIRF.pdf\n');

%% Numerical summary

fprintf('\n=================================================================\n');
fprintf(' Numerical summary at 1-year horizon (h = 4 quarters)\n');
fprintf('=================================================================\n\n');

h4 = 4;   % 1-year horizon

fprintf('--- SLOPE IRF at h=%dQ (1 year) ---\n', h4);
fprintf('%-22s  %10s  %10s  %10s  %10s\n', ...
    'Component', 'V6 Infl', 'V6 Unemp', 'V23 Infl', 'V23 Unemp');
fprintf('%s\n', repmat('-', 1, 66));
comps6  = {slope_y_v6,  slope_esr_v6,  slope_tp_v6};
comps23 = {slope_y_v23, slope_esr_v23, slope_tp_v23};
cnames  = {'Total', 'ESR', 'TP'};
for ic = 1:3
    fprintf('%-22s  %10.2f  %10.2f  %10.2f  %10.2f\n', cnames{ic}, ...
        comps6{ic}(h4,1),  comps6{ic}(h4,2), ...
        comps23{ic}(h4,1), comps23{ic}(h4,2));
end

fprintf('\n--- P vs Q SHORT RATE IRF at h=%dQ ---\n', h4);
fprintf('%-22s  %10s  %10s  %10s  %10s\n', ...
    '', 'V6 Infl', 'V6 Unemp', 'V23 Infl', 'V23 Unemp');
fprintf('%s\n', repmat('-', 1, 66));
fprintf('%-22s  %10.2f  %10.2f  %10.2f  %10.2f\n', 'Under P', ...
    IRF_rP_v6(h4,1),  IRF_rP_v6(h4,2), ...
    IRF_rP_v23(h4,1), IRF_rP_v23(h4,2));
fprintf('%-22s  %10.2f  %10.2f  %10.2f  %10.2f\n', 'Under Q', ...
    IRF_rQ_v6(h4,1),  IRF_rQ_v6(h4,2), ...
    IRF_rQ_v23(h4,1), IRF_rQ_v23(h4,2));
fprintf('%-22s  %10.2f  %10.2f  %10.2f  %10.2f\n', 'Risk premium gap (P-Q)', ...
    IRF_rP_v6(h4,1)-IRF_rQ_v6(h4,1),   IRF_rP_v6(h4,2)-IRF_rQ_v6(h4,2), ...
    IRF_rP_v23(h4,1)-IRF_rQ_v23(h4,1), IRF_rP_v23(h4,2)-IRF_rQ_v23(h4,2));

fprintf('\n--- MACRO BLOCK sub-IRF (V6) at h=%dQ, normalised s.d. ---\n', h4);
fprintf('%-28s  %12s  %12s\n', 'Response (factor <- shock)', 'Value (V6)', 'Value (V23)');
fprintf('%s\n', repmat('-', 1, 56));
for ki = 1:nxM
    for kj = 1:nxM
        fprintf('%-28s  %12.4f  %12.4f\n', ...
            sprintf('%s <- %s shock', macro_name{ki}, macro_name{kj}), ...
            IRF_xMacro_v6(ki, h4, kj), IRF_xMacro_v23(ki, h4, kj));
    end
end

fprintf('\n--- LATENT BLOCK sub-IRF (V6) at h=%dQ, normalised s.d. ---\n', h4);
fprintf('%-28s  %12s\n', 'Response (factor <- shock)', 'Value (V6)');
fprintf('%s\n', repmat('-', 1, 42));
for ki = 1:nxL
    for kj = 1:nxL
        fprintf('%-28s  %12.4f\n', ...
            sprintf('Latent %d <- Latent %d shock', ki, kj), ...
            IRF_xLatent_v6(ki, h4, kj));
    end
end

fprintf('\nDone.\n');

%% Local Functions

function [IRF_y, IRF_esr, IRF_tp] = ...
        irf_yields(gx, phiP, beta, sigma, matSel, ann_in, H_in, bps_in, nshk)
% Compute yield, ESR, and TP IRFs to the first nshk shocks.
%
% Inputs:
%   gx      — numObs x nx  annualised yield loadings
%   phiP    — nx x nx  companion matrix (physical measure)
%   beta    — nx x 1   short-rate factor loadings (per quarter)
%   sigma   — nx x nx  Cholesky shock matrix
%   matSel  — 1 x numObs  maturities in quarters
%   ann_in  — annualisation factor (4 for quarterly)
%   H_in    — number of horizons (quarters)
%   bps_in  — basis-point scaling (10000)
%   nshk    — number of shocks to compute (first nshk columns of sigma)
%
% IRF_y(im,h,k)   = annualised yield response of maturity im at horizon h
%                   to shock k, in basis points
% IRF_esr(im,h,k) = expected-short-rate component of IRF_y
% IRF_tp(im,h,k)  = term premium component (= IRF_y - IRF_esr)

nmat   = size(gx, 1);
nfac   = size(phiP, 1);
rx     = ann_in * beta';        % 1 x nfac  annualised short-rate loadings
IRF_y   = zeros(nmat, H_in, nshk);
IRF_esr = zeros(nmat, H_in, nshk);

for k = 1:nshk
    shock  = sigma(:, k);
    phiP_h = eye(nfac);
    for h = 1:H_in
        % Yield response at horizon h
        for im = 1:nmat
            IRF_y(im, h, k) = gx(im,:) * phiP_h * shock * bps_in;
        end
        % ESR: average expected future short-rate response from h to h+n-1
        for im = 1:nmat
            n_q     = matSel(im);   % bond maturity in quarters
            esr     = 0;
            phiP_hj = phiP_h;
            for j = 0:n_q-1
                if j > 0
                    phiP_hj = phiP_hj * phiP;
                end
                esr = esr + rx * phiP_hj * shock;
            end
            IRF_esr(im, h, k) = (esr / n_q) * bps_in;
        end
        phiP_h = phiP_h * phiP;
    end
end
IRF_tp = IRF_y - IRF_esr;
end


function fill_between(x, y1, y2, col, alpha_val)
% Shade the region between two curves y1 and y2.
xp = [x(:)', fliplr(x(:)')];
yp = [y1(:)', fliplr(y2(:)')];
fill(xp, yp, col, 'FaceAlpha', alpha_val, 'EdgeColor', 'none');
end
