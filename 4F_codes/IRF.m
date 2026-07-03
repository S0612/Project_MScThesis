% This script computes the IRFs
%
% FIGURES
%   Fig 1 — IRF Heatmap: all maturities x horizons, V6 vs V23
%   Fig 2 — ESR vs TP decomposition: 2yr and 10yr yields, both shocks
%   Fig 3 — Survey impact with MC bands: 2yr and 10yr, both shocks [headline]
%   Fig 4 — Term premium channel: 2yr and 10yr, both shocks
%
% The 2-year and 10-year maturities are chosen as representatives
%
% Files needed in dir
%   GATSM_4F_MacroFinance_Results_KO_proper_V6.mat
%   GATSM_4F_MacroFinance_Results_V23.mat
%   MonteCarlo_KO_3EST_T216.mat


clear; clc; close all;

%% User Settings

ann       = 4;
matSelect = [1 2 5 7 10 15] * ann;
mats_yr   = matSelect / ann;
numObs    = length(matSelect);
H         = 20;                  % quarters (5 years)
nxL       = 2;

pct_lo = 10;  pct_hi = 90;      % MC confidence band percentiles

% Focus maturities for line plots (2yr = short end, 10yr = long end)
sel_mats  = [2 10];
sel_idx   = arrayfun(@(m) find(mats_yr==m,1), sel_mats);
sel_label = {'2-year','10-year'};

% Colours
col_v6  = [0.12 0.47 0.71];     % blue  — 1Q (surveys)
col_v23 = [0.80 0.15 0.15];     % red   — NS (no surveys)
col_fill_v23 = [0.96 0.70 0.70];
col_fill_v6  = [0.65 0.82 0.93];
lw = 2.4;

% Shock and maturity labels
shock_name = {'Inflation shock', 'Unemployment shock'};
shock_short= {'Inflation', 'Unemployment'};

%% Load empirical estimates

r6  = load('GATSM_4F_MacroFinance_Results_KO_proper_V6.mat');
r23 = load('GATSM_4F_MacroFinance_Results_V23.mat');

% V6
phiP_v6  = r6.phiP;    phiQ_v6  = r6.phiQ;
beta_v6  = r6.beta(:); sigma_v6 = r6.sigma;
alpha_v6 = r6.alpha;   muQ_v6   = r6.muQ(:);
gx_v6    = r6.gx;      % numObs x nx (annualised, pre-computed)
data_v6  = r6.data;    % T x numObs
T        = size(data_v6, 1);

% V23
phiP_v23  = r23.phiP;    phiQ_v23  = r23.phiQ;
beta_v23  = r23.beta(:); sigma_v23 = r23.sigma;
alpha_v23 = r23.alpha;   muQ_v23   = r23.muQ(:);
gx_v23    = r23.gx;

fprintf('  V6  phiP_mm: [%.4f, %.4f]  (surveys)\n',    phiP_v6(1,1),  phiP_v6(2,2));
fprintf('  V23 phiP_mm: [%.4f, %.4f]  (no surveys)\n', phiP_v23(1,1), phiP_v23(2,2));
fprintf('  Difference:  [%+.4f, %+.4f] (V23 minus V6)\n\n', ...
    phiP_v23(1,1)-phiP_v6(1,1), phiP_v23(2,2)-phiP_v6(2,2));

%% Load MC distribution for confidence bands

fprintf('Loading MC distribution (R=2000)...\n');
mc = load('MonteCarlo_KO_3EST_T216.mat');

phiP_ll_true  = mc.true_params.phiP(3:4, 3:4);
phiQ_ll_true  = mc.true_params.phiQ(3:4, 3:4);
sigma_ll_true = mc.true_params.sigma(3:4, 3:4);

convA  = logical(mc.A.converged(:));
convB1 = logical(mc.B1.converged(:));
R_A    = sum(convA);
R_B1   = sum(convB1);
idx_A  = find(convA);
idx_B1 = find(convB1);
fprintf('  Converged: A=%d/2000  B1=%d/2000\n\n', R_A, R_B1);

%% Central IRFs

fprintf('Computing central IRFs...\n');
[IRF_y_v6,  IRF_esr_v6,  IRF_tp_v6 ] = compute_IRF(gx_v6,  phiP_v6,  beta_v6,  sigma_v6,  matSelect, ann, H);
[IRF_y_v23, IRF_esr_v23, IRF_tp_v23] = compute_IRF(gx_v23, phiP_v23, beta_v23, sigma_v23, matSelect, ann, H);
fprintf('  Done.\n\n');


%%MC IRF distributions for confidence bands
% Focus on 2yr and 10yr yields for inflation and unemployment shocks.
% Per replication: phiP = blkdiag(phiP_mm_r, phiP_ll_true).

nsel   = length(sel_mats);
nshock = 2;
mc_A_irf  = zeros(R_A,  nsel, nshock, H);
mc_B1_irf = zeros(R_B1, nsel, nshock, H);

build_phiP = @(pmm, ll) blkdiag(reshape(pmm,2,2), ll);
build_phiQ = @(pQmm,ll) blkdiag(reshape(pQmm,2,2), ll);
build_sig  = @(smm, ll) blkdiag([smm(1),0; smm(2),smm(3)], ll);

for ir = 1:R_A
    r      = idx_A(ir);
    phiP_r = build_phiP(mc.A.phiP_mm_hat(r,:)', phiP_ll_true);
    phiQ_r = build_phiQ(mc.A.phiQ_mm_hat(r,:)', phiQ_ll_true);
    sig_r  = build_sig(mc.A.sigma_mm_hat(r,:),  sigma_ll_true);
    beta_r = [mc.A.beta_m_hat(r,:)'; mc.A.beta_l_hat(r,:)'];
    muQ_r  = [mc.A.muQ_m_hat(r,:)'; zeros(nxL,1)];
    [~,gx_r] = compute_loadings(mc.A.alpha_hat(r), muQ_r, phiQ_r, beta_r, sig_r, matSelect, ann);
    [IRF_r,~,~] = compute_IRF(gx_r, phiP_r, beta_r, sig_r, matSelect, ann, H);
    for im = 1:nsel
        for ks = 1:nshock
            mc_A_irf(ir,im,ks,:) = IRF_r(sel_idx(im),:,ks);
        end
    end
end

for ir = 1:R_B1
    r      = idx_B1(ir);
    phiP_r = build_phiP(mc.B1.phiP_mm_hat(r,:)', phiP_ll_true);
    phiQ_r = build_phiQ(mc.B1.phiQ_mm_hat(r,:)', phiQ_ll_true);
    sig_r  = build_sig(mc.B1.sigma_mm_hat(r,:),  sigma_ll_true);
    beta_r = [mc.B1.beta_m_hat(r,:)'; mc.B1.beta_l_hat(r,:)'];
    muQ_r  = [mc.B1.muQ_m_hat(r,:)'; zeros(nxL,1)];
    [~,gx_r] = compute_loadings(mc.B1.alpha_hat(r), muQ_r, phiQ_r, beta_r, sig_r, matSelect, ann);
    [IRF_r,~,~] = compute_IRF(gx_r, phiP_r, beta_r, sig_r, matSelect, ann, H);
    for im = 1:nsel
        for ks = 1:nshock
            mc_B1_irf(ir,im,ks,:) = IRF_r(sel_idx(im),:,ks);
        end
    end
end

% Percentile bands: nsel x nshock x H
band_A_lo  = squeeze(prctile(mc_A_irf,  pct_lo, 1));
band_A_hi  = squeeze(prctile(mc_A_irf,  pct_hi, 1));
band_B1_lo = squeeze(prctile(mc_B1_irf, pct_lo, 1));
band_B1_hi = squeeze(prctile(mc_B1_irf, pct_hi, 1));

fprintf('Done.\n\n');

hyr = (1:H) / ann;   % horizon axis in years

%% Fig 1 — Yield Curve IRF Heatmap  (2x2)
%
% Layout:
%   [1,1] Inflation shock — 1Q     [1,2] Inflation shock — NS
%   [2,1] Unemployment shock — 1Q  [2,2] Unemployment shock — NS

fig1 = figure('Name','Fig1: Yield Curve IRF Heatmap', ...
    'Units','normalized','Position',[0.10 0.10 0.80 0.75]);

IRFs_heat  = {IRF_y_v6, IRF_y_v23};
est_labels = {'1Q', 'NS'};
shocks_h   = [1 2];

for ks = 1:2
    for ie = 1:2
        sp = subplot(2, 2, (ks-1)*2 + ie);
        D  = IRFs_heat{ie}(:,:,shocks_h(ks));

        % Horizon edges: half a cell before first, half after last
        dh       = hyr(2) - hyr(1);                     % 0.25 yr (1 quarter)
        hyr_edge = [hyr - dh/2,  hyr(end) + dh/2];      % 1 x (H+1)

        % Maturity edges, midpoints between consecutive maturities,
        % with symmetric outer edges
        m        = mats_yr(:);                           % 6 x 1
        mid      = (m(1:end-1) + m(2:end)) / 2;         % 5 interior edges
        m_edge   = [m(1) - (mid(1)-m(1));               % lower outer edge
                    mid;                                 % 5 interior edges
                    m(end) + (m(end)-mid(end))];         % upper outer edge
        % Pad D with a NaN row and NaN column (pcolor drops last row/col)
        % Signed sqrt compression, preserves sign, stretches small values
        D_scaled = sign(D) .* sqrt(abs(D));
        D_pad    = [D_scaled, nan(size(D_scaled,1),1);
            nan(1,    size(D_scaled,2)+1)];

        pcolor(hyr_edge, m_edge, D_pad);
        shading flat;

        cv_raw    = max(abs(D(:))) * 1.05;
        cv_scaled = sqrt(cv_raw);
        clim([-cv_scaled, cv_scaled]);
        colormap(sp, redblue_cmap(256));
        cb = colorbar;
        cb.FontSize = 10;

        % Colorbar ticks in original bps units
        tick_bps    = [-10 -5 -2 -1 0 1 2 5 10];
        tick_bps    = tick_bps(abs(tick_bps) <= cv_raw);
        tick_scaled = sign(tick_bps) .* sqrt(abs(tick_bps));
        cb.Ticks      = tick_scaled;
        cb.TickLabels = arrayfun(@(x) sprintf('%d', x), tick_bps, 'un', 0);
        ylabel(cb, 'bps', 'FontSize', 10);
        set(gca, 'FontSize', 12);
        yticks(mats_yr);
        yticklabels(arrayfun(@(x) sprintf('%dyr',x), mats_yr, 'un',0));
        xlabel('Years', 'FontSize', 10);
        hold on;
        yline(10, 'k--', 'LineWidth', 1.0, 'Alpha', 0.8);
        hold off;
        if ks == 1
            title(est_labels{ie}, 'FontSize', 12, 'FontWeight','bold');
        end
        if ie == 1
            ylabel(sprintf('%s\nMaturity', shock_name{ks}), 'FontSize', 14);
        end
    end
end

sgtitle({'Yield Curve Impulse Response Functions (basis points)', ...
    'Each cell = response of that yield at that horizon to a 1-std-dev shock', ...
    'Dashed line = 10-year maturity'}, ...
    'FontSize', 10, 'FontWeight', 'bold');

exportgraphics(fig1, 'Graphs/IRF_Heatmap.pdf', 'ContentType','vector');
fprintf('Saved: IRF_Heatmap.pdf\n');

% =========================================================================
%% Fig 2 — ESR vs TP Decomposition  (2x2)
%
% Layout:
%   [1,1] 2-year yield — Inflation     [1,2] 10-year yield — Inflation
%   [2,1] 2-year yield — Unemployment  [2,2] 10-year yield — Unemployment

fig2 = figure('Name','Fig2: ESR vs TP Decomposition', ...
    'Units','normalized','Position',[0.10 0.10 0.80 0.75]);

for ks = 1:2
    for im = 1:2
        i  = sel_idx(im);
        sp = (ks-1)*2 + im;
        subplot(2, 2, sp);
        hold on;

        % 1Q: solid total, dashed ESR, dotted TP
        p1 = plot(hyr, IRF_y_v6(i,:,ks),   '-',  'Color',col_v6,  'LineWidth',lw,     'DisplayName','1Q — Total');
        p2 = plot(hyr, IRF_esr_v6(i,:,ks), '--o', 'Color',col_v6,  'LineWidth',lw-0.4, 'DisplayName','1Q — ESR');
        p3 = plot(hyr, IRF_tp_v6(i,:,ks),  ':+',  'Color',col_v6,  'LineWidth',lw-0.4, 'DisplayName','1Q — TP');

        % NS
        p4 = plot(hyr, IRF_y_v23(i,:,ks),   '-',  'Color',col_v23, 'LineWidth',lw,     'DisplayName','NS — Total');
        p5 = plot(hyr, IRF_esr_v23(i,:,ks), '--o', 'Color',col_v23, 'LineWidth',lw-0.4, 'DisplayName','NS — ESR');
        p6 = plot(hyr, IRF_tp_v23(i,:,ks),  ':+',  'Color',col_v23, 'LineWidth',lw-0.4, 'DisplayName','NS — TP');

        % Shade gap in total yield IRF
        fill_between(hyr, IRF_y_v6(i,:,ks), IRF_y_v23(i,:,ks), [0.75 0.82 0.95], 0.25);

        yline(0, 'k-', 'LineWidth', 0.5, 'Alpha', 0.7);
        hold off;

        title(sprintf('%s — %s', sel_label{im}, shock_name{ks}), 'FontSize', 10);
        xlabel('Horizon (years)', 'FontSize', 9);
        ylabel('Response (bps)', 'FontSize', 9);
        xlim([0 H/ann]);
        grid on;  box on;  set(gca, 'FontSize', 9);

        if sp == 1
            legend([p1 p2 p3 p4 p5 p6], 'Location','best', ...
                'FontSize', 10, 'NumColumns', 2);
        end
    end
end

sgtitle({'IRF Decomposition: Expected Short Rate (ESR) vs Term Premium (TP)', ...
    'Solid = Total  |  Dashed = ESR  |  Dotted = TP', ...
    'Blue = 1Q  |  Red = NS  |  Shaded = gap'}, ...
    'FontSize', 14, 'FontWeight','bold');

exportgraphics(fig2, 'Graphs/IRF_ESR_TP_Decomposition.pdf', 'ContentType','vector');
fprintf('Saved: IRF_ESR_TP_Decomposition.pdf\n');

%% Fig 33 — Survey Impact with MC Confidence Bands  (2x2)
%
% Layout:
%   [1,1] 2-year yield — Inflation     [1,2] 10-year yield — Inflation
%   [2,1] 2-year yield — Unemployment  [2,2] 10-year yield — Unemployment
%
% Central lines: empirical V6 and V23 estimates.
% Shaded bands: 10th-90th percentile of MC replications.
%   Red band  = estimator A (no surveys, aligned with NS)
%   Blue band = estimator B1 (h=1 surveys, aligned with 1Q)

fig3 = figure('Name','Fig3: Survey Impact with MC Bands', ...
    'Units','normalized','Position',[0.10 0.10 0.80 0.75]);

for ks = 1:2
    for im = 1:2
        i  = sel_idx(im);
        sp = (ks-1)*2 + im;
        subplot(2, 2, sp);
        hold on;

        % MC confidence bands
        lo_A  = squeeze(band_A_lo(im,ks,:))';
        hi_A  = squeeze(band_A_hi(im,ks,:))';
        lo_B1 = squeeze(band_B1_lo(im,ks,:))';
        hi_B1 = squeeze(band_B1_hi(im,ks,:))';

        fill_between(hyr, lo_A,  hi_A,  col_fill_v23, 0.5);
        fill_between(hyr, lo_B1, hi_B1, col_fill_v6,  0.5);

        % Empirical central estimates
        p1 = plot(hyr, IRF_y_v6(i,:,ks),  '-',  'Color',col_v6,  'LineWidth',lw, ...
            'DisplayName','1Q');
        p2 = plot(hyr, IRF_y_v23(i,:,ks), '--', 'Color',col_v23, 'LineWidth',lw, ...
            'DisplayName','NS');

        % Annotate gap at Q4 (1-year horizon)
        h_ann = 4;
        gap   = IRF_y_v6(i,h_ann,ks) - IRF_y_v23(i,h_ann,ks);
        if abs(gap) > 0.3
            ymid = (IRF_y_v6(i,h_ann,ks) + IRF_y_v23(i,h_ann,ks)) / 2;
            text(hyr(h_ann) + 0.08, ymid, sprintf('%+.1f bps', gap), ...
                'FontSize', 12, 'Color', [0.2 0.2 0.5], 'FontWeight','bold');
        end

        yline(0, 'k-', 'LineWidth', 0.5, 'Alpha', 0.6);
        hold off;

        title(sprintf('%s — %s', sel_label{im}, shock_name{ks}), 'FontSize', 10);
        xlabel('Years', 'FontSize', 12);
        ylabel('Yield response (bps)', 'FontSize', 12);
        xlim([0 H/ann]);
        grid on;  box on;  set(gca, 'FontSize', 12);

        if sp == 1
            legend([p1 p2], 'Location','best', 'FontSize', 9);
            text(0.98, 0.06, ...
                sprintf('Bands: %dth–%dth percentile\n(MC, R = 2000)', pct_lo, pct_hi), ...
                'Units','normalized','FontSize',7.5,'HorizontalAlignment','right', ...
                'Color',[0.35 0.35 0.35]);
        end
    end
end

sgtitle({'Yield IRF: Survey-Augmented (V6) vs No-Survey (V23)', ...
    'Shaded bands = Monte Carlo estimation uncertainty (10th–90th percentile, R = 2000)', ...
    'Annotated gap = V6 minus V23 at the 1-year horizon'}, ...
    'FontSize', 10, 'FontWeight','bold');

exportgraphics(fig3, 'Graphs/IRF_SurveyComparison.pdf', 'ContentType','vector');
fprintf('Saved: IRF_SurveyComparison.pdf\n');

%% Fig 4 — Term Premium IRF  (2x2)
%
% Layout:
%   [1,1] 2-year TP — Inflation      [1,2] 10-year TP — Inflation
%   [2,1] 2-year TP — Unemployment   [2,2] 10-year TP — Unemployment

fig4 = figure('Name','Fig4: Term Premium IRF', ...
    'Units','normalized','Position',[0.10 0.10 0.80 0.75]);

for ks = 1:2
    for im = 1:2
        i  = sel_idx(im);
        sp = (ks-1)*2 + im;
        subplot(2, 2, sp);
        hold on;

        fill_between(hyr, IRF_tp_v6(i,:,ks), IRF_tp_v23(i,:,ks), [0.75 0.82 0.95], 0.30);

        p1 = plot(hyr, IRF_tp_v6(i,:,ks),  '-',  'Color',col_v6,  'LineWidth',lw, ...
            'DisplayName','1Q');
        p2 = plot(hyr, IRF_tp_v23(i,:,ks), '--', 'Color',col_v23, 'LineWidth',lw, ...
            'DisplayName','NS');

        yline(0, 'k-', 'LineWidth', 0.5, 'Alpha', 0.4);
        hold off;

        title(sprintf('%s TP — %s', sel_label{im}, shock_name{ks}), 'FontSize', 10);
        xlabel('Horizon (years)', 'FontSize', 10);
        ylabel('Term premium response (bps)', 'FontSize', 10);
        xlim([0 H/ann]);
        grid on;  box on;  set(gca, 'FontSize', 10);

        if sp == 1
            legend([p1 p2], 'Location','best', 'FontSize', 12);
        end
    end
end

sgtitle({'Term Premium Component of Yield IRF: V6 vs V23', ...
    'How much of the yield response reflects risk compensation (not revised rate expectations)', ...
    'Shaded area = gap attributable to survey correction'}, ...
    'FontSize', 10, 'FontWeight','bold');

exportgraphics(fig4, 'Graphs/IRF_TermPremium.pdf', 'ContentType','vector');
fprintf('Saved: IRF_TermPremium.pdf\n');

%% Numerical summary table

fprintf('\n=================================================================\n');
fprintf(' Numerical summary: 10-year yield IRF (basis points)\n');
fprintf('=================================================================\n\n');

i10   = find(mats_yr == 10);
fmt_h = '%-10s  %9s %9s %9s  %9s %9s %9s  %10s\n';
fmt_r = 'Q%-9d  %9.2f %9.2f %9.2f  %9.2f %9.2f %9.2f  %+10.2f\n';
cols  = {'Horizon','V6 Tot','V6 ESR','V6 TP','V23 Tot','V23 ESR','V23 TP','Gap V6-V23'};

for ks = 1:2
    fprintf('--- %s shock ---\n', shock_short{ks});
    fprintf(fmt_h, cols{:});
    fprintf('%s\n', repmat('-',1,100));
    for h = [1 2 4 8 12 16 20]
        fprintf(fmt_r, h, ...
            IRF_y_v6(i10,h,ks),  IRF_esr_v6(i10,h,ks),  IRF_tp_v6(i10,h,ks), ...
            IRF_y_v23(i10,h,ks), IRF_esr_v23(i10,h,ks), IRF_tp_v23(i10,h,ks), ...
            IRF_y_v6(i10,h,ks) - IRF_y_v23(i10,h,ks));
    end
    fprintf('\n');
end

fprintf('Done.\n');

%% Local Functions

function fill_between(x, y1, y2, col, alpha_val)
    % Shade the region between curves y1 and y2.
    xp = [x(:)', fliplr(x(:)')];
    yp = [y1(:)', fliplr(y2(:)')];
    fill(xp, yp, col, 'FaceAlpha', alpha_val, 'EdgeColor','none');
end

function cmap = redblue_cmap(n)
    % Diverging red-white-blue colormap.
    h  = floor(n/2);
    r1 = linspace(0.70,1.00,h)';   g1 = linspace(0.10,1.00,h)';   b1 = linspace(0.10,1.00,h)';
    r2 = linspace(1.00,0.10,n-h)'; g2 = linspace(1.00,0.20,n-h)'; b2 = linspace(1.00,0.70,n-h)';
    cmap = [r1,g1,b1; r2,g2,b2];
end

function [g0, gx] = compute_loadings(alpha, muQ, phiQ, beta, sigma, matSel, ann)
    nx   = length(beta);
    sig2 = sigma * sigma';
    nMax = max(matSel);
    A    = zeros(1, nMax);
    B    = zeros(nx, nMax);
    A(1) = -alpha;  B(:,1) = -beta;
    for k = 2:nMax
        A(k)   = -alpha + A(k-1) + B(:,k-1)'*muQ + 0.5*(B(:,k-1)'*sig2*B(:,k-1));
        B(:,k) = -beta + phiQ'*B(:,k-1);
    end
    g0 = (ann * (-A(matSel) ./ matSel))';
    gx = (ann * (-B(:,matSel) ./ matSel))';
end

function [IRF_y, IRF_esr, IRF_tp] = compute_IRF(gx, phiP, beta, sigma, matSel, ann, H)
    nmat = size(gx,1);  nx = size(phiP,1);  nfac = nx;
    rx   = ann * beta';
    bps  = 10000;
    IRF_y   = zeros(nmat, H, nfac);
    IRF_esr = zeros(nmat, H, nfac);
    for k = 1:nfac
        shock  = sigma(:, k);
        phiP_h = eye(nx);
        for h = 1:H
            for i = 1:nmat
                IRF_y(i,h,k) = gx(i,:) * phiP_h * shock * bps;
            end
            for i = 1:nmat
                n = matSel(i);  esr = 0;  phiP_hj = phiP_h;
                for j = 0:n-1
                    if j > 0,  phiP_hj = phiP_hj * phiP;  end
                    esr = esr + rx * phiP_hj * shock;
                end
                IRF_esr(i,h,k) = (esr / n) * bps;
            end
            phiP_h = phiP_h * phiP;
        end
    end
    IRF_tp = IRF_y - IRF_esr;
end

