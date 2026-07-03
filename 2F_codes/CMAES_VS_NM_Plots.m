% This script computes plot to compare the NM and CMAES estimator.
% Uses the saved workspace data "GATSM_2F_Estimates_US_CMAES.mat"
% and "GATSM_2F_Estimates_US_NelderMead.mat"

%   Figure 1  —  Yield fit: actual vs CMA-ES vs Nelder-Mead (6 maturities)
%   Figure 2  —  Pricing errors in basis points (6 maturities)
%   Figure 3  —  Term premia: CMA-ES vs Nelder-Mead (6 maturities)
%   Figure 4  —  Yield decomposition at 2, 5, 10 years (CMA-ES)
%   Figure 5  —  Yield decomposition at 2, 5, 10 years (Nelder-Mead)
%   Figure 6  —  Term premium difference: CMA-ES minus Nelder-Mead

% Used functions at the bottom of the MATLAB file 

clear; clc; close all;

%%  LOAD WORKSPACES

ws_cma = load('GATSM_2F_Estimates_US_CMAES.mat', 'outML');
outML_cma = ws_cma.outML;

ws_nm  = load('GATSM_2F_Estimates_US_NelderMead.mat', 'outML');
outML_nm  = ws_nm.outML;

%% Setp date date vector, maturities, color schemes 

% Date vector (Jan 1972 monthly)
T       = size(outML_cma.outKF.data, 2);
dateVec = datetime(1972, 1, 1) + calmonths(0:T-1);

% Maturities
matSelect  = outML_cma.outKF.model.matSelect;   % months
matYears   = matSelect / 12;
numObs     = length(matSelect);

% Colour scheme
colours = lines(numObs);

% rgb
col_cma = [0.13 0.47 0.71];   % blue CMA-ES
col_nm  = [0.84 0.15 0.16];   % red Nelder-Mead

%%  COMPUTE FITTED YIELDS AND PRICING ERRORS

% CMA-ES
outKF_cma    = outML_cma.outKF;
yHat_cma     = outKF_cma.g0 + outKF_cma.gx * outKF_cma.xHat;   % numObs x T
merrors_cma  = (outKF_cma.data - yHat_cma) * 10000;              % bps

% Nelder-Mead
outKF_nm     = outML_nm.outKF;
yHat_nm      = outKF_nm.g0  + outKF_nm.gx  * outKF_nm.xHat;    % numObs x T
merrors_nm   = (outKF_nm.data  - yHat_nm)  * 10000;              % bps

%%  COMPUTE YIELD CURVE DECOMPOSITIONS

resDecom_cma = local_yieldCurveDecom(outML_cma);
resDecom_nm  = local_yieldCurveDecom(outML_nm);

%% Fig 1 — YIELD FIT: ACTUAL vs CMA-ES vs NELDER-MEAD

selMatsYF  = [1 15];                          % years to show
selIdxYF   = arrayfun(@(m) find(matYears == m, 1), selMatsYF);

figure('Name', 'Yield Fit Comparison', ...
       'Units', 'normalized', 'Position', [0.10 0.10 0.80 0.70]);

optimLabels = {'CMA-ES', 'Nelder-Mead'};
yHats       = {yHat_cma, yHat_nm};
outKFs      = {outKF_cma, outKF_nm};
cols        = {col_cma, col_nm};

for col = 1:2           % col 1 = CMA-ES, col 2 = Nelder-Mead
    for row = 1:2       % row 1 = 1yr,    row 2 = 15yr
        subplot(2, 2, (row-1)*2 + col);
        i      = selIdxYF(row);
        actual = outKFs{col}.data(i, :) * 100;
        fitted = yHats{col}(i, :)       * 100;
        plot(dateVec, actual, 'Color', [0.65 0.65 0.65], 'LineWidth', 2.7); hold on;
        plot(dateVec, fitted, 'Color', cols{col},         'LineWidth', 0.5);
        hold off;
        title(sprintf('%s — %g-year yield', optimLabels{col}, matYears(i)), 'FontSize', 9);
        ylabel('Yield (%)', 'FontSize', 8);
        yLimits = ylim;
        yMin = floor(yLimits(1) / 2) * 2;
        yMax = ceil(yLimits(2)  / 2) * 2;
        yticks(yMin : 2 : yMax);
        ylim([min(yMin, 0) yMax]);
        legend({'Actual', optimLabels{col}}, 'Location', 'best', 'FontSize', 7);
        grid on; box on;
        set(gca, 'FontSize', 8);
    end
end

sgtitle('2-Factor GATSM — Yield Fit: CMA-ES vs Nelder-Mead', ...
        'FontSize', 11, 'FontWeight', 'bold');
exportgraphics(gcf, 'Graphs\Compare_YieldFit_2x2.pdf');

%%  Fig 2 — TERM PREMIA: CMA-ES vs NELDER-MEAD (1yr & 15yr)

selMatsTP = [1 15];
selIdxTP  = arrayfun(@(m) find(resDecom_cma.matSelect == m*12, 1), selMatsTP);
matYearsD = resDecom_cma.matSelect / 12;

figure('Name', 'Term Premia Comparison', ...
       'Units', 'normalized', 'Position', [0.15 0.15 0.70 0.55]);

for k = 1:length(selIdxTP)
    i = selIdxTP(k);
    subplot(1, 2, k);
    plot(dateVec, resDecom_cma.termPremia(:,i)*100, 'Color', col_cma, 'LineWidth', 1.1); hold on;
    plot(dateVec, resDecom_nm.termPremia(:,i) *100, 'Color', col_nm,  'LineWidth', 0.9, 'LineStyle', '--');
    yline(0, 'k-', 'LineWidth', 0.6);
    hold off;
    title(sprintf('%g-year term premium', matYearsD(i)), 'FontSize', 9);
    ylabel('Term premium (%)', 'FontSize', 8);
    xlim([dateVec(1) dateVec(end)]);
    legend({'CMA-ES', 'Nelder-Mead'}, 'Location', 'best', 'FontSize', 7);
    grid on; box on;
    set(gca, 'FontSize', 8);
end

sgtitle('Term Premia Comparison', ...
        'FontSize', 11, 'FontWeight', 'bold');
exportgraphics(gcf, 'Graphs\Compare_TermPremia.pdf');

%%  Figs 3 & 4 — YIELD DECOMPOSITION AT 2, 5, 10 YEARS

selMats = [2 5 10];
selIdx  = arrayfun(@(m) find(resDecom_cma.matSelect == m*12, 1, 'first'), ...
                   selMats(ismember(selMats*12, resDecom_cma.matSelect)));

for opt = 1:2   % 1 = CMA-ES, 2 = Nelder-Mead
    if opt == 1
        res   = resDecom_cma;
        label = 'CMA-ES';
        fname = 'Graphs\Compare_Decomposition_CMAES.pdf';
    else
        res   = resDecom_nm;
        label = 'Nelder-Mead';
        fname = 'Graphs\Compare_Decomposition_NelderMead.pdf';
    end

    figure('Name', sprintf('Yield Decomposition — %s', label), ...
           'Units', 'normalized', 'Position', [0.05 0.10 0.90 0.60]);

    for k = 1:length(selIdx)
        i = selIdx(k);
        subplot(1, length(selIdx), k);
        plot(dateVec, res.yHat(:,i)      * 100, 'k',   'LineWidth', 1.4); hold on;
        plot(dateVec, res.rExp(:,i)       * 100, 'b--', 'LineWidth', 1.2);
        plot(dateVec, res.termPremia(:,i) * 100, 'r:',  'LineWidth', 1.4);
        hold off;
        title(sprintf('%g-year yield (%s)', matYearsD(i), label), 'FontSize', 9);
        ylabel('Rate (%)', 'FontSize', 8);
        xlim([dateVec(1) dateVec(end)]);
        legend({'Fitted yield', 'Exp. short rate', 'Term premium'}, ...
               'Location', 'best', 'FontSize', 7);
        grid on; box on;
        set(gca, 'FontSize', 8);
    end
    sgtitle(sprintf('2-Factor GATSM — Yield Decomposition (%s)', label), ...
            'FontSize', 11, 'FontWeight', 'bold');

    exportgraphics(gcf, fname);
end

%% Summary Table

fprintf('\n=== Yield Fit RMSE Comparison (basis points) ===\n');
fprintf('%-10s  %12s  %12s  %12s\n', 'Mat (yr)', 'CMA-ES', 'Nelder-Mead', 'Difference');
fprintf('%s\n', repmat('-', 1, 50));
for i = 1:numObs
    rmse_c = sqrt(mean(merrors_cma(i,:).^2));
    rmse_n = sqrt(mean(merrors_nm(i,:).^2));
    fprintf('%-10.1f  %12.3f  %12.3f  %12.3f\n', matYears(i), rmse_c, rmse_n, rmse_c - rmse_n);
end
fprintf('%-10s  %12.3f  %12.3f  %12.3f\n', 'Overall', ...
    sqrt(mean(merrors_cma(:).^2)), sqrt(mean(merrors_nm(:).^2)), ...
    sqrt(mean(merrors_cma(:).^2)) - sqrt(mean(merrors_nm(:).^2)));

fprintf('\n=== Mean Term Premium Comparison (%%) ===\n');
fprintf('%-10s  %12s  %12s  %12s\n', 'Mat (yr)', 'CMA-ES', 'Nelder-Mead', 'Difference');
fprintf('%s\n', repmat('-', 1, 50));
for i = 1:ny
    tp_c = mean(resDecom_cma.termPremia(:,i)) * 100;
    tp_n = mean(resDecom_nm.termPremia(:,i))  * 100;
    fprintf('%-10.1f  %12.4f  %12.4f  %12.4f\n', matYearsD(i), tp_c, tp_n, tp_c - tp_n);
end

fprintf('\n=== Log-Likelihood Comparison ===\n');
fprintf('  CMA-ES       avg log L / T = %.6f\n', -outML_cma.avgLogL);
fprintf('  Nelder-Mead  avg log L / T = %.6f\n', -outML_nm.avgLogL);
fprintf('  Difference (CMA-ES - NM)   = %.6f\n', outML_nm.avgLogL - outML_cma.avgLogL);

fprintf('\nAll figures exported as PDF.\n');

%%  LOCAL FUNCTION — YIELD CURVE DECOMPOSITION

function res = local_yieldCurveDecom(outML)
% Decomposes model-implied yields into expected short rates and term premia.
%   res.yHat       : T x ny  fitted yields (annualised decimal)
%   res.rExp       : T x ny  average expected short rate (annualised decimal)
%   res.termPremia : T x ny  term premium = yHat - rExp
%   res.matSelect  : 1 x ny  maturities in months

model  = outML.outKF.model;
xhat   = outML.outKF.xHat;          % nx x T
[nx,T] = size(xhat);
muP    = model.muP;
phiP   = model.phiP;
maxMat = max(model.matSelect);

% Expected factors under P
xExp = nan(nx, maxMat, T);
for t = 1:T
    for i = 1:maxMat
        if i == 1
            xExp(:,1,t) = xhat(:,t);
        else
            xExp(:,i,t) = muP + phiP * xExp(:,i-1,t);
        end
    end
end

% Expected short rates under P
rExp = nan(maxMat, T);
for t = 1:T
    for i = 1:maxMat
        rExp(i,t) = model.r0 + model.rx * xExp(:,i,t);
    end
end

% Fitted yields and average expected short rate
yHat    = outML.outKF.g0 + outML.outKF.gx * outML.outKF.xHat;  % ny x T
ny      = length(model.matSelect);
rExpAvg = nan(ny, T);
for t = 1:T
    for i = 1:ny
        rExpAvg(i,t) = mean(rExp(1:model.matSelect(i), t));
    end
end

res.yHat       = yHat';
res.rExp       = rExpAvg';
res.termPremia = res.yHat - res.rExp;
res.matSelect  = model.matSelect;
end