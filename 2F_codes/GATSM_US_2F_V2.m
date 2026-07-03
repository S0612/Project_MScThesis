% Main script for estimating the two-factor latent GATSM on US data
% 
% Requires cmaes_dsgeDisplay.m or cmaes_dsgeDisplay_mp_v2.m by Andreasen (2010) for CMAES.
% All otherhelper functions are embedded at the bottom of this file so that no
% external files
%
% Data files expected in the MATLAB path / current directory:
%   UK_yields_1987_2025.csv          (T x M matrix of yields in %)
%   UK_yields_1987_2025_maturities.csv  (1 x M row of maturities in years)

addpath(genpath(pwd))
close all; clear; clc;

%% User Setting

matSelect       = [1 2 5 7 10 15]*12;     % Maturities in months

% Optimizer settings
optim.MaxIter     = 10000*10;
optim.MaxFunEvals = 10000*10;
optim.TolX        = 1e-4;
optim.TolFun      = 1e-4;
optim.optimizer   = 1;  
% 0 = use supplied starting values (no search)
% 1 = CMA-ES with multiprocessing
% 2 = CMA-ES without multiprocessing
% 3 = Nelder-Mead (fminsearch)

%% Parameter initialisation

nx = 2;   % Number of latent factors

% Calibrated parameters (fixed, not estimated) as per our restrictions.
calibrateParams.phiQ21 = 0;
calibrateParams.phiQ12 = 0;
calibrateParams.muQ1   = 0;
calibrateParams.muQ2   = 0;
calibrateParams.beta1  = 1;
calibrateParams.beta2  = 1;

% Starting values for estimated parameters

% Diagonal Q-dynamics (risk-neutral AR coefficients)
for i = 1:nx
    name = ['phiQ', num2str(i), num2str(i)];
    params0.(name) = 0.01 + i*0.02;
end

% Short-rate intercept
params0.alpha = 0.01;

% P-dynamics: mean and full AR matrix
for i = 1:nx
    name = ['muP', num2str(i)];
    params0.(name) = 0.0;
end
for i = 1:nx
    for j = 1:nx
        name = ['phiP', num2str(i), num2str(j)];
        params0.(name) = 0.1;
    end
end

% Lower-triangular Cholesky factor of innovation covariance
for i = 1:nx
    for j = i:nx
        name = ['sigma', num2str(j), num2str(i)];
        params0.(name) = 0.001;
    end
end

% This is a dead-value, not the actual starting value. This line purely
% exists such that stdY is included in params0. 
params0.stdY = 0.001;

% Starting values derived from StartingParams2F_US.m

%   phiQ11, phiQ22, alpha,
%   muP1, muP2,
%   phiP11, phiP12, phiP21, phiP22,
%   sigma11, sigma21, sigma22, stdY
params0Values = [ ...
        0.9970249269      0.9719181887      0.0040417407      0.0000055757  ...
       -0.0000626359      0.9970249269      0.0000000000      0.0000000000  ...
        0.9719181887      0.0003164546      0.0000000000      0.0001822510  ...
        0.0010356332]';

%% Load data

% Load UK yield data from CSV files.
%   yields_raw : T x M matrix of yields (%), rows = months, cols = maturities
%   mats_years : 1 x M vector of maturities in years
yields_raw = csvread('US_monthly_yields_Jan1972_Dec2025.csv');
mats_years = csvread('US_monthly_yields_Jan1972_Dec2025_maturities.csv');

% Convert maturities from years to months (must be integer-valued months)
mats_months = mats_years * 12;

% Select the requested maturities and transpose to numObs x T
% Yields are already in % units. Divide by 100 for annualised decimal units
numObs = length(matSelect);
T      = size(yields_raw, 1);
data   = zeros(numObs, T);
for i = 1:numObs
    col       = find(mats_months == matSelect(i));
    data(i,:) = yields_raw(:, col)' / 100;
end

% Construct a monthly date vector
% The CSV covers 1987m1 through 2025m12
dateStart      = datenum(1972, 1, 1);
setup_dates    = dateStart + (0:T-1)' * 30;

%% STEP 4: Construct the setup struct

[upperBounds, lowerBounds, Insigma] = local_paramsBoundsInsigma(nx);
setup.nx                = nx;
setup.matSelect         = matSelect;
setup.data              = data;
setup.timeIndex         = setup_dates;
setup.calibrateParams   = calibrateParams;
setup.epsValue          = 1e-4;
setup.selectParams      = fieldnames(params0);
setup.InsigmaValues     = local_struc2values(Insigma,     setup.selectParams);
setup.lowerBoundsValues = local_struc2values(lowerBounds, setup.selectParams);
setup.upperBoundsValues = local_struc2values(upperBounds, setup.selectParams);
setup.optimizer         = optim.optimizer;

% =========================================================================
%% STEP 5: Numerical optimisation -> MLE parameters & standard errors
% =========================================================================
outML = local_estimationMLE(params0Values, setup, optim);

% Display optimal parameters to screen
disp('=== Estimated parameters ===');
disp(local_struct2array(outML.paramsOpt)');

fprintf('Average log-likelihood at optimum: %.6f\n', outML.avgLogL);

%% STEP 6: Standard errors

% Standard errors are computed via the outer product of scores (OPG estimator).

paramNames = setup.selectParams;
paramsOpt  = local_struct2array(outML.paramsOpt)';
paramsSE   = local_struct2array(outML.paramsSE)';
 
fprintf('\n%-12s  %12s  %12s  %12s\n', 'Parameter', 'Estimate', 'Std Error', 't-stat');
fprintf('%s\n', repmat('-', 1, 52));
for i = 1:length(paramNames)
    tstat = paramsOpt(i) / paramsSE(i);
    fprintf('%-12s  %12.6f  %12.6f  %12.3f\n', ...
        paramNames{i}, paramsOpt(i), paramsSE(i), tstat);
end

%% LateX Table

% Collect point estimates and standard errors
est = local_struct2array(outML.paramsOpt)';   % column vector
se  = local_struct2array(outML.paramsSE)';    % column vector
 
% Map setup.selectParams to indices for readability
idx = @(name) find(strcmp(setup.selectParams, name));
 
% Helper: format one cell as "value\n(se)" in LaTeX
fmt = @(e, s) sprintf('%.6f \\\\\\\\ \n    (%.6f)', e, s);
 
% Helper: write one table row: label & value\\  then & (se)\\
% We use a two-line-per-parameter layout (value row + SE row).
latexRow = @(label, e, s) sprintf('%s & %.6f \\\\\\\\\n & (%.6f) \\\\\\\\', ...
    label, e, s);
 
% Choose filename based on optimizer
if optim.optimizer == 1 || optim.optimizer == 2
    texFilename = 'GATSM_2F_latex_table_CMAES.tex';
elseif optim.optimizer == 3
    texFilename = 'GATSM_2F_latex_table_NelderMead.tex';
else
    texFilename = 'GATSM_2F_latex_table.tex';
end
 
fid = fopen(texFilename, 'w');
 
fprintf(fid, '\\begin{table}[htbp]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Estimated parameters -- 2-factor GATSM (Nelder-Mead MLE)}\n');
fprintf(fid, '\\label{tab:gatsm_estimates}\n');
fprintf(fid, '\\begin{tabular}{lc}\n');
fprintf(fid, '\\toprule\n');
fprintf(fid, ' & Nelder-Mead \\\\\n');
fprintf(fid, ' & (1) \\\\\n');
fprintf(fid, '\\midrule\n');
 
% Q-dynamics
fprintf(fid, '$\\lambda_1^\\circ$ & %.6f \\\\\n', ...
    est(idx('phiQ11')));
fprintf(fid, ' & (%.6f) \\\\\n', se(idx('phiQ11')));
 
fprintf(fid, '$\\lambda_2^\\circ$ & %.6f \\\\\n', ...
    est(idx('phiQ22')));
fprintf(fid, ' & (%.6f) \\\\\n', se(idx('phiQ22')));
 
% Short-rate intercept
fprintf(fid, '$\\alpha$ & %.6f \\\\\n', est(idx('alpha')));
fprintf(fid, ' & (%.6f) \\\\\n',        se(idx('alpha')));
 
% P-mean
fprintf(fid, '$\\mu_1^P$ & %.6f \\\\\n', est(idx('muP1')));
fprintf(fid, ' & (%.6f) \\\\\n',          se(idx('muP1')));
 
fprintf(fid, '$\\mu_2^P$ & %.6f \\\\\n', est(idx('muP2')));
fprintf(fid, ' & (%.6f) \\\\\n',          se(idx('muP2')));
 
% P-AR matrix (rows then columns: Phi_ij = row i, col j)
fprintf(fid, '$\\Phi_{11}^P$ & %.6f \\\\\n', est(idx('phiP11')));
fprintf(fid, ' & (%.6f) \\\\\n',              se(idx('phiP11')));
 
fprintf(fid, '$\\Phi_{12}^P$ & %.6f \\\\\n', est(idx('phiP12')));
fprintf(fid, ' & (%.6f) \\\\\n',              se(idx('phiP12')));
 
fprintf(fid, '$\\Phi_{21}^P$ & %.6f \\\\\n', est(idx('phiP21')));
fprintf(fid, ' & (%.6f) \\\\\n',              se(idx('phiP21')));
 
fprintf(fid, '$\\Phi_{22}^P$ & %.6f \\\\\n', est(idx('phiP22')));
fprintf(fid, ' & (%.6f) \\\\\n',              se(idx('phiP22')));
 
% Cholesky factor of innovation covariance
fprintf(fid, '$\\Sigma_{11}$ & %.6f \\\\\n', est(idx('sigma11')));
fprintf(fid, ' & (%.6f) \\\\\n',              se(idx('sigma11')));
 
fprintf(fid, '$\\Sigma_{21}$ & %.6f \\\\\n', est(idx('sigma21')));
fprintf(fid, ' & (%.6f) \\\\\n',              se(idx('sigma21')));
 
fprintf(fid, '$\\Sigma_{22}$ & %.6f \\\\\n', est(idx('sigma22')));
fprintf(fid, ' & (%.6f) \\\\\n',              se(idx('sigma22')));
 
% Measurement error
fprintf(fid, '$\\sigma_v$ & %.6f \\\\\n', est(idx('stdY')));
fprintf(fid, ' & (%.6f) \\\\\n',           se(idx('stdY')));
 
% --- Log-likelihood ---
fprintf(fid, '\\midrule\n');
fprintf(fid, '$\\frac{\\log \\mathcal{L}}{T}$ & %.6f \\\\\n', -outML.avgLogL);
 
fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\begin{tablenotes}\\small\n');
fprintf(fid, '  \\item Standard errors (OPG estimator) in parentheses.\n');
fprintf(fid, '\\end{tablenotes}\n');
fprintf(fid, '\\end{table}\n');
 
fclose(fid);
fprintf('\nLaTeX table written to: %s\n', texFilename);
 

%% Post-estimation diagnostics

%% 7a: Pricing errors
% Fitted yields from the Kalman smoother: yHat = g0 + gx * xHat
outKF        = outML.outKF;
yHat         = outKF.g0 + outKF.gx * outKF.xHat;   % numObs x T
merrors      = outKF.data - yHat;                   % numObs x T  (decimal)
merrors_bps  = merrors * 10000;                      % convert to basis points
 
matYears     = setup.matSelect / 12;                 % maturities in years
numObs       = length(setup.matSelect);
 
fprintf('\n%-8s  %10s  %10s  %10s  %10s\n', ...
    'Mat(yr)', 'Mean(bps)', 'Std(bps)', 'RMSE(bps)', 'MaxAbs(bps)');
fprintf('%s\n', repmat('-', 1, 54));
for i = 1:numObs
    e    = merrors_bps(i, :);
    fprintf('%-8.1f  %10.3f  %10.3f  %10.3f  %10.3f\n', ...
        matYears(i), mean(e), std(e), sqrt(mean(e.^2)), max(abs(e)));
end
fprintf('%-8s  %10.3f  %10.3f  %10.3f  %10.3f\n', 'Overall', ...
    mean(merrors_bps(:)), std(merrors_bps(:)), ...
    sqrt(mean(merrors_bps(:).^2)), max(abs(merrors_bps(:))));
 
%% 7b: Yield fit plot

% Date vector (monthly, starting 1987-Jan)
T         = size(outKF.data, 2);
dateVec   = datetime(1972, 1, 1) + calmonths(0:T-1);
 
% Colour scheme: one colour per maturity
colours = lines(numObs);
 
% Figure 1: time-series overlay for each maturity
figure('Name', 'Yield fit: actual vs model-implied', ...
       'Units', 'normalized', 'Position', [0.05 0.05 0.90 0.85]);
 
nCols = 3;
nRows = ceil(numObs / nCols);
 
for i = 1:numObs
    subplot(nRows, nCols, i);
    actual = outKF.data(i, :) * 100;     % back to percent
    fitted = yHat(i, :)       * 100;
    plot(dateVec, actual, 'Color', [0.6 0.6 0.6], 'LineWidth', 2.0); hold on;
    plot(dateVec, fitted, 'Color', colours(i, :), 'LineWidth', 0.5);
    hold off;
    title(sprintf('%g-year yield', matYears(i)), 'FontSize', 9);
    ylabel('Yield (%)', 'FontSize', 8);
    xlabel('');
    xlim([dateVec(1) dateVec(end)]);
    legend({'Actual', 'Model'}, 'Location', 'best', 'FontSize', 7);
    grid on; box on;
    set(gca, 'FontSize', 8);
end
sgtitle('US Yield Curve — Actual vs Model-Implied Yields', 'FontSize', 11, 'FontWeight', 'bold');

exportgraphics(gcf, 'YieldFitPlot.pdf');
 
% Figure 2: pricing errors (basis points) over time
figure('Name', 'Pricing errors (basis points)', ...
       'Units', 'normalized', 'Position', [0.05 0.05 0.90 0.85]);
 
for i = 1:numObs
    subplot(nRows, nCols, i);
    plot(dateVec, merrors_bps(i, :), 'Color', colours(i, :), 'LineWidth', 1.2);
    hold on;
    yline(0, 'k--', 'LineWidth', 0.8);
    hold off;
    title(sprintf('%g-year: RMSE = %.2f bps', matYears(i), ...
          sqrt(mean(merrors_bps(i,:).^2))), 'FontSize', 9);
    ylabel('Error (bps)', 'FontSize', 8);
    xlim([dateVec(1) dateVec(end)]);
    grid on; box on;
    set(gca, 'FontSize', 8);
end
sgtitle('US Yield Curve — Pricing Errors (Actual minus Model-Implied)', ...
        'FontSize', 11, 'FontWeight', 'bold');

exportgraphics(gcf, 'PricingErrorPlot.pdf');


%% Save Data

%if optim.optimizer == 1 || optim.optimizer == 2
%    save("GATSM_2F_Estimates_US_CMAES") ;
%elseif optim.optimizer == 3
%    save("GATSM_2F_Estimates_US_NelderMead") ;
%else

%end


%% Yield curve decomposition (fitted yields, expected short rates, and term premia)

resDecom = local_yieldCurveDecom(outML);

% summary table
matYearsDecom = resDecom.matSelect / 12;
ny            = length(resDecom.matSelect);
T_decom       = size(resDecom.yHat, 1);

fprintf('\n=== Yield Curve Decomposition: time-averaged values (%%)\n');
fprintf('%-10s  %12s  %12s  %12s\n', 'Mat(yr)', 'FittedYield', 'ExpShortRate', 'TermPremium');
fprintf('%s\n', repmat('-', 1, 52));
for i = 1:ny
    fprintf('%-10.1f  %12.4f  %12.4f  %12.4f\n', ...
        matYearsDecom(i), ...
        mean(resDecom.yHat(:,i))   * 100, ...
        mean(resDecom.rExp(:,i))   * 100, ...
        mean(resDecom.termPremia(:,i)) * 100);
end

% Figure 3: Term premia over time
figure('Name', 'Term Premia', ...
       'Units', 'normalized', 'Position', [0.05 0.05 0.90 0.85]);

nCols3 = 3;
nRows3 = ceil(ny / nCols3);
colours3 = lines(ny);

for i = 1:ny
    subplot(nRows3, nCols3, i);
    plot(dateVec, resDecom.termPremia(:,i) * 100, ...
         'Color', colours3(i,:), 'LineWidth', 1.2);
    hold on;
    yline(0, 'k--', 'LineWidth', 0.8);
    hold off;
    title(sprintf('%g-year term premium', matYearsDecom(i)), 'FontSize', 9);
    ylabel('Term premium (%)', 'FontSize', 8);
    xlim([dateVec(1) dateVec(end)]);
    grid on; box on;
    set(gca, 'FontSize', 8);
end
sgtitle('US Yield Curve — Term Premia', 'FontSize', 11, 'FontWeight', 'bold');

exportgraphics(gcf, 'TermPremiaPlot.pdf');


% Figure 4: Decomposition for selected maturities
% Plots fitted yield, expected short rate, and term premium on one axis.
selMats   = [2 10];                         % years to highlight
selIdx    = arrayfun(@(m) find(resDecom.matSelect == m*12, 1, 'first'), ...
                     selMats(ismember(selMats*12, resDecom.matSelect)));

if ~isempty(selIdx)
    figure('Name', 'Yield decomposition: selected maturities', ...
           'Units', 'normalized', 'Position', [0.05 0.05 0.90 0.75]);
    for k = 1:length(selIdx)
        i = selIdx(k);
        subplot(1, length(selIdx), k);
        plot(dateVec, resDecom.yHat(:,i)      * 100, 'k',  'LineWidth', 1.5); hold on;
        plot(dateVec, resDecom.rExp(:,i)       * 100, 'b--','LineWidth', 1.5);
        plot(dateVec, resDecom.termPremia(:,i) * 100, 'r:', 'LineWidth', 1.5);
        hold off;
        title(sprintf('%g-year yield', matYearsDecom(i)), 'FontSize', 12);
        ylabel('Rate (%)', 'FontSize', 12);
        xlim([dateVec(1) dateVec(end)]);
        legend({'Fitted yield','Exp. short rate','Term premium'}, ...
               'Location', 'best', 'FontSize', 12);
        grid on; box on;
        set(gca, 'FontSize', 12);
    end
    sgtitle('US Yield Decomposition', 'FontSize', 12, 'FontWeight', 'bold');
end

exportgraphics(gcf, 'YieldDecompositionPlot.pdf');


%% LOCAL FUNCTIONS

% Wrapper around the chosen optimizer
function output = local_estimationMLE(params0Values, setup, optim)


if optim.optimizer == 1 || optim.optimizer == 2
    InsigmaValues    = setup.InsigmaValues;
    sigma            = 0.001 * 5;
    opts.SigmaMax    = 0.005 * 2;
    opts.LBounds     = setup.lowerBoundsValues;
    opts.UBounds     = setup.upperBoundsValues;
    opts.MaxIter     = optim.MaxIter;
    opts.MaxFunEvals = optim.MaxFunEvals;
    opts.PopSize     = 3 * length(params0Values);
    opts.VerboseModulo = 10;
    opts.TolFun      = optim.TolFun;
    opts.TolX        = optim.TolX;
    opts.Plotting    = 'off';
    opts.Saving      = 'off';
    if optim.optimizer == 1
        [paramsOptValues, ~, ~, ~] = cmaes_dsgeDisplay_mp_v2( ...
            @local_likelihood, params0Values, sigma, InsigmaValues, opts, setup);
    else
        [paramsOptValues, ~, ~, ~] = cmaes_dsgeDisplay( ...
            @local_likelihood, params0Values, sigma, InsigmaValues, opts, setup);
    end

elseif optim.optimizer == 3
    options = optimset('Display','iter', ...
        'MaxIter',     optim.MaxIter, ...
        'MaxFunEvals', optim.MaxFunEvals, ...
        'TolX',        optim.TolX, ...
        'TolFun',      optim.TolFun);
    [paramsOptValues, ~, ~, ~] = fminsearch( ...
        @local_likelihood, params0Values, options, setup);

else  % optimizer == 0: accept starting values without searching
    paramsOptValues = params0Values;
end

% Evaluate likelihood one final time at the optimum
[avgLogL, logL, ~, outKF] = local_likelihood(paramsOptValues, setup); %#ok<ASGLU>

% Compute standard errors via outer-product-of-scores
paramsSEvalues = local_getMLEse(paramsOptValues, setup);

% Pack output
output.paramsOpt = local_values2struct(paramsOptValues, setup.selectParams);
output.paramsSE  = local_values2struct(paramsSEvalues,  setup.selectParams);
output.avgLogL   = avgLogL;
output.outKF     = outKF;
end

% Evaluates the (negative average) log-likelihood; mirrors likelihood.m
function [avgLogL, logL, errorMes, outKF] = local_likelihood(paramsInputValues, setup)

% Bounds check
if any(paramsInputValues < setup.lowerBoundsValues) || ...
   any(paramsInputValues > setup.upperBoundsValues)
    errorMes = 1;
    if setup.optimizer == 1 || setup.optimizer == 2
        avgLogL = nan;
        logL    = nan(size(setup.data,2), 1);
    else
        avgLogL = 1e35;
        logL    = 1e35 * ones(size(setup.data,2), 1);
    end
    outKF = NaN;
    return;
end

% Unpack parameter vector into struct
for i = 1:size(setup.selectParams, 1)
    name = setup.selectParams(i, 1);
    params.(name{1}) = paramsInputValues(i, 1);
end

% Merge calibrated parameters (do not overwrite estimated ones)
if ~isempty(setup.calibrateParams)
    namesCalibrate = setdiff(fieldnames(setup.calibrateParams), setup.selectParams);
    for i = 1:size(namesCalibrate, 1)
        name = namesCalibrate{i};
        params.(name) = setup.calibrateParams.(name);
    end
end

% Solve the ATSM (bond pricing recursion)
[model, errorMes] = local_solveATSM(params, setup.nx, setup.matSelect);
if errorMes == 1
    if setup.optimizer == 1 || setup.optimizer == 2
        avgLogL = nan;
        logL    = nan(size(setup.data,2), 1);
    else
        avgLogL = 1e35;
        logL    = 1e35 * ones(size(setup.data,2), 1);
    end
    outKF = NaN;
    return;
end

% Kalman filter
numObs = length(setup.matSelect);
Sv     = eye(numObs) * params.stdY;
outKF  = local_KalmanFilter(setup.data, model.g0, model.gx, ...
                             Sv.^2, model.muP, model.phiP, model.sigma*model.sigma');
outKF.model = model;

% Return negative average log-likelihood (minimisation convention)
logL    = outKF.logL;
avgLogL = -outKF.sumLogL / length(outKF.logL);
errorMes = 0;
end

% Solves the ATSM bond pricing recursion; mirrors solveATSM.m
function [model, errorMes] = local_solveATSM(params, nx, matSelect)

maxMat = max(matSelect);

% Unpack parameters
alpha = params.alpha;
beta  = zeros(nx, 1);
phiQ  = zeros(nx, nx);
muQ   = zeros(nx, 1);
sigma = zeros(nx, nx);
muP   = zeros(nx, 1);
phiP  = zeros(nx, nx);

for i = 1:nx
    beta(i,1) = params.(['beta', num2str(i)]);
    muQ(i,1)  = params.(['muQ',  num2str(i)]);
    muP(i,1)  = params.(['muP',  num2str(i)]);
end
for i = 1:nx
    for j = 1:nx
        phiQ(i,j) = params.(['phiQ', num2str(i), num2str(j)]);
        phiP(i,j) = params.(['phiP', num2str(i), num2str(j)]);
        if i <= j
            sigma(j,i) = params.(['sigma', num2str(j), num2str(i)]);
        end
    end
end

% Normalization: enforce ordering of diagonal Q eigenvalues
for i = 1:nx-1
    if phiQ(i,i) < phiQ(i+1,i+1)
        errorMes = 1; model = []; return;
    end
    % Jordan-form correction for near-identical eigenvalues
    phiQ(i,i+1) = (1 - abs(phiQ(i,i) - phiQ(i+1,i+1)))^1000;
end

% Stationarity check under P
eigVals = eig(phiP);
if any(sqrt(real(eigVals).^2 + imag(eigVals).^2) > 1)
    errorMes = 1; model = []; return;
end

% Short-rate intercept must be non-negative
if alpha < 0
    errorMes = 1; model = []; return;
end

% Innovation covariance must be positive definite
sigma2 = sigma * sigma';
if any(eig(sigma2) <= 0)
    errorMes = 1; model = []; return;
end

% Bond pricing recursion (A and B coefficients)
A = zeros(1, maxMat);
B = zeros(nx, maxMat);

for k = 1:maxMat
    if k == 1
        A(1,k) = -alpha;
        B(:,k) = -beta;
    else
        A(1,k) = -alpha + A(1,k-1) + B(:,k-1)'*muQ + 0.5*B(:,k-1)'*sigma2*B(:,k-1);
        B(:,k) = -beta  + phiQ'*B(:,k-1);
    end
end

% Convert to annualised yield loadings
g0 = 12 * (-A(1, matSelect)' ./ matSelect');
gx = 12 * (-B(:, matSelect)' ./ repmat(matSelect', 1, nx));
r0 = 12 * alpha;
rx = 12 * beta';

model = struct('g0',g0,'gx',gx,'muP',muP,'phiP',phiP, ...
               'alpha',alpha,'beta',beta,'phiQ',phiQ,'muQ',muQ, ...
               'sigma',sigma,'A',A,'B',B,'r0',r0,'rx',rx,'matSelect',matSelect);
errorMes = 0;
end

% Standard Kalman filter; mirrors KalmanFilter.m
% Measurement:  y_t     = g0 + gx*x_t     + Sv*v_t
% Transition:   x_{t+1} = h0 + hx*x_t + Seps*w_{t+1}
function out = local_KalmanFilter(y, g0, gx, Rv, h0, hx, Reps)

[ny, T] = size(y);
nx      = size(hx, 1);

% Unconditional mean and variance as starting values
x0   = (eye(nx) - hx) \ h0;
vecP = (eye(nx^2) - kron(hx,hx)) \ reshape(Reps, nx^2, 1);
P0   = reshape(vecP, nx, nx);

% Pre-allocate
xHat      = zeros(nx, T);
xBar      = zeros(nx, T);
PHat      = zeros(nx, nx, T);
SxHat     = zeros(nx, nx, T);
PBar      = zeros(nx, nx, T);
VaryBar   = zeros(ny, ny, T);
yBar      = zeros(ny, T);
logL      = zeros(T, 1);
K         = zeros(nx, ny, T);
predError = zeros(ny, T);

for t = 1:T
    % Prediction step
    if t == 1
        xBar(:,t)   = h0 + hx*x0;
        PBar(:,:,t) = hx*P0*hx' + Reps;
    else
        xBar(:,t)   = h0 + hx*xHat(:,t-1);
        PBar(:,:,t) = hx*PHat(:,:,t-1)*hx' + Reps;
    end
    yBar(:,t)      = g0 + gx*xBar(:,t);
    VaryBar(:,:,t) = gx*PBar(:,:,t)*gx' + Rv;

    % -Update step (handle missing observations)
    selectY    = ~isnan(y(:,t));
    ny_t       = sum(selectY);
    invVaryBar = VaryBar(selectY, selectY, t) \ eye(ny_t, ny_t);

    K(:, selectY, t)  = PBar(:,:,t) * gx(selectY,:)' * invVaryBar;
    predError(:,t)    = y(:,t) - yBar(:,t);
    xHat(:,t)         = xBar(:,t) + K(:,selectY,t) * predError(selectY,t);
    PHat(:,:,t)       = PBar(:,:,t) - K(:,selectY,t)*VaryBar(selectY,selectY,t)*K(:,selectY,t)';

    % Ensure PHat stays numerically positive definite
    if min(eig(PHat(:,:,t))) < 1e-15
        idxScale = 0;
        while idxScale < 10
            [tmpSx, check] = chol(PHat(:,:,t) + eye(nx)*10^(10-idxScale), 'lower');
            if check == 0
                SxHat(:,:,t) = tmpSx;
            end
            idxScale = idxScale + 1;
        end
    else
        [SxHat(:,:,t), ~] = chol(PHat(:,:,t), 'lower');
    end

    % Log-likelihood contribution
    logL(t,1) = -(ny_t/2)*log(2*pi) ...
                - 0.5*log(det(VaryBar(selectY,selectY,t))) ...
                - 0.5*predError(selectY,t)'*invVaryBar*predError(selectY,t);
end
sumLogL = sum(logL, 1);

out.data      = y;
out.xHat      = xHat;
out.PHat      = PHat;
out.Sx        = SxHat;
out.logL      = logL;
out.sumLogL   = sumLogL;
out.Kgain     = K;
out.predError = predError;
out.VaryBar   = VaryBar;
out.xBar      = xBar;
out.PBar      = PBar;
out.g0        = g0;
out.gx        = gx;
out.h0        = h0;
out.hx        = hx;
out.Rv        = Rv;
out.Reps      = Reps;
out.yBar      = yBar;
end

% SE
function se = local_getMLEse(paramsValues,setup)

numParams  = length(paramsValues);
[~,logL]   = local_likelihood(paramsValues,setup);
T          = length(logL);
score      = zeros(T,numParams);
for i=1:numParams
    % positive step
    paramsValues_peps = paramsValues;
    paramsValues_peps(i) = paramsValues_peps(i) + setup.epsValue;
    [~,logL_peps,errorMes_peps] = local_likelihood(paramsValues_peps,setup);
    if errorMes_peps == 1
        logL_peps = NaN(T,1);
    end
    
    % negative step
    paramsValues_meps = paramsValues;
    paramsValues_meps(i) = paramsValues_meps(i) - setup.epsValue;
    [~,logL_meps,errorMes_meps] = local_likelihood(paramsValues_meps,setup);
    if errorMes_meps
        logL_meps = NaN(T,1);
    end
    % The score function
    score(:,i) = (logL_peps-logL_meps)/(2*setup.epsValue);
end
varScore = zeros(numParams,numParams);
score = score - mean(score,1);
for t=1:T
    varScore = varScore + score(t,:)'*score(t,:);
end
invVarScore = varScore\eye(numParams);
se          = sqrt(diag(invVarScore));


end

% Parameter bounds and CMA-ES initial step sizes
function [upper, lower, Insigma] = local_paramsBoundsInsigma(nx)

upper.stdY  = 1;     lower.stdY  = 0;     Insigma.stdY  = 0.005;
upper.alpha = 100;   lower.alpha = -10;   Insigma.alpha = 0.10;

for i = 1:nx
    nm = ['beta', num2str(i)];
    upper.(nm) = 100;  lower.(nm) = 0;    Insigma.(nm) = 10;
    nm = ['muQ', num2str(i)];
    upper.(nm) = 10;   lower.(nm) = 0;    Insigma.(nm) = 10;
end

for i = 1:nx
    for j = 1:i
        nm = ['sigma', num2str(i), num2str(j)];
        upper.(nm) = 100;
        lower.(nm) = (i == j) * 1e-8 + (i ~= j) * (-100);
        Insigma.(nm) = 0.01;

        nm = ['phiQ', num2str(i), num2str(j)];
        upper.(nm) = (i == j) * 1    + (i ~= j) * 10;
        lower.(nm) = (i == j) * 1e-4 + (i ~= j) * (-10);
        Insigma.(nm) = 0.50;
    end
end

for i = 1:nx
    nm = ['muP', num2str(i)];
    upper.(nm) = 10;  lower.(nm) = -10;  Insigma.(nm) = 0.5;
end

for i = 1:nx
    for j = 1:nx
        nm = ['phiP', num2str(i), num2str(j)];
        upper.(nm) = 10;  lower.(nm) = -10;  Insigma.(nm) = 0.5;
    end
end

end

% Extract struct fields into a column vector in the order given by names.
function valuesArray = local_struc2values(structEx, names)
valuesArray = zeros(size(names,1), 1);
for i = 1:size(names,1)
    valuesArray(i,1) = structEx.(names{i});
end
end

% Pack a numeric vector back into a struct using the supplied field names.
function out = local_values2struct(values, names)
for i = 1:size(names,1)
    out.(names{i}) = values(i,1);
end
end

% Convert a scalar struct of doubles to a row vector.
function a = local_struct2array(s)
c = struct2cell(s);
a = [c{:}];
end

% Decomposes model-implied yields into expected short rates and term premia.
function res = local_yieldCurveDecom(outML)
%   res.yHat       : T x ny  fitted yields (annualised decimal)
%   res.rExp       : T x ny  average expected short rate (annualised decimal)
%   res.termPremia : T x ny  term premium = yHat - rExp
%   res.matSelect  : 1 x ny  maturities in months

model = outML.outKF.model;

xhat   = outML.outKF.xHat;          % nx x T
[nx,T] = size(xhat);
muP    = model.muP;
phiP   = model.phiP;
maxMat = max(model.matSelect);

%% Expected factors under P
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

%% Expected short rates under P
rExp = nan(maxMat, T);
for t = 1:T
    for i = 1:maxMat
        rExp(i,t) = model.r0 + model.rx * xExp(:,i,t);
    end
end

%% Fitted yields and average expected short rate
yHat    = outML.outKF.g0 + outML.outKF.gx * outML.outKF.xHat;  % ny x T
ny      = length(model.matSelect);
rExpAvg = nan(ny, T);
for t = 1:T
    for i = 1:ny
        rExpAvg(i,t) = mean(rExp(1:model.matSelect(i), t));
    end
end

%% Output  (T x ny convention, consistent with GATSM_UK plotting loops)
res.yHat       = yHat';
res.rExp       = rExpAvg';
res.termPremia = res.yHat - res.rExp;
res.matSelect  = model.matSelect;
end

% Newey-West OLS with HAC standard errors.
function res = local_nwest(y, X, lag)
% INPUT:
%   y   : T x 1 dependent variable
%   X   : T x k regressors (including constant)
%   lag : bandwidth (number of lags)

[T, k] = size(X);
beta   = (X'*X) \ (X'*y);
e      = y - X*beta;

% Newey-West covariance
S = (e .* X)' * (e .* X);          % lag-0 term
for l = 1 : lag
    w   = 1 - l / (lag + 1);       % Bartlett kernel
    Xl  = X(l+1:T, :);
    el  = e(l+1:T);
    X0  = X(1:T-l, :);
    e0  = e(1:T-l);
    Gl  = (e0 .* X0)' * (el .* Xl);
    S   = S + w * (Gl + Gl');
end

XpX_inv = (X'*X) \ eye(k);
V       = T * XpX_inv * S * XpX_inv;   % sandwich
se      = sqrt(diag(V) / T);

res.beta  = beta;
res.tstat = beta ./ se;
res.se    = se;
res.rsqr  = 1 - var(e) / var(y);
end