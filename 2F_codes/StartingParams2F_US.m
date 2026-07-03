% This script computes data-consistent starting values for the 2-factor GATSM from
% US yield data. The output is a params0Values vector ready to paste
% directly into GATSM_US_2factor.m.
%
%
% Data files needed in directory:
%   UK_yields_1987_2025.csv             (T x M matrix of yields in %)
%   UK_yields_1987_2025_maturities.csv  (1 x M row of maturities in years)

close all; clear; clc;

%% User Settings

matSelect_yrs = [1 2 5 7 10 15];   % selected maturities in years
nx            = 2;                  % number of latent factors

%% Load data and select maturities

fprintf('Step 1: Loading data...\n');

yields_raw = csvread('US_monthly_yields_Jan1972_Dec2025.csv');           % T x M, in percent
mats_years = csvread('US_monthly_yields_Jan1972_Dec2025_maturities.csv'); % 1 x M, in years
mats_months = mats_years * 12;
matSelect   = matSelect_yrs * 12;                           % in months

% Validate requested maturities exist
assert(all(ismember(matSelect, mats_months)), ...
    'One or more maturities in matSelect_yrs are not in the CSV.');

% Extract selected maturities: Y is numObs x T, in annualised decimal units
numObs = length(matSelect);
T      = size(yields_raw, 1);
Y      = zeros(numObs, T);
for i = 1:numObs
    col     = find(mats_months == matSelect(i));
    Y(i, :) = yields_raw(:, col)' / 100;
end

fprintf('  Loaded: T=%d months, %d maturities selected (%s years)\n', ...
    T, numObs, num2str(matSelect_yrs));

%% STEP 2: Compute phiQ diagonal
%   Run PCA on demeaned yield levels. Fit a VAR(1) to the first nx PC
%   scores. The moduli of the VAR(1) eigenvalues give empirical P-measure
%   persistence. Add 0.004 to proxy Q-measure (risk-neutral) persistence,
%   which is typically greater due to the market price of risk.
%   Enforce strict descending order (required by GATSM normalisation).

fprintf('Step 2: Computing phiQ diagonal...\n');

Yc       = Y - mean(Y, 2);                  % demean each maturity row
CovY     = Yc * Yc' / T;                   % numObs x numObs covariance
[V, D]   = eig(CovY);                      % V: eigenvectors, D: eigenvalues
[~, idx] = sort(diag(D), 'descend');
V        = V(:, idx);                       % sort descending by variance

F        = V(:, 1:nx)' * Yc;              % nx x T: first nx PC scores

% VAR(1) on PC scores: F_{t+1} = Phi * F_t + eps with OLS, with no intercept needed
% since scores are already demeaned)
Ft  = F(:, 1:end-1);                       % nx x (T-1): regressors
Ft1 = F(:, 2:end);                         % nx x (T-1): dependent variable
PhiP_PC = Ft1 * Ft' / (Ft * Ft');          % nx x nx: VAR(1) coefficient matrix

% Eigenvalue moduli of PhiP_PC give empirical persistence
eigVals_modulus = abs(eig(PhiP_PC));
eigVals_modulus = sort(eigVals_modulus, 'descend');

fprintf('VAR(1) eigenvalue moduli on PC scores: %s\n', ...
    num2str(eigVals_modulus', '%.6f  '));

% Add buffer and enforce strict descending order
phiQ_diag = eigVals_modulus + 0.004;
phiQ_diag = min(phiQ_diag, 0.9999);        % cap below 1

% Enforce strictly descending (gap of at least 1e-4)
for i = 1:nx-1
    if phiQ_diag(i) <= phiQ_diag(i+1)
        phiQ_diag(i+1) = phiQ_diag(i) - 1e-4;
    end
end

fprintf('  phiQ diagonal (after +0.004 buffer, ordered desc): %s\n', ...
    num2str(phiQ_diag', '%.8f  '));

%% Compute alpha

mean_short_rate = mean(Y(1, :));            % mean 1-year yield, annualised decimal
alpha           = mean_short_rate / 12;    % monthly units

fprintf('  Mean 1yr yield (annualised): %.6f\n', mean_short_rate);
fprintf('  alpha = mean_1yr / 12     : %.8f\n', alpha);


%% Bond pricing recursion — g0 and gx
%   Given phiQ, alpha, beta=ones, muQ=zeros and a negligible placeholder
%   sigma (convexity term is second-order at starting values), iterate the
%   ATSM recursion up to the maximum maturity. Then convert bond price
%   coefficients A, B into annualised yield intercept g0 and factor
%   loading matrix gx.

beta   = ones(nx, 1);
muQ    = zeros(nx, 1);
sigma0 = eye(nx) * 1e-6;                   % negligible placeholder
sigma2 = sigma0 * sigma0';

% Build phiQ matrix with Jordan-form correction on super-diagonal
phiQ = diag(phiQ_diag);
for i = 1:nx-1
    phiQ(i, i+1) = (1 - abs(phiQ_diag(i) - phiQ_diag(i+1)))^1000;
end

maxMat = max(matSelect);
A      = zeros(1, maxMat);
B      = zeros(nx, maxMat);

for k = 1:maxMat
    if k == 1
        A(1, k) = -alpha;
        B(:, k) = -beta;
    else
        A(1, k) = -alpha + A(1,k-1) + B(:,k-1)'*muQ ...
                  + 0.5 * B(:,k-1)' * sigma2 * B(:,k-1);
        B(:, k) = -beta + phiQ' * B(:,k-1);
    end
end

% Annualised yield loadings at selected maturities
g0 = 12 * (-A(1, matSelect)' ./ matSelect');          % numObs x 1
gx = 12 * (-B(:, matSelect)' ./ repmat(matSelect', 1, nx));  % numObs x nx


%% Compute phiP

phiP = diag(phiQ_diag);

fprintf('  phiP diagonal: %s\n', num2str(diag(phiP)', '%.8f  '));


%% STEP 6: Compute muP

data_mean    = mean(Y, 2);                          % numObs x 1
IminusPhiP   = eye(nx) - phiP;
invIminusPhiP = IminusPhiP \ eye(nx);
M            = gx * invIminusPhiP;                  % numObs x nx
rhs          = data_mean - g0;                      % numObs x 1
muP          = M \ rhs;                             % least-squares, nx x 1

% Report fit quality
x0_implied   = invIminusPhiP * muP;
y_implied    = g0 + gx * x0_implied;
fit_error_bps = (y_implied - data_mean) * 10000;

%% Compute sigma diagonal

dY           = diff(Y, 1, 2);                       % numObs x (T-1)
CovdY        = dY * dY' / (T-1);                   % numObs x numObs
[Vd, Dd]     = eig(CovdY);
[~, idxd]    = sort(diag(Dd), 'descend');
lambda_chg   = diag(Dd);
lambda_chg   = lambda_chg(idxd);                   % nx largest eigenvalues
Vd           = Vd(:, idxd);

varExplained_chg = lambda_chg(1:nx) / sum(lambda_chg) * 100;
fprintf('  Variance of yield changes explained by first %d PCs: %s %%\n', ...
    nx, num2str(varExplained_chg', '%.1f  '));

% Back-solve sigma_ii via the gx loading norms
sigma_diag = zeros(nx, 1);
for i = 1:nx
    gx_norm      = norm(gx(:, i));
    sigma_diag(i) = sqrt(lambda_chg(i)) / gx_norm;
    fprintf('  sigma%d%d = sqrt(%.6e) / %.4f = %.10f\n', ...
        i, i, lambda_chg(i), gx_norm, sigma_diag(i));
end

sigma = diag(sigma_diag); % lower-triangular (diagonal here)

%% Compute stdY

fprintf('Step 8: Computing stdY...\n');

Y_fitted  = mean(Y, 2) + V(:, 1:nx) * (V(:, 1:nx)' * Yc);  % numObs x T
resid     = Y - Y_fitted;
stdY      = sqrt(mean(resid(:).^2));

fprintf('  RMS residual after 2-factor level fit: %.8f (%.2f bps)\n', ...
    stdY, stdY * 10000);

%% Assemble and display params0Values

params0Values = [ ...
    phiQ_diag(1);   phiQ_diag(2);                     % phiQ11, phiQ22
    alpha;                                              % alpha
    muP(1);         muP(2);                             % muP1, muP2
    phiP(1,1);      phiP(1,2);                          % phiP11, phiP12
    phiP(2,1);      phiP(2,2);                          % phiP21, phiP22
    sigma(1,1);                                         % sigma11
    sigma(2,1);     sigma(2,2);                         % sigma21, sigma22
    stdY ];                                             % stdY

% Display the vector
paramNames = { ...
    'phiQ11','phiQ22','alpha', ...
    'muP1','muP2', ...
    'phiP11','phiP12', ...
    'phiP21','phiP22', ...
    'sigma11','sigma21','sigma22', ...
    'stdY'};

%% Copy this into GATSM_US_2factor.m from the command window.

fprintf('\nparams0Values = [ ...\n');
lineItems = 0;
for i = 1:length(params0Values)
    if lineItems == 0
        fprintf('    ');
    end
    fprintf('%16.10f', params0Values(i));
    if i < length(params0Values)
        fprintf('  ');
        lineItems = lineItems + 1;
        if lineItems == 4
            fprintf('...\n');
            lineItems = 0;
        end
    end
end
fprintf(']'';\n');

%% Local Functions
function [logd, ld] = logdet_chol(M)
% Compute log-determinant of a symmetric positive definite matrix via
% Cholesky. Returns NaN cleanly if M is not PD.
[L, p] = chol(M, 'lower');
if p > 0
    logd = NaN; ld = NaN; return;
end
ld   = 2 * sum(log(diag(L)));
logd = ld;
end