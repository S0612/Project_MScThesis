%  ChiSquareTest.m
%
%  Hamilton & Wu (2012) over-identification test.
%
%  Re-estimates the structural parameters by minimising the efficiently-
%  weighted (information-matrix) minimum-chi-square criterion in HW's native
%  reduced-form coordinates, then reports the statistic at that minimiser:
%
%       chi2 = T1 * m(theta)' * R_hat * m(theta)  ~  chi2(q),  q = 13
%
%  Sune Grøn Pedersen, 2026

clear; clc; close all;

%% Load
load('GATSM_4F_MacroFinance_Results_V23.mat');
xMacro    = xMacro';            % 216x2  (rows = time)
data      = data';              % 216x6
matSelect = matSelect(:);
beta      = beta(:);            % 4x1
muQ       = muQ(:);             % 4x1
alpha     = alpha(1);

nxM = 2; nxL = 2; nx = 4; ann = 4; T1 = T - 1;
exactQ = [4;40]; idxExact = ismember(matSelect,exactQ); idxError = ~idxExact;
errorQ = matSelect(idxError);

%% Unrestricted reduced-form OLS (data side, fixed)
Y1 = data(:,idxExact);  Y2 = data(:,idxError);

% Y1 regression: Y1_t = A1* + Phi*_11 Y1_{t-1} + Psi*_1m xMacro_t + u1
y1 = Y1(2:end,:);  X1 = [ones(T1,1), Y1(1:end-1,:), xMacro(2:end,:)];
param1 = X1 \ y1;                       % 5x2
phiP_star11 = param1(2:3,:)';           % 2x2
u1 = y1 - X1*param1;  Omega1 = (u1'*u1)/T1;

% Y2 regression: Y2_t = A2* + Phi*_2m xMacro_t + Phi*_21 Y1_t + u2
y2 = Y2(2:end,:);  X2 = [ones(T1,1), xMacro(2:end,:), Y1(2:end,:)];
param2 = X2 \ y2;                       % 5x4
u2 = y2 - X2*param2;  Omega2 = diag(diag((u2'*u2)/T1));   % diagonal (HW-style)

Q1 = (X1'*X1)/T1;  Q2 = (X2'*X2)/T1;

% Macro VAR (just-identified): phiP_mm, Sigma_mm
ym = xMacro(2:end,:);  Xm = [ones(T1,1), xMacro(1:end-1,:)];
paramM = Xm \ ym;  phiP_mm = paramM(2:3,:)';
um = ym - Xm*paramM;  Sigma_mm = chol((um'*um)/T1,'lower');

% Latent P-persistence via similarity transform (just-identified)
B1l_chol = chol(Omega1,'lower');
phiP_ll  = B1l_chol \ (phiP_star11 * B1l_chol);

%% Efficient weighting matrix R_hat (block-diagonal, fixed)
D2 = [1 0 0; 0 1 0; 0 1 0; 0 0 1];                          % duplication (n=2)
W1  = kron(inv(Omega1), Q1);                                % Pi1 block (10x10)
Wom = 0.5 * D2' * kron(inv(Omega1), inv(Omega1)) * D2;      % Omega1 block (3x3)
W2  = kron(inv(Omega2), Q2);                                % Pi2 block (20x20)
R   = blkdiag(W1, Wom, W2);
R   = (R + R')/2;                                           % symmetrise
Uchol = chol(R);                                            % R = Uchol'*Uchol

% Pack everything the moment function needs
DAT = struct('phiP_mm',phiP_mm,'phiP_ll',phiP_ll,'Sigma_mm',Sigma_mm, ...
             'param1',param1,'param2',param2,'Omega1',Omega1, ...
             'exactQ',exactQ,'errorQ',errorQ,'maxMat',max(matSelect),'ann',ann);

%% Starting values from stored MCSE estimates
lam_mm0 = phiP_mm - phiQ(1:2,1:2);
lam_ll0 = phiP_ll - phiQ(3:4,3:4);
theta0  = [lam_mm0(:); beta(1:2); ...
           lam_ll0(1,1); lam_ll0(1,2); lam_ll0(2,1); ...
           beta(3:4); alpha; muQ(1:2)];           % 14x1

%% Minimise the efficiently-weighted objective
resid = @(th) Uchol * momvec(th, DAT);            % sum(resid.^2) = m'*R*m
opts  = optimoptions('lsqnonlin','Display','off', ...
            'FunctionTolerance',1e-15,'StepTolerance',1e-15, ...
            'OptimalityTolerance',1e-15,'MaxFunctionEvaluations',2e5, ...
            'MaxIterations',1e4);

[thHat,~] = lsqnonlin(resid, theta0, [], [], opts);
chi2_min  = T1 * sum(resid(thHat).^2);

% Multi-start for robustness
rng(0); best = chi2_min; thBest = thHat;
for it = 1:40
    pert = theta0 .* (1 + 0.25*randn(size(theta0)));
    th   = lsqnonlin(resid, pert, [], [], opts);
    val  = T1 * sum(resid(th).^2);
    if val < best, best = val; thBest = th; end
end
chi2_min = best;

% For comparison: weighted statistic at the STORED estimates (invalid as a test)
chi2_stored = T1 * (momvec(theta0,DAT)' * R * momvec(theta0,DAT));

%% Report
q = 13;
fprintf('\n  Hamilton & Wu (2012) formal over-identification test\n');
fprintf('  T1                                = %d\n', T1);
fprintf('  Over-id restrictions  q           = %d\n', q);
fprintf('  chi^2 at stored estimates (invalid) = %.2f\n', chi2_stored);
fprintf('  chi^2 at efficient minimiser        = %.4f\n', chi2_min);
fprintf('  chi^2 crit (5%%)                     = %.4f\n', chi2inv(0.95,q));
fprintf('  chi^2 crit (1%%)                     = %.4f\n', chi2inv(0.99,q));
fprintf('  p-value                             = %.3e\n', 1 - chi2cdf(chi2_min,q));
if chi2_min > chi2inv(0.99,q)
    fprintf('  Result: REJECTED at 1%%\n\n');
elseif chi2_min > chi2inv(0.95,q)
    fprintf('  Result: rejected at 5%% but not 1%%\n\n');
else
    fprintf('  Result: NOT rejected at 5%%\n\n');
end

%% Local Functions
function m = momvec(theta, D)
    lam_mm = reshape(theta(1:4),2,2);
    beta_m = theta(5:6);
    a = theta(7); b = theta(8); c = theta(9);
    beta_l = theta(10:11);
    alpha  = theta(12);
    muQ_m  = theta(13:14);

    phiQ_mm = D.phiP_mm - lam_mm;
    phiQ_ll = D.phiP_ll - [a b; c a];
    [B1m,B1l,A1,B2m,B2l,A2] = buildLoadings(phiQ_mm,phiQ_ll, ...
                               beta_m(:),beta_l(:),alpha,muQ_m(:),D);
    Bi = inv(B1l);

    % model-implied reduced form
    Psi1m = B1m;                 % G^{1,m}
    Om1m  = B1l*B1l';            % G^{1,l} G^{1,l}'
    phi21 = B2l*Bi;              % Phi*_21
    phi2m = B2m - B2l*Bi*B1m;    % Phi*_2m
    A2m   = A2 - B2l*Bi*A1;      % A2*

    % moment blocks (Pi1 rows 1-3 are just-identified -> zero)
    M1 = zeros(5,2);  M1(4:5,:) = D.param1(4:5,:) - Psi1m';
    m1 = M1(:);
    MOm   = D.Omega1 - Om1m;  mvech = [MOm(1,1); MOm(2,1); MOm(2,2)];
    M2 = D.param2 - [A2m'; phi2m'; phi21'];
    m2 = M2(:);
    m = [m1; mvech; m2];
end

function [B1m,B1l,A1,B2m,B2l,A2] = buildLoadings(phiQ_mm,phiQ_ll,beta_m,beta_l,alpha,muQ_m,D)
    phiQ = zeros(4); phiQ(1:2,1:2)=phiQ_mm; phiQ(3:4,3:4)=phiQ_ll;
    betaF = [beta_m; beta_l];  muQF = [muQ_m; 0; 0];
    sig = zeros(4); sig(1:2,1:2)=D.Sigma_mm; sig(3:4,3:4)=eye(2); sig2 = sig*sig';
    mx = D.maxMat;
    Bq = zeros(4,mx); Bq(:,1) = -betaF;
    Aq = zeros(1,mx); Aq(1) = -alpha;
    for k = 2:mx
        Bq(:,k) = -betaF + phiQ'*Bq(:,k-1);
        Aq(k)   = -alpha + Aq(k-1) + Bq(:,k-1)'*muQF + 0.5*Bq(:,k-1)'*sig2*Bq(:,k-1);
    end
    [B1m,B1l,A1] = pickLoad(D.exactQ, Bq, Aq, D.ann);
    [B2m,B2l,A2] = pickLoad(D.errorQ, Bq, Aq, D.ann);
end

function [Bm,Bl,Aa] = pickLoad(mats, Bq, Aq, ann)
    n = numel(mats); Bm = zeros(n,2); Bl = zeros(n,2); Aa = zeros(n,1);
    for i = 1:n
        k = mats(i); bb = ann*Bq(:,k)/k;
        Bm(i,:) = bb(1:2)'; Bl(i,:) = bb(3:4)'; Aa(i) = ann*Aq(k)/k;
    end
end