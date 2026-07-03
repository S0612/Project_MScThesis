% Computes FIML (Full-Information Maximum Likelihood) standard errors for
% the 4-Factor Macro-Finance GATSM point estimates from MCSE.
%
% NOTE: sigma_ll is identity by normalisation; sigma_ml = 0 by block
%       structure; muQ_l = 0 by identification.  These are not in theta.
%
% Files needed in dir
%   GATSM_4F_MacroFinance_Results_V23.mat
%   GATSM_4F_MacroFinance_Results_KO_proper_V5.mat
%   US_monthly_yields_Jan1972_Dec2025.csv
%   US_monthly_yields_Jan1972_Dec2025_maturities.csv
%   US_macro_inflation_unemployment_MF.csv

clear; clc;
fprintf('=============================================================\n');
fprintf(' FIML Standard Errors — 4-Factor MF-GATSM\n');
fprintf('=============================================================\n\n');

ann       = 4;
matSelect = [1 2 5 7 10 15] * ann;   % maturities in quarters
exactYears = [1 10];
nxM = 2; nxL = 2; nx = nxM + nxL;

%% Load data

fprintf('Loading data...\n');

yields_raw = readmatrix('US_monthly_yields_Jan1972_Dec2025.csv');
mats_raw   = readmatrix('US_monthly_yields_Jan1972_Dec2025_maturities.csv');
macro_raw  = readmatrix('US_macro_inflation_unemployment_MF.csv');

% Quarterly subsample (every 3rd month, end-of-quarter)
qi      = 3:3:size(yields_raw,1);
T       = length(qi);

% Yield data: select maturities and convert to decimal
data = zeros(length(matSelect), T);
for i = 1:length(matSelect)
    col = find(mats_raw == matSelect(i)/ann);
    data(i,:) = yields_raw(qi, col)' / 100;
end

% Macro data (already normalised)
xMacro = macro_raw(qi,:)';   % nxM x T

% Partition yield indices
idxExact = find(ismember(matSelect/ann, exactYears));
idxError = setdiff(1:length(matSelect), idxExact);
numObs   = length(matSelect);
Ne       = length(idxError);

fprintf('  T=%d quarters, %d maturities, %d exact, %d with error\n', ...
    T, numObs, length(idxExact), Ne);

%% Run for each model
%
models = {
    'GATSM_4F_MacroFinance_Results_V23.mat',          'Model A (V23)',  'Tables/FIML_SE_Model_A';
    'GATSM_4F_MacroFinance_Results_KO_proper_V5.mat', 'Model B (KO V5)','Tables/FIML_SE_Model_B_outdated';
};

for m = 1:size(models,1)
   
    res = load(models{m,1});

    % Pack MCSE point estimate into theta
    theta0 = pack_theta(res, nxM, nxL);
    nTheta = length(theta0);

    % Verify likelihood at theta0
    logL0 = kf_loglik(theta0, data, xMacro, matSelect, idxExact, idxError, ...
                      nxM, nxL, ann, T);

    % ---- Numerical Hessian via central finite differences ----
    fprintf('  Computing numerical Hessian (%d x %d)...\n', nTheta, nTheta);
    fprintf('  (This takes a few minutes — %d likelihood evaluations)\n', ...
            2*nTheta + 2*nTheta*(nTheta-1)/2 * 2);

    h_step = 1e-5;   % step size for finite differences
    H = numerical_hessian(@(t) kf_loglik(t, data, xMacro, matSelect, ...
                                          idxExact, idxError, nxM, nxL, ann, T), ...
                           theta0, h_step);

    fprintf('  Hessian computed.\n');

    % Covariance matrix and standard errors
    % Information matrix I = -H, H is Hessian of logL, so -H is positive definite
    % Asymptotic covariance: Cov(theta) = inv(I) = inv(-H)
    negH = -H;
    cond_negH = cond(negH);
    fprintf('  Condition number of -H: %.2e\n', cond_negH);

    if cond_negH > 1e12
        warning(['  -H is ill-conditioned. Some SEs may be unreliable.\n' ...
                 '  This typically affects near-unidentified parameters (muQ, muP_l).']);
    end

    try
        Cov = inv(negH);
        se  = sqrt(abs(diag(Cov)));   % abs() guards against tiny negative diag from numerical error
    catch
        warning('  Matrix inversion failed — using pinv.');
        Cov = pinv(negH);
        se  = sqrt(abs(diag(Cov)));
    end

    % Save results
    labels = theta_labels(nxM, nxL);
    out.theta0  = theta0;
    out.se      = se;
    out.H       = H;
    out.Cov     = Cov;
    out.logL0   = logL0;
    out.labels  = labels;
    out.res     = res;
    save([models{m,3} '.mat'], '-struct', 'out', '-v7.3');
    fprintf('  Saved %s.mat\n', models{m,3});

    % Print summary
    fprintf('\n  %-28s  %12s  %12s  %8s\n', 'Parameter', 'Estimate', 'FIML SE', 't-stat');
    fprintf('  %s\n', repmat('-',1,65));
    for k = 1:nTheta
        tstat = theta0(k) / se(k);
        fprintf('  %-28s  %12.6f  %12.6f  %8.3f\n', labels{k}, theta0(k), se(k), tstat);
    end

    % Write LaTeX table
    write_latex_table(res, theta0, se, labels, ann, nxM, nxL, ...
                      [models{m,3} '.tex'], models{m,2});
    fprintf('  LaTeX table written to %s.tex\n', models{m,3});
end

fprintf('\n\nDone.\n');

%% Local Functions

function theta = pack_theta(res, nxM, nxL)
% Pack all free GATSM parameters into a column vector.
% Every field from load() is forced to a column vector with (:) before
% concatenation to handle MATLAB's (1,1), (4,1) etc. shapes uniformly.
    phiQ_ll   = res.phiQ(nxM+1:end, nxM+1:end);
    phiP_ll   = res.phiP(nxM+1:end, nxM+1:end);
    Lambda_ll = phiP_ll - phiQ_ll;
    % Jordan form: Lambda_ll = [[a,b],[c,a]]
    a = Lambda_ll(1,1);
    b = Lambda_ll(1,2);
    c = Lambda_ll(2,1);

    phiP_mm  = res.phiP(1:nxM, 1:nxM);
    phiP_ll2 = res.phiP(nxM+1:end, nxM+1:end);
    phiQ_mm  = res.phiQ(1:nxM, 1:nxM);
    sigma_mm = res.sigma(1:nxM, 1:nxM);

    theta = [
        res.alpha(:);           % [1]       alpha
        phiP_mm(:);             % [2:5]     phiP_mm col-major: 11,21,12,22
        phiP_ll2(:);            % [6:9]     phiP_ll col-major: 11,21,12,22
        res.muP(1:nxM);         % [10:11]   muP_m
        res.muP(nxM+1:end);     % [12:13]   muP_l
        phiQ_mm(:);             % [14:17]   phiQ_mm col-major
        a(:); b(:); c(:);       % [18:20]   Jordan parameters
        res.beta(1:nxM);        % [21:22]   beta_m
        res.beta(nxM+1:end);    % [23:24]   beta_l
        res.muQ(1:nxM);         % [25:26]   muQ_m
        sigma_mm(1,1); sigma_mm(2,1); sigma_mm(2,2); % [27:29] Cholesky
        res.stdY(:);            % [30]      stdY
    ];

    theta = theta(:);   % guarantee column vector
end


function [phiP, phiQ, beta, alpha, muP, muQ, sigma, stdY] = unpack_theta(theta, nxM, nxL)
% Unpack parameter vector back into model matrices.
% theta is packed col-major (MATLAB default): phiP_mm(:) gives [11;21;12;22].
    theta = theta(:);   % ensure column
    nx = nxM + nxL;
    alpha    = theta(1);
    phiP_mm  = reshape(theta(2:5),   nxM, nxM);   % col-major: [11,21;12,22]
    phiP_ll  = reshape(theta(6:9),   nxL, nxL);
    muP_m    = theta(10:10+nxM-1);
    muP_l    = theta(12:12+nxL-1);
    phiQ_mm  = reshape(theta(14:17), nxM, nxM);
    a = theta(18); b = theta(19); c = theta(20);
    phiQ_ll  = phiP_ll - [a, b; c, a];
    beta_m   = theta(21:22);
    beta_l   = theta(23:24);
    muQ_m    = theta(25:26);
    s11 = theta(27); s21 = theta(28); s22 = theta(29);
    stdY     = theta(30);

    phiP  = blkdiag(phiP_mm, phiP_ll);
    phiQ  = blkdiag(phiQ_mm, phiQ_ll);
    beta  = [beta_m(:); beta_l(:)];
    muP   = [muP_m(:);  muP_l(:)];
    muQ   = [muQ_m(:);  zeros(nxL,1)];
    sigma = blkdiag([s11, 0; s21, s22], eye(nxL));
end


function logL = kf_loglik(theta, data, xMacro, matSelect, idxExact, idxError, ...
                           nxM, nxL, ann, T)
% Evaluate Kalman filter log-likelihood at parameter vector theta.
    try
        [phiP, phiQ, beta, alpha, muP, muQ, sigma, stdY] = ...
            unpack_theta(theta, nxM, nxL);

        % Guard: return -inf for non-finite or explosive parameters
        if ~all(isfinite(theta)), logL = -1e10; return; end
        if any(abs(eig(phiP)) >= 1.0), logL = -1e10; return; end

        % Compute bond pricing loadings
        sigma2  = sigma * sigma';
        nx      = nxM + nxL;
        nMax    = max(matSelect);
        A_bar   = -alpha;
        B_bar   = -beta;
        A_raw   = zeros(nMax, 1);
        B_raw   = zeros(nx, nMax);
        A_raw(1) = -A_bar;
        B_raw(:,1) = B_bar;
        for k = 2:nMax
            A_bar    = -alpha + A_bar + B_bar'*muQ + 0.5*(B_bar'*sigma2*B_bar);
            B_bar    = phiQ' * B_bar - beta;
            A_raw(k) = -A_bar / k;
            B_raw(:,k) = B_bar;
        end
        A_ann = ann * A_raw;
        B_ann = ann * B_raw;   % nx x nMax

        % g0, gx loadings (annualised)
        numObs = length(matSelect);
        g0 = zeros(numObs, 1);
        gx = zeros(numObs, nx);
        for i = 1:numObs
            n = matSelect(i);
            g0(i) = A_ann(n);
            gx(i,:) = (B_ann(:,n) / n)';   % already annualised by B_ann = ann*B_raw
        end

        % Measurement noise covariance
        Ne = length(idxError);
        Rv = zeros(numObs);
        Rv(idxError, idxError) = eye(Ne) * stdY^2;

        % Run Kalman filter
        out  = kalman_filter(data, g0, gx, Rv, muP, phiP, sigma2);
        logL = out.sumLogL;

        if ~isfinite(logL), logL = -1e10; end
    catch
        logL = -1e10;
    end
end


function out = kalman_filter(y, g0, gx, Rv, h0, hx, Reps)
% Standard Kalman filter (mirrors local_KalmanFilter from estimation script).
    [ny, T] = size(y);
    nx      = size(hx, 1);

    x0   = (eye(nx) - hx) \ h0;
    vecP = (eye(nx^2) - kron(hx,hx)) \ Reps(:);
    P0   = reshape(vecP, nx, nx);

    xHat  = zeros(nx, T);
    PHat  = zeros(nx, nx, T);
    PBar  = zeros(nx, nx, T);
    logL  = zeros(T, 1);

    for t = 1:T
        if t == 1
            xBar_t = h0 + hx*x0;
            PBar_t = hx*P0*hx' + Reps;
        else
            xBar_t = h0 + hx*xHat(:,t-1);
            PBar_t = hx*PHat(:,:,t-1)*hx' + Reps;
        end

        yBar_t    = g0 + gx*xBar_t;
        Vyy_t     = gx*PBar_t*gx' + Rv;

        sel       = ~isnan(y(:,t));
        ny_t      = sum(sel);
        Vyy_sel   = Vyy_t(sel,sel);

        % Symmetrise and regularise for numerical stability
        Vyy_sel = (Vyy_sel + Vyy_sel') / 2;
        [~,info] = chol(Vyy_sel);
        if info ~= 0
            Vyy_sel = Vyy_sel + eye(ny_t)*1e-12*trace(Vyy_sel);
        end

        invV      = Vyy_sel \ eye(ny_t);
        innov     = y(sel,t) - yBar_t(sel);

        K_t       = PBar_t * gx(sel,:)' * invV;
        xHat(:,t) = xBar_t + K_t * innov;
        PHat(:,:,t) = PBar_t - K_t * Vyy_sel * K_t';
        PHat(:,:,t) = (PHat(:,:,t) + PHat(:,:,t)') / 2;

        sign_det  = det(Vyy_sel);
        if sign_det <= 0, sign_det = abs(sign_det) + 1e-300; end
        logL(t) = -(ny_t/2)*log(2*pi) ...
                  - 0.5*log(sign_det) ...
                  - 0.5*innov'*invV*innov;
    end

    out.sumLogL = sum(logL);
    out.xHat    = xHat;
end


function H = numerical_hessian(f, theta, h)
% Hessian of scalar function f via central finite differences.
% H(i,j) = [f(t+hi+hj) - f(t+hi-hj) - f(t-hi+hj) + f(t-hi-hj)] / (4*h^2)
% Diagonal: H(i,i) = [f(t+2hi) - 2*f(t) + f(t-2hi)] / (4*h^2)
    n   = length(theta);
    H   = zeros(n, n);
    f0  = f(theta);

    % Progress reporting
    fprintf('  Hessian: ');
    pct_prev = 0;

    for i = 1:n
        ei = zeros(n,1); ei(i) = h;

        % Diagonal element
        H(i,i) = (f(theta + 2*ei) - 2*f0 + f(theta - 2*ei)) / (4*h^2);

        % Off-diagonal elements (upper triangle only, then symmetrise)
        for j = i+1:n
            ej = zeros(n,1); ej(j) = h;
            H(i,j) = (f(theta+ei+ej) - f(theta+ei-ej) ...
                     - f(theta-ei+ej) + f(theta-ei-ej)) / (4*h^2);
            H(j,i) = H(i,j);
        end

        pct = round(100*i/n);
        if pct >= pct_prev + 10
            fprintf('%d%%..', pct);
            pct_prev = pct;
        end
    end
    fprintf('done\n');
end


function lbl = theta_labels(nxM, nxL)
% Human-readable labels for each element of theta.
    lbl = {'\alpha', ...
           '\Phi^P_{mm,11}', '\Phi^P_{mm,21}', '\Phi^P_{mm,12}', '\Phi^P_{mm,22}', ...
           '\Phi^P_{ll,11}', '\Phi^P_{ll,21}', '\Phi^P_{ll,12}', '\Phi^P_{ll,22}', ...
           '\mu^P_1 (infl)', '\mu^P_2 (unemp)', '\mu^P_3 (lat1)', '\mu^P_4 (lat2)', ...
           '\Phi^Q_{mm,11}', '\Phi^Q_{mm,21}', '\Phi^Q_{mm,12}', '\Phi^Q_{mm,22}', ...
           'Jordan a', 'Jordan b', 'Jordan c', ...
           '\beta_1 (infl)', '\beta_2 (unemp)', '\beta_3 (lat1)', '\beta_4 (lat2)', ...
           '\mu^Q_1', '\mu^Q_2', ...
           '\Sigma_{11}', '\Sigma_{21}', '\Sigma_{22}', ...
           '\sigma_e'};
end


function write_latex_table(res, theta0, se, labels, ann, nxM, nxL, outfile, model_name)
% Write LaTeX table with MCSE estimates and FIML standard errors.
    fid = fopen(outfile, 'w');
    if fid < 0, error('Cannot open %s', outfile); end

    % Derived quantities
    eQ = sort(abs(eig(res.phiQ)), 'descend');
    phiQ_ll = res.phiQ(nxM+1:end, nxM+1:end);

    % SE lookup: theta indices
    %  1=alpha, 2-5=phiP_mm, 6-9=phiP_ll, 10-11=muP_m, 12-13=muP_l,
    % 14-17=phiQ_mm, 18-20=Jordan, 21-22=beta_m, 23-24=beta_l,
    % 25-26=muQ_m, 27-29=sigma, 30=stdY
    SE = @(k) se(k);

    % Q-eigenvalue SEs: propagate via delta method is complex;
    % report 'n/a' for eigenvalues (they are derived, not direct parameters)
    function write_row(label, val, se_val)
        fprintf(fid, '$%s$ & %.6f \\\\\n', label, val);
        if isnan(se_val) || se_val <= 0
            fprintf(fid, ' & \\\\\n');
        else
            fprintf(fid, ' & {\\footnotesize (%.6f)} \\\\\n', se_val);
        end
    end

    function write_sec(name)
        fprintf(fid, '\\multicolumn{2}{l}{\\textit{%s}} \\\\\n', name);
    end

    caption = sprintf(...
        'Estimated Parameters --- 4-Factor MF-GATSM, \\textbf{%s}', model_name);

    fprintf(fid, '%% Requires: \\usepackage{booktabs}\n');
    fprintf(fid, '\\begin{table}[htbp]\n\\centering\n');
    fprintf(fid, '\\caption{%s}\n', caption);
    fprintf(fid, '\\label{tab:params_%s}\n', strrep(model_name,' ','_'));
    fprintf(fid, '\\small\\setlength{\\tabcolsep}{8pt}\n');
    fprintf(fid, '\\begin{tabular}{lc}\n\\toprule\n');
    fprintf(fid, 'Parameter & MCSE \\\\\n\\midrule\n');

    write_sec('$\mathbb{Q}$-dynamics: eigenvalues');
    eQ_lbls = {'\lambda_1^{\mathbb{Q}}','\lambda_2^{\mathbb{Q}}', ...
               '\lambda_3^{\mathbb{Q}}','\lambda_4^{\mathbb{Q}}'};
    for i=1:4
        % Eigenvalues are functions of phiQ_mm (14-17) and Jordan (18-20)
        % Report without SE (derived quantity; delta method omitted)
        fprintf(fid, '$%s$ & %.6f \\\\\n', eQ_lbls{i}, eQ(i));
        fprintf(fid, ' & \\\\\n');
    end
    fprintf(fid, '\\addlinespace[3pt]\n');

    write_sec('Short-rate intercept');
    write_row('\alpha', theta0(1), SE(1));
    fprintf(fid, '\\addlinespace[3pt]\n');

    write_sec('$\mathbb{P}$-dynamics: mean ($\mu^{\mathbb{P}}$)');
    muP_lbls = {'\mu_1^{\mathbb{P}}\ \mathrm{(infl)}','\mu_2^{\mathbb{P}}\ \mathrm{(unemp)}', ...
                '\mu_3^{\mathbb{P}}\ \mathrm{(lat1)}','\mu_4^{\mathbb{P}}\ \mathrm{(lat2)}'};
    muP_idx  = [10,11,12,13];
    for i=1:4, write_row(muP_lbls{i}, theta0(muP_idx(i)), SE(muP_idx(i))); end
    fprintf(fid, '\\addlinespace[3pt]\n');

    write_sec('$\mathbb{P}$-dynamics: macro AR ($\Phi^{\mathbb{P}}_{mm}$)');
    mm_lbls = {'\Phi^{\mathbb{P}}_{mm,11}','\Phi^{\mathbb{P}}_{mm,21}', ...
               '\Phi^{\mathbb{P}}_{mm,12}','\Phi^{\mathbb{P}}_{mm,22}'};
    mm_idx  = [2,3,4,5];
    for i=1:4, write_row(mm_lbls{i}, theta0(mm_idx(i)), SE(mm_idx(i))); end
    fprintf(fid, '\\addlinespace[3pt]\n');

    write_sec('$\mathbb{P}$-dynamics: latent AR ($\Phi^{\mathbb{P}}_{ll}$)');
    ll_lbls = {'\Phi^{\mathbb{P}}_{ll,11}','\Phi^{\mathbb{P}}_{ll,21}', ...
               '\Phi^{\mathbb{P}}_{ll,12}','\Phi^{\mathbb{P}}_{ll,22}'};
    ll_idx  = [6,7,8,9];
    for i=1:4, write_row(ll_lbls{i}, theta0(ll_idx(i)), SE(ll_idx(i))); end
    fprintf(fid, '\\addlinespace[3pt]\n');

    write_sec('Volatility: macro Cholesky ($\Sigma_{mm}$)');
    sig_lbls = {'\Sigma_{11}','\Sigma_{21}','\Sigma_{22}'};
    sig_idx  = [27,28,29];
    for i=1:3, write_row(sig_lbls{i}, theta0(sig_idx(i)), SE(sig_idx(i))); end
    fprintf(fid, '\\addlinespace[3pt]\n');

    write_sec('$\mathbb{Q}$-dynamics: macro AR ($\Phi^{\mathbb{Q}}_{mm}$)');
    Qmm_lbls = {'\Phi^{\mathbb{Q}}_{mm,11}','\Phi^{\mathbb{Q}}_{mm,21}', ...
                '\Phi^{\mathbb{Q}}_{mm,12}','\Phi^{\mathbb{Q}}_{mm,22}'};
    Qmm_idx  = [14,15,16,17];
    for i=1:4, write_row(Qmm_lbls{i}, theta0(Qmm_idx(i)), SE(Qmm_idx(i))); end
    fprintf(fid, '\\addlinespace[3pt]\n');

    write_sec('$\mathbb{Q}$-dynamics: latent AR ($\Phi^{\mathbb{Q}}_{ll}$, Jordan)');
    Qll_lbls = {'\Phi^{\mathbb{Q}}_{ll,11}\ (=\Phi^{\mathbb{Q}}_{ll,22})', ...
                '\Phi^{\mathbb{Q}}_{ll,12}','\Phi^{\mathbb{Q}}_{ll,21}'};
    Qll_vals = [phiQ_ll(1,1); phiQ_ll(1,2); phiQ_ll(2,1)];
    Qll_idx  = [18,19,20];   % Jordan [a,b,c] -> phiQ_ll entries (sign-adjusted)
    for i=1:3, write_row(Qll_lbls{i}, Qll_vals(i), SE(Qll_idx(i))); end
    fprintf(fid, '\\addlinespace[3pt]\n');

    write_sec('Short-rate loadings ($\beta$)');
    b_lbls = {'\beta_1\ \mathrm{(infl)}','\beta_2\ \mathrm{(unemp)}', ...
              '\beta_3\ \mathrm{(lat1)}','\beta_4\ \mathrm{(lat2)}'};
    b_idx  = [21,22,23,24];
    for i=1:4, write_row(b_lbls{i}, theta0(b_idx(i)), SE(b_idx(i))); end
    fprintf(fid, '\\addlinespace[3pt]\n');

    write_sec('Risk-neutral intercept ($\mu^{\mathbb{Q}}$)');
    for i=1:2
        lbl = sprintf('\\mu_%d^{\\mathbb{Q}}', i);
        write_row(lbl, theta0(24+i), SE(24+i));
    end
    fprintf(fid, '\\addlinespace[3pt]\n');

    write_sec('Measurement error');
    fprintf(fid, '$\\sigma_e$ (bps) & %.4f \\\\\n', theta0(30)*ann*1e4);
    fprintf(fid, ' & {\\footnotesize (%.4f)} \\\\\n', SE(30)*ann*1e4);

    fprintf(fid, '\\midrule\n');
    fprintf(fid, '$\\frac{\\log\\mathcal{L}}{T}$ & %.6f \\\\\n', ...
            mean(res.outKF.logL));
    fprintf(fid, '\\bottomrule\n\\end{tabular}\n');
    fprintf(fid, '\\begin{flushleft}\n');
    fprintf(fid, ['{\\footnotesize \\textit{Notes:} Point estimates are from MCSE ' ...
                  '(Hamilton \\& Wu, 2012). Standard errors in parentheses are ' ...
                  'from the observed information matrix of the Kalman filter ' ...
                  'log-likelihood (FIML), evaluated numerically at the MCSE estimates ' ...
                  'via central finite differences ($h=10^{-5}$). ' ...
                  'Q-eigenvalues are derived from phiQ and have no direct standard error. ' ...
                  '$\\sigma_e$ reported in annualised basis points.}\n']);
    fprintf(fid, '\\end{flushleft}\n\\end{table}\n');
    fclose(fid);
end