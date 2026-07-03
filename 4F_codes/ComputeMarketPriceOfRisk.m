% Computes the market price of risk for the 1Q and NS estimates.
%
% files need in dir
%   GATSM_4F_MacroFinance_Results_V23.mat
%   GATSM_4F_MacroFinance_Results_KO_proper_V6.mat


clear; clc; close all;

%% Load saved model results

r23 = load('GATSM_4F_MacroFinance_Results_V23.mat');
r6  = load('GATSM_4F_MacroFinance_Results_KO_proper_V6.mat');

% Dimensions
nxM = 2;   % macro factors: 1 = inflation, 2 = unemployment
nxL = 2;   % latent factors
nx  = nxM + nxL;
ann = 4;   % quarterly -> annual

% Time axis (quarterly, 1972 Q1 onward)
T       = size(r23.outKF.xHat, 2);
dateVec = datetime(1972, 3, 1) + calmonths(3*(0:T-1));

fprintf('  T = %d quarters  (%s to %s)\n\n', T, ...
        datestr(dateVec(1),'mmm yyyy'), datestr(dateVec(end),'mmm yyyy'));

%% SECTION 2: Compute Lambda_t for each model

results = struct();

for mdl = 1:2

    if mdl == 1
        tag  = 'V23 (no surveys)';
        r    = r23;
    else
        tag  = 'V6 (survey-augmented)';
        r    = r6;
    end

    % Unpack — using identical variable names as in the estimation code
    sigma_mm = r.sigma(1:nxM, 1:nxM);          % lower-triangular Cholesky
    phiP_mm  = r.phiP(1:nxM, 1:nxM);
    phiQ_mm  = r.phiQ(1:nxM, 1:nxM);
    muP_m    = r.muP(1:nxM);
    muQ_m    = r.muQ(1:nxM);
    xHat     = r.outKF.xHat;                   % nx x T  filtered states

    
    % Step 1: Lambda_mm  (matches code line: phiQ_mm = phiP_mm - sigma_mm*Lambda_mm)
    
    Lambda_mm = sigma_mm \ (phiP_mm - phiQ_mm);   % nxM x nxM

    
    % Step 2: Lambda0_m  (constant part; analogous to muQ = muP - sigma*lambda0)
    
    Lambda0_m = sigma_mm \ (muP_m - muQ_m);        % nxM x 1

    
    % Step 3: Time-varying market price of risk  (nxM x T)
   
    x_macro  = xHat(1:nxM, :);                     % nxM x T
    Lambda_t = Lambda0_m + Lambda_mm * x_macro;     % nxM x T

    % Store
    results(mdl).tag       = tag;
    results(mdl).sigma_mm  = sigma_mm;
    results(mdl).phiP_mm   = phiP_mm;
    results(mdl).phiQ_mm   = phiQ_mm;
    results(mdl).muP_m     = muP_m;
    results(mdl).muQ_m     = muQ_m;
    results(mdl).Lambda_mm = Lambda_mm;
    results(mdl).Lambda0_m = Lambda0_m;
    results(mdl).Lambda_t  = Lambda_t;
    results(mdl).x_macro   = x_macro;

    
    % Console
    
    fprintf('========================================\n');
    fprintf('  %s\n', tag);
    fprintf('========================================\n');

    fprintf('\n  sigma_mm  (Cholesky volatility, macro block):\n');
    fprintf('    [ %+.6f   %+.6f ]\n', sigma_mm(1,1), sigma_mm(1,2));
    fprintf('    [ %+.6f   %+.6f ]\n', sigma_mm(2,1), sigma_mm(2,2));

    fprintf('\n  phiP_mm - phiQ_mm  (input to Lambda_mm):\n');
    D = phiP_mm - phiQ_mm;
    fprintf('    [ %+.6f   %+.6f ]\n', D(1,1), D(1,2));
    fprintf('    [ %+.6f   %+.6f ]\n', D(2,1), D(2,2));

    fprintf('\n  Lambda_mm = sigma_mm \\ (phiP_mm - phiQ_mm):\n');
    fprintf('  [row = which risk is priced; col = which factor drives it]\n');
    fprintf('             infl          unemp\n');
    fprintf('    infl   %+.6f    %+.6f\n', Lambda_mm(1,1), Lambda_mm(1,2));
    fprintf('    unemp  %+.6f    %+.6f\n', Lambda_mm(2,1), Lambda_mm(2,2));

    fprintf('\n  muP_m - muQ_m:\n');
    fprintf('    infl:  %+.6f\n', muP_m(1) - muQ_m(1));
    fprintf('    unemp: %+.6f\n', muP_m(2) - muQ_m(2));

    fprintf('\n  Lambda0_m = sigma_mm \\ (muP_m - muQ_m)  [constant part of Lambda_t]:\n');
    fprintf('    infl:  %+.6f\n', Lambda0_m(1));
    fprintf('    unemp: %+.6f\n', Lambda0_m(2));

    fprintf('\n  Time-series summary of Lambda_t (macro block):\n');
    fprintf('  %-14s  %8s  %8s  %8s  %8s\n', '', 'mean', 'std', 'min', 'max');
    lbl = {'Lambda_1t (infl)', 'Lambda_2t (unemp)'};
    for k = 1:nxM
        fprintf('  %-16s  %+7.4f  %7.4f  %+7.4f  %+7.4f\n', lbl{k}, ...
            mean(Lambda_t(k,:)), std(Lambda_t(k,:)), ...
            min(Lambda_t(k,:)), max(Lambda_t(k,:)));
    end

    fprintf('\n  Key sensitivities (Lambda_mm elements):\n');
    fprintf('    d(Lambda_1t)/d(pi_t)  [infl risk <- infl] : %+.6f\n', Lambda_mm(1,1));
    fprintf('    d(Lambda_1t)/d(u_t)   [infl risk <- unemp]: %+.6f\n', Lambda_mm(1,2));
    fprintf('    d(Lambda_2t)/d(pi_t)  [unemp risk <- infl]: %+.6f\n', Lambda_mm(2,1));
    fprintf('    d(Lambda_2t)/d(u_t)   [unemp risk <- unemp]: %+.6f\n', Lambda_mm(2,2));
    fprintf('\n');

end


%% H&W Table 7 reference values

fprintf('========================================\n');
fprintf('  Hamilton & Wu (2012) Table 7  —  Lambda_mm\n');
fprintf('  (AP2003 data: inflation + real activity)\n');
fprintf('========================================\n\n');

HW.Global = [ 2.8783,  0.4303; -6.1474, -0.8744];
HW.Local1 = [-0.3430,  0.1474;  1.7675, -0.0607];   % = AP2003 reported
HW.Local2 = [ 1.5633,  0.1341; 16.0624,  7.4290];

fields = {'Global','Local1','Local2'};
labels = {'Global minimum','Local1 (= AP2003 reported)','Local2'};
for f = 1:3
    L = HW.(fields{f});
    fprintf('  %s:\n', labels{f});
    fprintf('             infl        activity\n');
    fprintf('    infl   %+.4f    %+.4f\n', L(1,1), L(1,2));
    fprintf('    activ  %+.4f    %+.4f\n\n', L(2,1), L(2,2));
end


%% Summary comparison table

fprintf('=================================================================\n');
fprintf('  FULL COMPARISON: Lambda_mm\n');
fprintf('  Lambda_mm(i,j) = sensitivity of price of shock-i risk to factor j\n');
fprintf('=================================================================\n');
fprintf('  %-26s  %+13s  %+13s  %+13s  %+13s\n', ...
    'Model', 'L(1,1)', 'L(1,2)', 'L(2,1)', 'L(2,2)');
fprintf('  %-26s  %13s  %13s  %13s  %13s\n', ...
    '', 'infl<-infl', 'infl<-unemp', 'unemp<-infl', 'unemp<-unemp');
fprintf('  %s\n', repmat('-', 1, 84));
for mdl = 1:2
    L = results(mdl).Lambda_mm;
    fprintf('  %-26s  %+13.4f  %+13.4f  %+13.4f  %+13.4f\n', ...
        results(mdl).tag, L(1,1), L(1,2), L(2,1), L(2,2));
end
fprintf('  %s\n', repmat('-', 1, 84));
for f = 1:3
    L = HW.(fields{f});
    fprintf('  %-26s  %+13.4f  %+13.4f  %+13.4f  %+13.4f\n', ...
        ['H&W ', labels{f}], L(1,1), L(1,2), L(2,1), L(2,2));
end
fprintf('\n');


%% Print LaTeX table — Lambda_mm for NS and 1Q

fprintf('\n');
fprintf('=================================================================\n');
fprintf('  LaTeX table: Lambda_mm and Lambda0_m  (V23 and V6)\n');
fprintf('=================================================================\n\n');

L_v23 = results(1).Lambda_mm;
L0_v23 = results(1).Lambda0_m;
L_v6  = results(2).Lambda_mm;
L0_v6 = results(2).Lambda0_m;

% Open output file
texFile = 'Tables/Lambda_mm_table.tex';
fid = fopen(texFile, 'w');
if fid == -1
    warning('Could not open %s for writing; printing to console instead.', texFile);
    fid = 1;   % fall back to stdout
end

fprintf(fid, '%% =========================================================\n');
fprintf(fid, '%% Lambda_mm and Lambda_0 — Market Price of Risk\n');
fprintf(fid, '%% Generated by MarketPriceOfRisk.m\n');
fprintf(fid, '%% =========================================================\n\n');

fprintf(fid, '\\begin{table}[htbp]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Implied macro market price of risk: $\\bm{\\Lambda}_{mm}$ and $\\bm{\\Lambda}_{0,m}$}\n');
fprintf(fid, '\\label{tab:lambda_mm}\n');
fprintf(fid, '\\setlength{\\tabcolsep}{8pt}\n');
fprintf(fid, '\\renewcommand{\\arraystretch}{1.10}\n');
fprintf(fid, '\\begin{tabular}{lcccc}\n');
fprintf(fid, '\\toprule\n');

% Panel header
fprintf(fid, ' & \\multicolumn{2}{c}{No-Survey (V23)} & \\multicolumn{2}{c}{Survey-Augmented (V6)} \\\\\n');
fprintf(fid, '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n');
fprintf(fid, ' & $\\Lambda_{1t}$ (infl.\\ risk) & $\\Lambda_{2t}$ (unemp.\\ risk) & $\\Lambda_{1t}$ (infl.\\ risk) & $\\Lambda_{2t}$ (unemp.\\ risk) \\\\\n');
fprintf(fid, '\\midrule\n');

% Panel A: Lambda_mm rows
fprintf(fid, '\\multicolumn{5}{l}{\\textit{Panel A: Time-varying slope $\\bm{\\Lambda}_{mm} = \\bm{\\sigma}_{mm}^{-1}(\\bm{\\Phi}^{\\mathbb{P}}_{mm} - \\bm{\\Phi}^{\\mathbb{Q}}_{mm})$}} \\\\\n');
fprintf(fid, '[3pt]\n');

fprintf(fid, '$\\leftarrow$ Inflation ($\\pi_t$) & $%+.4f$ & $%+.4f$ & $%+.4f$ & $%+.4f$ \\\\\n', ...
    L_v23(1,1), L_v23(2,1), L_v6(1,1), L_v6(2,1));
fprintf(fid, '$\\leftarrow$ Unemployment ($u_t$) & $%+.4f$ & $%+.4f$ & $%+.4f$ & $%+.4f$ \\\\\n', ...
    L_v23(1,2), L_v23(2,2), L_v6(1,2), L_v6(2,2));

% Panel B: Lambda0_m
fprintf(fid, '[6pt]\n');
fprintf(fid, '\\multicolumn{5}{l}{\\textit{Panel B: Constant term $\\bm{\\Lambda}_{0,m} = \\bm{\\sigma}_{mm}^{-1}(\\bm{\\mu}^{\\mathbb{P}}_m - \\bm{\\mu}^{\\mathbb{Q}}_m)$}} \\\\\n');
fprintf(fid, '[3pt]\n');

fprintf(fid, '$\\Lambda_{0,m}$ & $%+.4f$ & $%+.4f$ & $%+.4f$ & $%+.4f$ \\\\\n', ...
    L0_v23(1), L0_v23(2), L0_v6(1), L0_v6(2));

fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');

% Notes
fprintf(fid, '\\begin{flushleft}\n');
fprintf(fid, '{\\scriptsize \\textbf{Notes:} ');
fprintf(fid, '$\\bm{\\Lambda}_{mm}(i,j)$ is the sensitivity of the price of shock-$i$ risk to a unit increase in macro factor $j$, ');
fprintf(fid, 'implied by the estimated $\\mathbb{P}$- and $\\mathbb{Q}$-dynamics via ');
fprintf(fid, '$\\bm{\\Phi}^{\\mathbb{Q}}_{mm} = \\bm{\\Phi}^{\\mathbb{P}}_{mm} - \\bm{\\sigma}_{mm}\\bm{\\Lambda}_{mm}$. ');
fprintf(fid, 'The full time-varying macro market price of risk vector is ');
fprintf(fid, '$\\bm{\\Lambda}_{t,m} = \\bm{\\Lambda}_{0,m} + \\bm{\\Lambda}_{mm}\\,\\mathbf{x}^{macro}_t$, ');
fprintf(fid, 'where $\\mathbf{x}^{macro}_t$ contains standardised CPI inflation and the unemployment rate. ');
fprintf(fid, 'No-Survey refers to the baseline MCSE model (V23); Survey-Augmented incorporates ');
fprintf(fid, '1-quarter-ahead SPF forecasts following \\citet{kim2012term} (V6).}\n');
fprintf(fid, '\\end{flushleft}\n');
fprintf(fid, '\\end{table}\n');

if fid ~= 1
    fclose(fid);
    fprintf('LaTeX table written to: %s\n\n', texFile);
end

% echo to console
fprintf('--- LaTeX output (also in %s) ---\n\n', texFile);
fid2 = fopen(texFile, 'r');
if fid2 ~= -1
    while ~feof(fid2)
        line = fgetl(fid2);
        if ischar(line); fprintf('%s\n', line); end
    end
    fclose(fid2);
end