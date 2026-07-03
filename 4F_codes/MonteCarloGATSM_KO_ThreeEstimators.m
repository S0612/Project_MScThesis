% =========================================================================
% Monte Carlo Simulations of NS, 1Q and 4Q estimators.
% Uses the 1Q as the DGP
% Internal names:
%   A  = standard MCSE (NS)
%   B1 = survey-augmented MCSE, h=1 (1Q)
%   B2 = survey-augmented MCSE, h=4 (4Q)
%
% We test over two differetn sample sizes: T=216 and T=100
%
%
% Files needed in directory.
%   GATSM_4F_MacroFinance_Results_KO_proper_V6.mat  (V6 DGP)
%   US_monthly_yields_Jan1972_Dec2025.csv
%   US_monthly_yields_Jan1972_Dec2025_maturities.csv
%   US_macro_inflation_unemployment_MF.csv
%   cmaes_dsgeDisplay.m

clear; clc;

%% User Settings

R          = 2000;
T_list     = [216, 100];
ann        = 4;
masterSeed = 42;
nxM = 2;  nxL = 2;  nx = nxM + nxL;

matSelect  = [1 2 5 7 10 15] * ann;
exactYears = [1 10];
idxExact   = find(ismember(matSelect/ann, exactYears));
idxError   = setdiff(1:length(matSelect), idxExact);
numObs     = length(matSelect);
Ne         = length(idxError);

% Survey calibration h=1 (dpgdp3/UNEMP3): s(t+1) ~ x_m(t)
svy1_mu  = [0.0003242844; -0.0034575621];  % nxM x 1
svy1_phi = [0.8812460015;  0.9954509693];  % nxM x 1
svy1_sig = [0.4749052732;  0.1161247741];  % nxM x 1

% Survey calibration h=4 (dpgdp6/UNEMP6): s(t+1) ~ x_m(t)
svy4_mu  = [-0.0009078313; -0.0019967736]; % nxM x 1
svy4_phi = [ 0.7879389772;  0.9578298412]; % nxM x 1
svy4_sig = [ 0.6183813795;  0.2937995231]; % nxM x 1


%% Load DGP

res        = load('GATSM_4F_MacroFinance_Results_KO_proper_V6.mat'); % V6 DGP
phiP_true  = res.phiP;
phiQ_true  = res.phiQ;
beta_true  = res.beta(:);
alpha_true = res.alpha;
muP_true   = res.muP(:);
muQ_true   = res.muQ(:);
sigma_true = res.sigma;
stdY_true  = res.stdY;
g0_true    = res.g0(:);
gx_true    = res.gx;

assert(all(abs(eig(phiP_true)) < 1), 'phiP not stationary');
fprintf('  max|eig(phiP)| = %.4f\n', max(abs(eig(phiP_true))));

% CMA-ES options
cmaOpts         = cmaes_dsgeDisplay('defaults');
cmaOpts.Quiet   = 1;
cmaOpts.TolFun  = 1e-10;
cmaOpts.TolX    = 1e-10;
cmaOpts.MaxIter = 3000;

% Pre-generate random seeds
rng(masterSeed);
seeds_factor = randi(2^31-1, R, 1);
seeds_survey = randi(2^31-1, R, 1);


%% Main loop over sample sizes

for iT = 1:length(T_list)

T_sim  = T_list(iT);
T_burn = 200;
fprintf('\n=== T = %d ===\n', T_sim);

% Pre-allocation matrices (structs not allowed as parfor output)
nPmm = nxM^2;  nPll = nxL^2;
A_phiP_mm  = NaN(R,nPmm);  B1_phiP_mm = NaN(R,nPmm);  B2_phiP_mm = NaN(R,nPmm);
A_phiP_ll  = NaN(R,nPll);  B1_phiP_ll = NaN(R,nPll);  B2_phiP_ll = NaN(R,nPll);
A_phiQ_mm  = NaN(R,nPmm);  B1_phiQ_mm = NaN(R,nPmm);  B2_phiQ_mm = NaN(R,nPmm);
A_jordan   = NaN(R,3);      B1_jordan  = NaN(R,3);     B2_jordan  = NaN(R,3);
A_beta_m   = NaN(R,nxM);   B1_beta_m  = NaN(R,nxM);   B2_beta_m  = NaN(R,nxM);
A_beta_l   = NaN(R,nxL);   B1_beta_l  = NaN(R,nxL);   B2_beta_l  = NaN(R,nxL);
A_alpha    = NaN(R,1);      B1_alpha   = NaN(R,1);     B2_alpha   = NaN(R,1);
A_muQ_m    = NaN(R,nxM);   B1_muQ_m   = NaN(R,nxM);   B2_muQ_m   = NaN(R,nxM);
A_muP      = NaN(R,nx);     B1_muP     = NaN(R,nx);    B2_muP     = NaN(R,nx);
A_sigma_mm = NaN(R,3);      B1_sigma_mm= NaN(R,3);     B2_sigma_mm= NaN(R,3);
A_stdY     = NaN(R,1);      B1_stdY    = NaN(R,1);     B2_stdY    = NaN(R,1);
A_conv     = false(R,1);    B1_conv    = false(R,1);   B2_conv    = false(R,1);

% Copy loop variables needed inside parfor
phiP_ = phiP_true;  sigma_ = sigma_true;  muP_ = muP_true;
g0_   = g0_true;    gx_    = gx_true;     stdY_= stdY_true;

parfor r = 1:R  % change to 'for' for no multiprocessing.

    % Simulate factors
    rng_f  = RandStream('mt19937ar','Seed',seeds_factor(r));
    x_path = zeros(nx, T_burn+T_sim);
    for t = 2:T_burn+T_sim
        x_path(:,t) = muP_ + phiP_*x_path(:,t-1) + sigma_*randn(rng_f,nx,1);
    end
    x_sim = x_path(:, T_burn+1:end);   % nx x T_sim

    % Yield panel
    noise = zeros(numObs, T_sim);
    noise(idxError,:) = stdY_ * randn(rng_f, Ne, T_sim);
    yields_sim = bsxfun(@plus, g0_, gx_*x_sim) + noise;

    % Survey forecasts (h=1 and h=4)
    rng_s   = RandStream('mt19937ar','Seed',seeds_survey(r));
    x_m     = x_sim(1:nxM,:);
    % h=1 surveys
    eta1    = diag(svy1_sig) * randn(rng_s, nxM, T_sim);
    svy1_sim = bsxfun(@plus, svy1_mu, bsxfun(@times, svy1_phi, x_m)) + eta1;
    % h=4 surveys (independent noise draw)
    eta4    = diag(svy4_sig) * randn(rng_s, nxM, T_sim);
    svy4_sim = bsxfun(@plus, svy4_mu, bsxfun(@times, svy4_phi, x_m)) + eta4;

    % Estimator A
    [cA, rA] = mc_run_mcse(yields_sim, x_m, [], matSelect, idxExact, idxError, ...
                            nxM, nxL, ann, cmaOpts, 1);
    A_conv(r) = cA;
    if cA
        A_phiP_mm(r,:) = rA(1:4);    A_phiP_ll(r,:) = rA(5:8);
        A_phiQ_mm(r,:) = rA(9:12);   A_jordan(r,:)  = rA(13:15);
        A_beta_m(r,:)  = rA(16:17);  A_beta_l(r,:)  = rA(18:19);
        A_alpha(r)     = rA(20);     A_muQ_m(r,:)   = rA(21:22);
        A_muP(r,:)     = rA(23:26);  A_sigma_mm(r,:)= rA(27:29);
        A_stdY(r)      = rA(30);
    end

    % Estimator B1
    [cB1, rB1] = mc_run_mcse(yields_sim, x_m, svy1_sim, matSelect, idxExact, idxError, ...
                             nxM, nxL, ann, cmaOpts, 1);
    B1_conv(r) = cB1;
    if cB1
        B1_phiP_mm(r,:) = rB1(1:4);    B1_phiP_ll(r,:) = rB1(5:8);
        B1_phiQ_mm(r,:) = rB1(9:12);   B1_jordan(r,:)  = rB1(13:15);
        B1_beta_m(r,:)  = rB1(16:17);  B1_beta_l(r,:)  = rB1(18:19);
        B1_alpha(r)     = rB1(20);     B1_muQ_m(r,:)   = rB1(21:22);
        B1_muP(r,:)     = rB1(23:26);  B1_sigma_mm(r,:)= rB1(27:29);
        B1_stdY(r)      = rB1(30);
    end

    % Estimator B2
    [cB2, rB2] = mc_run_mcse(yields_sim, x_m, svy4_sim, matSelect, idxExact, idxError, ...
                             nxM, nxL, ann, cmaOpts, 4);
    B2_conv(r) = cB2;
    if cB2
        B2_phiP_mm(r,:) = rB2(1:4);    B2_phiP_ll(r,:) = rB2(5:8);
        B2_phiQ_mm(r,:) = rB2(9:12);   B2_jordan(r,:)  = rB2(13:15);
        B2_beta_m(r,:)  = rB2(16:17);  B2_beta_l(r,:)  = rB2(18:19);
        B2_alpha(r)     = rB2(20);     B2_muQ_m(r,:)   = rB2(21:22);
        B2_muP(r,:)     = rB2(23:26);  B2_sigma_mm(r,:)= rB2(27:29);
        B2_stdY(r)      = rB2(30);
    end

end  % for r

% Prints how many replications converged
fprintf('T=%d after loop: convA=%d  convB1=%d  convB2=%d\n', T_sim, sum(A_conv), sum(B1_conv), sum(B2_conv));

% Pack into structs and save
A  = mc_pack(A_phiP_mm, A_phiP_ll, A_phiQ_mm, A_jordan, A_beta_m, A_beta_l,...
             A_alpha,  A_muQ_m,  A_muP,  A_sigma_mm,  A_stdY,  A_conv);
B1 = mc_pack(B1_phiP_mm,B1_phiP_ll,B1_phiQ_mm,B1_jordan,B1_beta_m,B1_beta_l,...
             B1_alpha, B1_muQ_m, B1_muP, B1_sigma_mm, B1_stdY, B1_conv);
B2 = mc_pack(B2_phiP_mm,B2_phiP_ll,B2_phiQ_mm,B2_jordan,B2_beta_m,B2_beta_l,...
             B2_alpha, B2_muQ_m, B2_muP, B2_sigma_mm, B2_stdY, B2_conv);

true_params = struct('phiP',phiP_true,'phiQ',phiQ_true,'beta',beta_true,...
    'alpha',alpha_true,'muP',muP_true,'muQ',muQ_true,'sigma',sigma_true,'stdY',stdY_true);

outfile = sprintf('MonteCarlo_KO_3EST_T%d.mat', T_sim);
save(outfile,'A','B1','B2','true_params','R','T_sim',...
     'svy1_mu','svy1_phi','svy1_sig','svy4_mu','svy4_phi','svy4_sig',...
     'masterSeed','-v7.3');
fprintf('T=%d: convA=%d/%d  convB1=%d/%d  convB2=%d/%d  -> %s\n', ...
    T_sim, sum(A_conv),R, sum(B1_conv),R, sum(B2_conv),R, outfile);

% Bias preview on phiP_mm diagonal
fprintf('  h=1 surveys (B1):\n');
mc_bias_preview(A_phiP_mm, B1_phiP_mm, A_alpha, B1_alpha, A_conv, B1_conv, ...
                phiP_true, alpha_true, T_sim);
fprintf('  h=4 surveys (B2):\n');
mc_bias_preview(A_phiP_mm, B2_phiP_mm, A_alpha, B2_alpha, A_conv, B2_conv, ...
                phiP_true, alpha_true, T_sim);

end  % T_list
fprintf('\nDone.\n');

%% LOCAL FUNCTIONS
function [conv, results] = mc_run_mcse(yields, xMacro, surveys, matSelect, ...
    idxExact, idxError, nxM, nxL, ann, cmaOpts, h_svy)
if nargin < 11, h_svy = 1; end
% Run MCSE estimation for one replication.
% Returns conv flag and results vector of length 30.
% results = [phiP_mm(4) | phiP_ll(4) | phiQ_mm(4) | jordan(3) |
%            beta_m(2) | beta_l(2) | alpha | muQ_m(2) | muP(4) |
%            sigma_mm(3) | stdY]  = 30 elements

conv    = false;
results = NaN(30,1);

try
    T  = size(yields,2);
    T1 = T-1;
    nx = nxM+nxL;
    w_svy = ~isempty(surveys);

    Y1 = yields(idxExact,:);
    Y2 = yields(idxError,:);
    Ne = size(Y2,1);

    mature.exact = matSelect(idxExact);
    mature.error = matSelect(idxError);

    % Block m OLS (with optional survey augmentation)
    ym = xMacro(:,2:end)';
    xm = [ones(T1,1), xMacro(:,1:end-1)'];
    if w_svy
        % Detect horizon from survey series name passed via size:
        % Both h=1 and h=4 surveys have T_sim columns.
        % h=1: ym_svy = surveys(:,1:end-1)'  -> T1 rows, same xm
        % h=4: ym_svy = surveys(:,1:T1-3)'   -> T1-3 rows, shifted xm
        % The caller passes h as the 11th argument
        if h_svy == 1
            ym_aug = [ym; surveys(:,1:end-1)'];
            xm_aug = [xm; xm];
        else  % h=4
            T1_svy = T1 - 3;
            ym_svy = surveys(:, 1:T1_svy)';
            xm_svy = [ones(T1_svy,1), xMacro(:, 4:T1)'];
            ym_aug = [ym; ym_svy];
            xm_aug = [xm; xm_svy];
        end
        paramM = xm_aug \ ym_aug;
    else
        paramM = xm \ ym;
    end
    Am_star     = paramM(1,:)';
    phiP_starmm = paramM(2:end,:)';
    umt         = ym - xm*paramM;
    Omega_starm = (umt'*umt)/T1;

    % Block 1 OLS
    y1     = Y1(:,2:end)';
    x1     = [ones(T1,1), Y1(:,1:end-1)', xMacro(:,2:end)'];
    param1 = x1\y1;
    phiP_star11  = param1(2:1+nxL,:)';
    psiP_star1m  = param1(2+nxL:end,:)';
    u1t          = y1 - x1*param1;
    Omega_star1  = (u1t'*u1t)/T1;

    % Block 2 OLS
    y2     = Y2(:,2:end)';
    x2     = [ones(T1,1), xMacro(:,2:end)', Y1(:,2:end)'];
    param2 = x2\y2;
    A2_star      = param2(1,:)';
    phiP_star2m  = param2(2:1+nxM,:)';
    phiP_star21  = param2(2+nxM:end,:)';
    u2t          = y2 - x2*param2;

    % Step 4: sigma_mm, phiP_mm
    sigma_mm = chol(Omega_starm,'lower');
    phiP_mm  = phiP_starmm;

    B1m_OLS  = psiP_star1m;
    B2m_OLS  = phiP_star2m + phiP_star21*B1m_OLS;
    B1B1_OLS = Omega_star1;
    B2B1_OLS = phiP_star21*Omega_star1;
    stdY     = mean(std(u2t,0,1));

    % Step 6: latent phiP via similarity
    B1l_chol = chol(Omega_star1,'lower');
    Madj     = B1l_chol \ (phiP_star11*B1l_chol);
    [Veig,Deig] = eig(Madj);
    [~,isrt]    = sort(abs(diag(Deig)),'descend');
    Veig   = Veig(:,isrt);
    B1l_ann= B1l_chol*Veig;
    phiP_ll= Veig\Madj*Veig;

    % Step 5A: macro block CMA-ES
    bm_sign  = sign(sum(B1m_OLS,1));
    bm_scale = max(abs(B1m_OLS),[],1)' / (ann*0.86);
    bm_scale = max(bm_scale,1e-5);
    x0A  = [zeros(nxM^2,1); bm_scale.*bm_sign(:)];
    s0A  = [0.3*ones(nxM^2,1); bm_scale];
    lbA  = [-5*ones(nxM^2,1); zeros(nxM,1)];
    ubA  = [ 5*ones(nxM^2,1); 10*bm_scale];
    cmaA = cmaOpts; cmaA.LBounds=lbA; cmaA.UBounds=ubA;
    cmaA.PopSize = 4+floor(3*log(nxM^2+nxM));
    objA = @(p) sum(mc_macroRes(p,phiP_mm,sigma_mm,B1m_OLS,B2m_OLS,mature,nxM,ann).^2);
    [xA,~] = cmaes_dsgeDisplay(objA,x0A,1,s0A,cmaA);
    Lambda_mm = reshape(xA(1:nxM^2),nxM,nxM);
    beta_m    = xA(nxM^2+1:end);
    phiQ_mm   = phiP_mm - sigma_mm*Lambda_mm;

    % Step 5B: latent block CMA-ES (two starts)
    bl_scale = diag(B1l_chol)/(ann*0.86);
    bl_scale = max(bl_scale,0.0001);
    lbB = [-5;-5;-5; 0.0001; 0.0001];
    ubB = [ 5; 5; 5; bl_scale(1)*5; bl_scale(2)*5];
    cmaB = cmaOpts; cmaB.LBounds=lbB; cmaB.UBounds=ubB;
    cmaB.PopSize = 4+floor(3*log(5));
    objB = @(p) sum(mc_latentRes(p,phiP_ll,B1B1_OLS,B2B1_OLS,mature,nxL,ann).^2);
    [xBs1,fBs1] = cmaes_dsgeDisplay(objB,[-0.0689;0.1351;0.1520;0.00226;0.00062],...
                                     1,[0.03;0.03;0.03;0.0005;0.00015],cmaB);
    [xBs2,fBs2] = cmaes_dsgeDisplay(objB,[0;0;0;bl_scale],...
                                     1,[0.3;0.1;0.1;bl_scale],cmaB);
    xB = xBs1; if fBs2 < fBs1, xB = xBs2; end
    lev=xB(1); lb_j=xB(2); lc_j=xB(3);
    if lb_j > lc_j, [lb_j,lc_j]=deal(lc_j,lb_j); xB([4,5])=xB([5,4]); end
    beta_l    = xB(4:5);
    phiQ_ll   = phiP_ll - [lev,lb_j; lc_j,lev];
    if any(abs(eig(phiQ_ll)) >= 0.9999), return; end

    % Step 7: alpha, muQ_macro
    phiQ_full  = blkdiag(phiQ_mm, phiQ_ll);
    beta_full  = [beta_m; beta_l];
    sigma_full = blkdiag(sigma_mm, eye(nxL));
    [~,~,B1l_ann_s,B2l_ann_s] = mc_computeB(phiQ_full,beta_full,mature,nxM,nxL,ann);
    mean_y1yr = mean(yields(idxExact(1),:));
    x07  = [mean_y1yr/ann; zeros(nxM,1)];
    lb7  = [-0.03; -3*ones(nxM,1)];
    ub7  = [ 0.03;  3*ones(nxM,1)];
    cma7 = cmaOpts; cma7.LBounds=lb7; cma7.UBounds=ub7;
    cma7.PopSize=4+floor(3*log(3)); cma7.TolFun=1e-10; cma7.TolX=1e-10;
    obj7 = @(x) sum(mc_alphaRes(x,mature,phiQ_full,beta_full,sigma_full,...
                                A2_star,B1l_ann_s,B2l_ann_s,nxM,nxL,ann).^2) ...
               + (1e6)^2 * (x(1) - mean_y1yr/ann)^2 ...
               + (1e4)^2 * sum(x(2:end).^2);
    [x7,~] = cmaes_dsgeDisplay(obj7,x07,1,[0.005;0.3*ones(nxM,1)],cma7);
    alpha_est = x7(1);
    muQ_m_est = x7(2:end);

    % Step 8: muP recovery
    muP_m_est = Am_star;
    muP_l_est = zeros(nxL,1);   % simplified (latent means near zero)
    muP_est   = [muP_m_est; muP_l_est];

    % Pack results vector (30 elements)
    results = [phiP_mm(:); phiP_ll(:); phiQ_mm(:); lev;lb_j;lc_j;
               beta_m; beta_l; alpha_est; muQ_m_est;
               muP_est; sigma_mm(1,1);sigma_mm(2,1);sigma_mm(2,2); stdY];
    conv = true;

catch ME
    % Print error for first 3 failures to diagnose
    if r <= 3
        fprintf('  REP %d FAILED: %s\n', r, ME.message);
        for k = 1:min(3,length(ME.stack))
            fprintf('    at %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
        end
    end
    % results remains NaN
end
end


function S = mc_pack(pmm,pll,pQmm,jor,bm,bl,al,mq,mp,sm,sy,cv)
    S.phiP_mm_hat=pmm; S.phiP_ll_hat=pll; S.phiQ_mm_hat=pQmm;
    S.jordan_hat=jor;  S.beta_m_hat=bm;   S.beta_l_hat=bl;
    S.alpha_hat=al;    S.muQ_m_hat=mq;    S.muP_hat=mp;
    S.sigma_mm_hat=sm; S.stdY_hat=sy;     S.converged=cv;
end


function mc_bias_preview(Apmm,Bpmm,Aal,Bal,Acv,Bcv,phiP_t,al_t,T)
    fprintf('  Bias preview (T=%d):  True  BiasA  BiasB\n',T);
    labs={'\Phi_mm,11','\Phi_mm,22'}; cols=[1,4]; tv=[phiP_t(1,1),phiP_t(2,2)];
    for k=1:2
        bA=mean(Apmm(Acv,cols(k)))-tv(k); bB=mean(Bpmm(Bcv,cols(k)))-tv(k);
        fprintf('    %-12s  %7.4f  %+7.4f  %+7.4f\n',labs{k},tv(k),bA,bB);
    end
    bA=mean(Aal(Acv))-al_t; bB=mean(Bal(Bcv))-al_t;
    fprintf('    %-12s  %7.4f  %+7.4f  %+7.4f\n','alpha',al_t,bA,bB);
end


function F = mc_macroRes(para,phiP_mm,sigma_mm,B1m_OLS,B2m_OLS,mature,nxM,ann)
    Lambda_mm = reshape(para(1:nxM^2),nxM,nxM);
    beta_m    = para(nxM^2+1:end);
    phiQ_mm   = phiP_mm - sigma_mm*Lambda_mm;
    mats = [mature.exact, mature.error];
    Bm   = zeros(length(mats),nxM);
    Qt   = phiQ_mm';
    for im=1:length(mats)
        n=mats(im); S=zeros(nxM,1); P=eye(nxM);
        for j=0:n-1; S=S+P*beta_m; P=P*Qt; end
        Bm(im,:)=(ann*S/n)';
    end
    if ~all(isfinite(Bm(:))), F=1e6*ones(numel(B1m_OLS)+numel(B2m_OLS),1); return; end
    nE=length(mature.exact);
    F=[(B1m_OLS-Bm(1:nE,:))*1e5; (B2m_OLS-Bm(nE+1:end,:))*1e5]; F=F(:);
end


function F = mc_latentRes(para,phiP_ll,B1B1_OLS,B2B1_OLS,mature,nxL,ann)
    lev=para(1); lb=para(2); lc=para(3); beta_l=para(4:end);
    Lambda_ll=[lev,lb;lc,lev]; phiQ_ll=phiP_ll-Lambda_ll;
    nEq=nxL*(nxL+1)/2+numel(B2B1_OLS);
    if any(abs(eig(phiQ_ll))>=0.9999), F=1e8*ones(nEq,1); return; end
    mats=[mature.exact,mature.error]; Bl=zeros(length(mats),nxL); Qt=phiQ_ll';
    for im=1:length(mats)
        n=mats(im); S=zeros(nxL,1); P=eye(nxL);
        for j=0:n-1; S=S+P*beta_l; P=P*Qt; end
        Bl(im,:)=(ann*S/n)';
    end
    if ~all(isfinite(Bl(:))), F=1e8*ones(nEq,1); return; end
    nE=length(mature.exact); B1l_s=Bl(1:nE,:); B2l_s=Bl(nE+1:end,:);
    M1=B1B1_OLS-B1l_s*B1l_s';
    r1=M1([1,2,4])*1e5;       % [M(1,1) M(2,1) M(2,2)] 3 unique elements
    M2=B2B1_OLS-B2l_s*B1l_s';
    r2=M2(:)*1e5;              % Ne*nxL elements
    F=[r1(:); r2(:)];
end


function F = mc_alphaRes(x,mature,phiQ,beta,sigma,A2_star,B1l_ann,B2l_ann,...
                          nxM,nxL,ann)
    nx=nxM+nxL; alpha=x(1); muQ=zeros(nx,1); muQ(1:nxM)=x(2:end);
    try
        sigma2=sigma*sigma'; nMax=max([mature.exact,mature.error]);
        A_raw=zeros(1,nMax); A_bar=-alpha; B_bar=-beta; A_raw(1)=-A_bar;
        for k=2:nMax
            A_bar=-alpha+A_bar+B_bar'*muQ+0.5*(B_bar'*sigma2*B_bar);
            B_bar=phiQ'*B_bar-beta; A_raw(k)=-A_bar/k;
        end
        A_ann=ann*A_raw;
        A1=A_ann(mature.exact)'; A2=A_ann(mature.error)';
        F = (A2_star - A2 + B2l_ann/B1l_ann*A1)*1e6;
    catch
        F=1e10*ones(length(mature.error),1);
    end
end


function [B1m,B2m,B1l,B2l] = mc_computeB(phiQ,beta,mature,nxM,nxL,ann)
    nx=nxM+nxL; mats=[mature.exact,mature.error];
    Ball=zeros(length(mats),nx); Qt=phiQ';
    for im=1:length(mats)
        n=mats(im); S=zeros(nx,1); pow=eye(nx);
        for j=0:n-1; S=S+pow*beta; pow=pow*Qt; end
        Ball(im,:)=(ann*S/n)';
    end
    nE=length(mature.exact); B1=Ball(1:nE,:); B2=Ball(nE+1:end,:);
    B1m=B1(:,1:nxM); B1l=B1(:,nxM+1:end);
    B2m=B2(:,1:nxM); B2l=B2(:,nxM+1:end);
end