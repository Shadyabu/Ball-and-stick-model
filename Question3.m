% Code was written with the help of AI. All instructions of what to code,
% analysis and write up was done manually, without AI.

% PLEASE ADD data.mat INTO THE DIRECTORY BEFORE RUNNING THE CODE


% Load the diffusion signal
fid = fopen('isbi2015_data_normalised.txt', 'r', 'b');
fgetl(fid); % Read in the header
D = fscanf(fid, '%f', [6, inf])'; % Read in the data
fclose(fid);
% Select the first of the 6 voxels
meas = D(:,1);
% Load the protocol
fid = fopen('isbi2015_protocol.txt', 'r', 'b');
fgetl(fid);
A = fscanf(fid, '%f', [7, inf]);
fclose(fid);
% Create the protocol
grad_dirs = A(1:3,:);
G = A(4,:);
delta = A(5,:);
smalldel = A(6,:);
TE = A(7,:);
GAMMA = 2.675987E8;
bvals = ((GAMMA*smalldel.*G).^2).*(delta-smalldel/3);
% convert bvals units from s/m^2 to s/mm^2
bvals = bvals/10^6;

%% -------------------------------------------------------------------------
% Question 3.1
% -------------------------------------------------------------------------
fprintf('\n%s\n', 'QUESTION 3.1 --------------')

% --- Step 1: Compute qhat from the new protocol ---
% grad_dirs is 3 x 3612 (each column is a direction vector)
% We need to normalise each column to unit length
% b=0 measurements have zero vectors — we protect against dividing by zero

grad_norms = vecnorm(grad_dirs, 2, 1);      % 1x3612: length of each direction vector
safe_norms = grad_norms;
safe_norms(safe_norms == 0) = 1;            % replace zeros with 1 (avoids NaN)

qhat_isbi = grad_dirs ./ safe_norms;        % 3x3612: each row is a unit direction vector

% Signal for voxel 1
Avox_isbi = meas;                           % 3612x1 vector, already loaded
K = length(Avox_isbi);                      % = 3612
fprintf('Number of measurements K = %d\n', K);

% --- Step 2: Expected RESNORM ---
% If the model fits perfectly, remaining error is just noise.
% With Gaussian noise, each squared residual ≈ sigma^2 on average.
% So expected RESNORM ≈ K * sigma^2
sigma = 0.04;
expected_RESNORM = K * sigma^2;
fprintf('Expected RESNORM (noise only) = %.4f\n', expected_RESNORM);
% This should be approximately 3612 * 0.0016 = 5.78

% --- Step 3: Parameter encoding (same transformation as Q1.2) ---
% x(1) = sqrt(S0)      → S0    = x(1)^2         forces S0 > 0
% x(2) = sqrt(diff)    → diff  = x(2)^2         forces diff > 0
% x(3) = logit(f)      → f     = 1/(1+exp(-x3)) forces 0 < f < 1
% x(4) = theta         → unconstrained angle
% x(5) = phi           → unconstrained angle
%
% Starting point tuned for normalised data (S0≈1) and this b-value range:
%   sqrt(1.0) = 1.0       → S0 ≈ 1.0
%   sqrt(1e-3) ≈ 0.0316   → diff ≈ 1e-3 s/mm^2 (sensible for brain tissue)
%   log(1) = 0            → f = 0.5 (equal ball and stick to start)
%   0, 0                  → theta and phi start at zero

startx_isbi = [sqrt(1.0), sqrt(1e-3), 0, 0, 0];

% --- Step 4: Optimizer settings ---
h_isbi = optimset('Display',      'off',   ...
                  'MaxFunEvals',   20000,   ...
                  'Algorithm',     'quasi-newton', ...
                  'TolX',          1e-10,   ...
                  'TolFun',        1e-10);

isQ1_1_isbi = false;   % false = use constrained (transformed) parameter version

% --- Step 5: Multi-start fitting to find global minimum ---
% We run 100 times from randomly perturbed starting points.
% Each run may land in a different local minimum — we keep the best.
nRuns = 100;
results_isbi = zeros(nRuns, 1);
params_isbi  = zeros(nRuns, 5);

fprintf('Running %d random restarts...\n', nRuns);
rng(42)   % fix seed so results are reproducible
for i = 1:nRuns
    % Perturb each parameter in transformed space:
    %   S0 term: small perturbation around 1
    %   diff term: small perturbation around 0.03
    %   f term: larger perturbation to explore full 0-1 range
    %   angles: full random coverage of sphere
    perturbation = [randn*0.3,    ...   
                    randn*0.01,   ...   
                    randn*1.5,    ...   
                    randn*pi,     ...   
                    randn*2*pi];        

    startx_rand = startx_isbi + perturbation;

    try
        [ph, rn] = fminunc('BallStickSSD', startx_rand, h_isbi, ...
                           Avox_isbi, bvals, qhat_isbi, isQ1_1_isbi);
        results_isbi(i) = rn;
        params_isbi(i,:) = ph;
    catch
        results_isbi(i) = inf;   % mark failed runs as very bad
        params_isbi(i,:) = nan;
    end

    if mod(i, 10) == 0
        fprintf('  Completed %d/%d runs\n', i, nRuns);
    end
end

% --- Step 6: Identify the global minimum ---
[best_RESNORM_isbi, best_idx] = min(results_isbi);
best_params_raw = params_isbi(best_idx, :);

% Convert from transformed space back to real parameter values
S0_isbi    = best_params_raw(1)^2;
diff_isbi  = best_params_raw(2)^2;
f_isbi     = 1 / (1 + exp(-best_params_raw(3)));
theta_isbi = best_params_raw(4);
phi_isbi   = best_params_raw(5);

% --- Step 7: Frequency of finding the global minimum ---
% Count runs that got within 1% of the best RESNORM
tolerance       = best_RESNORM_isbi * 0.01;
n_found         = sum(results_isbi < best_RESNORM_isbi + tolerance);
freq_global_min = n_found / nRuns;

% --- Step 8: Report all results ---
fprintf('\n========== Q3.1 Results ==========\n')
fprintf('Best RESNORM:            %.6f\n', best_RESNORM_isbi)
fprintf('Expected RESNORM:        %.6f\n', expected_RESNORM)
fprintf('Ratio actual/expected:   %.3f\n',  best_RESNORM_isbi / expected_RESNORM)
fprintf('\nBest-fit parameters:\n')
fprintf('  S0:    %.6f\n',  S0_isbi)
fprintf('  diff:  %.4e  (units: s/mm^2)\n', diff_isbi)
fprintf('  f:     %.6f\n',  f_isbi)
fprintf('  theta: %.6f  (radians)\n', theta_isbi)
fprintf('  phi:   %.6f  (radians)\n', phi_isbi)
fprintf('\nRuns finding global min:  %d / %d  (%.0f%%)\n', ...
        n_found, nRuns, freq_global_min*100)

% --- Step 9: Plot the fit ---
fibdir_isbi     = [cos(phi_isbi)*sin(theta_isbi), ...
                   sin(phi_isbi)*sin(theta_isbi), ...
                   cos(theta_isbi)];
% fibdotgrad: dot product of each gradient direction with the fibre direction
fibdotgrad_isbi = sum(qhat_isbi .* repmat(fibdir_isbi', [1, K]), 1);

S_pred_isbi = S0_isbi * ( f_isbi     * exp(-bvals .* diff_isbi .* fibdotgrad_isbi.^2) ...
                        + (1-f_isbi) * exp(-bvals .* diff_isbi) );

figure;
plot(Avox_isbi, 'bs', 'MarkerSize', 5, 'LineWidth', 1);
hold on;
plot(S_pred_isbi, 'rx', 'MarkerSize', 5, 'LineWidth', 1);
legend('Data (ISBI voxel 1)', 'Ball-and-Stick fit');
xlabel('Measurement index k');
ylabel('Normalised signal S');
title('Q3.1: Ball-and-Stick fit to ISBI normalised data');

%% -------------------------------------------------------------------------
% Question 3.2
% -------------------------------------------------------------------------
fprintf('\n%s\n', 'QUESTION 3.2 --------------')

% Shared settings
h_q32 = optimset('Display',    'off', ...
                 'MaxFunEvals', 20000, ...
                 'Algorithm',  'quasi-newton', ...
                 'TolX',       1e-10, ...
                 'TolFun',     1e-10);
nRuns_q32 = 100;
rng(42)

%% --- MODEL 1: Diffusion Tensor (linear start → nonlinear fit) ---
fprintf('\n--- Model 1: Diffusion Tensor ---\n')

qx = qhat_isbi(1,:);
qy = qhat_isbi(2,:);
qz = qhat_isbi(3,:);

% Step A: Build linear design matrix and get starting point
% log(S) = log(S0) - b*(Dxx*qx^2 + 2Dxy*qx*qy + 2Dxz*qx*qz + Dyy*qy^2 + 2Dyz*qy*qz + Dzz*qz^2)
G_dt = [ones(K,1), ...
        (-bvals.*qx.^2)', ...
        (-2*bvals.*qx.*qy)', ...
        (-2*bvals.*qx.*qz)', ...
        (-bvals.*qy.^2)', ...
        (-2*bvals.*qy.*qz)', ...
        (-bvals.*qz.^2)'];

% Solve in log space (linear) — only valid measurements
valid = Avox_isbi > 0;
x_lin_dt = G_dt(valid,:) \ log(Avox_isbi(valid));

% Convert log-space S0 to real S0 for nonlinear starting point
startx_dt = x_lin_dt;
startx_dt(1) = exp(x_lin_dt(1));   % x(1) = S0 directly (not log)

% Step B: Nonlinear fit using DT_SSD — this is the correct RESNORM
[params_dt, RESNORM_dt] = fminunc('DT_SSD', startx_dt, h_q32, ...
                                   Avox_isbi, bvals, qhat_isbi);

% Extract parameters
S0_dt  = abs(params_dt(1));
D_mat  = [params_dt(2) params_dt(3) params_dt(4);
          params_dt(3) params_dt(5) params_dt(6);
          params_dt(4) params_dt(6) params_dt(7)];

[~, E_dt]  = eig(D_mat);
evals_dt   = sort(diag(E_dt), 'descend');

% Compute predictions for plotting
qdotDq_dt = sum(qhat_isbi .* (D_mat * qhat_isbi), 1);   % 1xK
S_pred_dt = S0_dt * exp(-bvals .* abs(qdotDq_dt));

fprintf('RESNORM: %.6f\n', RESNORM_dt)
fprintf('S0: %.6f\n', S0_dt)
fprintf('Eigenvalues: %.4e  %.4e  %.4e\n', evals_dt(1), evals_dt(2), evals_dt(3))

% --- MODEL 2: Ball and Stick (already done in Q3.1 - just report) ---
fprintf('\n--- Model 2: Ball and Stick (from Q3.1) ---\n')
fprintf('RESNORM: %.6f\n', best_RESNORM_isbi)
fprintf('S0=%.4f  d=%.4e  f=%.4f  theta=%.4f  phi=%.4f\n', ...
        S0_isbi, diff_isbi, f_isbi, theta_isbi, phi_isbi)

% --- MODEL 3: Zeppelin and Stick ---
fprintf('\n--- Model 3: Zeppelin and Stick ---\n')

% Starting point:
%   Use ball&stick result as base - d and f and angles are the same
%   lambda2 starts at half of d (lambda2/d = 0.5, so logit = 0)
startx_zs = [sqrt(S0_isbi), ...           % sqrt(S0)
             sqrt(diff_isbi), ...         % sqrt(d)
             0, ...                       % logit(lambda2/d) = 0 -> lambda2=d/2
             best_params_raw(3), ...      % logit(f) - reuse from Q3.1
             theta_isbi, ...              % theta
             phi_isbi];                   % phi

results_zs = zeros(nRuns_q32, 1);
params_zs  = zeros(nRuns_q32, 6);

fprintf('Running %d restarts...\n', nRuns_q32)
for i = 1:nRuns_q32
    perturb = [randn*0.3, randn*0.01, randn*1.0, ...
               randn*1.5, randn*pi,   randn*2*pi];
    sx = startx_zs + perturb;
    try
        [ph, rn] = fminunc('ZeppelinStickSSD', sx, h_q32, ...
                           Avox_isbi, bvals, qhat_isbi);
        results_zs(i) = rn;
        params_zs(i,:) = ph;
    catch
        results_zs(i) = inf;
    end
end

[best_RN_zs, idx_zs]  = min(results_zs);
best_p_zs             = params_zs(idx_zs,:);

% Convert back to real values
S0_zs      = best_p_zs(1)^2;
d_zs       = best_p_zs(2)^2;
lambda2_zs = d_zs / (1 + exp(-best_p_zs(3)));
f_zs       = 1 / (1 + exp(-best_p_zs(4)));
theta_zs   = best_p_zs(5);
phi_zs     = best_p_zs(6);

tol_zs   = best_RN_zs * 0.01;
freq_zs  = sum(results_zs < best_RN_zs + tol_zs) / nRuns_q32;

fprintf('RESNORM: %.6f\n', best_RN_zs)
fprintf('S0=%.4f  d=%.4e  lambda2=%.4e  f=%.4f  theta=%.4f  phi=%.4f\n', ...
        S0_zs, d_zs, lambda2_zs, f_zs, theta_zs, phi_zs)
fprintf('Frequency finding global min: %.0f%%\n', freq_zs*100)

% --- MODEL 4: Zeppelin and Stick with Tortuosity ---
fprintf('\n--- Model 4: Zeppelin + Stick + Tortuosity ---\n')

% Same 5 params as ball&stick - reuse that starting point
startx_zst = [sqrt(S0_isbi), sqrt(diff_isbi), best_params_raw(3), ...
              theta_isbi, phi_isbi];

results_zst = zeros(nRuns_q32, 1);
params_zst  = zeros(nRuns_q32, 5);

fprintf('Running %d restarts...\n', nRuns_q32)
for i = 1:nRuns_q32
    perturb = [randn*0.3, randn*0.01, randn*1.5, randn*pi, randn*2*pi];
    sx = startx_zst + perturb;
    try
        [ph, rn] = fminunc('ZeppelinStickTortSSD', sx, h_q32, ...
                           Avox_isbi, bvals, qhat_isbi);
        results_zst(i) = rn;
        params_zst(i,:) = ph;
    catch
        results_zst(i) = inf;
    end
end

[best_RN_zst, idx_zst] = min(results_zst);
best_p_zst             = params_zst(idx_zst,:);

S0_zst      = best_p_zst(1)^2;
d_zst       = best_p_zst(2)^2;
f_zst       = 1 / (1 + exp(-best_p_zst(3)));
lambda2_zst = (1 - f_zst) * d_zst;   % tortuosity - derived, not fitted
theta_zst   = best_p_zst(4);
phi_zst     = best_p_zst(5);

tol_zst  = best_RN_zst * 0.01;
freq_zst = sum(results_zst < best_RN_zst + tol_zst) / nRuns_q32;

fprintf('RESNORM: %.6f\n', best_RN_zst)
fprintf('S0=%.4f  d=%.4e  f=%.4f  lambda2=(1-f)*d=%.4e  theta=%.4f  phi=%.4f\n', ...
        S0_zst, d_zst, f_zst, lambda2_zst, theta_zst, phi_zst)
fprintf('Frequency finding global min: %.0f%%\n', freq_zst*100)

% --- PLOTS for all 4 models ---
figure('Position', [100 100 1200 900]);

% Precompute predictions for each model
bvals = bvals(:)';
fibdir_zs     = [cos(phi_zs)*sin(theta_zs); sin(phi_zs)*sin(theta_zs); cos(theta_zs)];
fibdotgrad_zs = sum(qhat_isbi .* repmat(fibdir_zs,[1,K]),1);
S_pred_zs     = S0_zs * (f_zs * exp(-bvals*d_zs.*fibdotgrad_zs.^2) + ...
                (1-f_zs)*exp(-bvals.*(lambda2_zs+(d_zs-lambda2_zs).*fibdotgrad_zs.^2)));

fibdir_zst     = [cos(phi_zst)*sin(theta_zst); sin(phi_zst)*sin(theta_zst); cos(theta_zst)];
fibdotgrad_zst = sum(qhat_isbi .* repmat(fibdir_zst,[1,K]),1);
S_pred_zst     = S0_zst * (f_zst * exp(-bvals*d_zst.*fibdotgrad_zst.^2) + ...
                 (1-f_zst)*exp(-bvals.*(lambda2_zst+(d_zst-lambda2_zst).*fibdotgrad_zst.^2)));

fibdir_bs      = [cos(phi_isbi)*sin(theta_isbi); sin(phi_isbi)*sin(theta_isbi); cos(theta_isbi)];
fibdotgrad_bs  = sum(qhat_isbi .* repmat(fibdir_bs,[1,K]),1);
S_pred_bs      = S0_isbi*(f_isbi*exp(-bvals*diff_isbi.*fibdotgrad_bs.^2) + ...
                 (1-f_isbi)*exp(-bvals*diff_isbi));

models     = {'Diffusion Tensor', 'Ball & Stick', 'Zeppelin & Stick', 'Zeppelin+Stick+Tort'};
preds      = {S_pred_dt, S_pred_bs, S_pred_zs, S_pred_zst};
resnorms   = {RESNORM_dt, best_RESNORM_isbi, best_RN_zs, best_RN_zst};

for m = 1:4
    subplot(2,2,m)
    plot(Avox_isbi, 'bs', 'MarkerSize', 3);
    hold on
    plot(preds{m}, 'rx', 'MarkerSize', 3);
    title(sprintf('%s\nRESNORM=%.4f', models{m}, resnorms{m}))
    xlabel('Measurement index k')
    ylabel('Signal S')
    legend('Data','Model','Location','northeast')
end
sgtitle('Q3.2: Model comparison on ISBI voxel 1')

% --- Summary table ---
fprintf('\n========== Q3.2 Summary ==========\n')
fprintf('Expected RESNORM (noise): %.4f\n', expected_RESNORM)
fprintf('%-35s  RESNORM\n', 'Model')
fprintf('%-35s  %.4f\n', 'Diffusion Tensor',               RESNORM_dt)
fprintf('%-35s  %.4f\n', 'Ball & Stick',                   best_RESNORM_isbi)
fprintf('%-35s  %.4f\n', 'Zeppelin & Stick',               best_RN_zs)
fprintf('%-35s  %.4f\n', 'Zeppelin & Stick + Tortuosity',  best_RN_zst)

%% -------------------------------------------------------------------------
% Question 3.3 — AIC and BIC model comparison
% -------------------------------------------------------------------------
fprintf('\n%s\n', 'QUESTION 3.3 --------------')

% Known noise standard deviation
sigma = 0.04;
K     = length(Avox_isbi);   % number of data points = 3612

% --- Step 1: Compute log-likelihood for each model ---
% For Gaussian noise:
%   ln(L) = -K/2 * ln(2*pi*sigma^2)  -  RESNORM / (2*sigma^2)
% The first term is constant across all models, so it cancels in comparisons.
% We include it for completeness so AIC/BIC values are on the correct scale.

const_term = -K/2 * log(2*pi*sigma^2);

logL_dt  = const_term - RESNORM_dt        / (2*sigma^2);
logL_bs  = const_term - best_RESNORM_isbi / (2*sigma^2);
logL_zs  = const_term - best_RN_zs        / (2*sigma^2);
logL_zst = const_term - best_RN_zst       / (2*sigma^2);

% --- Step 2: Number of free parameters per model ---
k_dt  = 7;   % S0, Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
k_bs  = 5;   % S0, d, f, theta, phi
k_zs  = 6;   % S0, d, lambda2, f, theta, phi
k_zst = 5;   % S0, d, f, theta, phi  (lambda2 is derived, not free)

% --- Step 3: Compute AIC = 2k - 2*ln(L) ---
AIC_dt  = 2*k_dt  - 2*logL_dt;
AIC_bs  = 2*k_bs  - 2*logL_bs;
AIC_zs  = 2*k_zs  - 2*logL_zs;
AIC_zst = 2*k_zst - 2*logL_zst;

% --- Step 4: Compute BIC = ln(n)*k - 2*ln(L) ---
% n = K = number of measurements
BIC_dt  = log(K)*k_dt  - 2*logL_dt;
BIC_bs  = log(K)*k_bs  - 2*logL_bs;
BIC_zs  = log(K)*k_zs  - 2*logL_zs;
BIC_zst = log(K)*k_zst - 2*logL_zst;

% --- Step 5: Report results ---
fprintf('\n%-35s  %4s  %10s  %10s  %10s  %10s\n', ...
        'Model', 'k', 'RESNORM', 'logL', 'AIC', 'BIC')
fprintf('%-35s  %4d  %10.4f  %10.1f  %10.1f  %10.1f\n', ...
        'Diffusion Tensor',              k_dt,  RESNORM_dt,        logL_dt,  AIC_dt,  BIC_dt)
fprintf('%-35s  %4d  %10.4f  %10.1f  %10.1f  %10.1f\n', ...
        'Ball & Stick',                  k_bs,  best_RESNORM_isbi, logL_bs,  AIC_bs,  BIC_bs)
fprintf('%-35s  %4d  %10.4f  %10.1f  %10.1f  %10.1f\n', ...
        'Zeppelin & Stick',              k_zs,  best_RN_zs,        logL_zs,  AIC_zs,  BIC_zs)
fprintf('%-35s  %4d  %10.4f  %10.1f  %10.1f  %10.1f\n', ...
        'Zeppelin & Stick + Tortuosity', k_zst, best_RN_zst,       logL_zst, AIC_zst, BIC_zst)

% --- Step 6: Rank the models (1 = best) ---
[~, aic_rank] = sort([AIC_dt, AIC_bs, AIC_zs, AIC_zst]);
[~, bic_rank] = sort([BIC_dt, BIC_bs, BIC_zs, BIC_zst]);
model_names = {'DT', 'Ball&Stick', 'Zeppelin&Stick', 'Zeppelin+Tort'};

fprintf('\nAIC ranking (best to worst):\n')
for r = 1:4
    fprintf('  %d. %s\n', r, model_names{aic_rank(r)})
end
fprintf('\nBIC ranking (best to worst):\n')
for r = 1:4
    fprintf('  %d. %s\n', r, model_names{bic_rank(r)})
end

fprintf('\nln(K) = ln(%d) = %.4f  (BIC penalty per parameter)\n', K, log(K))
fprintf('AIC penalty per parameter = 2\n')
fprintf('-> BIC penalises complexity %.1fx more than AIC\n', log(K)/2)