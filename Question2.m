% Code was written with the help of AI. All instructions of what to code,
% analysis and write up was done manually, without AI.

% PLEASE ADD data.mat INTO THE DIRECTORY BEFORE RUNNING THE CODE

% =========================================================================
% Q2.1 - Classical Bootstrap (STANDALONE SCRIPT)
% Run this independently - does NOT require Q1 to have run first.
% =========================================================================

clear; clc;

% --- Load data (same as your main script) ---
load('data');
dwis  = double(dwis);
dwis  = permute(dwis, [4,1,2,3]);
qhat  = load('bvecs');
bvals = 1000 * sum(qhat .* qhat);

% =========================================================================
% SETTINGS — change these to tune speed vs accuracy
% =========================================================================
T         = 1000;  % bootstrap samples  (use 50 for a quick test run)
nRestarts = 10;    % random restarts per bootstrap fit

% =========================================================================
% VOXELS TO ANALYSE
% =========================================================================
voxel_list = {
    92,  65, 72, 'Q1.1 voxel (92,65,72)';
    85, 128, 72, 'Voxel (85,128,72)';
    43,  64, 82, 'Voxel (43,64,82)';
};
nVoxels = size(voxel_list, 1);

% =========================================================================
% OPTIMISER OPTIONS
% =========================================================================
h_boot = optimset('Display',    'off', ...
                  'MaxFunEvals', 20000, ...
                  'Algorithm',   'quasi-newton', ...
                  'TolX',        1e-10, ...
                  'TolFun',      1e-10);

% =========================================================================
% MAIN LOOP OVER VOXELS
% =========================================================================
rng(42);

for v = 1:nVoxels

    vx    = voxel_list{v,1};
    vy    = voxel_list{v,2};
    vz    = voxel_list{v,3};
    vname = voxel_list{v,4};

    Avox_orig = dwis(:, vx, vy, vz);
    K = length(Avox_orig);

    fprintf('\n--- %s ---\n', vname);
    fprintf('Running %d bootstrap samples, %d restarts each...\n', T, nRestarts);

    boot_S0 = zeros(T,1);
    boot_d  = zeros(T,1);
    boot_f  = zeros(T,1);

    for t = 1:T

        % --- Classical bootstrap: resample with replacement ---
        idx        = ceil(rand(1,K) * K);
        Avox_boot  = Avox_orig(idx);
        bvals_boot = bvals(idx);
        qhat_boot  = qhat(:, idx);   % qhat is 3xK

        % --- Fit with multiple restarts (Q1.2 transformation, isQ1_1=false) ---
        best_resnorm = inf;
        best_params  = zeros(1,5);

        for r = 1:nRestarts
            % Starting point in transformed space, with random perturbation
            startx_rand = [randn*3.5, randn*1e-3, randn, randn*pi, randn*2*pi];                       
            try
                [p_hat, resnorm] = fminunc('BallStickSSD', startx_rand, h_boot, ...
                                           Avox_boot, bvals_boot, qhat_boot, false);
                if resnorm < best_resnorm
                    best_resnorm = resnorm;
                    best_params  = p_hat;
                end
            catch
            end
        end

        % --- Back-transform to physical parameters ---

        boot_S0(t) = best_params(1)^2;
        boot_d(t)  = best_params(2)^2;
        boot_f(t)  = 1 / (1 + exp(-best_params(3)));

        if mod(t,100) == 0
            fprintf('  Sample %d / %d\n', t, T);
        end
    end

    % =====================================================================
    % COMPUTE AND REPORT RANGES
    % =====================================================================
    param_names   = {'S0',    'd',    'f'};
    param_samples = {boot_S0, boot_d, boot_f};

    fprintf('\nResults for %s:\n', vname);
    fprintf('%-4s  %-12s  %-26s  %-26s\n', ...
            'Par', 'Mean', '2-sigma [lo,  hi]', '95% range [lo,  hi]');

    sigma_ranges = zeros(3,2);
    pct_ranges   = zeros(3,2);

    for p = 1:3
        samp = param_samples{p};

        % 2-sigma range
        mu      = mean(samp);
        sigma   = std(samp);
        lo_2sig = mu - 2*sigma;
        hi_2sig = mu + 2*sigma;

        % 95% range (sort and index)
        sorted = sort(samp);
        lo_95  = sorted(max(1, round(0.025*T)));
        hi_95  = sorted(min(T, round(0.975*T)));

        sigma_ranges(p,:) = [lo_2sig, hi_2sig];
        pct_ranges(p,:)   = [lo_95,   hi_95];

        fprintf('%-4s  %-12.4e  [%-10.4e, %-10.4e]  [%-10.4e, %-10.4e]\n', ...
                param_names{p}, mu, lo_2sig, hi_2sig, lo_95, hi_95);
    end

    % =====================================================================
    % PLOT
    % =====================================================================
    figure('Name', ['Bootstrap - ' vname], 'NumberTitle', 'off');

    for p = 1:3
        samp = param_samples{p};
        subplot(1,3,p);
        histogram(samp, 40, 'Normalization', 'probability', ...
                  'FaceColor', [0.4 0.6 0.8], 'EdgeColor', 'white');
        hold on;
        xline(sigma_ranges(p,1), 'b--', 'LineWidth', 1.5, 'DisplayName', '2\sigma lo');
        xline(sigma_ranges(p,2), 'b--', 'LineWidth', 1.5, 'DisplayName', '2\sigma hi');
        xline(pct_ranges(p,1),   'g-',  'LineWidth', 1.5, 'DisplayName', '95% lo');
        xline(pct_ranges(p,2),   'g-',  'LineWidth', 1.5, 'DisplayName', '95% hi');
        xline(mean(samp),        'r-',  'LineWidth', 2,   'DisplayName', 'Mean');
        title(param_names{p});
        xlabel('Value'); ylabel('Probability');
        legend('Location','best');
        hold off;
    end
    sgtitle(['Classical Bootstrap - ' vname]);

end

fprintf('\nDone.\n');

%% =========================================================================
% Q2.2 - MCMC (Metropolis-Hastings) - STANDALONE SCRIPT
% =========================================================================
% Estimates the 2-sigma and 95% ranges for S0, d, f using MCMC, then
% compares with the bootstrap results from Q2.1.
%
% HOW MCMC WORKS (Metropolis-Hastings):
%   Start at best-fit parameters from Q1.
%   Each step: propose a small random perturbation (the "candidate").
%   Accept the candidate if it fits better, or with probability equal to
%   the likelihood ratio if it fits worse. This allows exploration.
%   After many steps, the visited positions map out the posterior p(x|data).
% =========================================================================
% --- Load data ---
load('data');
dwis  = double(dwis);
dwis  = permute(dwis, [4,1,2,3]);
qhat  = load('bvecs');
bvals = 1000 * sum(qhat .* qhat);

% =========================================================================
% MCMC SETTINGS
% =========================================================================
T_total  = 50000;  % total MCMC steps to run
burnin   = 5000;   % discard first this many samples (chain finding its way)
thin     = 5;      % keep every nth sample (reduces correlation)
                   % effective samples = (T_total - burnin) / thin

% Proposal standard deviations - these control step size for each parameter
% These are in TRANSFORMED space (sqrt(S0), sqrt(d), logit(f), theta, phi)
% Tune these to get acceptance rate of 20-50%
% If acceptance too high -> increase step sizes
% If acceptance too low  -> decrease step sizes
prop_std = [0.05,    ...  % sqrt(S0):   S0 is ~4000, sqrt(S0)~63, so 0.05 is small step
            0.001,   ...  % sqrt(d):    d~1e-3, sqrt(d)~0.032, so 0.001 is small step
            0.1,     ...  % logit(f):   logit is unbounded, 0.1 reasonable
            0.1,     ...  % theta
            0.1];         % phi

% =========================================================================
% VOXELS TO ANALYSE (same as Q2.1 for direct comparison)
% =========================================================================
voxel_list = {
    92,  65, 72, 'Q1.1 voxel (92,65,72)';
    85, 128, 72, 'Voxel (85,128,72)';
    43,  64, 82, 'Voxel (43,64,82)';
};
nVoxels = size(voxel_list, 1);

% Bootstrap results from Q2.1 for comparison (paste your Q2.1 results here)
% Format: [S0_lo, S0_hi; d_lo, d_hi; f_lo, f_hi] for 95% range
boot_95 = {
    [4.1477e+03, 4.3678e+03; 1.0908e-03, 1.2012e-03; 3.1597e-01, 4.0193e-01], ...  % voxel 1
    [3.9253e+03, 4.1236e+03; 1.3390e-03, 1.5055e-03; 4.7516e-01, 5.5413e-01], ...  % voxel 2
    [3.7927e+03, 3.9255e+03; 1.1739e-03, 1.3283e-03; 4.6989e-01, 5.6172e-01]       % voxel 3
};

% =========================================================================
% OPTIMISER to get starting point (best fit from Q1)
% =========================================================================
h_opt = optimset('Display', 'off', 'MaxFunEvals', 20000, ...
                 'Algorithm', 'quasi-newton', 'TolX', 1e-10, 'TolFun', 1e-10);

rng(42);

% =========================================================================
% MAIN LOOP OVER VOXELS
% =========================================================================
for v = 1:nVoxels

    vx    = voxel_list{v,1};
    vy    = voxel_list{v,2};
    vz    = voxel_list{v,3};
    vname = voxel_list{v,4};

    Avox = dwis(:, vx, vy, vz);
    K    = length(Avox);

    fprintf('\n=== %s ===\n', vname);

    % ------------------------------------------------------------------
    % STEP 1: Find best-fit parameters to use as starting point
    % ------------------------------------------------------------------
    fprintf('Finding best-fit starting point...\n');
    best_resnorm = inf;
    best_params  = zeros(1,5);
    startx_base  = [sqrt(3.5), sqrt(3e-3), log(0.25/0.75), 0, 0];

    for r = 1:10  % 10 random restarts to find global min
        s = startx_base + [randn*0.5, randn*0.01, randn*0.5, randn*pi, randn*2*pi];
        try
            [p, rn] = fminunc('BallStickSSD', s, h_opt, Avox, bvals, qhat, false);
            if rn < best_resnorm
                best_resnorm = rn;
                best_params  = p;
            end
        catch
        end
    end

    % Estimate noise std from RESNORM at best fit:
    % Under Gaussian noise, E[SSD] = K * sigma^2, so sigma = sqrt(SSD/K)
    sigma_noise = sqrt(best_resnorm / K);
    fprintf('  Best RESNORM: %.4e,  sigma_noise estimate: %.2f\n', ...
            best_resnorm, sigma_noise);

% ------------------------------------------------------------------
% STEP 2: MCMC - Metropolis-Hastings in TRANSFORMED parameter space
% ------------------------------------------------------------------
% We run MCMC in transformed space (same as Q1.2) so that parameters
% always stay physically realistic - the chain never proposes e.g.
% negative diffusivity.
% ------------------------------------------------------------------
fprintf('Running MCMC (%d steps, burn-in %d, thin %d)...\n', ...
        T_total, burnin, thin);
prop_std = [0.02, 0.0003, 0.04, 0.05, 0.05];
% Storage for ALL steps (we'll thin after)
all_samples = zeros(T_total, 5);  % transformed parameters
n_accepted  = 0;

% Current position = best fit
x_current = best_params;
ssd_current = BallStickSSD(x_current, Avox, bvals, qhat, false);

for t = 1:T_total

    % --- Propose a candidate (random walk step) ---
    x_candidate = x_current + randn(1,5) .* prop_std;

    % --- Compute SSD for candidate ---
    ssd_candidate = BallStickSSD(x_candidate, Avox, bvals, qhat, false);

    % --- Compute log likelihood ratio ---
    % log p(data|x) = -SSD / (2*sigma^2)  (up to a constant)
    % log alpha = log p(data|x_c) - log p(data|x_current)
    %           = -(SSD_c - SSD_current) / (2*sigma^2)
    log_alpha = -(ssd_candidate - ssd_current) / (2 * sigma_noise^2);

    % --- Accept or reject ---
    if log(rand) < log_alpha
        % Accept: move to candidate
        x_current   = x_candidate;
        ssd_current = ssd_candidate;
        n_accepted  = n_accepted + 1;
    end
    % If rejected: stay at x_current (implicitly, by not updating)

    all_samples(t, :) = x_current;
end

acceptance_rate = n_accepted / T_total;
fprintf('  Acceptance rate: %.1f%%\n', acceptance_rate * 100);
if acceptance_rate < 0.15
    fprintf('  WARNING: acceptance rate too low - consider decreasing prop_std\n');
elseif acceptance_rate > 0.6
    fprintf('  WARNING: acceptance rate too high - consider increasing prop_std\n');
end

% ------------------------------------------------------------------
% STEP 3: Apply burn-in and thinning
% ------------------------------------------------------------------
samples_after_burnin = all_samples(burnin+1:end, :);
samples_thinned      = samples_after_burnin(1:thin:end, :);
n_eff = size(samples_thinned, 1);
fprintf('  Effective samples after burn-in and thinning: %d\n', n_eff);

% ------------------------------------------------------------------
% STEP 4: Back-transform to physical parameters
% ------------------------------------------------------------------
mcmc_S0 = samples_thinned(:,1).^2;
mcmc_d  = samples_thinned(:,2).^2;
mcmc_f  = 1 ./ (1 + exp(-samples_thinned(:,3)));

% ------------------------------------------------------------------
% STEP 5: Compute 2-sigma and 95% ranges (same method as Q2.1)
% ------------------------------------------------------------------
param_names   = {'S0',   'd',    'f'};
param_samples = {mcmc_S0, mcmc_d, mcmc_f};

fprintf('\nResults for %s:\n', vname);
fprintf('%-4s  %-12s  %-26s  %-26s\n', ...
        'Par', 'Mean', '2-sigma [lo,  hi]', '95%% range [lo,  hi]');

sigma_ranges = zeros(3,2);
pct_ranges   = zeros(3,2);

for p = 1:3
    samp = param_samples{p};
    mu      = mean(samp);
    sigma   = std(samp);
    lo_2sig = mu - 2*sigma;
    hi_2sig = mu + 2*sigma;
    sorted  = sort(samp);
    lo_95   = sorted(max(1,   round(0.025*n_eff)));
    hi_95   = sorted(min(n_eff, round(0.975*n_eff)));

    sigma_ranges(p,:) = [lo_2sig, hi_2sig];
    pct_ranges(p,:)   = [lo_95,   hi_95];

    fprintf('%-4s  %-12.4e  [%-10.4e, %-10.4e]  [%-10.4e, %-10.4e]\n', ...
            param_names{p}, mu, lo_2sig, hi_2sig, lo_95, hi_95);
end

% ------------------------------------------------------------------
% STEP 6: Plot histograms
% ------------------------------------------------------------------
figure('Name', ['MCMC - ' vname], 'NumberTitle', 'off');
for p = 1:3
    samp = param_samples{p};
    subplot(1,3,p);
    histogram(samp, 60, 'Normalization', 'probability', ...
              'FaceColor', [0.8 0.4 0.4], 'EdgeColor', 'white');
    hold on;
    xline(sigma_ranges(p,1), 'b--', 'LineWidth', 1.5, 'DisplayName', '2\sigma lo');
    xline(sigma_ranges(p,2), 'b--', 'LineWidth', 1.5, 'DisplayName', '2\sigma hi');
    xline(pct_ranges(p,1),   'g-',  'LineWidth', 1.5, 'DisplayName', '95% lo');
    xline(pct_ranges(p,2),   'g-',  'LineWidth', 1.5, 'DisplayName', '95% hi');
    xline(mean(samp),        'r-',  'LineWidth', 2,   'DisplayName', 'Mean');
    title(param_names{p});
    xlabel('Value'); ylabel('Probability');
    legend('Location', 'best');
    hold off;
end
sgtitle(['MCMC (Metropolis-Hastings) - ' vname]);

% ------------------------------------------------------------------
% STEP 7: Plot the chain traces (important diagnostic!)
% ------------------------------------------------------------------
% These show how the chain moved over time. A good chain should look
% like "fuzzy caterpillar" - no long trends, no getting stuck.
figure('Name', ['MCMC chains - ' vname], 'NumberTitle', 'off');
chain_params = {mcmc_S0, mcmc_d, mcmc_f};
for p = 1:3
    subplot(3,1,p);
    plot(chain_params{p}, 'Color', [0.6 0.6 0.6], 'LineWidth', 0.5);
    ylabel(param_names{p});
    xlabel('Sample index (after burn-in and thinning)');
    title(['Chain trace: ' param_names{p}]);
end
sgtitle(['Chain traces - ' vname]);

% ------------------------------------------------------------------
% STEP 8: Compare MCMC vs Bootstrap 95% ranges
% ------------------------------------------------------------------
fprintf('\nComparison with Bootstrap 95%% range for %s:\n', vname);
fprintf('%-4s  %-30s  %-30s\n', 'Par', 'MCMC 95% [lo, hi]', 'Bootstrap 95% [lo, hi]');
boot = boot_95{v};
for p = 1:3
    fprintf('%-4s  [%-10.4e, %-10.4e]    [%-10.4e, %-10.4e]\n', ...
            param_names{p}, ...
            pct_ranges(p,1), pct_ranges(p,2), ...
            boot(p,1), boot(p,2));
end
end
fprintf('\nDone. Check acceptance rates and chain traces!\n');
fprintf('If acceptance rate is wrong, adjust prop_std at the top of this script.\n');