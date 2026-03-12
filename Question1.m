% Code was written with the help of AI. All instructions of what to code,
% analysis and write up was done manually, without AI.

% PLEASE ADD data.mat INTO THE DIRECTORY BEFORE RUNNING THE CODE

load('data');
dwis=double(dwis);
dwis=permute(dwis,[4,1,2,3]);

% Middle slice of the 1st image volume, which has b=0
figure;
imshow(flipud(squeeze(dwis(1,:,:,72))'), []);
% Middle slice of the 2nd image volume, which has b=1000
figure;
imshow(flipud(squeeze(dwis(2,:,:,72))'), []);
qhat = load('bvecs');
bvals = 1000*sum(qhat.*qhat);

Avox = dwis(:,92,65,72);
% Define a starting point for the non-linear fit
startx = [3.5e+00 3e-03 2.5e-01 0 0];
% Define various options for the non-linear fitting
% algorithm.
isQ1_1 = true;
h=optimset('Display', 'iter', ...
'MaxFunEvals',20000,...
'Algorithm','quasi-newton',...
'TolX',1e-10,...
'TolFun',1e-10);
% Now run the fitting
disp('Running the first fit from Q1.1')
[parameter_hat,RESNORM,EXITFLAG,OUTPUT]=fminunc('BallStickSSD',startx,h,Avox,bvals,qhat,isQ1_1);

format short e

% Display parameter_hat and RESNORM
disp(parameter_hat)

disp(RESNORM)

% Experimenting with different fminunc options
disp('Experimenting with fminunc (Q1.1)')
h = optimset('Display', 'iter', ...
             'MaxFunEvals', 500, ...
             'MaxIter', 200, ...
             'TolX', 1e-10, ...
             'TolFun', 1e-10);
[parameter_hat,RESNORM,EXITFLAG,OUTPUT]=fminunc('BallStickSSD',startx,h,Avox,bvals,qhat,true);

% Running original fit again to plot it
disp('Rerunning original fit to plot')
h=optimset('Display', 'iter', ...
'MaxFunEvals',20000,...
'Algorithm','quasi-newton',...
'TolX',1e-10,...
'TolFun',1e-10);
[parameter_hat,RESNORM,EXITFLAG,OUTPUT]=fminunc('BallStickSSD',startx,h,Avox,bvals,qhat,true);

% PLOT THE FIT
% Step 1: Extract the fitted parameters from parameter_hat
S0    = parameter_hat(1);
diff  = parameter_hat(2);
f     = parameter_hat(3);
theta = parameter_hat(4);
phi   = parameter_hat(5);

% Step 2: Recompute the model predictions (same formula as in BallStickSSD)
fibdir     = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
fibdotgrad = sum(qhat .* repmat(fibdir, [length(qhat) 1])', 1);
S_pred     = S0 * (f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));

% Step 3: Plot actual data vs model predictions
figure;
plot(Avox, 'bs', 'MarkerSize', 10, 'LineWidth', 2);   % blue squares = data
hold on;
plot(S_pred, 'rx', 'MarkerSize', 10, 'LineWidth', 2); % red crosses = model
legend('Data', 'Model');
xlabel('Measurement index k');
ylabel('Signal S');
title('Ball-and-Stick model fit vs actual data');

%% -------------------------------------------------------------------------
% Question 1.2
% -------------------------------------------------------------------------
fprintf('\n%s\n', 'QUESTION 1.2 --------------')

isQ1_1 = false;
% Run fit with only realistic values
disp('Running fit with only realistic values')
[parameter_hat,RESNORM,EXITFLAG,OUTPUT]=fminunc('BallStickSSD',startx,h,Avox,bvals,qhat,isQ1_1);

% Change starting point
startx = [sqrt(3.5e+00), sqrt(3e-03), log(0.25/0.75), 0, 0];
% Run fit again with different starting point
disp('Running realistic fit with different starting point (Q1.1)')
[parameter_hat,RESNORM,EXITFLAG,OUTPUT]=fminunc('BallStickSSD',startx,h,Avox,bvals,qhat,isQ1_1);

% PLOT NEW FIT
% Step 1: Extract the fitted parameters from parameter_hat
S0    = parameter_hat(1)^2;
diff  = parameter_hat(2)^2;
f     = 1 / (1 + exp(-parameter_hat(3)));
theta = parameter_hat(4);
phi   = parameter_hat(5);

% Step 2: Recompute the model predictions (same formula as in BallStickSSD)
fibdir     = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
fibdotgrad = sum(qhat .* repmat(fibdir, [length(qhat) 1])', 1);
S_pred     = S0 * (f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));

% Step 3: Plot actual data vs model predictions
figure;
plot(Avox, 'bs', 'MarkerSize', 10, 'LineWidth', 2);   % blue squares = data
hold on;
plot(S_pred, 'rx', 'MarkerSize', 10, 'LineWidth', 2); % red crosses = model
legend('Data', 'Model');
xlabel('Measurement index k');
ylabel('Signal S');
title('Ball-and-Stick model fit vs actual data');

disp('Parameter values for Q1.2')
fprintf('S0: %e\n', S0)
fprintf('diff: %e\n', diff)
fprintf('f: %e\n', f)
fprintf('theta: %e\n', theta)
fprintf('phi: %e\n', phi)
fprintf('RESNORM: %e\n', RESNORM)

%% -------------------------------------------------------------------------
% Question 1.3
% -------------------------------------------------------------------------

fprintf('\n%s\n', 'QUESTION 1.3 --------------')

nRuns = 100;  % number of random starts
results = zeros(nRuns, 1);  % store RESNORM for each run
params = zeros(nRuns, 5);   % store parameter_hat for each run

% Turn off display for speed
h=optimset('Display', 'off', ...
'MaxFunEvals',20000,...
'Algorithm','quasi-newton',...
'TolX',1e-10,...
'TolFun',1e-10); 

rng(42)
for i = 1:nRuns
    % Add random perturbation scaled to each parameter
    startx_rand = startx + [randn*3.5, randn*1e-3, randn*0.5, randn*pi, randn*2*pi];    

    % Run the fit
    [ph, rn] = fminunc('BallStickSSD', startx_rand, h, Avox, bvals, qhat, isQ1_1);
    
    results(i) = rn;
    params(i,:) = ph;
end

% Find the best (lowest) RESNORM
[best_RESNORM, best_idx] = min(results);
best_params = params(best_idx, :);

% Count how many runs found the global min (within a small tolerance)
tolerance = 1e3;  % runs within 1000 of best count as "finding" it
n_found = sum(results < best_RESNORM + tolerance);
proportion = n_found / nRuns;

fprintf('Best RESNORM: %.4e\n', best_RESNORM);
fprintf('Proportion finding global min: %.2f\n', proportion);

n_needed = ceil(log(0.05) / log(1 - proportion));
fprintf('Runs needed for 95%% confidence: %d\n', n_needed);

% Trying fit on other Voxels
fprintf('\n%s\n', 'Trying fit on other Voxels')
voxels = {
    dwis(:, 85, 128, 72),  'Voxel (85, 128, 72)';
    dwis(:, 43, 64, 82), 'Voxel (43, 64, 82)';
};

for v = 1:size(voxels, 1)
    Avox_test = voxels{v, 1};
    voxel_name = voxels{v, 2};
    
    results_v = zeros(nRuns, 1);

    rng(42)
    for i = 1:nRuns
        startx_rand = startx + [randn*3.5, randn*1e-3, randn, randn*pi, randn*2*pi];
        [~, rn] = fminunc('BallStickSSD', startx_rand, h, Avox_test, bvals, qhat, isQ1_1);
        results_v(i) = rn;
    end
    
    best_RESNORM_v = min(results_v);
    n_found_v = sum(results_v < best_RESNORM_v + tolerance);
    proportion_v = n_found_v / nRuns;
    n_needed_v = ceil(log(0.05) / log(1 - proportion_v));
    
    fprintf('\n%s\n', voxel_name);
    fprintf('  Best RESNORM:               %.4e\n', best_RESNORM_v);
    fprintf('  Expected RESNORM (noise):   %.4e\n', 108 * 200^2);
    fprintf('  Proportion finding global min: %.2f\n', proportion_v);
    fprintf('  Runs needed for 95%% confidence: %d\n', n_needed_v);
end

%% -------------------------------------------------------------------------
% Question 1.4
% -------------------------------------------------------------------------
fprintf('\n%s\n', 'QUESTION 1.4 --------------')

nX = 145;
nY = 174;

% Pre-allocate maps (filled with zeros to start)
S0_map    = zeros(nX, nY);
d_map     = zeros(nX, nY);
f_map     = zeros(nX, nY);
theta_map = zeros(nX, nY);
phi_map   = zeros(nX, nY);
resnorm_map = zeros(nX, nY);

nRuns = 3; % Number of random restarts per voxel to get 95% confidence of global minimum

tic
rng(42)
for x = 1:nX
    for y = 1:nY
        Avox = dwis(:, x, y, 72);  % 108-element signal vector for this voxel

        % Skip empty/background voxels to save time
        if mean(Avox) < 50
            continue
        end

        bestResnorm = inf;
        bestParams  = zeros(1,5);

        for r = 1:nRuns
            % Perturbed starting point (same approach as Q1.3)
            startx_rand = startx + [randn*3.5, randn*1e-3, randn, randn*pi, randn*2*pi];

            % Apply inverse transformation for constrained version (Q1.2)
            % e.g., startx_transformed(1) = sqrt(startx(1)), etc.

            try
                [p_hat, resnorm] = fminunc('BallStickSSD', ...
                                            startx_rand, h, ...
                                            Avox, bvals, qhat, isQ1_1);
                if resnorm < bestResnorm
                    bestResnorm = resnorm;
                    bestParams  = p_hat;
                end
            catch
                % If fitting fails for some reason, just skip
            end
        end

        % Apply forward transformation to get actual parameter values
        S0_map(x,y)      = bestParams(1)^2;
        d_map(x,y)       = bestParams(2)^2;
        f_map(x,y)       = 1 / (1 + exp(-bestParams(3)));
        theta_map(x,y)   = bestParams(4);
        phi_map(x,y)     = bestParams(5);
        resnorm_map(x,y) = bestResnorm;
    end
    fprintf('Row %d/%d done\n', x, nX); % Progress indicator
end
t_random = toc;
% figure of parameter maps
figure;
subplot(2,2,1); imagesc(S0_map');    colorbar; title('S0');       colormap(gca, 'gray');
subplot(2,2,2); imagesc(d_map', [0, 5e-3]);     colorbar; title('diff');        colormap(gca, 'hot');
subplot(2,2,3); imagesc(f_map');     colorbar; title('f');        colormap(gca, 'hot');
subplot(2,2,4); imagesc(resnorm_map'); colorbar; title('RESNORM'); colormap(gca, 'hot');


% Converting theta and phi to vector n
nx = cos(phi_map) .* sin(theta_map);
ny = sin(phi_map) .* sin(theta_map);
nz = cos(theta_map);

% FIBRE DIRECTION MAP
% Force consistent direction: flip if nz < 0
flip_mask = nz < 0;
nx(flip_mask) = -nx(flip_mask);
ny(flip_mask) = -ny(flip_mask);

% Weight by volume fraction f
nx_weighted = nx .* f_map;
ny_weighted = ny .* f_map;

% Create coordinate grids for quiver
[Y_grid, X_grid] = meshgrid(1:nY, 1:nX);

figure;
% Show anatomy as background
imagesc(S0_map');
colormap(gray); axis image; hold on;
quiver(X_grid, Y_grid, nx_weighted, ny_weighted, 4);
% The 0.5 is a scale factor — adjust to taste
axis equal;
title('Fibre direction n weighted by f');

%% -------------------------------------------------------------------------
% Question 1.5
% -------------------------------------------------------------------------
fprintf('\n%s\n', 'QUESTION 1.5 --------------')
startx_con = [3.5e+00, 3e-03, 0.25, 0, 0];

% ---- Pre-allocate maps ----
S0_map_dt    = zeros(nX, nY);
d_map_dt     = zeros(nX, nY);
f_map_dt     = zeros(nX, nY);
theta_map_dt = zeros(nX, nY);
phi_map_dt   = zeros(nX, nY);
resnorm_map_dt = zeros(nX, nY);
% Calculating likelihood of global minimum
Avox = dwis(:, 85, 128, 72);

nRuns = 100;
resnorms_fminunc = zeros(nRuns,1);
resnorms_fmincon = zeros(nRuns,1);
resnorms_dt1     = zeros(nRuns,1);
resnorms_dt2     = zeros(nRuns,1);
resnorms_dt3     = zeros(nRuns,1);

% Pre-compute DT-informed starting points for this voxel
G = zeros(length(bvals), 7);
for k = 1:length(bvals)
    qx = qhat(1,k); qy = qhat(2,k); qz = qhat(3,k);
    G(k,:) = [1, -bvals(k)*qx^2, -bvals(k)*qy^2, -bvals(k)*qz^2, ...
                 -bvals(k)*2*qx*qy, -bvals(k)*2*qx*qz, -bvals(k)*2*qy*qz];
end
logAvox = log(max(Avox, 1));
xDT = G \ logAvox;

S0_init = exp(xDT(1));
DT = [xDT(2), xDT(5), xDT(6);
      xDT(5), xDT(3), xDT(7);
      xDT(6), xDT(7), xDT(4)];

[V, E] = eig(DT);
eigenvalues = diag(E);
[~, idx] = max(eigenvalues);
e1 = V(:, idx);

theta_init = acos(max(min(e1(3), 1), -1));
phi_init   = atan2(e1(2), e1(1));
MD         = mean(eigenvalues);
FA         = sqrt(1.5) * sqrt(sum((eigenvalues - MD).^2)) / sqrt(sum(eigenvalues.^2));

startx_dt1 = [sqrt(max(S0_init,0)), sqrt(max(MD,1e-6)),               log(0.5/0.5),       theta_init, phi_init];
startx_dt2 = [sqrt(max(S0_init,0)), sqrt(max(eigenvalues(idx),1e-6)), log(0.5/0.5),       theta_init, phi_init];
f_fa = max(min(FA, 0.99), 0.01);
startx_dt3 = [sqrt(max(S0_init,0)), sqrt(max(MD,1e-6)),               log(f_fa/(1-f_fa)), theta_init, phi_init];


% Calculating likelihood of global minimum
Avox = dwis(:, 85, 128, 72);

nRuns = 100;
resnorms_fminunc = zeros(nRuns,1);
resnorms_fmincon = zeros(nRuns,1);
resnorms_dt1     = zeros(nRuns,1);
resnorms_dt2     = zeros(nRuns,1);
resnorms_dt3     = zeros(nRuns,1);

% Pre-compute DT-informed starting points for this voxel
G = zeros(length(bvals), 7);
for k = 1:length(bvals)
    qx = qhat(1,k); qy = qhat(2,k); qz = qhat(3,k);
    G(k,:) = [1, -bvals(k)*qx^2, -bvals(k)*qy^2, -bvals(k)*qz^2, ...
                 -bvals(k)*2*qx*qy, -bvals(k)*2*qx*qz, -bvals(k)*2*qy*qz];
end
logAvox = log(max(Avox, 1));
xDT = G \ logAvox;

S0_init = exp(xDT(1));
DT = [xDT(2), xDT(5), xDT(6);
      xDT(5), xDT(3), xDT(7);
      xDT(6), xDT(7), xDT(4)];

[V, E] = eig(DT);
eigenvalues = diag(E);
[~, idx] = max(eigenvalues);
e1 = V(:, idx);

theta_init = acos(max(min(e1(3), 1), -1));
phi_init   = atan2(e1(2), e1(1));
MD         = mean(eigenvalues);
FA         = sqrt(1.5) * sqrt(sum((eigenvalues - MD).^2)) / sqrt(sum(eigenvalues.^2));

startx_dt1 = [sqrt(max(S0_init,0)), sqrt(max(MD,1e-6)),               log(0.5/0.5),       theta_init, phi_init];
startx_dt2 = [sqrt(max(S0_init,0)), sqrt(max(eigenvalues(idx),1e-6)), log(0.5/0.5),       theta_init, phi_init];
f_fa = max(min(FA, 0.99), 0.01);
startx_dt3 = [sqrt(max(S0_init,0)), sqrt(max(MD,1e-6)),               log(f_fa/(1-f_fa)), theta_init, phi_init];

rng(42)
for i = 1:nRuns
    % fminunc random restarts (reparameterised)
    startx_rand = startx + [randn*3.5, randn*1e-3, randn, randn*pi, randn*2*pi];
    [~, resnorms_fminunc(i)] = fminunc('BallStickSSD', startx_rand, h, Avox, bvals, qhat, false);

    % Bounded starting point, transformed to reparameterised space, then fminunc
    startx_rand_con = startx_con + [randn*3.5, randn*1e-3, randn*0.1, randn*pi, randn*2*pi];
    startx_rand_con(1) = max(startx_rand_con(1), 1e-6);
    startx_rand_con(2) = max(min(startx_rand_con(2), 0.009), 1e-6);
    startx_rand_con(3) = max(min(startx_rand_con(3), 0.99), 0.01);
    startx_rand_con_reparam = [sqrt(startx_rand_con(1)), ...
                                sqrt(startx_rand_con(2)), ...
                                log(startx_rand_con(3) / (1 - startx_rand_con(3))), ...
                                startx_rand_con(4), ...
                                startx_rand_con(5)];
    [~, resnorms_fmincon(i)] = fminunc('BallStickSSD', startx_rand_con_reparam, h, Avox, bvals, qhat, false);

    % DT Mapping 1
    startx_rand_dt1 = startx_dt1 + [randn*0.5, randn*0.01, randn*0.5, randn*0.3, randn*0.3];
    [~, resnorms_dt1(i)] = fminunc('BallStickSSD', startx_rand_dt1, h, Avox, bvals, qhat, false);

    % DT Mapping 2
    startx_rand_dt2 = startx_dt2 + [randn*0.5, randn*0.01, randn*0.5, randn*0.3, randn*0.3];
    [~, resnorms_dt2(i)] = fminunc('BallStickSSD', startx_rand_dt2, h, Avox, bvals, qhat, false);

    % DT Mapping 3
    startx_rand_dt3 = startx_dt3 + [randn*0.5, randn*0.01, randn*0.5, randn*0.3, randn*0.3];
    [~, resnorms_dt3(i)] = fminunc('BallStickSSD', startx_rand_dt3, h, Avox, bvals, qhat, false);
end

global_min = min([resnorms_fminunc; resnorms_fmincon; resnorms_dt1; resnorms_dt2; resnorms_dt3]);
tol = 1e3;

p_fminunc = mean(resnorms_fminunc < global_min + tol);
p_fmincon = mean(resnorms_fmincon < global_min + tol);
p_dt1     = mean(resnorms_dt1     < global_min + tol);
p_dt2     = mean(resnorms_dt2     < global_min + tol);
p_dt3     = mean(resnorms_dt3     < global_min + tol);

n_fminunc = ceil(log(0.05) / log(1 - p_fminunc));
n_fmincon = ceil(log(0.05) / log(1 - p_fmincon));
n_dt1     = ceil(log(0.05) / log(1 - p_dt1));
n_dt2     = ceil(log(0.05) / log(1 - p_dt2));
n_dt3     = ceil(log(0.05) / log(1 - p_dt3));

fprintf('%-35s  p: %.2f,  N needed: %d\n', 'fminunc (random restarts)',        p_fminunc, n_fminunc);
fprintf('%-35s  p: %.2f,  N needed: %d\n', 'bounded startx + fminunc',         p_fmincon, n_fmincon);
fprintf('%-35s  p: %.2f,  N needed: %d\n', 'DT Mapping 1 (mean d)',            p_dt1,     n_dt1);
fprintf('%-35s  p: %.2f,  N needed: %d\n', 'DT Mapping 2 (max eigenval)',      p_dt2,     n_dt2);
fprintf('%-35s  p: %.2f,  N needed: %d\n', 'DT Mapping 3 (FA-based f)',        p_dt3,     n_dt3);



rng(42)
for i = 1:nRuns
    % fminunc random restarts (reparameterised)
    startx_rand = startx + [randn*3.5, randn*1e-3, randn, randn*pi, randn*2*pi];
    [~, resnorms_fminunc(i)] = fminunc('BallStickSSD', startx_rand, h, Avox, bvals, qhat, false);

    % Bounded starting point, transformed to reparameterised space, then fminunc
    startx_rand_con = startx_con + [randn*3.5, randn*1e-3, randn*0.1, randn*pi, randn*2*pi];
    startx_rand_con(1) = max(startx_rand_con(1), 1e-6);
    startx_rand_con(2) = max(min(startx_rand_con(2), 0.009), 1e-6);
    startx_rand_con(3) = max(min(startx_rand_con(3), 0.99), 0.01);
    startx_rand_con_reparam = [sqrt(startx_rand_con(1)), ...
                                sqrt(startx_rand_con(2)), ...
                                log(startx_rand_con(3) / (1 - startx_rand_con(3))), ...
                                startx_rand_con(4), ...
                                startx_rand_con(5)];
    [~, resnorms_fmincon(i)] = fminunc('BallStickSSD', startx_rand_con_reparam, h, Avox, bvals, qhat, false);

    % DT Mapping 1
    startx_rand_dt1 = startx_dt1 + [randn*0.5, randn*0.01, randn*0.5, randn*0.3, randn*0.3];
    [~, resnorms_dt1(i)] = fminunc('BallStickSSD', startx_rand_dt1, h, Avox, bvals, qhat, false);

    % DT Mapping 2
    startx_rand_dt2 = startx_dt2 + [randn*0.5, randn*0.01, randn*0.5, randn*0.3, randn*0.3];
    [~, resnorms_dt2(i)] = fminunc('BallStickSSD', startx_rand_dt2, h, Avox, bvals, qhat, false);

    % DT Mapping 3
    startx_rand_dt3 = startx_dt3 + [randn*0.5, randn*0.01, randn*0.5, randn*0.3, randn*0.3];
    [~, resnorms_dt3(i)] = fminunc('BallStickSSD', startx_rand_dt3, h, Avox, bvals, qhat, false);
end

global_min = min([resnorms_fminunc; resnorms_fmincon; resnorms_dt1; resnorms_dt2; resnorms_dt3]);
tol = 1e3;

p_fminunc = mean(resnorms_fminunc < global_min + tol);
p_fmincon = mean(resnorms_fmincon < global_min + tol);
p_dt1     = mean(resnorms_dt1     < global_min + tol);
p_dt2     = mean(resnorms_dt2     < global_min + tol);
p_dt3     = mean(resnorms_dt3     < global_min + tol);

n_fminunc = ceil(log(0.05) / log(1 - p_fminunc));
n_fmincon = ceil(log(0.05) / log(1 - p_fmincon));
n_dt1     = ceil(log(0.05) / log(1 - p_dt1));
n_dt2     = ceil(log(0.05) / log(1 - p_dt2));
n_dt3     = ceil(log(0.05) / log(1 - p_dt3));

fprintf('%-35s  p: %.2f,  N needed: %d\n', 'fminunc (random restarts)',        p_fminunc, n_fminunc);
fprintf('%-35s  p: %.2f,  N needed: %d\n', 'bounded startx + fminunc',         p_fmincon, n_fmincon);
fprintf('%-35s  p: %.2f,  N needed: %d\n', 'DT Mapping 1 (mean d)',            p_dt1,     n_dt1);
fprintf('%-35s  p: %.2f,  N needed: %d\n', 'DT Mapping 2 (max eigenval)',      p_dt2,     n_dt2);
fprintf('%-35s  p: %.2f,  N needed: %d\n', 'DT Mapping 3 (FA-based f)',        p_dt3,     n_dt3);

% Mappings:
% Mapping 1: mean d, fixed f=0.5
% Mapping 2: largest eigenvalue as d, fixed f=0.5
%Mapping 3: mean d, FA-estimated f
mapping_names = {'Mapping 1', ...
                 'Mapping 2', ...
                 'Mapping 3'};

nMappings = 3;
nRuns_dt = 1; 

fprintf('Starting DT-informed mapping...\n');
% Store times for comparison
t_DT_all = zeros(1, nMappings);

rng(42)
for m = 1:nMappings

    fprintf('\nRunning %s...\n', mapping_names{m});

    % Pre-allocate maps for this mapping
    S0_map_dt    = zeros(nX, nY);
    d_map_dt     = zeros(nX, nY);
    f_map_dt     = zeros(nX, nY);
    resnorm_map_dt = zeros(nX, nY);

    tic;

    for x = 1:nX
        for y = 1:nY
            Avox = dwis(:, x, y, 72);

            if mean(Avox) < 50
                continue
            end

            % Step 1: Fit diffusion tensor
            G = zeros(length(bvals), 7);
            for k = 1:length(bvals)
                qx = qhat(1,k); qy = qhat(2,k); qz = qhat(3,k);
                G(k,:) = [1, -bvals(k)*qx^2, -bvals(k)*qy^2, -bvals(k)*qz^2, ...
                             -bvals(k)*2*qx*qy, -bvals(k)*2*qx*qz, -bvals(k)*2*qy*qz];
            end
            logAvox = log(max(Avox, 1));
            xDT = G \ logAvox;

            S0_init = exp(xDT(1));
            DT = [xDT(2), xDT(5), xDT(6);
                  xDT(5), xDT(3), xDT(7);
                  xDT(6), xDT(7), xDT(4)];

            % Step 2: Eigendecomposition
            [V, E] = eig(DT);
            eigenvalues = diag(E);
            [~, idx] = max(eigenvalues);
            e1 = V(:, idx);

            % Step 3: Choose mapping
            if m == 1
                % Mapping 1: mean diffusivity, fixed f
                d_init = max(mean(eigenvalues), 1e-6);
                f_init = 0.5;

            elseif m == 2
                % Mapping 2: largest eigenvalue as d, fixed f
                d_init = max(eigenvalues(idx), 1e-6);
                f_init = 0.5;

            elseif m == 3
                % Mapping 3: mean diffusivity, FA-estimated f
                d_init = max(mean(eigenvalues), 1e-6);
                MD = mean(eigenvalues);
                FA = sqrt(1.5) * sqrt(sum((eigenvalues - MD).^2)) / ...
                     sqrt(sum(eigenvalues.^2));
                f_init = max(min(FA, 0.99), 0.01);
            end

            theta_init = acos(max(min(e1(3), 1), -1));
            phi_init   = atan2(e1(2), e1(1));

            % Step 4: Transform to constrained space
            startx_dt = [sqrt(max(S0_init, 0)), ...
                         sqrt(d_init), ...
                         log(f_init / (1 - f_init)), ...
                         theta_init, ...
                         phi_init];

            % Step 5: Run fminunc
            bestResnorm = inf;
            bestParams  = zeros(1, 5);

            for r = 1:nRuns_dt
                startx_rand = startx_dt + [randn*0.5, randn*0.01, randn*0.5, ...
                                            randn*0.3, randn*0.3];
                try
                    [p_hat, resnorm] = fminunc('BallStickSSD', startx_rand, h, ...
                                                Avox, bvals, qhat, false);
                    if resnorm < bestResnorm
                        bestResnorm = resnorm;
                        bestParams  = p_hat;
                    end
                catch
                end
            end

            % Step 6: Store
            S0_map_dt(x,y)      = bestParams(1)^2;
            d_map_dt(x,y)       = bestParams(2)^2;
            f_map_dt(x,y)       = 1 / (1 + exp(-bestParams(3)));
            resnorm_map_dt(x,y) = bestResnorm;
        end
        fprintf('Row %d/%d done\n', x, nX);
    end

    t_DT_all(m) = toc;
    fprintf('%s took: %.1f seconds\n', mapping_names{m}, t_DT_all(m));

    % Plot results for this mapping
    figure;
    subplot(2,2,1); imagesc(S0_map_dt');         colorbar; colormap(gca,'gray');
                    title(['S0 - ' mapping_names{m}]);
    subplot(2,2,2); imagesc(d_map_dt',[0,5e-3]); colorbar; colormap(gca,'hot');
                    title(['d - ' mapping_names{m}]);
    subplot(2,2,3); imagesc(f_map_dt');           colorbar; colormap(gca,'hot');
                    title(['f - ' mapping_names{m}]);
    subplot(2,2,4); imagesc(resnorm_map_dt');     colorbar; colormap(gca,'hot');
                    title(['RESNORM - ' mapping_names{m}]);
end

% Summary comparison
fprintf('\n--- SUMMARY ---\n');
fprintf('Q1.4 random restarts (3 runs): %.1f seconds\n', t_random);
for m = 1:nMappings
    fprintf('%s: %.1f seconds\n', mapping_names{m}, t_DT_all(m));
end

fprintf('\n%s\n', 'Q1.5 fmincon approach')

% Define bounds directly on the physical parameters
lb = [0,     0,     0,    0, 0];  % lower bounds
ub = [50000,   0.01,  1,     2*pi,  2*pi];  % upper bounds
% S0: must be positive, upper bound of 50 000 (quite generous, considering no S0 value is over 18 000)
% d:  must be positive, upper bound 0.01 mm2/s (generous for brain tissue)
% f:  must be in [0,1]
% theta, phi: between 0 and 2pi (all values of the angle)

% fmincon options — similar tolerances to before
h_con = optimoptions('fmincon', ...
    'Display', 'off', ...
    'MaxFunctionEvaluations', 20000, ...
    'Algorithm', 'interior-point', ...
    'OptimalityTolerance', 1e-10, ...
    'StepTolerance', 1e-10);

% Starting point — now in REAL parameter space (no transformation needed)
startx_con = [3.5e+00, 3e-03, 0.25, 0, 0];

nRuns_con = 3;  % start with same as Q1.4, then assess

% Pre-allocate maps
S0_map_con     = zeros(nX, nY);
d_map_con      = zeros(nX, nY);
f_map_con      = zeros(nX, nY);
resnorm_map_con = zeros(nX, nY);

fprintf('Starting fmincon mapping...\n');
tic;

for x = 1:nX
    for y = 1:nY
        Avox = dwis(:, x, y, 72);

        if mean(Avox) < 50
            continue
        end

        bestResnorm = inf;
        bestParams  = zeros(1, 5);

        for r = 1:nRuns_con
            % Perturb starting point — in real parameter space this time
            % so perturbations must keep values within reasonable ranges
            startx_rand = startx_con + ...
                [randn*3.5, randn*1e-3, randn*0.1, randn*pi, randn*2*pi];

            % Clamp starting point to be within bounds before passing in
            % (fmincon will enforce bounds, but starting outside can cause issues)
            startx_rand(1) = max(startx_rand(1), 1e-6);   % S0 > 0
            startx_rand(2) = max(min(startx_rand(2), 0.009), 1e-6); % 0 < d < 0.01
            startx_rand(3) = max(min(startx_rand(3), 0.99), 0.01);  % 0 < f < 1

            try
                [p_hat, resnorm] = fmincon(@(x) BallStickSSD(x, Avox, bvals, qhat, true), ...
                                            startx_rand, ...
                                            [], [], [], [], ...  % no linear constraints
                                            lb, ub, [], ...      % bounds
                                            h_con);
                if resnorm < bestResnorm
                    bestResnorm = resnorm;
                    bestParams  = p_hat;
                end
            catch
            end
        end

        % No back-transformation needed — parameters are already real values
        S0_map_con(x,y)      = bestParams(1);
        d_map_con(x,y)       = bestParams(2);
        f_map_con(x,y)       = bestParams(3);
        resnorm_map_con(x,y) = bestResnorm;
    end
    fprintf('Row %d/%d done\n', x, nX);
end

t_fmincon = toc;
fprintf('fmincon mapping took: %.1f seconds\n', t_fmincon);
fprintf('Compare: fminunc random restarts took: %.1f seconds\n', t_random);
fprintf('Compare: DT-informed took: %.1f seconds\n', min(t_DT_all));

% Plot
figure;
subplot(2,2,1); imagesc(S0_map_con');          colorbar; colormap(gca,'gray');
                title('S0 (fmincon)');
subplot(2,2,2); imagesc(d_map_con',[0,5e-3]);  colorbar; colormap(gca,'hot');
                title('d (fmincon)');
subplot(2,2,3); imagesc(f_map_con');           colorbar; colormap(gca,'hot');
                title('f (fmincon)');
subplot(2,2,4); imagesc(resnorm_map_con');     colorbar; colormap(gca,'hot');
                title('RESNORM (fmincon)');
