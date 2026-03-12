function sumRes = ZeppelinStickSSD(x, meas, bvals, qhat)
% Zeppelin and Stick model
% x = [sqrt(S0), sqrt(d), logit(lambda2/d), logit(f), theta, phi]

bvals = bvals(:)';   % CRITICAL: force bvals to be 1xK row vector

S0      = x(1)^2;
d       = x(2)^2;
lambda2 = d / (1 + exp(-x(3)));   % 0 < lambda2 <= d
f       = 1 / (1 + exp(-x(4)));
theta   = x(5);
phi     = x(6);

fibdir     = [cos(phi)*sin(theta); sin(phi)*sin(theta); cos(theta)];  % 3x1
fibdotgrad = sum(qhat .* repmat(fibdir, [1, size(qhat,2)]), 1);       % 1xK

SI = exp(-bvals * d   .* fibdotgrad.^2);
SE = exp(-bvals .* (lambda2 + (d - lambda2) .* fibdotgrad.^2));

S = S0 * (f * SI + (1 - f) * SE);

sumRes = sum((meas' - S).^2);
end