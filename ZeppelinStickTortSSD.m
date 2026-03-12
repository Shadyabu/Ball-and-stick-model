function sumRes = ZeppelinStickTortSSD(x, meas, bvals, qhat)
% Zeppelin and Stick with Tortuosity: lambda2 = (1-f)*d

bvals = bvals(:)';   % CRITICAL: force bvals to be 1xK row vector

S0    = x(1)^2;
d     = x(2)^2;
f     = 1 / (1 + exp(-x(3)));
theta = x(4);
phi   = x(5);

lambda2 = (1 - f) * d;

fibdir     = [cos(phi)*sin(theta); sin(phi)*sin(theta); cos(theta)];
fibdotgrad = sum(qhat .* repmat(fibdir, [1, size(qhat,2)]), 1);

SI = exp(-bvals * d   .* fibdotgrad.^2);
SE = exp(-bvals .* (lambda2 + (d - lambda2) .* fibdotgrad.^2));

S = S0 * (f * SI + (1 - f) * SE);

sumRes = sum((meas' - S).^2);
end