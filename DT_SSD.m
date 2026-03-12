function sumRes = DT_SSD(x, meas, bvals, qhat)
% Diffusion Tensor model - nonlinear SSD objective function
% x = [S0, Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
% qhat: 3 x K

bvals = bvals(:)';   % CRITICAL: force bvals to be 1xK row vector

S0  = abs(x(1));     % abs instead of squaring to keep same parameterisation
Dxx = x(2); Dxy = x(3); Dxz = x(4);
Dyy = x(5); Dyz = x(6); Dzz = x(7);

D = [Dxx Dxy Dxz;
     Dxy Dyy Dyz;
     Dxz Dyz Dzz];

% q̂ᵀDq̂ for each measurement — sum over the 3 spatial dimensions (rows)
qdotDq = sum(qhat .* (D * qhat), 1);  % 1 x K

% abs() adds numerical stability when D becomes non-positive-definite
S = S0 * exp(-bvals .* abs(qdotDq));  % 1 x K

sumRes = sum((meas' - S).^2);
end