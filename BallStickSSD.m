function sumRes = BallStickSSD(x, Avox, bvals, qhat, isQ1_1)
if isQ1_1
    S0 = x(1);
    diff = x(2);
    f = x(3);
else 
    S0 = x(1)^2;
    diff = x(2)^2;
    f = 1 / (1 + exp(-x(3)));
end
% Extract the parameters
theta = x(4);
phi = x(5);
% Synthesize the signals according to the model
fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
% Compute the sum of square differences
sumRes = sum((Avox - S').^2);