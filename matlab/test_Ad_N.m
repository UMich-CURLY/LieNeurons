clear;
clc;

num_sample = 100000;
eps = 1e-5;
for i=1:num_sample
    %% generate N
    n = rand(3,1);
    N = [1 n(1) n(2); 0 1 n(3); 0 0 1];
    N = 1/det(N)^(1/3) * N;
    n1 = N(1,2);
    n2 = N(1,3);
    n3 = N(2,3);

    Ad_N =...
    [[           1,                     n1,                   n1,     0, 0,  0, n2/2 + (n1*n3)/2,  -n3/2];
    [         -n1,             1 - n1^2/2,              -n1^2/2,     0, 0,  0, n3/2 - (n1*n2)/2,   n2/2];
    [          n1,                 n1^2/2,           n1^2/2 + 1,     0, 0,  0, n3/2 + (n1*n2)/2,  -n2/2];
    [           0,                      0,                    0,     1, 0,  0, n2/2 - (n1*n3)/2,   n3/2];
    [2*n1*n3 - n2, - n3 - n1*(n2 - n1*n3), n3 - n1*(n2 - n1*n3), -3*n2, 1, n1, -n2*(n2 - n1*n3), -n2*n3];
    [          n3,             n1*n3 - n2,           n1*n3 - n2, -3*n3, 0,  1, -n3*(n2 - n1*n3),  -n3^2];
    [           0,                      0,                    0,     0, 0,  0,                1,      0];
    [           0,                      0,                    0,     0, 0,  0,              -n1,      1]];

    %% generate sl(3)
    v = rand(8,1);
    v_hat = hat_sl3(v);
    v_hat_ad = N * v_hat * inv(N);
    v_ad = Ad_N * v;                    % Linear map on v
    v_hat_ad_vee = vee_sl3(v_hat_ad);   % vee after adjoint on v_hat

    diff = v_ad - v_hat_ad_vee;         % They should be the same
    
    if norm(diff) > eps
       disp('error bigger than eps');
       disp(diff)
    elseif i==num_sample
       disp('Passed all test case!');
    end
    
end

