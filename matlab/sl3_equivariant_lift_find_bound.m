clc;
clear; 

syms v1 v2 v3 v4 v5 v6 v7 v8
assumeAlso([v1 v2 v3 v4 v5 v6 v7 v8],'real')

syms h1 h2 h3 h4 h5 h6 h7 h8 h9
assumeAlso([h1 h2 h3 h4 h5 h6 h7 h8 h9],'real')
%%

E1 = [1, 0, 0; 0, -1, 0; 0, 0, 0];
E2 = [0, 1, 0; 1, 0, 0; 0, 0, 0];
E3 = [0, -1, 0; 1, 0, 0; 0, 0, 0];
E4 = [1, 0, 0; 0, 1, 0; 0, 0, -2];
E5 = [0, 0, 1; 0, 0, 0; 0, 0, 0];
E6 = [0, 0, 0; 0, 0, 1; 0, 0, 0];
E7 = [0, 0, 0; 0, 0, 0; 1, 0, 0];
E8 = [0, 0, 0; 0, 0, 0; 0, 1, 0];

Ex = [0, 0, 0;0, 0, -1;0, 1, 0];
Ey = [0, 0, 1;0, 0, 0;-1, 0, 0];
Ez = [0, -1, 0;1, 0, 0;0, 0, 0];

% E1 = [0, 0, 1; 0, 0, 0; 0, 0, 0];
% E2 = [0, 0, 0; 0, 0, 1; 0, 0, 0];
% E3 = [0, -1, 0; 1, 0, 0; 0, 0, 0];
% E4 = [0, 0, 0; 0, 0, 0; 0, 0, -1];
% E5 = [1, 0, 0; 0, -1, 0; 0, 0, 0];
% E6 = [0, 1, 0; 0, 0, 0; 0, 0, 0];
% E7 = [0, 0, 0; 0, 0, 0; 1, 0, 0];
% E8 = [0, 0, 0; 0, 0, 0; 0, 1, 0];

E = {E1,E2,E3,E4,E5,E6,E7,E8};

E1_vec = reshape(E1,1,[])';
E2_vec = reshape(E2,1,[])';
E3_vec = reshape(E3,1,[])';
E4_vec = reshape(E4,1,[])';
E5_vec = reshape(E5,1,[])';
E6_vec = reshape(E6,1,[])';
E7_vec = reshape(E7,1,[])';
E8_vec = reshape(E8,1,[])';


E_vec = [E1_vec, E2_vec, E3_vec, E4_vec, E5_vec, E6_vec, E7_vec, E8_vec]; 

x_hat = v1*E1+v2*E2+v3*E3+v4*E4+v5*E5+v6*E6+v7*E7+v8*E8;


%% find h
H = [h1,h2,h3;h4,h5,h6;h7,h8,h9];
Ad_H_hat = H*x_hat*inv(H);
Ad_H_hat_vec = reshape(Ad_H_hat,1,[])';

% solve least square to obtain x
x = inv(E_vec'*E_vec)*(E_vec')*Ad_H_hat_vec;
var = [v1,v2,v3,v4,v5,v6,v7,v8];
[Ad_H_sym,b]=equationsToMatrix(x,var);

syms f(h1,h2,h3,h4,h5,h6,h7,h8,h9);
f(h1,h2,h3,h4,h5,h6,h7,h8,h9) = Ad_H_sym;

%% K
num_K = 1000;
rnd_scale_K = 100;
rnd = rnd_scale_K*rand(num_K,3);
K_results = zeros(num_K,4);
K_color = zeros(num_K,3);
K_norm = zeros(num_K,1);
for i=1:num_K
    H = expm(hat_so3(rnd(i,:)));
    Ad_test = double(f(H(1,1),H(1,2),H(1,3),H(2,1),H(2,2),H(2,3),H(3,1),H(3,2),H(3,3)));
    D = kron(inv(H'),Ad_test)-eye(24);
    
    K_results(i,1) = rnd(i,1);
    K_results(i,2) = rnd(i,2);
    K_results(i,3) = rnd(i,3);
    K_results(i,4) = rank(D);
    K_norm(i,1) = norm(K_results(i,1:3));
    if K_results(i,4)<24
        K_results(i,5) = 0;
        K_color(i,:) = [0, 0.4470, 0.7410];
    elseif K_results(i,4) == 24
        H
        K_results(i,5) = 1;
        K_color(i,:) = [0.8500, 0.3250, 0.0980];
    end
end

K = figure(1);
S = repmat(25,num_K,1);
scatter3(K_results(:,1),K_results(:,2),K_results(:,3),S,K_color(:,:),'filled');
xlabel("x")
ylabel("y")
zlabel("z")

K2 = figure(2);
scatter(K_norm,K_results(:,5),S,K_color,'filled');

%% A
num_A = 100;
rnd_scale_A = 100;
% rnd = 