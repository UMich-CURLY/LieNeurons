clear;
clc;

syms n1 n2 n3 a1 a2 v1 v2 v3 v4 v5 v6 v7 v8
assumeAlso([n1 n2 n3 a1 a2 v1 v2 v3 v4 v5 v6 v7 v8],'real')

syms h1 h2 h3 h4 h5 h6 h7 h8 h9
assumeAlso([h1 h2 h3 h4 h5 h6 h7 h8 h9],'real')
%% basis

E1 = [1, 0, 0; 0, -1, 0; 0, 0, 0];
E2 = [0, 1, 0; 1, 0, 0; 0, 0, 0];
E3 = [0, -1, 0; 1, 0, 0; 0, 0, 0];
E4 = [1, 0, 0; 0, 1, 0; 0, 0, -2];
E5 = [0, 0, 1; 0, 0, 0; 0, 0, 0];
E6 = [0, 0, 0; 0, 0, 1; 0, 0, 0];
E7 = [0, 0, 0; 0, 0, 0; 1, 0, 0];
E8 = [0, 0, 0; 0, 0, 0; 0, 1, 0];

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

%% find Ad_N
N = [1 n1 n2; 0 1 n3; 0 0 1];
Ad_N_hat = N*x_hat*inv(N);
Ad_N_hat_vec = reshape(Ad_N_hat,1,[])';

% solve for least square
x = inv(E_vec'*E_vec)*(E_vec')*Ad_N_hat_vec;
var = [v1,v2,v3,v4,v5,v6,v7,v8];
[Ad_N,b]=equationsToMatrix(x,var);

%% find Ad_A
A = [a1, 0,0; 0,a2,0;0,0,1/a1/a2];
Ad_A_hat = A*x_hat*inv(A);
Ad_A_hat_vec = reshape(Ad_A_hat,1,[])';

% solve for least square
x = inv(E_vec'*E_vec)*(E_vec')*Ad_A_hat_vec;
var = [v1,v2,v3,v4,v5,v6,v7,v8];
[Ad_A,b]=equationsToMatrix(x,var);

%% find h
H = [h1,h2,h3;h4,h5,h6;h7,h8,h9];
Ad_H_hat = H*x_hat*inv(H);
Ad_H_hat_vec = reshape(Ad_H_hat,1,[])';

% solve for least square
x = inv(E_vec'*E_vec)*(E_vec')*Ad_H_hat_vec;
var = [v1,v2,v3,v4,v5,v6,v7,v8];
[Ad_H,b]=equationsToMatrix(x,var);

Ad_kron_H = kron(inv(H)',Ad_H);

%%
% sigma = svd(Ad_kron_H);
% z = null(Ad_kron_H - eye(24))