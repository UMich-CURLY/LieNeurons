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

% solve for least square
x = inv(E_vec'*E_vec)*(E_vec')*Ad_H_hat_vec;
var = [v1,v2,v3,v4,v5,v6,v7,v8];
[Ad_H_sym,b]=equationsToMatrix(x,var);

syms f(h1,h2,h3,h4,h5,h6,h7,h8,h9);
f(h1,h2,h3,h4,h5,h6,h7,h8,h9) = Ad_H_sym;

%% find Ad_Ei
Ad_E = {};
dAd_E = {};
for i=1:8
    % Find Ad_E_i
    H = expm(E{i}); % Using numerical exponential for now
    Ad_E{i} = double(f(H(1,1),H(1,2),H(1,3),H(2,1),H(2,2),H(2,3),H(3,1),H(3,2),H(3,3)));
    dAd_E{i} = logm(Ad_E{i});
end

%% construct dro_3

dro_3 = {};
C = [];
for i =1:8
   dro_3{i} = kron(-E{i}',eye(8))+kron(eye(3),dAd_E{i});
   C = [C;dro_3{i}];
end

%% 
C_minus_I = C-eye(size(C));
rank(C_minus_I)
[U,S,V] = svd(C);


Q = null(C);





