clc;
clear; 

syms v1 v2 v3 v4 v5 v6 v7 v8
assumeAlso([v1 v2 v3 v4 v5 v6 v7 v8],'real')

syms h1 h2 h3 h4 h5 h6 h7 h8 h9
assumeAlso([h1 h2 h3 h4 h5 h6 h7 h8 h9],'real')
%%

% E1 = [1, 0, 0; 0, -1, 0; 0, 0, 0];
% E2 = [0, 1, 0; 1, 0, 0; 0, 0, 0];
% E3 = [0, -1, 0; 1, 0, 0; 0, 0, 0];
% E4 = [1, 0, 0; 0, 1, 0; 0, 0, -2];
% E5 = [0, 0, 1; 0, 0, 0; 0, 0, 0];
% E6 = [0, 0, 0; 0, 0, 1; 0, 0, 0];
% E7 = [0, 0, 0; 0, 0, 0; 1, 0, 0];
% E8 = [0, 0, 0; 0, 0, 0; 0, 1, 0];

Ekx = [0, 0, 0;0, 0, -1;0, 1, 0];
Eky = [0, 0, 1;0, 0, 0;-1, 0, 0];
Ekz = [0, -1, 0;1, 0, 0;0, 0, 0];

Ea1 = [1,0,0;0,0,0;0,0,-1];
Ea2 = [0,0,0;0,1,0;0,0,-1];

Ea3 = [1,0,0;0,-1,0;0,0,0];

Enx = [0,0,1;0,0,0;0,0,0];
Eny = [0,0,0;0,0,1;0,0,0];
Enz = [0,1,0;0,0,0;0,0,0];

E1 = Ekx;
E2 = Eky;
E3 = Ekz;
E4 = Ea1;
E5 = Ea2;
E6 = Enx;
E7 = Eny;
E8 = Enz;

% E1 = [0, 0, 1; 0, 0, 0; 0, 0, 0];
% E2 = [0, 0, 0; 0, 0, 1; 0, 0, 0];
% E3 = [0, -1, 0; 1, 0, 0; 0, 0, 0];
% E4 = [0, 0, 0; 0, 0, 0; 0, 0, -1];
% E5 = [1, 0, 0; 0, -1, 0; 0, 0, 0];
% E6 = [0, 1, 0; 0, 0, 0; 0, 0, 0];
% E7 = [0, 0, 0; 0, 0, 0; 1, 0, 0];
% E8 = [0, 0, 0; 0, 0, 0; 0, 1, 0];

% E = {E1,E2,E3,E4,E5,E6,E7,E8};
E = {Ekx,Eky,Ekz,Ea1,Ea2,Enx,Eny,Enz};
% E = {Ekx,Eky,Ekz};

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

%% find Ad_Ei
Ad_E = {};
dAd_E = {};
for i=1:size(E,2)
    % Find Ad_E_i
    H = expm(E{i}); % Using numerical exponential for now
    Ad_E{i} = double(f(H(1,1),H(1,2),H(1,3),H(2,1),H(2,2),H(2,3),H(3,1),H(3,2),H(3,3)));
    dAd_E{i} = logm(Ad_E{i});
end

%% construct dro_3

dro_3 = {};
C = [];
for i =1:size(E,2)
   dro_3{i} = kron(-E{i}',eye(8))+kron(eye(3),dAd_E{i});
   C = [C;dro_3{i}];
end

%% solve for the null space
[U,S,V] = svd(C);


Q = null(C);

%%
H = expm(hat_sl3(rand(8,1)));
Ad_test = double(f(H(1,1),H(1,2),H(1,3),H(2,1),H(2,2),H(2,3),H(3,1),H(3,2),H(3,3)));

% w = Q*Q'*rand(24,1)*10;
w = Q*Q'*ones(24,1)*10;

W = reshape(w,[8,3]);

v_test = [2,3,1]';
H_v_test = H*v_test;

x_test = W*v_test;
x_H_test = W*H_v_test;
x_ad_test = Ad_test*x_test;
disp([x_H_test,x_ad_test])

%%
% rank(C,1e-10)

%% test solving for one h
% % a = E1+E2+E3+E4+E5+E6+E7+E8;
% 
% % a = Ekx+Eky+Ekz+Ea1+Ea2+Enx+Eny+Enz;
% 
% a = Ekx+Eky+Ekz+Ea1+Ea2+Enx+Eny+Enz;
% % a = Ea1+Ea2;
% % a = E4
% H = expm(a);
% % rnd = 100*rand(3);
% % x=rnd(1);
% % y=rnd(2);
% % z=rnd(3);
% % H = expm(x*Ekx+y*Eky+z*Ekz);
% 
% % rnd = 100*randn(3);
% % n1=rnd(1);
% % n2=rnd(2);
% % n3=rnd(3);
% % H = [1 n1 n2; 0 1 n3; 0 0 1];
% 
% % rnd = 2*randn(3);
% % a1=rnd(1);
% % a2=rnd(2);
% % H = [a1, 0,0; 0,a2,0;0,0,1/a1/a2];
% 
% % H = [0.5, 0,0; 0,2,0;0,0,1/0.5/2];
% Ad_test = double(f(H(1,1),H(1,2),H(1,3),H(2,1),H(2,2),H(2,3),H(3,1),H(3,2),H(3,3)));
% 
% D = kron(inv(H'),Ad_test)-eye(24);
% % D = kron(Ad_test,inv(H'))-eye(24);
% % [DU,DS,DV] = svd(D);
% 
% DQ = null(D);
% 
% rank(D)
% 
% wd = DQ*DQ'*ones(24,1);
% 
% Wd = reshape(wd,[8,3]);
% 
% v_test = [2,3,1]';
% H_v_test = H*v_test;
% 
% x_test = Wd*v_test;
% x_H_test = Wd*H_v_test;
% x_ad_test = Ad_test*x_test;
% disp([x_H_test,x_ad_test])
% disp(x_H_test - x_ad_test)

% rank([E1;E2;E3;E4;E5;E6;E7;E8])

%% 
