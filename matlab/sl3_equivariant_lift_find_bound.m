clc;
clear; 

compute_K = false;
num_K = 1000;
rnd_scale_K = 100;

compute_A = false;
num_A = 1000;
rnd_scale_A = 100;


compute_N = false;
num_N = 1000;
rnd_scale_N = 100;

compute_KN = true;
num_KN = 1000;
rnd_scale_KN = 1;

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
if compute_K
    rnd = rnd_scale_K*rand(num_K,3);
    tol = 1e-8;
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
        K_results(i,4) = rank(D,tol);
        K_norm(i,1) = norm(K_results(i,1:3));
        if K_results(i,4)<24
            K_results(i,5) = 0;
            K_color(i,:) = [0, 0.4470, 0.7410];
        elseif K_results(i,4) == 24
    %         H
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
end
%%
% idx = K_results(:,5) == 1;
% no_sol_K = K_results(idx,:);
% no_sol_norm = K_norm(idx,:);
% no_sol_norm_mod_pi = mod(no_sol_norm,pi);
% K3 = figure(3);
% histogram(no_sol_norm_mod_pi,314);

%% A
if compute_A
    rnd = rnd_scale_A*rand(num_A,2);
    tol = 1e-8;
    A_results = zeros(num_A,4);
    A_color = zeros(num_A,3);
    A_norm = zeros(num_A,1);
    for i = 1:num_A
        a1 = rnd(i,1);
%         a2 = rnd(i,2);
        a2 = 1/a1;
        H = [a1, 0,0; 0,a2,0;0,0,1/a1/a2];
%         H = [a1,0,0; 0,1/a1,0;0,0,1];
        
        Ad_test = double(f(H(1,1),H(1,2),H(1,3),H(2,1),H(2,2),H(2,3),H(3,1),H(3,2),H(3,3)));
        D = kron(inv(H'),Ad_test)-eye(24);

        A_results(i,1) = a1;
        A_results(i,2) = a2;
        A_results(i,4) = rank(D,tol);
        A_norm(i,1) = norm(A_results(i,1:2));
        if A_results(i,4)<24
            A_results(i,5) = 0;
            A_color(i,:) = [0, 0.4470, 0.7410];
        elseif A_results(i,4) == 24
%             H
            A_results(i,5) = 1;
            A_color(i,:) = [0.8500, 0.3250, 0.0980];
        end
    end

    A_figure1 = figure(1);
    S = repmat(25,num_A,1);
    scatter(A_results(:,1),A_results(:,2),S,A_color(:,:),'filled');
    xlabel("x")
    ylabel("y")
    zlabel("z")
    title("Solution existence for A")

    A_figure2 = figure(2);
    scatter(A_norm,A_results(:,5),S,A_color,'filled');
    
    
    idx_A = A_results(:,5) == 1;
    no_sol_A = A_results(idx_A,:);
end

%% N
if compute_N
    rnd = rnd_scale_N*rand(num_N,3);
    tol = 1e-8;
    N_results = zeros(num_N,4);
    N_color = zeros(num_N,3);
    N_norm = zeros(num_N,1);
    for i = 1:num_N
        n1 = rnd(i,1);
        n2 = rnd(i,2);
        n3 = rnd(i,3);
        H = [1 n1 n2; 0 1 n3; 0 0 1];
        
        Ad_test = double(f(H(1,1),H(1,2),H(1,3),H(2,1),H(2,2),H(2,3),H(3,1),H(3,2),H(3,3)));
        D = kron(inv(H'),Ad_test)-eye(24);

        N_results(i,1) = n1;
        N_results(i,2) = n2;
        N_results(i,3) = n3;
        N_results(i,4) = rank(D,tol);
        N_norm(i,1) = norm(N_results(i,1:3));
        if N_results(i,4)<24
            N_results(i,5) = 0;
            N_color(i,:) = [0, 0.4470, 0.7410];
        elseif N_results(i,4) == 24
%             H
            N_results(i,5) = 1;
            N_color(i,:) = [0.8500, 0.3250, 0.0980];
        end
    end

    N_figure1 = figure(1);
    S = repmat(25,num_N,1);
    scatter3(N_results(:,1),N_results(:,2),N_results(:,3),S,N_color(:,:),'filled');
    xlabel("x")
    ylabel("y")
    zlabel("z")
    title("Solution existence for N")

    N_figure2 = figure(2);
    scatter(N_norm,N_results(:,5),S,N_color,'filled');
    
    
    idx_N = N_results(:,5) == 1;
    no_sol_N = N_results(idx_N,:);
end

%% KN
if compute_KN
    rnd = rnd_scale_KN*rand(num_KN,3);
    rnd2 = rnd_scale_KN*rand(num_KN,3);
    tol = 1e-8;
    KN_results = zeros(num_KN,4);
    KN_color = zeros(num_KN,3);
    KN_norm = zeros(num_KN,1);
    for i = 1:num_KN
        n1 = rnd(i,1);
        n2 = rnd(i,2);
        n3 = rnd(i,3);
        N = [1 n1 n2; 0 1 n3; 0 0 1];
        K = expm(hat_so3(rnd2(i,:)));

        H = K*N;

        Ad_test = double(f(H(1,1),H(1,2),H(1,3),H(2,1),H(2,2),H(2,3),H(3,1),H(3,2),H(3,3)));
        D = kron(inv(H'),Ad_test)-eye(24);

        KN_results(i,1) = n1;
        KN_results(i,2) = n2;
        KN_results(i,3) = n3;
        KN_results(i,4) = rank(D,tol);
        KN_results(i,6) = rnd2(i,1);
        KN_results(i,7) = rnd2(i,2);
        KN_results(i,8) = rnd2(i,3);
        KN_norm(i,1) = norm(KN_results(i,1:3));
        if KN_results(i,4)<24
            KN_results(i,5) = 0;
            KN_color(i,:) = [0, 0.4470, 0.7410];
        elseif KN_results(i,4) == 24
%             H
            KN_results(i,5) = 1;
            KN_color(i,:) = [0.8500, 0.3250, 0.0980];
        end
    end

    KN_figure1 = figure(1);
    S = repmat(25,num_N,1);
    scatter3(KN_results(:,1),KN_results(:,2),KN_results(:,3),S,KN_color(:,:),'filled');
    xlabel("x")
    ylabel("y")
    zlabel("z")
    title("Solution existence for KN")

    N_figure2 = figure(2);
    scatter(KN_norm,KN_results(:,5),S,KN_color,'filled');
    
    
    idx_KN = KN_results(:,5) == 1;
    no_sol_KN = KN_results(idx_KN,:);
end