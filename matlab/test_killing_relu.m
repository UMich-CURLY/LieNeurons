clear;
clc;

%% sl(3) generators
G1 = [1,0,0;
      0, -1, 0;
      0,0,0];
G2 = [0, 1,0;
      1, 0 ,0;
      0, 0 ,0];
G3 = [0, -1, 0;
      1, 0, 0;
      0, 0, 0];
G4 = [1, 0, 0;
      0, 1, 0;
      0, 0, -2];
G5 = [0,0,1;
      0,0,0;
      0,0,0];
G6 = [0,0,0;
      0,0,1;
      0,0,0];
G7 = [0,0,0;
      0,0,0;
      1,0,0];
G8 = [0,0,0;
      0,0,0;
      0,1,0];
  
%%
num_sample = 100000;
out_sample = zeros(num_sample,2);
for i=1:num_sample

    a = 2*(rand(8,1)-0.5);
%     b = 2*(rand(8,1)-0.5);
    b = a;
    
    A = a(1)*G1+a(2)*G2+a(3)*G3+a(4)*G4+a(5)*G5+a(6)*G6+a(7)*G7+a(8)*G8;
    B = b(1)*G1+b(2)*G2+b(3)*G3+b(4)*G4+b(5)*G5+b(6)*G6+b(7)*G7+b(8)*G8;
    
    if(trace(A)~=0)
        print("no");
    end
    K = -6*trace(A*B);
    
    % relu
    if(K>0)
       out_sample(i,1)=K;
       out_sample(i,2)=K;
    else
       out_sample(i,1)=K;
       out_sample(i,2)=0;
    end
end


%%
figure(1)
scatter(out_sample(:,1),out_sample(:,2));

%%
G = [G1,G2,G3,G4,G5,G6,G7,G8];
a2 = A\G;
a3 = inv(G'*G)*G'*A;