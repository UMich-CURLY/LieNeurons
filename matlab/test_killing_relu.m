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
% 
% def sl3_to_R8(M):
%     # [a1 + a4, a2 - a3,    a5]
%     # [a2 + a3, a4 - a1,    a6]
%     # [     a7,      a8, -2*a4]
%     v = torch.zeros(8).to(M.device)
%     v[3] = -0.5*M[2,2]
%     v[4] = M[0,2]
%     v[5] = M[1,2]
%     v[6] = M[2,0]
%     v[7] = M[2,1]
%     v[0] = (M[0,0] - v[3])
% 
%     v[1] = 0.5*(M[0,1] + M[1,0])
%     v[2] = 0.5*(M[1,0] - M[0,1])
%     return v