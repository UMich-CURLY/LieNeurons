


true_y0 = [2., 1.,3.0];
t = [0:0.025:25];


[t,y] = ode45(@EulerPoincare, t, true_y0);


f1 = figure(1);
plot(t,y);
legend("x","y","z");
xlabel("time")
ylabel("x,y,z")

f2 = figure(2);
plot(y(:,1),y(:,2));
xlabel("x")
ylabel("y")



function w_wedge = wedge(w)
  w_wedge = zeros(3,3);
  w_wedge(1,2) = -w(3);
  w_wedge(1,3) = w(2);
  w_wedge(2,1) = w(3);
  w_wedge(2,3) = -w(1);
  w_wedge(3,1) = -w(2);
  w_wedge(3,2) = w(1);
end

function dwdt = EulerPoincare(t,w)
  I = [[12, 0, 0];[0, 20., 0];[0, 0, 5.]];
%   I = [[12, -5., 7.];[-5., 20., -2.];[7., -2., 5.]];
  w_wedge = wedge(w);
  
  dwdt = -I\w_wedge*I*w;
end