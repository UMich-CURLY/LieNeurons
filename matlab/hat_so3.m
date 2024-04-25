function R = hat_so3(v)
    Ex = [0, 0, 0;0, 0, -1;0, 1, 0];
    Ey = [0, 0, 1;0, 0, 0;-1, 0, 0];
    Ez = [0, -1, 0;1, 0, 0;0, 0, 0];
    
    R = v(1)*Ex+v(2)*Ey+v(3)*Ez;
    
%     if ndims(v)==1
%         R = v(1)*Ex+v(2)*Ey+v(3)*Ez;
%     elseif ndims(v) > 1
%         R = v(:,1).*Ex+v(:,2).*Ey+v(:,3).*Ez;
%     end
    
end