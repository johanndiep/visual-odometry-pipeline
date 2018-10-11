function [ T_F2F1 ] = tf2invtf( T_F1F2 )
% Returns the inv. homogeneous transformation
% Input:
%  - T_F1F2(4x4) : homogeneous transformation from Frame F2 to F1
%
% Output:
%  - T_F2F1(4x4) : homogeneous transformation from Frame F1 to F2

T_F2F1 = [T_F1F2(1:3,1:3)',  -T_F1F2(1:3,1:3)'*T_F1F2(1:3,4);
          zeros(1,3),                1];   

end
