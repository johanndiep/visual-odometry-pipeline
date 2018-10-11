function [bearing_angle_deg] = calcBearingangle(kp_img1, kp_img2, K)
% Returns pair-wise angles between keypoint rays.
%
% Input:
%  - kp_img1(2xN) : keypoints image 1 [v u]
%  - kp_img2(2xN) : keypoints image 2 [v u]
%  - K(3x3) : camera calibration matrix
%
% Output:
%  - bearing_angle_deg(1xN) : in-between angles between kp pairs

vector_first = [kp_img1; repmat(K(1,1), [1, size(kp_img1,2)])];
vector_j = [kp_img2; repmat(K(1,1), [1, size(kp_img2,2)])];

bearing_angle_deg = atan2d(twoNormMatrix(cross(vector_j, vector_first)), dot(vector_j, vector_first));

end