function [E, inliers_index] = eightPointRansac(p1, p2, K1, K2, max_error)
% estimateEssentialMatrix_normalized: estimates the essential matrix
% given matching point coordinates, and the camera calibration K
%
% Input: point correspondences
%  - p1(3,N): homogeneous coordinates of 2-D points in image 1
%  - p2(3,N): homogeneous coordinates of 2-D points in image 2
%  - K1(3,3): calibration matrix of camera 1
%  - K2(3,3): calibration matrix of camera 2
%
% Output:
%  - E(3,3) : fundamental matrix
%

error_method = 1;
num_iter = 2000;
num_points = size(p1, 2);
k = 8; % sample_size

% Initialize RANSAC.
best_inliers = zeros(1, num_points);
% matched_query_keypoints = flipud(matched_query_keypoints);
max_num_inliers_history = zeros(1, num_iter);
max_num_inliers = 0;
% Replace the following with the path to your camera projection code:
% addpath('../../01_camera_projection/code');

% RANSAC
for i = 1:num_iter
    % Model from k samples (DLT or P3P)
    [p1_sample, idx] = datasample(p1, k, 2, 'Replace', false);
    p2_sample = p2(:, idx);
    
    F_estimate = fundamentalEightPoint_normalized(p1_sample, p2_sample);
    
    % Calculate error. TODO: replace with reprojection error
    if error_method == 0
        homog_points = [p1, p2];
        epi_lines = [F_estimate.'*p2, F_estimate*p1];
        denom = epi_lines(1,:).^2 + epi_lines(2,:).^2;
        cost = sqrt((sum(epi_lines.*homog_points,1).^2)./denom );
        errors = sum(reshape(cost,num_points,2)',1);
    elseif error_method == 1
        a1 = F_estimate*p1; % epipolar lines in image 2
        a2 = F_estimate.'*p2; % epipolar lines in image 1
        Numer = sum(p2.*a1, 1).^2; 
        Denom = a1(1,:).^2 + a1(2,:).^2 + a2(1,:).^2 + a2(2,:).^2;
        dSquared = Numer./Denom;
        errors = sqrt(dSquared);
    end
    
    inliers = errors <= max_error;
    num_inliers = nnz(inliers);
    
    if num_inliers > max_num_inliers
        max_num_inliers = num_inliers;
        best_inliers = inliers;
        max_num_inliers_history(i) = num_inliers;
    end
    
    max_num_inliers_history(i) = max_num_inliers;
    
end

if max_num_inliers == 0
    error('no inliers found!');
else
%     p1_inliers = p1(:,best_inliers);
%     p2_inliers = p2(:,best_inliers);
    inliers_index = best_inliers;
    best_F = fundamentalEightPoint_normalized(p1(:,inliers_index), p2(:,inliers_index));

    E = K2'*best_F*K1;
end

end