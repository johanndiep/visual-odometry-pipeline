function keypoints = selectKeypoints(scores, num, r)
% Selects the num best scores as keypoints and performs non-maximum 
% supression of a (2r + 1)*(2r + 1) box around the current maximum.

keypoints = zeros(2, num); % array to store the keypoints
temp_scores = padarray(scores, [r r]); % pad the array such that there is no problem with the neighborhood zero setting

for i = 1:num
    % get coordinate of the keypoint in the padded array
    [~, kp] = max(temp_scores(:));
    [row, col] = ind2sub(size(temp_scores), kp);
    kp = [row;col];
    
    keypoints(:, i) = kp - r; % remove the boundary created by the padarray
    temp_scores(kp(1)-r:kp(1)+r, kp(2)-r:kp(2)+r) = zeros(2*r + 1, 2*r + 1); % set neighborhood to zero
end

end