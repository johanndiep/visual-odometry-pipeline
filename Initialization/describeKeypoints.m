function descriptors = describeKeypoints(img, keypoints, r)
% Returns a (2r+1)^2xN matrix of image patch vectors based on image
% img0 and a 2xN matrix containing the keypoint coordinates.
% r is the patch "radius".

N = size(keypoints, 2); % number of keypoints
descriptors = uint8(zeros((2*r+1) ^ 2, N)); % store all the patches here
padded = padarray(img, [r, r]); % pad such that we dont have to deal with cases where keypoint is closer to the image border than the descriptor patch radius

for i = 1:N
    kp = keypoints(:, i) + r; % because of padding
    descriptors(:,i) = reshape(padded(kp(1)-r:kp(1)+r, kp(2)-r:kp(2)+r), [], 1); % reshape it into the right form -> see exercise 3
end

end