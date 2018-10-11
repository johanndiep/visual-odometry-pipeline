function scores = harris(img, patch_size, kappa)

% first convolves each column of img with the first vector (sobel_orth'/sobel_para), 
% and then convolves each row of the result with the second vector (sobel_para/sobel_orth) 
% -> more efficient since filter is separable
% valid because it return only parts of the convolution that are computed without zero-padded edges
sobel_para = [-1 0 1];
sobel_orth = [1 2 1];
Ix = conv2(sobel_orth', sobel_para, img, 'valid'); 
Iy = conv2(sobel_para', sobel_orth, img, 'valid');

% Pixel-wise products and not matrix multiplications!
Ixx = double(Ix .^ 2);
Iyy = double(Iy .^ 2);
Ixy = double(Ix .* Iy);

% convolve with box filter to get entries of M, 
% alternative use Gaussian filter instead to avoid aliasing
% and give more weight to central pixel
patch = ones(patch_size, patch_size);
pr = floor(patch_size / 2);  % patch radius
sIxx = conv2(Ixx, patch, 'valid');
sIyy = conv2(Iyy, patch, 'valid');
sIxy = conv2(Ixy, patch, 'valid');

scores = (sIxx .* sIyy - sIxy .^ 2) ... % determinant
    - kappa * (sIxx + sIyy) .^ 2;  % kappa times square trace

scores(scores<0) = 0; % setting scores below 0 to 0

scores = padarray(scores, [1+pr 1+pr]); % pad the array such that it is the same size as the image

end
