% Returns the column-wise norm of a matrix in a new Matrix M_Norm

% Input:
% - M : (NxN) Matrix

% Output:
% - M_Norm: (1xN) Matrix with column-wise norms

% Source: http://stackoverflow.com/questions/7209521/vector-norm-of-an-array-of-vectors-in-matlab

function M_Norm = twoNormMatrix (M)
M_Norm = sqrt(sum( real(M .* conj(M)),  1 ));
end

