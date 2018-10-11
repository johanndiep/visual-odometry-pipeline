function matches = matchDescriptors(query_descriptors, database_descriptors, lambda)
% Returns a 1xQ matrix where the i-th coefficient is the index of the
% database descriptor which matches to the i-th query descriptor.
% The descriptor vectors are MxQ and MxD where M is the descriptor
% dimension and Q and D the amount of query and database descriptors
% respectively. matches(i) will be zero if there is no database descriptor
% with an SSD < lambda * min(SSD). No two non-zero elements of matches will
% be equal.

[dists,matches] = pdist2(double(database_descriptors)', ...
double(query_descriptors)', 'euclidean', 'Smallest', 1); % gives out distance and matched index of the database descriptor

sorted_dists = sort(dists); % it should be sorted by default already
sorted_dists = sorted_dists(sorted_dists~=0); % neglect 0 distance
min_non_zero_dist = sorted_dists(1); % take minimum value
matches(dists >= lambda * min_non_zero_dist) = 0; % matches with distance above a threshold will be neglected

% remove double matches
unique_matches = zeros(size(matches));
[~,unique_match_idxs,~] = unique(matches, 'stable');
unique_matches(unique_match_idxs) = matches(unique_match_idxs);

matches = unique_matches;

end