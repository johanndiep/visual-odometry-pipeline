%% Setup
clc
clear all

%% Actual Thresholds (to be tuned)
% Bearing angle: Limit when to triangulate
bearingangle_limit_normal=2.0;
bearingangle_limit_lower=0.1;

% Delete Landmarks with z coord <0 or >upperlimit_zcoord
upperlimit_zcoord=100000;
% New Keypoint when pixel in difference in x and y coord to existing 
% ones is larger then pixeldiff_keypoint
pixeldiff_keypoint=2;

%Harris parameter
harris_patch_size = 9; % taken from exercise 3
harris_kappa = 0.08; % magic number taken from exercise 3
num_keypoints = 400; % tune this in trade-off with computational power
nonmaximum_supression_radius = 8; % taken from exercise 3
%descriptor_radius = 9;
%match_lambda = 5;

ransac_limit=0.1;
%others: KLT

%video recording on/off
record_frames = false;

%last_frame_boot=200;

%%
ds = 3; % 0: KITTI, 1: Malaga, 2: parking, 3: iphone
addpath('Initialization');
setPaths();
if ds == 0
    % need to set kitti_path to folder containing "00" and "poses"
    assert(exist('kitti_path', 'var') ~= 0);
    ground_truth = load([kitti_path '/poses/00.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
elseif ds == 1
    % Path containing the many files of Malaga 7.
    assert(exist('malaga_path', 'var') ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    % Path containing images, depths and all...
    assert(exist('parking_path', 'var') ~= 0);
    last_frame = 598;
    K = load([parking_path '/K.txt']);
     
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
elseif ds == 3
    K = [3298.58359 0 2017.62693
        0 3288.0025 1533.14381
        0 0 1];
    assert(exist('iphone_path', 'var') ~= 0);
    last_frame = 426;
else
    assert(false);
end

%allocate for video recording
if record_frames
%     recording(last_frame) = struct('cdata',[],'colormap',[]);
    v = VideoWriter('myVideo.avi');
    open(v);
end

%% Bootstrap
% need to set bootstrap_frames
bootstrap_frames(1)=1;
bootstrap_frames(2)=3;
if ds == 0
    img0 = imread([kitti_path '/00/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(1))]);
    img1 = imread([kitti_path '/00/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(2))]);
elseif ds == 1
    img0 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(1)).name]));
    img1 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(2)).name]));
elseif ds == 2
    img0 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(1))]));
    img1 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(2))]));
elseif ds == 3
    img0 = flipud(rgb2gray(imread([iphone_path ...
        sprintf('/IMG_%04d.JPG',bootstrap_frames(1)+2908)])));
    img0 = imresize(img0, 0.25);
    img1 = flipud(rgb2gray(imread([iphone_path ...
        sprintf('/IMG_%04d.JPG',bootstrap_frames(2)+2908)])));
    img1 = imresize(img1, 0.25);
else
    assert(false);
end

%% Bootstrap: keypoint detection and matching

% function that calculates the harris score for each pixel
harris_scores = harris(img0, harris_patch_size, harris_kappa);

% function that selects the keypoints, coordinates
% and keypoint index in degrading intensity 
% -> [2, num_keypoints]
keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius);

% % function that calculates the descriptors (single patches) for each
% % keypoint, patch in reshaped form and keypoint index (in degrading intensity) 
% % -> [(2*descriptor_radius+1) ^ 2, num_keypoints] 
% descriptors = describeKeypoints(img0, keypoints, descriptor_radius);
% 
% % function that matches the descriptors from img0 and img1, matches has
% % same length as descriptors_2 and consists of indexes of matched keypoint
% % from img0
% % -> [(2*descriptor_radius+1) ^ 2,1]
% harris_scores_2 = harris(img1, harris_patch_size, harris_kappa);
% keypoints_2 = selectKeypoints(harris_scores_2, num_keypoints, nonmaximum_supression_radius);
% descriptors_2 = describeKeypoints(img1, keypoints_2, descriptor_radius);
% matches = matchDescriptors(descriptors_2, descriptors, match_lambda);


% switch from (y,x) to (x,y)
keypoints=flipud(keypoints);

%% Initialization of keypoints with KLT

klt_tracker = vision.PointTracker('NumPyramidLevels', 4, 'MaxBidirectionalError', 2); 
%2 Pixels error allowed (back projection)
%4 pyramidlevels-> work at higher resolution->can handle larger
%displacements (1-4 recommended)

% initialize tracker with the query kp locations
initialize(klt_tracker, keypoints', img0);

% track keypoints
[kp_tracked, validIdx, ~] = step(klt_tracker, img1);
matched_kp_tracks = kp_tracked(validIdx, :)'; %tracked keypoints in new image

idx_p1 = find(validIdx'); %index of database keypoints that were tracked to new image
p1=[keypoints(:,idx_p1);ones(length(matched_kp_tracks),1)'];
p2=[matched_kp_tracks;ones(length(matched_kp_tracks),1)'];
     
%verifiy matches
subplot(1,2,1)
imshow(img0,[]);
hold on
plot(p1(1,:),p1(2,:), 'ys');
title('Image 1')

subplot(1,2,2)
imshow(img1,[]);
hold on
plot(p2(1,:), p2(2,:), 'ys');
title('Image 2')


%% Bootstrap: Find relative poses 

% function to obtain E [3x3] (8-point-Algorithm)
% E = estimateEssentialMatrix(p1, p2, K, K); %p1 & p2 Point correspondances [3xnum_pointorrespond]
% TODO: parameters for estimateFundamentalMatrix may require tweaking
% [F_ransac, inliers] = estimateFundamentalMatrix(p1(1:2,:)', p2(1:2,:)','Method','RANSAC','NumTrials',2000,'DistanceThreshold',1e-4);
[E_ransac, inliers] = eightPointRansac(p1, p2, K, K, ransac_limit);
p1 = p1(:, inliers);
p2 = p2(:, inliers);
% E_ransac = K'*F_ransac*K;
%F_no_ransac = K'\E/K;

% function to obtain extrinsic parameters (Rots,t) [3x3x2 and 3x1] from E
%[Rots,u3] = decomposeEssentialMatrix(E); % Rots are 2 possible solutions
[Rots_ransac,u3_ransac] = decomposeEssentialMatrix(E_ransac); % Rots are 2 possible solutions
 
% function to get the relative camera pose (R and t) [3x3 and 3x1]
% Disambiguate among the four possible configurations
% you get the relative poses between the 2 camera frames
%from W(=C1) to C2
%[R_C2_W,T_C2_W] = disambiguateRelativePose(Rots,u3,p1,p2,K,K);
[R_C2_W_ransac,T_C2_W_ransac] = disambiguateRelativePose(Rots_ransac,u3_ransac,p1,p2,K,K);
camerapose=[R_C2_W_ransac T_C2_W_ransac; 0 0 0 1];
 
%% Bootstrap: Triangulate landmarks
% function to triangulate 3D Points from point corespondances p1&p2
% Triangulate a point cloud using the final transformation (R,T)
M1 = K * eye(3,4);                    % camera 1 at world frame
%M2 = K * [R_C2_W, T_C2_W];
M2_ransac= K * [R_C2_W_ransac, T_C2_W_ransac];
%P = linearTriangulation(p1,p2,M1,M2); %[4xnum_pointcorrespond, last col ones]
P_ransac=linearTriangulation(p1,p2,M1,M2_ransac); %[4xnum_pointcorrespond, last col ones]

%% Manually remove outliers
%remove negative Landmarks
validPoints = P_ransac(3,:) > 0; 
P_ransac=P_ransac(:,validPoints);
p1=p1(:,validPoints);
p2=p2(:,validPoints);
%remove landmarks too far away
validPoints2 = P_ransac(3,:) < upperlimit_zcoord;
P_ransac=P_ransac(:,validPoints2);
p1=p1(:,validPoints2);
p2=p2(:,validPoints2);

%% Visualize the 3-D scene

figure(1),
subplot(1,3,1)

% P is a [4xN] matrix containing the triangulated point cloud (in
% homogeneous coordinates), given by the function linearTriangulation
% plot3(P(1,:), P(2,:), P(3,:), 'o');
plot3(P_ransac(1,:), P_ransac(2,:), P_ransac(3,:), 'o');

% Display camera pose
plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');

% center_cam2_W = -R_C2_W'*T_C2_W;
center_cam2_W = -R_C2_W_ransac'*T_C2_W_ransac;
plotCoordinateFrame(R_C2_W_ransac',center_cam2_W, 0.8);
text(center_cam2_W(1)-0.1, center_cam2_W(2)-0.1, center_cam2_W(3)-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');

axis equal
rotate3d on;
grid

% Display matched points
subplot(1,3,2)
imshow(img0,[]);
hold on
plot(p1(1,:), p1(2,:), 'ys');
title('Image 1')

subplot(1,3,3)
imshow(img1,[]);
hold on
plot(p2(1,:), p2(2,:), 'ys');
title('Image 2')


%% Continuous operation
% p1: picture coordinates in prev_img
% p2: picture coordinates in image

% Initialize state S
S.P=p2(1:2,:);         % Keypoints in use (current frame) [2xnumkeypoints]
S.X=P_ransac(1:3,:);   % 3D Landmarks to current keypoints P [3xnumkeypoints]
S.T=camerapose(:);     % History of acutal absolute camera poses world to Ci [16xnumerRuns]
S.C=[];                % candidate keypoints (in current frame coordinates) [2xnumCandidates]
S.TC=[];               % absolute camera pose world to Ci for each candidatekeypoint at first observation [16xnumCandidates]
S.F=[];                % coordinates of candidate keypoints at first observation [2xnumCandidates]

%initialize prev_img
prev_img=img1;

%initialize cameracentre matrix:
cameracentre=[];
    
range = (bootstrap_frames(2)+1):last_frame; %last_frame_boot
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if ds == 0
        image = imread([kitti_path '/00/image_0/' sprintf('%06d.png',i)]);
    elseif ds == 1
        image = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
    elseif ds == 2
        image = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
    elseif ds == 3
        image = flipud(rgb2gray(imread([iphone_path ...
            sprintf('/IMG_%04d.JPG',i+2908)])));
        image = imresize(image, 0.25);
    else
        assert(false);
    end
    pause(0.01);
    
%Define new possible candidate keypoints 
    harris_scores_candidate = harris(prev_img, harris_patch_size, harris_kappa);
    keypoints_candidate = selectKeypoints(harris_scores_candidate, num_keypoints, nonmaximum_supression_radius);
    keypoints_candidate(1:2,:)=flipud(keypoints_candidate(1:2,:));
    
%KLT
    %Track actual keypoints P from state vector
       % klt_tracker = vision.PointTracker('NumPyramidLevels', 4, 'MaxBidirectionalError', 2);
        %2 Pixels error allowed
        %4 pyramidlevels-> work at higher resolution->can handle larger
        %displacements (1-4 recommended)

        % track keypoints
        %initialize(klt_tracker, S.P', prev_img); % initialize tracker with the query kp locations
        setPoints(klt_tracker,S.P')

        [kp_tracked, validIdx, ~] = step(klt_tracker, image);
        matched_kp_tracks = kp_tracked(validIdx, :)';

        idx_p1 = find(validIdx'); %index of p1 keypoints
        p1_boot=[S.P(:,idx_p1);ones(length(matched_kp_tracks),1)'];     %tracked keypoints in previous frame coords
        p2_boot=[matched_kp_tracks;ones(length(matched_kp_tracks),1)']; %tracked keypoints in actual frame coords
        
     %Track existing candidate keypoints C from state vector
     if i==bootstrap_frames(2)+1
      klt_tracker_candidates=vision.PointTracker('NumPyramidLevels', 4, 'MaxBidirectionalError', 2);
      initialize(klt_tracker_candidates, keypoints_candidate', prev_img);
     end
        if nnz(S.C)>0    %if not empty
           %release(klt_tracker_candidates); %needed in order to reinitialize klt tracker
          %initialize(klt_tracker_candidates, S.C', prev_img);
          setPoints(klt_tracker_candidates,S.C')

            % track keypoints
            [kp_candidateExist_tracked, validIdx_candidateExist, ~] = step(klt_tracker_candidates, image);
            matched_kp_candidateExist_tracks = kp_candidateExist_tracked(validIdx_candidateExist, :)';

            idx_p1_candidateExist = find(validIdx_candidateExist'); %index of p1 keypoints
            p1_boot_candidateExist=[S.C(:,idx_p1_candidateExist);ones(length(matched_kp_candidateExist_tracks),1)']; %tracked keypoints in previous frame coords
            p2_boot_candidateExist=[matched_kp_candidateExist_tracks;ones(length(matched_kp_candidateExist_tracks),1)']; %tracked keypoints in actual frame coords
            S.F=S.F(:,idx_p1_candidateExist);  %remove not tracked ones also from state
            S.TC=S.TC(:,idx_p1_candidateExist); %remove not tracked ones also from state
        end
    
    %Track candidate keypoints
       
        %release(klt_tracker_candidates);
        %initialize(klt_tracker_candidates, keypoints_candidate', prev_img);
        setPoints(klt_tracker_candidates,keypoints_candidate')
        
        [kp_candidate_tracked, validIdx_candidate, ~] = step(klt_tracker_candidates, image);
        matched_kp_candidate_tracks = kp_candidate_tracked(validIdx_candidate, :)';

        idx_p1_candidate = find(validIdx_candidate'); %index of p1 keypoints
        p1_boot_candidate=[keypoints_candidate(:,idx_p1_candidate);ones(length(matched_kp_candidate_tracks),1)']; %tracked keypoints in previous frame coords
        p2_boot_candidate=[matched_kp_candidate_tracks;ones(length(matched_kp_candidate_tracks),1)']; %tracked keypoints in actual frame coords
  
   
%Ransac for existing keypoints P, existing candidate keypoints C
%and new candidate keypoints
    % function to obtain E [3x3] (8-point-Algorithm)
    % [F_ransac_boot,index_inlier_boot] = estimateFundamentalMatrix(p1_boot(1:2,:)', p2_boot(1:2,:)','Method','RANSAC');%,'NumTrials',2000,'DistanceThreshold',1e-4);
    % E_ransac_boot = K'*F_ransac*K;
    [E_ransac_boot, index_inlier_boot] = eightPointRansac(p1_boot, p2_boot, K, K, ransac_limit);
    if nnz(S.C)>0
        [E_ransac_boot_candidateExist, index_inlier_boot_candidateExist] = eightPointRansac(p1_boot_candidateExist, p2_boot_candidateExist, K, K, ransac_limit);
    end
     [E_ransac_boot_candidate, index_inlier_boot_candidate] = eightPointRansac(p1_boot_candidate, p2_boot_candidate, K, K, ransac_limit);

    %keep only inliers
    p1_boot=p1_boot(:,index_inlier_boot);
    p2_boot=p2_boot(:,index_inlier_boot);
    if nnz(S.C)>0
    p1_boot_candidateExist=p1_boot_candidateExist(:,index_inlier_boot_candidateExist);
    p2_boot_candidateExist=p2_boot_candidateExist(:,index_inlier_boot_candidateExist);
    S.C=p2_boot_candidateExist(1:2,:);
    S.F=S.F(:,index_inlier_boot_candidateExist);
    S.TC=S.TC(:,index_inlier_boot_candidateExist);
    end
    p1_boot_candidate=p1_boot_candidate(:,index_inlier_boot_candidate);
    p2_boot_candidate=p2_boot_candidate(:,index_inlier_boot_candidate);
    
% Save updated landmarks: leave only the ones whos keypoints were tracked
    landmarks_boot=S.X(:,idx_p1);   %what stayed after KLT
    landmarks_boot=landmarks_boot(:,index_inlier_boot); %what stayed after Ransac

    
% New Poses   
    % Obtain R&t actual relative frame (mapping from C1 to C2)
    [Rots_ransac_boot,u3_ransac_boot] = decomposeEssentialMatrix(E_ransac_boot); % Rots are 2 possible solutions
    [R_C2_C1_ransac_boot,T_C2_C1_ransac_boot] = disambiguateRelativePose(Rots_ransac_boot,u3_ransac_boot,p1_boot,p2_boot,K,K);
    
    % construct new camera pose
    T_C1W=reshape(S.T(:,end),[4,4]);
    T_WC1 = tf2invtf(T_C1W); 
    T_C2C1 = [R_C2_C1_ransac_boot   T_C2_C1_ransac_boot; % from C1 to C2
              zeros(1,3)       1];
    %T_C1C2 = tf2invtf(T_C2C1);                 % from C2 to C1
    T_C2W = T_C2C1 * T_C1W;                    %from world to C2
    %T_WC2 = tf2invtf(T_C2W); 
    

% New Triangulation 
    
    if nnz(S.C)>0
      if length(S.X)<100 %less than 100 landscape points left
            bearingangle_limit=bearingangle_limit_lower; %lower threshold to triangulate enough points
      else 
            bearingangle_limit=bearingangle_limit_normal;
      end  
      bearingangle= calcBearingangle(S.C,S.F,K); %angles between all candidate keypoints and their first observation
      threshold_reached = bearingangle > bearingangle_limit; 
      index_reached=find(threshold_reached>0);   % index of candidate keypoints that reached large enough distance to be triangulated
      P_ransac_boot=[]; % New triangulated Points
      p1_newtriang=[];  % coords of this keypoints in frames of their first observation
      p2_newtriang=[];  % coords of this keypoints in the actual frame
      for j=1:length(index_reached)
           TC_matrix=reshape(S.TC(:,index_reached(j)), [4,4] ); % pose of this keypoints at their first observation
           M1_boot = K* TC_matrix(1:3,:);               % M for Camera pose at first observation expressed in world
           M2_ransac_boot= K * T_C2W(1:3,:);            % M for Actual camera pose expressed in world
           p1_newtriang=[S.F(:,index_reached(j));1];
           p2_newtriang=[S.C(:,index_reached(j));1];
           P_ransac_boot(:,j)=linearTriangulation(p1_newtriang,p2_newtriang,M1_boot,M2_ransac_boot); %[4xnum_pointcorrespond, last col ones]
      end   
     
    end

% check if new tracked new candidate keypoint is already in S.C (after KLT
% and Ransac)
% if not-> write them to newkeypoints
    newkeypoints=[];
    if nnz(S.C)==0
        Compare=[0;0]; %in order to avoid crash when no C in state
    else
        Compare=S.C;
    end
    for j=1:length(p1_boot_candidate)
        %test one new keypoint to all other existing keypoints
        differencevector=abs(Compare-p1_boot_candidate(1:2,j));
        %check if difference in x and y is smaller than 2 pixel
        %this 2 can be updated aswell
        testx=differencevector(1,:) > pixeldiff_keypoint;
        testy=differencevector(2,:) > pixeldiff_keypoint;

        %check if both differences are smaller
        newkeypoint=true;
        for k=1:length(testx)
            if testx(k)~= 1 && testy(k)~= 1
                newkeypoint=false;
            end
        end

        %Save the non redundant new keypoints
        if  newkeypoint 
            newkeypoints=[newkeypoints p2_boot_candidate(1:2,j)];
        end
    end
    
%Update State and parameters (shrinkes over time without adding new keypoints)
    prev_img=image;
    if exist('bearingangle') %if new triangulated points
    S.P=[p2_boot(1:2,:) S.C(:,bearingangle>bearingangle_limit)]; % save tracked old ones and add the used candidate keypoints for triangulation
    if isempty(P_ransac_boot) 
         S.X=[landmarks_boot];
    else
    S.X=[landmarks_boot P_ransac_boot(1:3,:)];% save the ones of the old tracked keypoints and add new triangulated Landmarks from candidate keypoints
    end
    keepvector= bearingangle < bearingangle_limit;
    S.C=S.C(:,keepvector); %remove candidate keypoints use for triangulation
    S.F=S.F(:,keepvector); %remove candidate keypoints use for triangulation
    S.TC=S.TC(:,keepvector); %remove candidate keypoints use for triangulation
    end
    S.T=[S.T T_C2W(:)]; %new absolute camera pose in actual frame

%Add new keypoints and their first observation/pose at first observation to state 
   S.C=[S.C newkeypoints];
   S.TC=[S.TC repmat(T_C2W(:),1,length(newkeypoints))]; %absolute from W to Camera j
   S.F=[S.F newkeypoints];
   
% Manually remove outliers
    %remove negative Landmarks
    validPoints_boot = S.X(3,:) > 0; 
    S.X=S.X(:,validPoints_boot);
    S.P=S.P(:,validPoints_boot);
    %remove landmarks too far away
    validPoints_boot2 = S.X(3,:) < upperlimit_zcoord; 
    S.X=S.X(:,validPoints_boot2);
    S.P=S.P(:,validPoints_boot2);
    
% Plot and update plot with each new frame
    T_actual=reshape(S.T(:,i-2),[4,4]); % i-1 because we are missing one image at initialisation
    cameracentre(:,i)= -T_actual(1:3,1:3)'*T_actual(1:3,4);
    plotAll();
end
close(v);


%% Visualize the 3-D scene continuous

%calculate camera centres
% cameracentre=[];
% for i=1:size(S.T,2)
%     T_actual=reshape(S.T(:,i),[4,4]);
%     cameracentre(:,i)= -T_actual(1:3,1:3)'*T_actual(1:3,4);
%     %cameracentre(:,i)= T_actual(1:3,1:3)*T_actual(1:3,4);
% end

figure(4)
plot3(cameracentre(1,:),cameracentre(2,:),cameracentre(3,:), 'o')
axis equal
rotate3d on;
grid
title('Camera poses over time')
% 
figure(1)
hold on
%plot(cameracentre(3,:), cameracentre(1,:),'r');
plot(cameracentre(2,:), cameracentre(3,:),'r');
%plot(ground_truth(1:(length(range)+2),2)',ground_truth(1:(length(range)+2),1)','b');
axis equal
grid
title('Camera poses over time')
%ylim([-2 2])

figure(2)
subplot(1,3,1)


%actual 3D Landmarks
plot3(S.X(1,:), S.X(2,:), S.X(3,:), 'o');


% Display camera pose 0
plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
text(-0.1,-0.1,-0.1,'Cam 0','fontsize',10,'color','k','FontWeight','bold');
% Display camera pose last
T_last=reshape(S.T(:,end),[4,4]);
center_cam2_W = -T_last(1:3,1:3)'*T_last(1:3,4);
plotCoordinateFrame(R_C2_W_ransac',center_cam2_W, 0.8);
text(center_cam2_W(1)-0.1, center_cam2_W(2)-0.1, center_cam2_W(3)-0.1,'Cam Last','fontsize',10,'color','k','FontWeight','bold');

axis equal
rotate3d on;
grid

% Display keypoints in last frame

subplot(1,2,2)
imshow(image,[]);
hold on
plot(S.P(1,:),S.P(2,:), 'ys');
title('Image actual')