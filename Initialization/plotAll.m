figure(3);
set(gcf, 'Position', [20, 50, 1800, 900]);

% plot current image with keypoints
subplot('Position',[0.1 0.55 0.4 0.4]);
imHandle = imshow(image);
hold on;
plot(S.P(1,:),S.P(2,:), 'gx', 'MarkerSize',3);
title(['frame number ' num2str(i) ' with keypoints'])

% plot trajectory last 20 frames
if i < 21
    trajectory20 = [cameracentre(1, :); cameracentre(3, :)];
else
    trajectory20 = [cameracentre(1, end-20:end); cameracentre(3, end-20:end)];
end
subplot('Position',[0.1 0.1 0.4 0.4])
plot(smooth(trajectory20(1,:),10),smooth(trajectory20(2,:),10), '-x','MarkerSize', 2) %plot last 20 positions
axis equal
hold on
set(gcf, 'GraphicsSmoothing', 'on');
view(0,90);
title('Trajectory last 20 frames')
hold off

% plot entire trajectory

%landmarkplotx=sort(S.X(1,:))
[~,index_plot]=sort(abs(S.X(3,:)));
landmarkplotz=S.X(3,index_plot);
landmarkplotx=S.X(1,index_plot);
subplot('Position',[0.55 0.55 0.4 0.4]);
plot(smooth(cameracentre(1,:),10),smooth(cameracentre(3,:),10), 'color', 'r', 'Linewidth',2);
scale_diff = 1;
hold on
%plot(1/scale_diff*ground_truth(1:i+2,1)',1/scale_diff*ground_truth(1:i+2,2)','g');
plot(landmarkplotx(1:end-length(landmarkplotx)/10),landmarkplotz(1:end-length(landmarkplotx)/10),'b.');
hold off
legend('visual odometry', 'current landmarks'); % 'ground truth', 'current landmarks');
axis equal;
title('Full Trajectory')

if record_frames
    drawnow
    current_frame = getframe(gcf);
    writeVideo(v,current_frame);
end