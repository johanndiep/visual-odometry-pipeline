figure(3);
set(gcf, 'Position', [20, 50, 1800, 900]);


% plot entire trajectory

%landmarkplotx=sort(S.X(1,:))
scale_diff = 0.152;
plot(smooth(scale_diff*cameracentre(1,:),10),smooth(scale_diff*cameracentre(3,:),10), 'color', 'r', 'Linewidth',2);

hold on
plot(ground_truth(1:599,1)',ground_truth(1:599,2)','g');
hold off
legend('visual odometry', 'ground truth');
axis equal;
title('Full Trajectory')
