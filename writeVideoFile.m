%BEFORE RUNNING THIS:
% type recording_crop = recording(4:xx) where xx = last nonempty entry
% adjust length of for loop below
% now run the script
% afterwards rename the video file so it won't get overwritten later

v = VideoWriter('myVideo.avi');

open(v);
for i = 1:1330 %adjust to number of frames in recording_crop
    writeVideo(v,recording_crop(i));
end
close(v);

