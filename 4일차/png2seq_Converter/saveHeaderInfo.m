
function saveHeaderInfo(fid, nImg, width, height, channel)


version = 3;
%width = 320;
%height = 256;
imageBitDepth = 8*channel;
imageBitDepthReal = 8;
imageSizeBytes = width*height*channel;
imageFormat = 100; %'raw';
numFrames = nImg;
trueImageSize = width*height*channel+16;
validCheck = 1024;
descr = zeros(1,256);

fseek(fid,0,'bof');
fwrite(fid,65261,'uint32'); %feed
S = 'Norpix seq';
fwrite(fid,S,'uint16');

fwrite(fid,0,'int32');
fwrite(fid,version,'int32');
fwrite(fid,validCheck,'uint32');
fwrite(fid,descr,'uint16');

fwrite(fid,width,'uint32');
fwrite(fid,height,'uint32');
fwrite(fid,imageBitDepth,'uint32');
fwrite(fid,imageBitDepthReal,'uint32');
fwrite(fid,imageSizeBytes,'uint32');
fwrite(fid,imageFormat,'uint32');
fwrite(fid,numFrames,'uint32');
fwrite(fid,0,'uint32');
fwrite(fid,trueImageSize,'uint32');

fwrite(fid,1,'float64');

fwrite(fid,zeros(1,432),'uint8');




end