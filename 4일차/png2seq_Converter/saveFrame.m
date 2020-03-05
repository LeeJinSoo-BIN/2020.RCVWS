
function saveFrame(fid, strFileName, width, height, channel)
    % img :불러온 png 이미지 파일
    img = imread(strFileName);
    if channel==1
        Timg=img'; % reshape(I,siz(2),siz(1))
    elseif channel == 3
        Timg = permute(img,[3,2,1]);
        I=Timg(1,:,:);
        Timg(1,:,:)=Timg(3,:,:);
        Timg(3,:,:)=I;
    end
    changeimg = reshape(Timg,[1, width*height*channel]);
    fwrite(fid,changeimg,'*uint8');
    fwrite(fid,1,'uint32');
    fwrite(fid,1,'uint16');
    fwrite(fid,1,'uint16');
    fwrite(fid,1,'uint32');
    fwrite(fid,1,'uint32');

end