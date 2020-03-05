
function saveFrame_JPG(fid, strFileName)

    fwrite(fid,nImg,'uint32'); % 실제로 읽어야 하는 비트
    img = imread(strFileName);
    img = reshape(img,[1, 320*256]);
    fwrite(fid,img,'*uint8'); % nBytes 제외하고 
    fwrite(fid,1,'uint32');
    fwrite(fid,1,'uint16');
    fwrite(fid,1,'uint16');
    fwrite(fid,1,'uint32');
    fwrite(fid,1,'uint32');
    
end