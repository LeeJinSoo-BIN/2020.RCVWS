
function saveFrame_JPG(fid, strFileName)

    fwrite(fid,nImg,'uint32'); % ������ �о�� �ϴ� ��Ʈ
    img = imread(strFileName);
    img = reshape(img,[1, 320*256]);
    fwrite(fid,img,'*uint8'); % nBytes �����ϰ� 
    fwrite(fid,1,'uint32');
    fwrite(fid,1,'uint16');
    fwrite(fid,1,'uint16');
    fwrite(fid,1,'uint32');
    fwrite(fid,1,'uint32');
    
end