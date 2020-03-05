% ������ ����Ǿ��� â�� �ʱ�ȭ
clc; close all; clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% parameter setting %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ext = 'raw';
width_RGB =680;  % width of your RGB image
height_RGB =498; % height of your RGB image
width_Thr =713;  % width of your Thermal image
height_Thr =535; %height of your Thermal image

strDir = 'D:\PRC_PNG_DIR';   % RGB.png ���ϵ��� ��� �ִ� �θ� ���
structSets = dir([strDir '\*']);        % ������� �Ʒ��� �ִ� dir ���� (Set01, Set02 ...)
SAVE = 'D:\final';                      % ������ ���� ��ġ, �������� ������ �ڵ� ����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RGB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
width = width_RGB;
height = height_RGB;
channel = 3;
% RGB Left png ���ϵ��� �ϳ��� seq�� ��ȯ�ϴ� �κ� (22-57��)
for i = 3:length(structSets)
    % ���� ��� �ܶ��� �� �ܶ��� �����Ƿ� �󼼼����� �����մϴ�.
    
    subDir = structSets(i).name; 
    strPreFix = [subDir '\decode\RGB_L\'];
    strSaveDirName = SAVE;
    strSaveDirName = [strSaveDirName '\RGB_L'];
    strSaveFileName = [strSaveDirName '\' subDir '.seq']
    
    % ������ ������ �������� ������ ����
    if ~exist(strSaveDirName)
        mkdir(strSaveDirName)
    end
    % ������ ũ��� ü�� �� ���� ����� Header ���� ����
    strLeftFileList = dir([strDir '\' strPreFix '*.png']);  % ��� png ������ ����Ʈ 
    nImg = length(strLeftFileList);                         % png ������ ����
    fid = fopen(strSaveFileName, 'wb');                     % ������ ������ ��Ʈ ��������� ����
    saveHeaderInfo(fid, nImg, width, height, channel)       % Header ���� ����
    % png ������ seq ���Ͽ� ����(43-55��)
    for i = 1:nImg
        % strFileName :������ path
        strFileName = [strDir '\' strPreFix strLeftFileList(i).name];
        
        if strcmp(ext,'JPG')
            saveFrame_JPG(fid, strFileName); 
        else
            %saveFrame.m�� �ڵ带 ������� png to seq ��ȯ�� ����
            saveFrame(fid, strFileName, width, height, channel); 
        end

       fprintf("%d/%d..\n",i,nImg);

    end

    fclose(fid);
end
% RGB Right png ���ϵ��� �ϳ��� seq�� ��ȯ�ϴ� �κ� (59-91��)
for i = 3:length(structSets)
    
    subDir = structSets(i).name;
    strPreFix = [subDir '\decode\RGB_R\'];
    strSaveDirName = SAVE;
    strSaveDirName = [strSaveDirName '\RGB_R'];
    strSaveFileName = [strSaveDirName '\' subDir '.seq']
    
    if ~exist(strSaveDirName)
        mkdir(strSaveDirName)
    end
    
    strLeftFileList = dir([strDir '\' strPreFix '*.png']);
    nImg = length(strLeftFileList);
    fid = fopen(strSaveFileName, 'wb');
    saveHeaderInfo(fid, nImg, width, height, channel)

    for i = 1:nImg

        strFileName = [strDir '\' strPreFix strLeftFileList(i).name];

        if strcmp(ext,'JPG')
            saveFrame_JPG(fid, strFileName); 
        else %raw: uncompressed image
            saveFrame(fid, strFileName, width, height, channel); 
        end

       fprintf("%d/%d..\n",i,nImg);

    end

    fclose(fid);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Thermal %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ( Set01~ end ) Thr.png to Thr.seq
width = width_Thr;
height = height_Thr;
channel = 1;
% Thermal Left png ���ϵ��� �ϳ��� seq�� ��ȯ�ϴ� �κ� (102-134��)
for i = 3:length(structSets)
    
    subDir = structSets(i).name;
    strPreFix = [subDir '\decode\Thr_L\'];
    strSaveDirName = SAVE;
    strSaveDirName = [strSaveDirName '\Thr_L'];
    strSaveFileName = [strSaveDirName '\' subDir '.seq']
    
    if ~exist(strSaveDirName)
        mkdir(strSaveDirName)
    end
    
    strLeftFileList = dir([strDir '\' strPreFix '*.png']);
    nImg = length(strLeftFileList);
    fid = fopen(strSaveFileName, 'wb');
    saveHeaderInfo(fid, nImg, width, height, channel)

    for i = 1:nImg

        strFileName = [strDir '\' strPreFix strLeftFileList(i).name];

        if strcmp(ext,'JPG')
            saveFrame_JPG(fid, strFileName); 
        else %raw: uncompressed image
            saveFrame(fid, strFileName, width, height, channel); 
        end

       fprintf("%d/%d..\n",i,nImg);

    end

    fclose(fid);
end
% Thermal Right png ���ϵ��� �ϳ��� seq�� ��ȯ�ϴ� �κ� (136-168��)
for i = 3:length(structSets)
    
    subDir = structSets(i).name;
    strPreFix = [subDir '\decode\Thr_R\'];
    strSaveDirName = SAVE;
    strSaveDirName = [strSaveDirName '\Thr_R'];
    strSaveFileName = [strSaveDirName '\' subDir '.seq']
    
    if ~exist(strSaveDirName)
        mkdir(strSaveDirName)
    end
    
    strLeftFileList = dir([strDir '\' strPreFix '*.png']);
    nImg = length(strLeftFileList);
    fid = fopen(strSaveFileName, 'wb');
    saveHeaderInfo(fid, nImg, width, height, channel)

    for i = 1:nImg

        strFileName = [strDir '\' strPreFix strLeftFileList(i).name];

        if strcmp(ext,'JPG')
            saveFrame_JPG(fid, strFileName); 
        else %raw: uncompressed image
            saveFrame(fid, strFileName, width, height, channel); 
        end

       fprintf("%d/%d..\n",i,nImg);

    end

    fclose(fid);
end

