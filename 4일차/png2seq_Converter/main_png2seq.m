% 이전에 실행되었던 창을 초기화
clc; close all; clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% parameter setting %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ext = 'raw';
width_RGB =680;  % width of your RGB image
height_RGB =498; % height of your RGB image
width_Thr =713;  % width of your Thermal image
height_Thr =535; %height of your Thermal image

strDir = 'D:\PRC_PNG_DIR';   % RGB.png 파일들이 모두 있는 부모 경로
structSets = dir([strDir '\*']);        % 하위경로 아래에 있는 dir 정보 (Set01, Set02 ...)
SAVE = 'D:\final';                      % 저장할 파일 위치, 존재하지 않으면 자동 생성
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RGB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
width = width_RGB;
height = height_RGB;
channel = 3;
% RGB Left png 파일들을 하나의 seq로 변환하는 부분 (22-57줄)
for i = 3:length(structSets)
    % 뒤의 모든 단락은 본 단락과 같으므로 상세설명을 생략합니다.
    
    subDir = structSets(i).name; 
    strPreFix = [subDir '\decode\RGB_L\'];
    strSaveDirName = SAVE;
    strSaveDirName = [strSaveDirName '\RGB_L'];
    strSaveFileName = [strSaveDirName '\' subDir '.seq']
    
    % 저장할 폴더가 존재하지 않으면 생성
    if ~exist(strSaveDirName)
        mkdir(strSaveDirName)
    end
    % 영상의 크기와 체널 수 등을 기록한 Header 정보 저장
    strLeftFileList = dir([strDir '\' strPreFix '*.png']);  % 모든 png 파일의 리스트 
    nImg = length(strLeftFileList);                         % png 파일의 갯수
    fid = fopen(strSaveFileName, 'wb');                     % 저장할 파일을 비트 쓰기용으로 열기
    saveHeaderInfo(fid, nImg, width, height, channel)       % Header 정보 설정
    % png 영상을 seq 파일에 쓰기(43-55줄)
    for i = 1:nImg
        % strFileName :파일의 path
        strFileName = [strDir '\' strPreFix strLeftFileList(i).name];
        
        if strcmp(ext,'JPG')
            saveFrame_JPG(fid, strFileName); 
        else
            %saveFrame.m의 코드를 실행시켜 png to seq 변환을 진행
            saveFrame(fid, strFileName, width, height, channel); 
        end

       fprintf("%d/%d..\n",i,nImg);

    end

    fclose(fid);
end
% RGB Right png 파일들을 하나의 seq로 변환하는 부분 (59-91줄)
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
% Thermal Left png 파일들을 하나의 seq로 변환하는 부분 (102-134줄)
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
% Thermal Right png 파일들을 하나의 seq로 변환하는 부분 (136-168줄)
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

