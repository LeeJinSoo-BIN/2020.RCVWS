%clf; clear all;
function RCVSS_GUI()
    
    [hFig, pLf, pMenu, poss ] = deal([]);
    pLf=makeLayout();
    filesApi = filesMakeApi();
    objApi=objMakeApi()
    nframe = 7;
    w = 1238;
    h = 638;
    function pLf = makeLayout()
        hFig = figure( 'Name','VBB Labeler', 'NumberTitle','off', ...
          'Toolbar','auto', 'MenuBar','none','Color','k', ...
          'Visible','on');
        pMenu.hVid    = uimenu(hFig,'Label','Image');
        pMenu.hVidOpn = uimenu(pMenu.hVid,'Label','Open');
        pMenu.hVidCls = uimenu(pMenu.hVid,'Label','Close');
        pMenu.hAnn    = uimenu(hFig,'Label','Annotation');
        pMenu.hAnnNew = uimenu(pMenu.hAnn,'Label','New');
        pMenu.hAnnOpn = uimenu(pMenu.hAnn,'Label','Open');
        pMenu.hAnnSav = uimenu(pMenu.hAnn,'Label','Save');

        
        pLf.h=uipanel('parent', hFig, 'background', 'k');
        pLf.hAx=axes('Units','pixels', 'parent', pLf.h);
        cla(pLf.hAx)

    end
    
    function api = filesMakeApi()
        [fVid, fAnn, tSave, tSave1] = deal([]);
        set( pMenu.hVidOpn, 'Callback', @(h,e) openVid() );
        set( pMenu.hVidCls, 'Callback', @(h,e) closeVid() );
        set( pMenu.hAnnOpn, 'Callback', @(h,e) openAnn() );
        set( pMenu.hAnnSav, 'Callback', @(h,e) saveAnn() );

        
        api = struct('closeVid',@closeVid, 'openVid',@openVid, 'openAnn',@openAnn );
        function openVid( f )
            poss=[]
            d = './Image';
            [f,d]=uigetfile('*.jpg','Select visible image',[d '/*.jpg']);
            fVid=[d f];
            image(imread(fVid));
        end
         function closeVid()      
            fVid=[];   if(~isempty(fAnn)), closeAnn(); end
                cla(pLf.hAx)
         end
         
         function openAnn( f )
             d = './vbb'; 
            assert(~isempty(fVid)); if(~isempty(fAnn)), closeAnn(); end
            [f,d]=uigetfile('*.vbb;*.txt','Select Annotation',fVid(1:end-4));
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          % 저장한 text파일을 읽어 다시 rect를 그려라
          % https://kr.mathworks.com/help/matlab/ref/rectangle.html
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end

        function saveAnn()
          % poss is [x y w h x2 y2 w2 h2 ... xn yn wn hn];
          assert(~isempty(fVid))
          d = './vbb';
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          % poss를 text 파일 형식으로 저장하라
          % https://kr.mathworks.com/help/matlab/ref/fprintf.html
          
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end   
       
   function api = objMakeApi()
        set( pMenu.hAnnNew, 'Callback', @(h,e) newAnn() );
        
         api=struct( 'newAnn',@newAnn);
        function newAnn()
            A=vbb('init',nframe);
            pos=getrect;
            rectangle(pLf.hAx,'Position', pos);
            poss=[poss pos]
        end
    end
    
    
end

