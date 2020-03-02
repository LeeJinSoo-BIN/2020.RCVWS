#### [Adaptive Fusion 설명 바로가기](https://github.com/sejong-rcv/AdaptiveFusion)

#### 1. Git clone
```
git clone https://github.com/sejong-rcv/StaticFusion.git
(sudo)chmod -R 777 StaticFusion
cd StaticFusion
```

#### 2.Docker Build & Make container

```
cd docker
```

##### Steps to setup
###### 해당 설정은 자신의 로컬PC 환경에 따라 달라질 수 있음(도커에 대한 학습이 필요)
------------------------------------------------------------------------------ 
#### SF/AF 통합 한번만 진행(만약 이미 실행했을경우 바로 3번부터 진행) </br>
1. `01_build_base_docker.sh`
2. `02_build_user_docker.sh {VSCODE_PASSWORD}`
------------------------------------------------------------------------------
3. `03_run_container.sh {TENSORBOARD_PORT} {VSCODE_PORT} {POS_GPUS}`
    - caution
        - local과 docker container내의 volume을 link를 수정할 필요 있음
        - 기본적으로 `-v {local path}:{docker path}`에서 docker path는 수정 불필요
        - `-v /home/$USER/workspace/$PROJECT_NAME:/home/$USER/workspace/$PROJECT_NAME`
            - local에 StaticFusion을 clone받은 경로 (상황에 맞게 수정)
        - `-v /raid/$USER/workspace/jobs:/home/$USER/workspace/$PROJECT_NAME/jobs`
            - local에서 jobs를 저장하는 경로 (상황에 맞게 수정)
        - `-v /raid/datasets:/home/$USER/workspace/$PROJECT_NAME/datasets`
            - local에서 datasets 경로 (상황에 맞게 수정)
        - `-v /usr/share/zoneinfo:/usr/share/zoneinfo`
            - 수정하지 말것

##### How to use

1. `code-server --port {VSCODE_PORT}`
2. webbrowser에서 `(server.ip.address):{VSCODE_PORT}`
3. `{VSCODE_PASSWORD}` 입력
    - password를 수정하려면 `~/.bashrc`에서 `PASSWORD`를 수정하면 됨

#### 3. Evaluation

##### 정상 상황에 대한 평가
```
python eval_coco.py
```
##### 비정상 상황에 대한 평가(단일 케이스)
##### synth_fail case

```
python eval_coco.py --synth_fail blackout None
```
None</br>
blackout</br>
crack1.jpg</br>
bug1.png</br>
dust2.png </br>
....
(synthetic_failure_masks 참고)

예시</br>
(RGB) (Thermal)</br>
blackout None</br>
None blackout</br>
bug2.png crack2.jpg</br>
crack2.jpg dust2.png</br>

##### 비정상 상황에 대한 평가(복수 케이스)
```
./fail_test.sh
```
#### 4. Code Detail

##### train_eval.py
[학습 및 평가 코드](https://github.com/sejong-rcv/AdaptiveFusion/blob/master/train_eval.py)</br>
[코드설명](https://github.com/sejong-rcv/AdaptiveFusion/blob/tmp/doc/Train_eval.md)</br>

##### model.py
[모델 코드](https://github.com/sejong-rcv/StaticFusion/blob/master/model.py)</br>
[코드설명](https://github.com/sejong-rcv/StaticFusion/blob/master/doc/model.md)</br>

##### eval_coco.py
[평가코드](https://github.com/sejong-rcv/AdaptiveFusion/blob/master/eval_coco.py)</br>
[코드설명](https://github.com/sejong-rcv/AdaptiveFusion/issues/3)</br>

##### dataset.py
[데이터 로드 코드](https://github.com/sejong-rcv/AdaptiveFusion/blob/master/datasets.py)</br>
[코드설명](https://github.com/sejong-rcv/AdaptiveFusion/blob/tmp/doc/dataset.md)</br>

#### 5. Option

##### [make_annotation.py](https://github.com/sejong-rcv/AdaptiveFusion/blob/master/make_annotation.py)

```
DB_ROOT = './datasets/New_Sejong_RCV_dataset/RGBTDv3' # 데이터 경로 설정
image_set = 'test_20.txt' #GT 경로가 있는 txt
filename = 'Potenit_20.json' # GT박스로 만들어진 정답 json의 
cat_id_path = 'catids.json' #COCO 데이터셋에서 정의하는 카테고리 id 파일 (해당 파일은 COCO 평가코드를 위해 필요함)
```

##### [find_mean_std.py](https://github.com/sejong-rcv/AdaptiveFusion/blob/master/find_mean_std.py)

새로운 데이터셋의 mean과 std를 구함 이를 통한 mean과 std를 통해 정규화 진행
```
    transforms2 = Compose([ RandomHorizontalFlip(), \
                            RandomResizedCrop( [512,640], scale=(0.25, 4.0), ratio=(0.8, 1.2)), \
                            ToTensor(), \
                            Normalize( [0.3465,0.3219,0.2842], [0.2358,0.2265,0.2274], 'R'), \
                            Normalize([0.1598], [0.0813], 'T')\
                            ])
```

##### [Draw_GT.py](https://github.com/sejong-rcv/AdaptiveFusion/blob/master/Draw_GT.py)

모델이 예측한 박스와 GT를 함께 그려서 시각화 진행</br>
아래 체크포인트 [수정](https://github.com/sejong-rcv/AdaptiveFusion/blob/0b43fc7d562d0933da1b538cb8938051ff0856a9/Draw_GT.py#L58)이 필요함

```
checkpoint = './jobs/2020-02-11_17h20m_train_SF+_POTENIT_Hard_OCc/checkpoint_ssd300.pth.tar017'
```
