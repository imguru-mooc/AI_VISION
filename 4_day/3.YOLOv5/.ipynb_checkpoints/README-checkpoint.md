# YOLOv5_Custom_data_Training

## 데이터셋 준비



> ### 데이터셋 선정 및 다운로드

- 데이터셋은 어떤 것을 사용하셔도 되지만 간편하게 roboflow 라는 사이트에서 object detection 을 위한 여러가지 퍼블릭 데이터셋을 다운받아 사용할 수 있음
- 그 중에서 저는 드론으로 부둣가를 촬영하여 자동차, 배, 부두 등을 검출하기 위해 만들어진 'Aerial Maritime Drone Dataset' 이라는 데이터셋을 사용
- 아래 사이트에 접속해서 데이터셋을 검색하고 다운받을 수 있음
- https://public.roboflow.com/object-detection
- 다운로드 받을 데이터셋 포맷은 TXT - YOLOv5 pytorch 포맷


<p align = "center">
  <img width="618" alt="image" src="https://user-images.githubusercontent.com/96943196/208346488-3a24718e-9c9e-467e-9fd3-6790ac7625f4.png">
</p>
<p align ="center"> Roboflow 퍼블릭 데이터셋
</p>
<p align = "center">
  <img width="1477" alt="image" src="https://user-images.githubusercontent.com/96943196/208346578-921c0634-47f6-4389-9b75-3721e5e67744.png">
</p>
<p align ="center"> Aerial Maritime Drone Dataset
</p>
<p align = "center">
  <img width="837" alt="image" src="https://user-images.githubusercontent.com/96943196/208346627-cec1623e-60ff-46b7-a5e5-128d0564a740.png">
</p>
<p align ="center"> 데이터셋 포맷
</p>    
<br>
<br>

> ### 다운받은 데이터셋 확인

- 다운받은 압축 파일을 해제하면 아래와 같이 학습/평가 데이터와 정답 정보가 들어있는 test, train, valid 폴더와 데이터셋에 대한 정보가 들어있는 data.yaml 파일이 있음
- yaml 파일은 코드 에디터나 텍스트 편집기로 열어보면 아래처럼 확인이 가능
- 나중에 yaml 파일에서 train, val 경로를 변경해야함 (밑에서 설명)


![image](https://user-images.githubusercontent.com/96943196/208346904-732566ae-c0b2-4f7c-b952-0291d5d02966.png)
<p align ="center"> Roboflow 퍼블릭 데이터셋
</p>  
<br>
<br>


> ### 데이터셋, 학습 코드 → 구글 드라이브 업로드

- 다운 받은 데이터셋 폴더를 구글 드라이브에 업로드 (YOLO 모델을 구글 코랩이라는 환경에서 학습시킬 예정이기 때문에 데이터셋을 구글 드라이브에 업로드하는 것)
- 구글 드라이브의 '내 드라이브'에 'dataset' 이라는 폴더를 하나 만들고 그 안에 다운받은 데이터셋을 업로드.(폴더 이름에 점(.)이나 각종 특수 문자가 있는 경우 코드 실행 시 문제가 되므로 이름을 변경)
- 'YOLOv5_학습_커스텀데이터셋.ipynb' 파일또한 '내 드라이브' 경로에 업로드

<p align = "center">
  <img width="795" alt="image" src="https://user-images.githubusercontent.com/96943196/208346880-a3bf7615-8aef-48e4-96ba-378341c0b1cd.png">
</p>
<p align ="center"> 구글드라이브 - 내 드라이브에 dataset 폴더 생성, YOLOv5_학습_커스텀데이터셋.ipynb 파일 업로드
</p>

이제 데이터셋과 학습 코드준비가 끝이 났습니다.  

<br>
<br>

## YOLOv5 모델 학습 및 테스트 - 구글 코랩 사용

- YOLOv5 모델은 구글 코랩이라는 환경에서 학습
- 구글 코랩을 사용 시 개발 환경을 따로 세팅해줄 필요가없어서 예상치 못한 에러를 만날 확률이 적기 때문에 코딩을 할 줄 모르는 분들도 쉽게 따라하실 수 있음

<br>

> ### 1. 구글 드라이브 마운트
- 구글 코랩과 구글 드라이브를 연결시켜주기 위해 마운트 필요. (아래 화면에서 재생표시를 눌러서 진행)
- 빠른 학습을 위해 GPU를 사용 (런타임 - 런타임 유형 변경 - 하드웨어 가속기 → GPU)

<p align = "center">
  <img width="750" alt="image" src="https://user-images.githubusercontent.com/96943196/208346933-b1748ba4-7c00-48b0-934f-0873a4538ae2.png">
</p>  

<br>
<br>

> ### 2. YOLOv5 레퍼지토리 다운로드

<p align = "center">
  <img width="750" alt="image" src="https://user-images.githubusercontent.com/96943196/208346982-3403429b-85f9-4219-a9c7-422b1a98af7c.png">
</p>

- YOLOv5 를 사용하기 위해 깃허브에 올려진 YOLOv5 레퍼지토리에서 코드들을 다운로드
- 이는 한번만 실행해주면 구글 드라이브 내 드라이브에 yolov5가 다운로드됨 (두번째 학습부터는 건너뛰어도 되는 부분이고, 패키지 다운로드 및 임포트 하는 부분은 매번 실행 필요)

<br>
<br>

> ### 3. 데이터셋 다운로드, yaml 파일 수정

- 데이터셋은 이미 다운로드 받고 구글 드라이브에 업로드시켜서 아래와같이 데이터셋과 yaml 파일을 확인할 수 있음 
- 여기서 데이터셋의 경로를 담고 있는 yaml 파일을 조금 수정해줘야 학습이 가능 
- 아래 화면에서 data.yaml 을 클릭하셔서 직접 수정할 수 있지만, 코드로 yaml 파일을 수정하도록 구현해두었음


<p align = "center">
  <img width="750" alt="image" src="https://user-images.githubusercontent.com/96943196/208347040-0a1adb39-5200-468c-bd82-143e3413ad9f.png">
</p>
<p align ="center"> 구글 코랩에서 구글 드라이브 파일 확인
</p>

- 아래와 같이 data_dir 에는 데이터셋의 경로를 data_yaml 변수에 yaml 파일의 경로를 지정 \
- 그리고 나서 아래 코드를 수행하면 yaml 파일의 정보가 나오는데, 우리는 데이터셋을 구글 드라이브에 올렸기 때문에 여기서 train, val 의 경로를 수정해줘야합니다.

- 코드를 실행하여 yaml 파일 정보를 확인하고, film['train'], film['val'] 부분을 수정하여 yaml 파일을 수정, 변경된 정보를 확인

<p align = "center">
<img width="750" alt="image" src="https://user-images.githubusercontent.com/96943196/208347132-8a76c507-7c06-468f-bd9a-bed14b1f876b.png">
</p>
<p align ="center"> 기존 yaml 파일의 정보 확인
</p>

<p align = "center">
<img width="750" alt="image" src="https://user-images.githubusercontent.com/96943196/208347135-72bbebe6-9512-4020-a743-2e93ad8f1cdf.png">
</p>
<p align ="center"> yaml 파일 정보 수정
</p>  

<br>
<br>

> ### 4. YOLOv5 모델 학습

- 아래 학습 파라미터들을 지정해주고 모델 학습을 시작
- 지정한 epoch 만큼 학습을 진행하고 나면 학습이 종료되고 'Results saved to runs/train/exp*' 이라는 문구가 뜨는데 이 경로에 학습한 정보가 저장되었다는 뜻. 
- exp 몇번에 저장되었는지 확인 필요 (테스트 시)

<p align = "center">
<img width="750" alt="image" src="https://user-images.githubusercontent.com/96943196/208347156-e48de24c-b4c8-4606-bfee-0f294883d21f.png">
</p>
<p align ="center"> yaml YOLOv5 모델 하이퍼 파라미터 및 학습 실행 코드
</p>

<p align = "center">
<img width="750" alt="image" src="https://user-images.githubusercontent.com/96943196/208347170-c5494c39-ed7b-47ff-8ae4-e7e5193f6253.png">
</p>
<p align ="center"> 학습 종료 후 저장된 경로 확인 필수 !
</p>  

<br>
<br>

> ### 5. 텐서보드 시작하기

- 텐서보드는 학습시키는 딥러닝 모델이 학습되는 과정을 여러 가지 metric으로 기록하여 시각적으로 볼 수 있는 도구
- 앞서 확인한 exp 번호를 보고 왼쪽 아래에서 원하는 학습 정보만 필터링

<p align = "center">
<img width="750" alt="image" src="https://user-images.githubusercontent.com/96943196/208347210-c65c37d0-bdbd-47ea-bc05-abd67dc96725.png">
</p>  

<br>
<br>

> ### 6. 학습한 모델 테스트

- 학습한 모델을 테스트 데이터셋에서 테스트
- 앞서 확인한 exp 번호를 train_exp_num에 저장해두고 테스트를 실행합니다. 
- 테스트 또한 끝나고 나면 테스트 정보가 저장된 경로가 나옵니다. 이 exp 번호 또한 확인

<p align = "center">
<img width="600" alt="image" src="https://user-images.githubusercontent.com/96943196/208347246-0d635c27-c86d-48cc-a96e-2051ae3af3a1.png">
</p>
<p align ="center"> 학습 exp 번호를 지정
</p>

<p align = "center">
<img width="600" alt="image" src="https://user-images.githubusercontent.com/96943196/208347300-dab66d2b-fc40-46bb-81a9-b72b4d0a2aba.png">
</p>
<p align ="center"> 테스트 정보가 저장된 exp 번호 확인
</p>

- 확인한 테스트 exp 번호를 text_exp_num에 저장하고 테스트 결과 확인 코드를 실행하면 아래 그림과 같이 이미지에서 객체를 검출한 결과를 확인할 수 있음

<p align = "center">
<img width="750" alt="image" src="https://user-images.githubusercontent.com/96943196/208347395-d434c65b-e1c1-47b2-9c3b-478c1d1dd49e.png">
</p>
<p align ="center"> 테스트 결과 확인 코드
</p>

<p align = "center">
<img width="750" alt="image" src="https://user-images.githubusercontent.com/96943196/208347410-439a4e04-b59e-4545-996d-7c99cdf806e5.png">
</p>
<p align ="center"> 테스트 결과 예시
</p>  

<br>
<br>

> 7. 베스트 모델 저장

- 마지막으로 학습 중 결과가 가장 좋은 베스트 모델을 저장할 수 있음. 
- 저장한 모델은 언제든 불러와서 사용 가능

