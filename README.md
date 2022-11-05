# Federated Learning Simulator _ Korean ver
## Project Purpose 

논문 `학습 에이전트를 고려한 연합학습 시뮬레이터 고안`에서 제안한 시뮬레이터로 자세한 내용은 논문을 참고하면 된다.

## Requirements

see `requirements.txt`

## Configurations

See `batch file` or `shell script`
- situation_script
  - save loacation
    - file_root
      - output 저장 위치
    - file_name_device
      - server, client, learning agent 성능을 볼 수 있는 csv 파일 이름
    - file_name_round
      - round별로 출력되는 결과를 확인할 수 있는 csv 파일 이름
    - file_name_graph
      -  라운드별 정확도를 확인할 수 있도록 저장하는 png 파일 이름 
  - parameters
    - edge_num
      - client 개수
    - round_num
      - 진행 round 횟수
    - model
      - 미리 지정해 둔 MobileNet_v2, Inception_v3, Vgg19, SRGAN
    - dataset
      - 미리 지정해 둔 MNIST, Cifar10, CelebA
    - server_comm
      - server가 사용하는 통신 방법
    - client_comm
      - client가 사용하는 통신 방법
    - l_agent_comm
      - learning agent가 사용하는 통신 방법
    - l_agent_num
      - client가 가지는 learning agent 개수
    - a
      - 시작 정확도
    - b
      - 수렴 정확도
    - p
      - 로컬 학습 가중치
    - q
      - 학습에 참여하는 클라이언트 수 가중치
- accuracy_script
  - accuracy
    - 사용자가 미리 학습한 정확도
  - predict_round
    - 예측하고 싶은 round

## File

### device.py
server, client, learning agent의 성능, 통신방법, 데이터셋 등을 선언
### helper.py
main 코드에 사용하는 함수 정의
### main.py
연합학습 시뮬레이션이 작동

## Run
- batch
    ``` 
    ### batch
    situation_script.bat
    accuracy_script.bat
    ```
- shell script
    ```
    ### sh
    sh situation_script.sh
    sh accuracy_script.sh
    ```

## Results
- situation_script  
![simulation](C:\Users\hyeonbae\PycharmProjects\Federated_Learning_Simulator\asset\FL_accuracy_Graph.png)
- accuracy_script  
![simulation](C:\Users\hyeonbae\PycharmProjects\Federated_Learning_Simulator\asset\Predict_accuracy.png)
