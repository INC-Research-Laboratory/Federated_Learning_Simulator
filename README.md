# Federated Learning Simulator _ Korean ver
## Project Purpose 

논문 `학습 에이전트를 고려한 연합학습 시뮬레이터 고안`에서 제안한 시뮬레이터로 자세한 내용은 논문을 참고하면 된다.  

최근 4차산업혁명으로 인해 인공지능, 사물인터넷 분야에 많은 관심이 모이고 있다. 
이에 따라 인공지능 모델 학습에 사용자의 개인 정보를 사용하지만 프라이버시 문제를 피할 수 있는 학습 방법인 
연합학습에도 관심이 많아지고 있다.  
  
연합학습은 라운드 단위로 학습이 진행된다. 첫번째 라운드가 시작될 때, 서버에서 글로벌 모델을 구축하고 
학습에 참여할 클라이언트를 선택한다. 선택된 클라이언트로 모델을 전송하여 클라이언트가 가지고 있는 로컬 데이터를 
활용해 학습을 진행하여 도출된 모델의 가중치 값을 서버로 전송한다. 그러면 서버에서는 선택된 클라이언트들의 
가중치들을 집계(aggregation)하여 글로벌 모델을 업데이트하며 다음 라운드에 전송하는 활동이 반복된다.  

이와 같은 연합학습을 위해서 많은 고려사항이 존재한다. 우선, 연합학습에 참여할 많은 기기들이 필요하다. 
예를 들어 통신사에서 고객 데이터를 이용한 연합학습을 진행할 경우를 가정한다면 통신사를 사용하는 모든 고객이 
클라이언트가 될 수 있다. 그리고 Wifi,LTE,5G와 같은 네트워크 상황과 사용하는 기기의 성능 등을 고려해야 한다.
이런 제약으로 인해 연합학습을 실제에 바로 도입하기 보다는 모의 실험을 통해 성능이 어떻게 도출되는지 확인해 볼 필요가 있다.  

연합학습을 구현할 때는 flower 같은 파이썬 프레임워크를 사용하면 된다. 하지만 코드를 볼 줄 알아야 하며, 
프레임워크에 대한 전문지식이 있어야 하기 때문에 간단하지는 않다. 그래서 시뮬레이션에 활용하기에 적합하지 않다.  

위와 같은 이유로 본 프로젝트는 연합학습을 간단히 실험해 볼 수 있는 시뮬레이터를 구현하는 것이 목적이다.
프레임워크를 다루는 것과는 다르게 몇가지 파라미터만을 조정하면 연합학습 시뮬레이션이 가능하다는 장점이 있다.

추가기능으로는 프레임워크를 사용할 수 있지만 컴퓨팅 파워나 네트워크 상황, 클라이언트의 수를 다르게 지정한 결과를 보고
싶은 경우에도 활용할 수 있도록 기능을 추가했다. 그리고 기존 연합학습 시뮬레이터와의 차별점인 `학습 에이전트`개념을 
도입했다. 
  
학습 에이전트는 연합학습에 참여하는 기기의 주변에 있는 가용 기기를 활용하지 않는다는 기존 실험의 한계점을 극복할
수 있는 개념으로, 연합학습 참여 기기의 사용자의 성능이 더 좋은 기기를 이용해 대신 학습하도록 한다는 것이다.
이로써 보다 빠른 학습이 가능하며 연합학습의 시간도 단축할 수 있다는 장점을 가지게 된다. 이 시뮬레이터에 학습 에이전트
기능을 탑재했으므로 사용자 주변 기기를 활용했을 때의 연합학습 성능도 확인 가능하다.

## Requirements

see `requirements.txt`

## Configurations

See `shell script`
- parameters
  - file_name_device
    - server, client, learning agent 성능을 볼 수 있는 csv 파일 이름
  - file_name_round
    - round별로 출력되는 결과를 확인할 수 있는 csv 파일 이름
  - file_name_graph
    -  라운드별 정확도를 확인할 수 있도록 저장하는 png 파일 이름 
  - edge_num
    - client 개수
  - round_num
    - 진행 round 횟수
  - model
    - 미리 지정해 둔 MobileNet_v2, Inception_v3, Vgg19, SRGAN / 사용자 모델 mymodel
  - model_size
    - 사용자 모델을 사용할 경우 모델 용량 설정
  - dataset
    - 미리 지정해 둔 MNIST, Cifar10, CelebA / 사용자 데이터셋 mydataset
  - dataset_vol
    - 사용자 데이터셋 용량
  - dataset_imgs
    - 사용자 데이터셋 이미지 개수
  - dataset_size
    - 사용자 데이터셋 이미지 사이즈
  - server_comm
    - server가 사용하는 통신 방법
  - client_comm
    - client가 사용하는 통신 방법
  - l_agent_comm
    - learning agent가 사용하는 통신 방법
  - comm_speed
    - 통신 속도를 지정할 경우 기입
  - l_agent_num
    - client가 가지는 learning agent 개수


## File

### device.py

### helper.py
 
### main.py

## Run

    - move to `pkg` folder
    ```
    main.py --edge_num 3 --round_num 4 --model_size 20 --server_speed 10 --max_edge_speed 10 --edge_data 3 --max_learning_agent_speed 20 --max_learning_agent_num 1
    ```

## Results




