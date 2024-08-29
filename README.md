# Curiosity-driven Exploration by Self-supervised Prediction

[Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363) 논문의 PyTorch 구현입니다.
Linux 환경에서 개발하였습니다.

### Quick guide to start (with conda)
- conda 환경 구축

        conda create --name <env_name> python=3.9
        conda activate <env_name>

- Super Mario Bros 설치

        pip install gym-super-mario-bros
    
- OpenCV 설치

        conda install opencv

- Pytorch 설치
    
        conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

- 동영상 저장을 위한 라이브러리

        pip install ffmpeg-python


### Dev FAQ

1. Super Mario Bros. 실험 환경에 관해:
    1. 원 논문은 [Paquette](https://github.com/ppaquette/gym-super-mario)의 환경을 사용하는데 왜 여기서는 [Kautenja](https://github.com/Kautenja/gym-super-mario-bros)의 환경을 사용하나요?
  
        : Paquette은 2018년 7월 이후로 레포 업데이트를 중단했고, Kautenja는 2022년 7월까지 활발한 업데이트를 제공했습니다. 또한, PyTorch에서 제공하는 [Train a Mario-playing RL Agent](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) 예제를 참고할 수도 있는데, 이 역시 Kautenja의 환경을 사용하였습니다.

    1. random action을 취하는 마리오는 금방 죽을 것 같은데 그렇지 않네요. 이유가 뭔가요?

        : 마리오가 죽지 않는 가장 쉬운 방법은 시작과 동시에 왼쪽으로 가서 timeout될 때까지 밖을 탐험하지 않는 것입니다.
    
    1. action space에 관하여 - `gym_super_mario_bros.actions`

        : [action의 종류](https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/actions.py)에서, `RIGHT_ONLY`는 5개, `SIMPLE_MOVEMENT`는 7개, `COMPLEX_MOVEMENT`는 12개의 action입니다.
        
        | Combination       | Description           |
        | ------------------|:-------------:        |
        | `NOOP`            | Stay (No operation)   |
        | `right`           | Walk right            |
        | `right`, `A`      | Jump right            |
        | `right`, `B`      | Run right             |
        | `right`, `A`, `B` | Run & jump right      |
        | | |
        | `A`               | Jump                  |
        | `left`            | Walk left             |
        | | |
        | `left`, `A`       | Jump left             |
        | `left`, `B`       | Run left              |
        | `left`, `A`, `B`  | Run & jump left       |
        | `down`            | Crouch                |
        | `up`              | Climb                 |
        

    1. 각종 wrapper들

        1. `nes_py.wrappers.JoypadSpace` : integer action을 조이패드 input으로 변환한다. nes_py 환경에서 action이 잘 먹히도록 한다.
        1. `gym.wrappers.FrameStack` : 논문에서 최근 4개 frame을 연결하여 state로 사용하였다.
        1. `gym.wrappers.GrayScaleObservation` : 논문에서 RGB image를 흑백 image로 변환하여 사용하였다.
        1. `gym.wrappers.ResizeObservation` : 논문에서 image를 42x42로 변환하여 사용하였다.
        1. `SkipFrame` : 논문에서 training 과정에서 VizDoom은 4번, Mario는 6번동안 한 action을 반복하도록 하였다 (TAS 플레이 방지). 하지만 inference 시에는 이 제한을 두지 않았다!

    1. `gym.wrappers`를 사용하는데 NumPy 관련 에러가 자꾸 발생해요
        
        : NumPy 버전을 의심해보세요. 2024년 초에 NumPy v2.0이 배포되었는데, 이건 너무 최근입니다. gym wrappers는 2022년 10월에 마지막으로 업데이트되었습니다. 일단 NumPy v1.24에서는 이 레포가 잘 작동합니다!

1. 왜 `gymnasium`을 사용하지 않고 `gym`을 사용하나요?

    : Kautenja의 환경을 사용하기 위해 `gym-super-mario-bros`를 설치하면, 자동으로 gym이 설치됩니다 ;) 혹시 모를 충돌을 방지하기 위해 gymnasium은 추가설치하지 않았습니다.

1. Python 버전에 관한 가이드가 있나요?

    : `gym-super-mario-bros`를 설치하면, 자동으로 NES-py가 설치됩니다. [NES-py](https://github.com/Kautenja/nes-py)가 3.5에서 3.9까지 지원합니다. 적어도 이 범위에는 맞춰야 할 것 같습니다.

1. PyTorch 버전에 관한 가이드가 있나요?

    : 모르겠네요. 일단 제 GPU에 맞췄습니다. CUDA Version이 11.4라, 그에 해당하는 버전으로 설치했습니다. 이 레포의 초기 버전이 PyTorch v2.1.1에서도 실행되는 것을 확인한 적이 있습니다. 마음껏 선택하세요.

1. Windows에서 개발 중인데, `ffmpeg`쪽에서 에러가 나요.

    : 여기를 한번 확인해보세요. https://stackoverflow.com/a/78172102/14257620

### references

- pytorch.org - [Train a Mario-playing RL Agent](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
