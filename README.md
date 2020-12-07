<p align='center'>
  <img src='https://github.com/prasunroy/openpose-pytorch/raw/master/assets/image_1.jpg' />
</p>

# 인간 행동기반 낙상 감지 기법
**PyTorch implementation of OpenPose.**

## Introduction
최근에는 혼자 생활하는 직장인, 맞벌이하는 가정집에 남겨진 아이들, 독거노인 등 1인 가구 및 장시간 보호자가 없는 가구가 증가하고 있다. 따라서, 실내 공간에서 지병으로 인해 의식을 잃어 쓰러지거나, 부주의로 인한 낙상사고가 발생했을 때 주변에서는 이를 전혀 인지할 수가 없기 때문에, 위와 같은 경우의 해결책을 찾기 위한 인간 행동 인식 기반 낙상 감지 방법이다.


## 구현
* Install [PyTorch]
* Install [OpenCV]
```
### Option 1: bodybox 정의
```
<p align='center'>
  <img src='https://github.com/rookey2/Capston2020/raw/master/assets/fig01.jpg' />    
</p>

```
### Option 2: bodybox의 형태 감지
```
<p align='center'>
  <img src='https://github.com/rookey2/Capston2020/raw/master/assets/fig02.jpg' />
</p>

```
### Option 3: bodybox의 각도 감지
```
<p align='center'>
  <img src='https://github.com/rookey2/Capston2020/raw/master/assets/fig03.jpg' />
</p>

```
### Option 4: bodybox의 움직임벡터(MV) 감지
```
<p align='center'>
  <img src='https://github.com/rookey2/Capston2020/raw/master/assets/fig04.jpg' />
</p>


## 실행
```
python video_detect.py
```

<p align='center'>
  <img src='https://github.com/rookey2/Capston2020/raw/master/assets/fig05.jpg' />
</p>


## References
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* [PyTorch OpenPose](https://github.com/Hzzone/pytorch-openpose)

## License
* [OpenPose License](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)
* [OpenPose PyTorch License](https://github.com/prasunroy/openpose-pytorch/blob/master/LICENSE)

<br />
<br />
