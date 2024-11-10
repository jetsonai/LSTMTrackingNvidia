# LSTMTrackingNvidia

## Section 1 과 Section 2 예제를 위해 다음의 링크에서 Section1.tar.gz 와  Section2.tar.gz 를 다운받습니다.

https://drive.google.com/file/d/1TIDsQKEr5ndU2sr4fIpSarrtY1Kv0Xfv/view?usp=sharing

https://drive.google.com/file/d/1lP5K7xpWXidyZdlIdO5tKrqMRRQ-OsSe/view?usp=sharing

## LSTMTrackingNvidia 깃헙 다운로드

cd

git clone https://github.com/jetsonai/LSTMTrackingNvidia

cd LSTMTrackingNvidia

## 다운로드 폴더의 예제 파일을 옮겨온 후 압축을 해제합니다.

mv ~/Downloads/Section1.tar.gz ./

tar xzf Section1.tar.gz

mv ~/Downloads/Section2.tar.gz ./

tar xzf Section2.tar.gz

-----------------------------

# Section 1 예제 

cd Section1

rnn lstm gru 순서로 train 실행

-----------------------------

# Section 2 예제 

cd Section2

train test 실행

-----------------------------

# Section 3 예제 

cd Section3

train test 실행

-----------------------------

# Section 4 예제 

cd Section4

train test 실행

-----------------------------

# Section 5 예제 

cd Section5

### 순서대로 실행

python3 crop_video.py

python3 visualize_object_tracking.py

python3 train.py --total-epoch 50

python3 inference_on_val.py

python3 inference_on_test.py
