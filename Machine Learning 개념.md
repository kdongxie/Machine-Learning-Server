# Machine Learning

## Contents

- Machine Learning의 개요
- Machine Learning의 정의
- Machine Learning의 분류
- Supervised Learning (지도학습)
- Unsupervised Learning (비지도학습)
- Reinforcement Learning (강화학습)

### Machine Learning 의 개요

- Machine Learning(머신러닝) 이라는 용어는 IBM의 인공지능 분야 연구원이였던 ''아서 사무엘" 이 자신의 논문 "Studies in Machine Learning Using the Game of Checkers"에서 처음으로 사용됨.
- 머신러님의 3가지 접근법
  - 신경 모형 패러다임:
    - 신경 모형은 퍼셉 트론에서 출발해서 지금은 딥러닝으로 이어지고 있음
  - 심볼릭 개념의 학급 패러다임
    - 숫자나 통계이론 대신 논리학이나 그래프 구조를 사용하는 것으로 1970년대 중반부터 1980년대 후반까지 인공지능의 핵심적인 접근법이었음.
  - 현대지식의 집약적 패러다임
    - 1970년대 중반부터 시작된 이 패러다임은 백지상태에서 학습을 시작하는 신경모형을 지양하고 이미 학습된 지식은 재활용 해야 한다는 이론이 대두되면서 시작됨.

### Machine Learning의 정의

- 카네기 멜론 대학교의 톰 미첼(Tome Mitchell)교수는 자신의 저서 "머신러닝"에서 러닝, 즉 학습의 정의를 다음과 같이 내렸다. 
- A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E
- **<u>[만약 컴퓨터 프로그램이 특정한 테스크 T를 수행할때, 성능 P 만큼 개선되는 경험 E를 보이면, 그 컴퓨터 프로그램을 테스크 T와 성능 P에 대해 경험 E를 학습 했다라고 할 수 있다.]</u>**
  - 테스크 T : 필기체를 인식하고 분류하는 것
  - 성능 P : 필기체를 정확히 구분한 확률
  - 학습경험 E : 필기체와 정확한 글자를 표시한 데이터 세트 
- 실무적 관점의 러닝, 즉 학습의 정의는 
  - **학습(Learning) = 표현 (Representation) + 평가 (Evaluation) + 최적화 (Optimization)**
- 머신러닝은 종종 데이터 마이닝과 홍용이 되곤 한다. 아마도 머신러닝에서 사용하는 분류나 군집같은 방법을 데이터 마이닝에서도 똑같이 사용하기 때문일 것임.
- 분류나 예측, 군집과 같은 기술, 모델, 알고리즘을 이용해 문제를 해결하는 것을 컴퓨터 과학 관점에서는 머신러닝이라고 하고, 통계학 관점에서는 데이터 마이닝 이라고 함. 
- 머신러닝과 데이터 마이닝의 차이점은 데이터 마이닝은 가지고 있는 데이터에서 현상과 특성을 발견하는 것이 목적이나 **<u>머신러닝은 기존 데이터를 통해 학습시킨 후 새로운 데이터에 대한 예측값을 알아내는 데 목적이 있음.</u>**

### Machine Learning의 분류

- **지도학습 (Supervised Learning)**
  - 분류 (Classification)
    - kNN (k nearest neighbor)
    - 서포트 벡터 머신 (Support Vector Machine)
    - 의사 결정 트리 (Decision Tree) 모델
  - 예측 (Prediction)
    - Regression

- **비지도 학습 (Unsupervised Learning)**
  - Clustering
    - k-means
  - Neural Networks

- **강화학습 (reinforcement Learning)** 

  >  나누자면 비지도 학습에 들어가고 데이터를 넣었을 때 스스로 알고리즘에 따라학습하게된다.
  >
  > Deep Learning 이며 아래의 두가지가 존재함
  >
  > - Supervised Deep Learning 
  > - Unsupervised Deep Learning
  >
  > 

  

  - 머신러닝 분류기준으로 지도학습에 포함 가능함.
  - 지도학습으로 분류가능한 이유는 에이전트가 취한 모든 행동에 대한 환경으로 부터 보상화 벌칙을 사전에 사람으로부터 가이드를 받고 사람이 아닌 환경으로 부터 보상과 벌칙을 피드백 받는다. 

![image](https://user-images.githubusercontent.com/46669551/55443077-4b91cd00-55ec-11e9-92a1-f17cfe26b545.png)

![image](https://user-images.githubusercontent.com/46669551/55443723-b6440800-55ee-11e9-8e9c-de69557ad6c2.png)

### 지도학습 (Supervised Learning)

- 광고 인기도
- 스팸 분류
- 얼굴 인식

### 비지도 학습 (Unsupervised Learning)

- 추천 시스템
- 구매 습관
- 사용자 로그 그룹화

![image](https://user-images.githubusercontent.com/46669551/55443785-e1c6f280-55ee-11e9-8493-f8559998e747.png)

### 강화 학습 (Reinforcement Learning)

![image](https://user-images.githubusercontent.com/46669551/55443871-47b37a00-55ef-11e9-86a8-53975964fce0.png) 





#### [딥러닝](<http://hellogohn.com/post_one18>)

![image](https://user-images.githubusercontent.com/46669551/55443923-7c273600-55ef-11e9-97b3-a8d6c23f6fa0.png)