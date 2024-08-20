# 정신건강 고위험군 선별 알고리즘 개발
- 2024 C&I Lab Project (24.05.01 ~ 진행중)
- 충청남도-단국대학교 공동 참여 프로젝트
- 지역사회 기반 AI기술을 활용한 정신건강 고위험군 조기 발견 및 중재 플랫폼 개발
- 다양한 알고리즘을 적용한 모델을 비교 및 분석하여 가장 효과적인 방법을 모색

### Loneliness 예측

**데이터셋**
- 충남 정신건강 플랫폼 SIMS에서 수집
- raw_data1.csv : 환자 별 설문조사 결과와 받은 서비스 내역
- raw_data2.csv : 관리자가 환자별로 해당 질환을 갖고 있는지 판단하여 체크한 내역
- 추후 데이터셋 추가로 부가적인 테스트 진행 예정
  
**테스트에 사용한 Method**
- Naive Bayes Classifier
- Support Vector Machine (RBF)
- ElasticNet Model
- Feed-Forward Neural Network
- Bi-LSTM
- XGBoost
<img src="https://github.com/user-attachments/assets/189f4ab6-481e-4771-a1e2-9d0529caede4" width="700" height="400"/>

**Feature Engineering**

<img src="https://github.com/user-attachments/assets/b0639b6f-2498-445f-95d2-d0d3ec58ae8c" width="700" height="800"/>

**현재까지 Result**

<img src="https://github.com/user-attachments/assets/6d3134a2-8f27-4db9-b705-4f9c1bf9d5ac" width="700" height="150"/>



