- dataset : CIFAR10
            05/12 ImageNet으로 변경

- main.py: 메인함수 

- model.py: 05/10 embedding_net2 (최대한 가벼운 모델로 하기 위하여 Resnet 18 로 일단 설정해둠)

            05/12 embedding_net 1, 2 모두 mobilenetV3 L 로 바꿔주고, EntireNet으로 두 모델을 묶어준뒤 트레이닝

- predict_only.py: only for testing

- train.py: 모델 fit

- utils.py: 05/10 smooth 하는 함수 , MOD_CrossEntropyLoss 함수

            05/12 LabelSmoothingCross_Entropy_Loss 로 로스 함수 변경 
            

** 05/12 스무딩 계수가 계속 0.9로 수렴되는 문제점 
