dataset : CIFAR10

main.py: 메인함수 

model.py: embedding_net2 (최대한 가벼운 모델로 하기 위하여 Resnet 18 로 일단 설정해둠)

predict_only.py: only for testing

train.py: 모델 fit

utils.py: smooth 하는 함수 , MOD_CrossEntropyLoss 함수

** 현재 train 할때 smoothing 계수 true label에 적용하는 부분에서 shape error나서 수정중 
