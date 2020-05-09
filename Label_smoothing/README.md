dataset : CIFAR10

main: 메인함수 

model: embedding_net2 (Resnet 18 로 일단 설정해둠)

predict_only: only for testing

train: 모델 fit

utils: smooth 하는 함수 , MOD_CrossEntropyLoss 함수

** 현재 train 할때 smoothing 계수 true label에 적용하는 부분에서 shape error나서 수정중 
