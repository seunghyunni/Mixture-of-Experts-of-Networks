dataset: CIFAR10
         --> 05/12 ImageNet으로 변경
         --> 06/02 ImageNet중 10개의 클라스에서만 train, test

main.py: train / validate code

         05/12 training 과정에서 gating network 학습시에 balance loss 추가해서 시도중
         06/02 dataset 변경 이후 재 학습, Gumbel Net에 softmax dimension 변경 

model.py: MixNet S , MixNet M 
       
       두 모델 모두 5번째 layer에서 끊어서 gating 삽입 시도 
       
utils.py: gating network , gumbel softmax 함수  
