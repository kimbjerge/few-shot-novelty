#python FewShotTraining.py --model resnet18 --dataset CUB --mode classic --epochs 10 --shot 1
#python FewShotTraining.py --model resnet34 --dataset CUB --mode classic --epochs 10 --shot 1
#python FewShotTraining.py --model resnet50 --dataset CUB --mode classic --epochs 10 --shot 1
python FewShotTraining.py --model resnet18 --dataset euMoths --mode classic --epochs 1 --shot 5 --softmax True
python FewShotTraining.py --model resnet34 --dataset euMoths --mode classic --epochs 1 --shot 5 --softmax True
python FewShotTraining.py --model resnet50 --dataset euMoths --mode classic --epochs 1 --shot 5 --softmax True
