cd ..
python TestFewShotNoveltyModel.py --model resnet18 --weights mini_imagenet --dataset miniImagenet --way 5 --shot 1
python TestFewShotNoveltyModel.py --model resnet18 --weights mini_imagenet --dataset miniImagenet --way 5 --shot 5
python TestFewShotNoveltyModel.py --model resnet34 --weights mini_imagenet --dataset miniImagenet --way 5 --shot 1
python TestFewShotNoveltyModel.py --model resnet34 --weights mini_imagenet --dataset miniImagenet --way 5 --shot 5
python TestFewShotNoveltyModel.py --model resnet50 --weights mini_imagenet --dataset miniImagenet --way 5 --shot 1
python TestFewShotNoveltyModel.py --model resnet50 --weights mini_imagenet --dataset miniImagenet --way 5 --shot 5
cd scripts