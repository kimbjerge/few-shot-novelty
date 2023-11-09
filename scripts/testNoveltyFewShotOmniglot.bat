cd ..
python TestFewShotNoveltyModel.py --model resnet12 --weights Omniglot --dataset Omniglot --novelty True --way 6 --shot 1 --threshold bayes
python TestFewShotNoveltyModel.py --model resnet12 --weights Omniglot --dataset Omniglot --novelty True --way 6 --shot 1 --threshold std
python TestFewShotNoveltyModel.py --model resnet12 --weights Omniglot --dataset Omniglot --novelty True --way 6 --shot 5 --threshold bayes
python TestFewShotNoveltyModel.py --model resnet12 --weights Omniglot --dataset Omniglot --novelty True --way 6 --shot 5 --threshold std
REM python TestFewShotNoveltyModel.py --model resnet12 --weights Omniglot --dataset Omniglot --way 5 --shot 1
REM python TestFewShotNoveltyModel.py --model resnet12 --weights Omniglot --dataset Omniglot --way 5 --shot 5
cd scripts