cd ..
python TestFewShotNoveltyModelAdv.py --model resnet12 --weights Omniglot --dataset Omniglot --alpha 0.7 --novelty True --way 5 --shot 1 --threshold bayes
python TestFewShotNoveltyModelAdv.py --model resnet12 --weights Omniglot --dataset Omniglot --alpha 0.7 --novelty True --way 5 --shot 1 --threshold std
python TestFewShotNoveltyModelAdv.py --model resnet12 --weights Omniglot --dataset Omniglot --alpha 0.7 --novelty True --way 5 --shot 5 --threshold bayes
python TestFewShotNoveltyModelAdv.py --model resnet12 --weights Omniglot --dataset Omniglot --alpha 0.7 --novelty True --way 5 --shot 5 --threshold std
python TestFewShotNoveltyModelAdv.py --model resnet12 --weights Omniglot --dataset Omniglot --alpha 0.7 --way 5 --shot 1
python TestFewShotNoveltyModelAdv.py --model resnet12 --weights Omniglot --dataset Omniglot --alpha 0.7 --way 5 --shot 5
python TestFewShotNoveltyModelAdv.py --model resnet12 --weights Omniglot --dataset Omniglot --alpha 0.9 --novelty True --way 5 --shot 1 --threshold bayes
python TestFewShotNoveltyModelAdv.py --model resnet12 --weights Omniglot --dataset Omniglot --alpha 0.9 --novelty True --way 5 --shot 1 --threshold std
python TestFewShotNoveltyModelAdv.py --model resnet12 --weights Omniglot --dataset Omniglot --alpha 0.9 --novelty True --way 5 --shot 5 --threshold bayes
python TestFewShotNoveltyModelAdv.py --model resnet12 --weights Omniglot --dataset Omniglot --alpha 0.9 --novelty True --way 5 --shot 5 --threshold std
python TestFewShotNoveltyModelAdv.py --model resnet12 --weights Omniglot --dataset Omniglot --alpha 0.9 --way 5 --shot 1
python TestFewShotNoveltyModelAdv.py --model resnet12 --weights Omniglot --dataset Omniglot --alpha 0.9 --way 5 --shot 5
cd scripts