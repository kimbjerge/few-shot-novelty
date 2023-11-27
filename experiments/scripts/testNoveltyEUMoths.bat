REM python TestFewShotNoveltyModel.py --model resnet18 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 1 --threshold bayes
REM python TestFewShotNoveltyModel.py --model resnet18 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 1 --threshold std
REM python TestFewShotNoveltyModel.py --model resnet18 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 5 --threshold bayes
REM python TestFewShotNoveltyModel.py --model resnet18 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 5 --threshold std
REM python TestFewShotNoveltyModel.py --model resnet34 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 1 --threshold bayes
REM python TestFewShotNoveltyModel.py --model resnet34 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 1 --threshold std
REM python TestFewShotNoveltyModel.py --model resnet34 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 5 --threshold bayes
REM python TestFewShotNoveltyModel.py --model resnet34 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 5 --threshold std
REM python TestFewShotNoveltyModel.py --model resnet50 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 1 --threshold bayes
REM python TestFewShotNoveltyModel.py --model resnet50 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 1 --threshold std
REM python TestFewShotNoveltyModel.py --model resnet50 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 5 --threshold bayes
REM python TestFewShotNoveltyModel.py --model resnet50 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 5 --threshold std

python TestFewShotNoveltyModelAdv.py --model resnet18 --weights ImageNet --dataset euMoths --novelty True --way 5 --shot 5 --query 6 --threshold bayes
python TestFewShotNoveltyModelAdv.py --model resnet18 --weights ImageNet --dataset euMoths --novelty True --way 5 --shot 1 --query 6 --threshold bayes
python TestFewShotNoveltyModelAdv.py --model resnet34 --weights ImageNet --dataset euMoths --novelty True --way 5 --shot 5 --query 6 --threshold bayes
python TestFewShotNoveltyModelAdv.py --model resnet34 --weights ImageNet --dataset euMoths --novelty True --way 5 --shot 1 --query 6 --threshold bayes
python TestFewShotNoveltyModelAdv.py --model resnet50 --weights ImageNet --dataset euMoths --novelty True --way 5 --shot 5 --query 6 --threshold bayes
python TestFewShotNoveltyModelAdv.py --model resnet50 --weights ImageNet --dataset euMoths --novelty True --way 5 --shot 1 --query 6 --threshold bayes
