cd ..
REM python TestFewShotNoveltyModel.py --model resnet18 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 1 --threshold bayes
REM python TestFewShotNoveltyModel.py --model resnet18 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 1 --threshold std
REM python TestFewShotNoveltyModel.py --model resnet18 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 5 --threshold bayes
REM python TestFewShotNoveltyModel.py --model resnet18 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 5 --threshold std
REM python TestFewShotNoveltyModel.py --model resnet34 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 1 --threshold bayes
REM python TestFewShotNoveltyModel.py --model resnet34 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 1 --threshold std
REM python TestFewShotNoveltyModel.py --model resnet34 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 5 --threshold bayes
REM python TestFewShotNoveltyModel.py --model resnet34 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 5 --threshold std
python TestFewShotNoveltyModel.py --model resnet50 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 1 --threshold bayes
python TestFewShotNoveltyModel.py --model resnet50 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 1 --threshold std
python TestFewShotNoveltyModel.py --model resnet50 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 5 --threshold bayes
python TestFewShotNoveltyModel.py --model resnet50 --weights euMoths --dataset euMoths --novelty True --way 6 --shot 5 --threshold std
cd scripts