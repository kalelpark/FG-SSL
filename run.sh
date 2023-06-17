python main.py --seed 0 --model pmg --patches 32 16 8 --dataset aptos --gpu_ids 4
python main.py --seed 0 --model pmg --patches 8 4 2 --dataset aptos --imgsize 450 --gpu_ids 3
python main.py --seed 0 --model pmg --patches 32 16 8 --dataset aptos --imgsize 450 --gpu_ids 2 
python main.py --seed 0 --model pmg --patches 16 8 4 --dataset isic2017 --gpu_ids 6
python main.py --seed 0 --model pmg --patches 16 8 4 --dataset isic2018 --gpu_ids 7


python main.py --seed 0 --model resnet --dataset aptos --gpu_ids 0
python main.py --seed 0 --model resnet --dataset isic2017 --gpu_ids 1
python main.py --seed 0 --model resnet --dataset isic2018 --gpu_ids 2