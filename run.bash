env=RandomMaze-v0
# model=Impala-PPO
model=HELM-llm
seed=0

CUDA_VISIBLE_DEVICES=0 \
nohup python main.py \
--seed ${seed} \
--run-id 0 \
--var env=${env} \
--var model=${model} \
> logs/${env}_${model}_${seed}.log 2>&1 &
