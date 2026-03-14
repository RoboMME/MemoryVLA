# micromamba activate robomme
# when the server is running, then run this script

python evaluation/robomme/eval.py \
    --args.model_seed 7 \
    --args.model_ckpt_id 160000 \
    --args.policy_name memvla \
    --args.port 8051