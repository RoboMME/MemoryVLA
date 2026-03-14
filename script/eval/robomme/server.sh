# We follow the LIBERO Setup
# micromamba activate memvla

python deploy.py \
    --saved_model_path memoryvla_ckpt/checkpoints/step-160500-epoch-13-loss=0.0051.pt \
    --unnorm_key robomme \
    --adaptive_ensemble_alpha 0.1 \
    --cfg_scale 1.5 \
    --port 8051 \
    --action_chunking \
    --action_chunking_window 8 \
    --seed 7

