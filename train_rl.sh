# reformat the command in lines
python train_sac.py \
    --model-name=debug \
    --width 192 --height 192 \
    --map Town03 \
    --spectator \
    --repeat-action 5 \
    --start-location custom \
    --episode-length 200 \
    --sensor rgb \
    --action-type continuous
