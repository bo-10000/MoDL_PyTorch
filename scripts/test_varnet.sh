GPU_NUM=0
TEST_CONFIG_YAML="configs/base_varnet,k=10.yaml"

CUDA_VISIBLE_DEVICES=$GPU_NUM python test.py \
    --config=$TEST_CONFIG_YAML \
    --batch_size=32 \
    --write_image=1