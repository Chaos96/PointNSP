# NeRF Version
torchrun --nproc_per_node=1 --nnodes=1 \
  --rdzv-endpoint=localhost:2980 --rdzv_backend=c10d test.py \
  --depth=16 --fp16=2  \
  --vqvae_pretrained_path ./checkpoint/vqvae-ckpt.pt \
  --ar_ckpt_path ./checkpoint/text-condition-ckpt.pth \
  --save_path ./eval \
  --flexicubes False \
  --text_conditioned True \
  --text_json_path ./test_files/test_text.json

# Flexicubes Version
# torchrun --nproc_per_node=1 --nnodes=1 \
#   --rdzv-endpoint=localhost:2980 --rdzv_backend=c10d test.py \
#   --depth=16 --fp16=2  \
#   --vqvae_pretrained_path ./checkpoint/vqvae-flexicubes-ckpt.pt \
#   --ar_ckpt_path ./checkpoint/text-condition-ckpt.pth \
#   --save_path ./eval \
#   --flexicubes True \
#   --text_conditioned True \
#   --text_json_path ./test_files/test_text.json