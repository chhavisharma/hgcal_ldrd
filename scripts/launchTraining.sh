# python scripts/heptrx_nnconv.py --categorized --forcecats --cats 4 --optimizer AdamW --model EdgeNetWithCategories --loss nll_loss

python scripts/heptrx_nnconv_test_plot.py --categorized --forcecats --cats 4 --model  EdgeNetWithCategories   --model_weights  ./models/checkpoints/model_checkpoint_EdgeNetWithCategories_264403_5b5c05404f_csharma.best.pth.tar   --dataset  ./data/   --loss nll_loss