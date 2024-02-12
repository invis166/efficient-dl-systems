# Changes 
Modelling:
1. fixed noise distribution in the diffusion's forward pass from uniform to normal (torch.rand_like -> torch.randn_like); source: paper
2. fixed formula in the diffusion's forward pass (one_minus_alpha_over_prod -> sqrt_one_minus_alpha_prod); source: paper
3. fixed time embedding size in unet; source: tried to run forward pass and it didn't work
4. added support for GPU training

Tests:
1. fixed seed
2. changed normalization constants
3. disabled randomness
4. ensured that model trains on the correct device

Training:
1. changed normalization constants
2. added inverse normalization at the sample generation stage

# How to run code
1. install requirements from requirements.txt: `pip install -r requirements.txt`
2. set `WANDB_API_KEY` environment variable (or set `WANDB_MODE=offline` if you won't use wandb)
3. init dvc repo: `dvc init`
4. run training: `dvc exp run` (you can change training parameters with `--set-param` argument, [here is](https://dvc.org/doc/user-guide/experiment-management/hydra-composition#running-experiments) an example)

# Process of solving Task 1
Firstly, I ran tests, and they failed. I began by fixing the tests that failed with a runtime error using a debugger. After completing that, I proceeded to fix the tests that failed due to assertion errors. I referred to a DDPM paper, examining the training and sampling algorithms, and compared them with the code, identifying additional errors. Once I finished fixing the existing tests, I added a test for training the entire pipeline. I copied the code from main.py, made slight modifications, and included some assertions. As a result, I achieved 99% coverage.

Coverage report:
```
collected 9 items                                                                                                                                                                                    

tests/test_model.py .....                                                                                                                                                                      [ 55%]
tests/test_pipeline.py ....                                                                                                                                                                    [100%]

---------- coverage: platform linux, python 3.10.12-final-0 ----------
Name                    Stmts   Miss  Cover
-------------------------------------------
modeling/__init__.py        0      0   100%
modeling/diffusion.py      35      0   100%
modeling/training.py       33      1    97%
modeling/unet.py           68      0   100%
-------------------------------------------
TOTAL                     136      1    99%
```

# W&B 
[project with all runs](https://wandb.ai/kolesnikovv028/effdl_example)