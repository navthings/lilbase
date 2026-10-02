# lilbase

a 297m param language model i trained from scratch on a free kaggle tpu. it's a base model, so it continues whatever text you give it instead of answering questions. the chat version is [lilchat](https://ollama.com/navthings/lilchat), and the bigger follow-up is [sprout](https://github.com/navthings/sprout).

## try it

```
ollama run navthings/lilbase "The water cycle begins when"
```

give it the start of a sentence, not a question. tags are `latest` (q8_0, 379mb, basically no loss vs f16), `q4_k_m` (274mb, a tiny bit worse) and `f16` (595mb). weights and ggufs are on [hugging face](https://huggingface.co/navthings/lilbase).

## what it writes

> The heart is a muscular organ that has two lobes, one on the right side and one on the left. It pumps blood into the body by way of the arteries. The right side is connected to the lungs and the left side connects with the heart. The heart contains two main chambers, the ventricles (smaller) and the mitral valve (large).

> Here are some tips for studying effectively:
>
> 1. Keep your schedule. If you're not sure how to study effectively, remember that studying is a mental exercise. Take some time off, relax and be more organized.
> 2. Use flash cards. Using flash cards can help you to study better.
> 3. Study early in the day. So you should try to study before you have any breakfast.

the grammar holds up, the facts don't always (the mitral valve is not a chamber).

## vs gpt-2

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/vs_gpt2_dark2.png">
  <img alt="lilbase vs gpt-2 small" src="assets/vs_gpt2_light2.png">
</picture>

a quick first comparison against gpt-2 small, using [`eval/compare.py`](eval/compare.py). the bigger, more careful one, against all four gpt-2 sizes with more questions, is in [sprout](https://github.com/navthings/sprout#how-it-did), and lilbase is in there too.

## what it is

llama style: grouped query attention, rope, rmsnorm, swiglu, tied embeddings. 24 layers, 1024 wide, 16 query heads and 4 kv heads, 1024 token context, llama tokenizer (32k vocab).

it read 6.1 billion tokens of [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT), which is about 20 tokens per parameter, roughly the chinchilla amount for this size.

## training

- 524,288 tokens a step, data parallel over all 8 chips of a tpu v5e-8, bf16 matmuls with some fp32
- adamw at 4e-4, 1000 warmup steps, then cosine down to 10%
- planned 11,634 steps. the 8.4 hour session limit stopped it at 11,043, but the learning rate was already at its floor by then so the last 5% wouldn't have changed much
- about 193k tokens a second across the 8 chips the whole way
- held-out loss 2.608 (perplexity 13.6)

## running it yourself

easiest is kaggle. import `tpu/lilbase_tpu_kaggle.ipynb`, set the accelerator to tpu v5e-8, then save & run all. or just open [my notebook](https://www.kaggle.com/code/navneetdagdiya/base-tpu-kaggle). kaggle already has everything installed.

it refuses to start unless jax sees all 8 chips. kaggle sometimes hands out a broken allocation with 1 chip, and its better to restart than to train at 1/8th speed without noticing.

if you have your own 8 chip tpu: `pip install -r tpu/requirements.txt`, then `python tpu/lilbase.py`.

## knobs

all at the top of `tpu/lilbase.py`. the ones worth touching:

- `BS, GRAD_ACC`: 8x8. 16x4 ran out of tpu memory by about 400mb. if it overflows on the first compile anyway, it halves the batch and tries again on its own.
- `MAX_HOURS`: 8.4. lower it if checkpoint saves start getting cut close.
- `LR`: 4e-4. gpt-3 350m used 3e-4 at a similar batch size, so this is a bit optimistic.
- `SAVE_EVERY`: 500 steps.

the prompts it prints at the end are at the bottom of the file, change them to test the model on different things.

## logs

every 50 steps (`PRINT_EVERY`) it prints the average loss. every 250 (`STATS_EVERY`) that line also shows grad norm, learning rate, tokens a second, time left, host ram, tpu memory peak and `q`. if `q` sits near 0, tokenizing is the bottleneck and the tpu is waiting on the cpu.

## branches

`main` is the tpu training and the eval. `mac` has the scripts for turning the weights into hugging face and gguf files and sampling on a mac.
