# Haskell GPT

Super minimal implementation of [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) in Haskell.

Highly inspired by:
- [openai/gpt-2](https://github.com/openai/gpt-2).
- [karpathy/minGPT](https://github.com/karpathy/mingpt).
- [picoGPT](https://github.com/jaymody/picoGPT).

Structure is similar to [picoGPT](https://github.com/jaymody/picoGPT). Code contains:
- Translation of OpenAI's BPE Tokenizer
- Decoder-only transformer
- GPT-2 generation

You can run tests with

```
cabal build
cabal test
```

(You'll first need to download GPT-2 weights, tokenizer, and hyperparams into `/models`)

## TODO
- [ ] Merge changes containing tensorflow inference for actual text generation
