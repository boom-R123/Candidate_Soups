
This repository contains the implementation of our paper:

Candidate Soups: Fusing Candidate Results Improves Translation Quality for Non-Autoregressive Translation

Our code is implemented based on DSLP open source code(https://github.com/chenyangh/DSLP). 
And we only changed the inference process of the 4 base models of Vanilla NAT, CMLM, GLAT and GLAT + DSLP.
We train them with reference to the tutorials published by DSLP.




### Inference
Using Candidate Soups without AT model for re-scoring:
```bash
fairseq-generate data-bin/wmt14.en-de_kd  --path PATH_TO_NAT_MODEL \
    --batch-size 1 --gen-subset test --task translation_lev --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 --beam 1 --remove-bpe --iter-decode-with-beam 5 --search-mode CDS
```

Using Candidate Soups with AT model for re-scoring:
```bash
fairseq-generate data-bin/wmt14.en-de_kd  --path PATH_TO_NAT_MODEL:PATH_TO_AT_MODEL --iter-decode-with-external-reranker \
    --batch-size 1 --gen-subset test --task translation_lev --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 --beam 1 --remove-bpe --iter-decode-with-beam 5 --search-mode CDS
```

**Note**: 1) Add `--search-mode CDS ` to indicate the use of the Candidate Soups algorithm.




