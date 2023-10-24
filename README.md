# FREMAX

FREMAX: A SIMPLE METHOD TOWARDS TRULY SECURE GENERATIVE LINGUISTIC STEGANOGRAPHY

Generative Linguistic Steganography (GLS) is a popular technique that utilizes Language Models (LMs) to hide secret messages within seemingly innocuous texts. GLS aims to generate steganographic texts (stegos) similar to normal carrier texts (covers). The mainstream GLS method, Provably Secure Steganography(PSS), has developed to guarantee imperceptibility between stegos and covers. However, there exists a statical difference between LM-generated texts and human texts, making the stegos more detectable. To bridge the gap between stegos and covers, this paper proposed a probability reformation method named Frequency REformed Softmax (FREmax) for generating highly imperceptible stegos aligned to natural language. We reformed the softmax function based on the tokens’ frequency distribution of human corpus. Extensive experimental results show that FREmax can improve the linguistic quality and imperceptibility of the generated stegos, proving it a valuable remedy to existing GLS methods.

## Usage
### Preparation
### Run an exmaple
```shell
python Generate.py \
--mode=G
--gpuid=0
--seed=42
--out_dir=path/to/your/output
--generation_config=generate.json
--
 ``` 
