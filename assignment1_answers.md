## 2 Byte-Pair Encoding (BPE) Tokenizer

### 2.1 The Unicode Standard

#### Problem (unicode1)

- (a) `chr(0)` returns the `NUL` character (`'\x00'`). It's a control character, not a printable character.
- (b) `print()` shows the content; `repr()` shows a representation that makes the content unambiguous to a programmer. So for this character, `print()` produces no visiable character, `repr()` returns `\x00`.
- (c) When `U+0000 (NUL)` occurs inside a Python string, it is simply a real character in the string. It does not automatically terminate the Python string (unlike languages such as C).

   When debugging tokenizer code, prefer:

   ```python
   print(repr(text))
   ```

   rather than:

   ```python
   print(text)
   ```

  because repr() makes characters such as:

  ```
  \n
  \t
  \r
  \x00
  ```
  visible.

### 2.2 Unicode Encodings

#### Problem (unicode2)

- (a) 1. UTF-8 is space-efficient for typical text.
  
    UTF-8 uses a variable number of bytes:
    - ASCII characters -> 1 byte
    - Many European characters -> 2 bytes
    - Many CJK characters -> 3 bytes
      
    UTF-16 generally uses 2 or 4 bytes per character, and UTF-32 always uses 4 bytes.

    So UTF-8 tends to give you **shorter sequences before BPE merging**, especially for English.

  2. UTF-8 has a very small, fixed base vocabulary
  
      With UTF-8, you can start a byte-level tokenizer with only 256 possible byte values.

  UTF-8 isn't always better in terms of **number of bytes per character**. However, it gives you a simple, tiny, universal base vocabulary while remaining reasonably compact for the world's most common text formats.

- (b) Because UTF-8 encodes a Unicode character using 1–4 bytes, and this function tries to decode each byte independently. An example input byte string could be "你好".encode("utf-8").

- (c) In UTF-8, a 2-byte sequence must have the form:

      ```
      110xxxxx 10xxxxxx
      ```

  So `b"\xE0\x80"` will be an invalid UTF-8 character.

  It shows that **not every arbitrary sequence of bytes is valid UTF-8**.

### 2.5 Experimenting with BPE Tokenizer Training

#### Problem (train_bpe_tinystories)

- (a) `$ uv run python -m cs336_basics.train_bpe --num-processes 32`

  - Total time: ~49s using 32 processes.
  - Memory consumption: N/A.
  - The longest token: b' accomplishment'
  - Does it make sense?

- (b) Time breakdown - pre-tokenization: 31s, merges: 18s.

  - Parallelization helps reduce pre-tokenization time. As a comparison, it took 202s for pre-tokenization when using 4 processes.
  - Using heap for pair counts (O(1) to find the pair with the largest count) greatly reduces merge time. As a comparison, it took 114s in merge without heap.

#### Problem (train_bpe_expts_owt)

- (a) `uv run python -m cs336_basics.train_bpe --input-path "data/owt_train.txt" --num-processes 64 --vocab-size 32000`

  - Total time: 6182s (~1.7h) using 64 processes.
  - Time breakdown - pre-tokenization: 183s, merges: 5999s

    > It took 31505s (~8.8h) in merge without heap.

  - The longest token: ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ

    ```python
    b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'
    ```
  - Does it make sense?

- (b) OpenWebText is more complex, it comes from the real world.

### 2.7 Experiments

#### Problem (tokenizer_experiments)

- (a) TinyStories tokenizer's compression ratio is 0.2448, OpenWebText tokenizer's compression ratio is 0.2284.
- (b) The compression ratio is 0.3159, which is much larger. It's likely because TinyStories dataset is simpler than OpenWebText dataset, so texts from OpenWebText can't be merged efficiently with the TinyStories tokenizer.
- (c) The TinyStories tokenizer throughput: 5.81 MB/s, OpenWebText tokenizer throughput: 5.65 MB/s. It will take ~1.8 days to tokenize the Pile dataset.
- (d) Because uint16 can represent up to 65536 integers, which is enough for 32K vocabulary size.

## 3 Transformer Language Model Architecture

### 3.5 The Full Transformer LM

#### Problem (transformer_accounting)

- (a) Our model will have 1,640,452,800 trainable parameters:
  
  - embeddings: 80,411,200  (4.90%)
  - rms: 153,600  (0.01%)
  - ffn: 987,955,200  (60.22%)
  - mha: 491,520,000  (29.96%)
  - final_rms: 1,600  (0.00%)
  - lm_head: 80,411,200  (0.05%)

  It will take ~6.4G memory to just load this model. See `scripts/transformer_accounting.py` for detailed calculations.

- (b) The matrix multiplies require in total 3,517,087,744,000 FLOPs:

  - embeddings: 0  (0.00%)
  - rms: 314,572,800  (0.01%)
  - ffn: 2,023,332,249,600  (57.53%)
  - mha: 1,328,755,507,200  (37.78%)
  - final rms: 3,276,800  (0.00%)
  - lm_head: 164,682,137,600  (0.05%)

- (c) The FFN and MHA require the most FLOPs.
- (d) Breakdown of model components and its associated FLOPs:

      GPT-2 small:
      The matrix multiplies require in total 291,687,628,800 FLOPs:
        - embeddings: 0  (0.00%)
        - rms: 37,748,736  (0.01%)
        - ffn: 115,964,116,992  (39.76%)
        - mha: 96,636,764,160  (33.13%)
        - final rms: 1,572,864  (0.00%)
        - lm_head: 79,047,426,048  (0.27%)

      GPT-2 medium:
      The matrix multiplies require in total 830,275,059,712 FLOPs:
        - embeddings: 0  (0.00%)
        - rms: 100,663,296  (0.01%)
        - ffn: 415,538,085,888  (50.05%)
        - mha: 309,237,645,312  (37.25%)
        - final rms: 2,097,152  (0.00%)
        - lm_head: 105,396,568,064  (0.13%)

      GPT-2 large:
      The matrix multiplies require in total 1,768,722,268,160 FLOPs:
        - embeddings: 0  (0.00%)
        - rms: 188,743,680  (0.01%)
        - ffn: 960,327,843,840  (54.30%)
        - mha: 676,457,349,120  (38.25%)
        - final rms: 2,621,440  (0.00%)
        - lm_head: 131,745,710,080  (0.07%)

    When the model size grows, the FFN FLOPs grows rapidly.
- (e) GPT-2 XL with 16X context length (1024->16384):
  - Total FLOPs: 133,582,815,232,000 (~38X compared to 1024 context length).
  - embeddings: 0  (0.00%)
  - rms: 5033164800  (0.00%)
  - ffn: 32373315993600  (24.23%)
  - mha: 98569499443200  (73.79%)
  - final rms: 52428800  (0.00%)
  - lm_head: 2634914201600  (0.02%)

  The MHA becomes the majority part of FLOPs.