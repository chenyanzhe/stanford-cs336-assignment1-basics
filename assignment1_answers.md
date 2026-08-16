## Byte-Pair Encoding (BPE) Tokenizer

### The Unicode Standard

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

### Unicode Encodings

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

### BPE Tokenizer Training