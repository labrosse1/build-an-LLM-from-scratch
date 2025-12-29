import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "Akwirw ier"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
words = tokenizer.decode(integers)
print(words)