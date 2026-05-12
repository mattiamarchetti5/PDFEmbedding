from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
tokenizer.save_pretrained(r"C:\Users\matti\Documents\modelli\nomic-tokenizer")