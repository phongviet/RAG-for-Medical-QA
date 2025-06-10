examples = [{"prompt": e["prompt"]} for e in mapping]
train_ds = Dataset.from_list(examples)

policy_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
if policy_tokenizer.pad_token_id is None:
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

def tokenize_policy(batch):
    toks = policy_tokenizer(
        batch["prompt"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    return {
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
    }

train_ds = train_ds.map(
    tokenize_policy,
    batched=True,
)
train_ds.set_format(type="torch", columns=["prompt", "input_ids", "attention_mask"])