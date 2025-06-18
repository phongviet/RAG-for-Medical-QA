import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    GenerationConfig
)
from trl import PPOConfig, PPOTrainer


def prepare_preference_data(mapping_path: str, output_path: str, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    # load mapping of prompts to gold completions
    mapping = json.load(open(mapping_path, "r"))

    # load generator
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    def generate_answer(prompt: str, max_new_tokens: int = 256) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # build preference dataset
    pref_data = []
    for entry in mapping:
        prompt = entry["prompt"]
        gold = entry.get("completion", "")
        sample = generate_answer(prompt)
        pref_data.append({
            "prompt": prompt,
            "answer_A": gold,
            "answer_B": sample,
            "preference": 0
        })

    # save to jsonl
    with open(output_path, "w", encoding="utf-8") as fout:
        for d in pref_data:
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")



def train_reward_model(
    data_path: str,
    output_dir: str,
    base_model: str = "bert-base-uncased"
):
    """
    Train a reward model from preference data.
    """
    # load dataset
    ds = load_dataset(
        "json",
        data_files={"train": data_path}
    )["train"]

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def preprocess(example):
        enc = tokenizer(
            example["answer_A"], example["answer_B"],
            max_length=256, truncation=True, padding="max_length"
        )
        enc["labels"] = example["preference"]
        return enc

    ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=50,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds
    )
    trainer.train()
    trainer.save_model(output_dir)



def prepare_rlhf_data(
    mapping_path: str,
    max_length: int = 256
) -> Dataset:
    """
    Create a torch Dataset of prompts for RLHF.
    """
    mapping = json.load(open(mapping_path, "r"))
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
            max_length=max_length
        )
        return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}

    train_ds = train_ds.map(tokenize_policy, batched=True, remove_columns=["prompt"])
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return train_ds


def run_ppo(
    train_dataset: Dataset,
    reward_model_dir: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ppo_lr: float = 1e-5,
    batch_size: int = 4,
    mini_batch_size: int = 1,
    device: torch.device = None
):
    """
    Run RLHF using PPOTrainer.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_dir,
        local_files_only=True
    ).to(device)

    # patch forward to remove unused keys
    _orig_forward = reward_model.forward
    def _patched(self, *args, **kwargs):
        for k in ("use_cache","output_attentions","output_hidden_states"):
            kwargs.pop(k, None)
        return _orig_forward(self, *args, **kwargs)
    reward_model.forward = types.MethodType(_patched, reward_model)

    reward_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # score function
    def reward_fn(prompts: list[str], responses: list[str]) -> list[float]:
        return reward_model.score(prompts, responses)

    # adapter for classifier
    adapter = torch.nn.Linear(
        reward_model.config.hidden_size, reward_model.config.hidden_size
    ).to(device)

    _orig_cls = reward_model.classifier
    def score(self, hidden_states):
        dtype = _orig_cls.weight.dtype
        hs = hidden_states.to(dtype)
        b, s, h = hs.size()
        flat = hs.view(-1, h)
        projected = adapter(flat)
        flat_logits = _orig_cls(projected)
        return flat_logits.view(b, s, -1)
    reward_model.score = types.MethodType(score, reward_model)

    # load policy and ref models
    policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)
    policy.generation_config = GenerationConfig.from_model_config(policy.config)

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for m in (getattr(policy, n) for n in dir(policy)):
        if hasattr(m, "base_model_prefix"):
            tokenizer.base_model_prefix = m.base_model_prefix
            break

    # data collator
    def collator_fn(features):
        texts = [f["prompt"] for f in features]
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

    ppo_config = PPOConfig(
        learning_rate=ppo_lr,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size
    )

    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=train_dataset,
        value_model=policy,
        data_collator=collator_fn
    )

    # redirect score methods
    ppo_trainer.model.policy.score = types.MethodType(reward_model.score, ppo_trainer.model.policy)
    ppo_trainer.model.value_model.score = types.MethodType(reward_model.score, ppo_trainer.model.value_model)

    # train and save
    ppo_trainer.train()
    ppo_trainer.model.save_pretrained(output_dir)


if __name__ == "__main__":
    # file paths (modify as needed)
    qa_mapping = "./data/faiss_index/qa_mapping.json"
    pref_path = "./data/preference_data.jsonl"
    reward_dir = "./data/reward-model"
    rlhf_out = "./data/qwen-rlhf"

    # 1. Prepare preference data
    prepare_preference_data(qa_mapping, pref_path)

    # 2. Train reward model
    train_reward_model(pref_path, reward_dir)

    # 3. Prepare RLHF data
    train_ds = prepare_rlhf_data(qa_mapping)

    # 4. Run PPO
    run_ppo(train_ds, reward_dir, rlhf_out)