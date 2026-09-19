# Fine-Tuning `qwen2.5` for the Family Travel Agent (Unsloth + Colab Free T4)

This guide fine-tunes Qwen2.5 (3B or 7B) on your own family travel Q&A pairs
so the model internalizes the regional priorities and translation-flexibility
behavior directly, instead of relying entirely on the system prompt every
call. Everything below runs on Colab's free T4 GPU tier.

## 1. Dataset Preparation

Format 30–50 Q&A pairs in Qwen ChatML JSONL, one JSON object per line. Cover:
travel recommendations, regional priority reasoning (unvisited > visited),
and the translation dynamic (grandparents' language is not a blocker).

```jsonl
{"messages": [{"role": "system", "content": "You are the family travel assistant. Language is never a blocking factor — the family translates on-site. Prioritize unvisited high-value regions over repeat destinations."}, {"role": "user", "content": "We haven't done Western Europe yet — what would you suggest?"}, {"role": "assistant", "content": "Western Europe is a strong pick since you haven't been — a Rhine or Rhine/Mosel river cruise would fit the family's low-walking, fixed-cost pattern well, similar to the Danube trip. Paris/Amsterdam by rail is a solid backup if you'd rather do cities."}]}
{"messages": [{"role": "system", "content": "You are the family travel assistant. Language is never a blocking factor — the family translates on-site. Prioritize unvisited high-value regions over repeat destinations."}, {"role": "user", "content": "Is Singapore worth a repeat visit?"}, {"role": "assistant", "content": "Singapore was great last time, but since you've already been, I'd lean toward a new high-priority region like the Baltics or Western Mediterranean first, and save a Singapore repeat for a shorter trip later."}]}
```

Save this as `training_data.jsonl` with 30–50 lines like the examples above,
varying phrasing, budget figures, and regions so the model generalizes
rather than memorizing exact wording.

## 2. Google Colab Setup

1. Open a new Colab notebook (or the official Unsloth Qwen2.5 example notebook).
2. **Runtime → Change runtime type → T4 GPU**.
3. Install Unsloth and dependencies:

   ```python
   !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   !pip install --no-deps trl peft accelerate bitsandbytes
   ```

4. Load the base model (swap to the 3B variant if you're on a slower/free
   tier quota, or if you want a smaller model to match a lighter local Ollama
   deployment):

   ```python
   from unsloth import FastLanguageModel

   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",  # or "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
       max_seq_length=2048,
       dtype=None,
       load_in_4bit=True,
   )
   ```

## 3. LoRA Configuration

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

## 4. Training

Load `training_data.jsonl` as a Hugging Face `Dataset`, then:

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,  # your loaded training_data.jsonl
    dataset_text_field="text",  # after applying the ChatML template
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=200,  # ~100-300 steps is enough for 30-50 examples
        learning_rate=2e-4,
        optim="adamw_8bit",
        logging_steps=10,
        output_dir="outputs",
    ),
)
trainer.train()
```

With 30–50 examples, 100–300 steps is plenty — more risks overfitting to
the exact training phrasing rather than the underlying pattern.

## 5. Export to Ollama

```python
model.save_pretrained_gguf(
    "qwen_family_agent",
    tokenizer,
    quantization_method="q4_k_m",
)
```

Download the resulting `qwen_family_agent-unsloth.Q4_K_M.gguf` file from
Colab to your local machine, then:

1. Create a `Modelfile` next to the downloaded `.gguf`:

   ```
   FROM ./qwen_family_agent-unsloth.Q4_K_M.gguf
   ```

2. Import it into your local Ollama:

   ```bash
   ollama create family-qwen -f Modelfile
   ```

3. Point `router.py` at it by setting the env var before running Streamlit:

   ```bash
   export FAMILY_AGENT_QWEN_MODEL=family-qwen
   streamlit run app.py
   ```

## Notes

- If Colab's free T4 disconnects mid-training, `output_dir="outputs"`
  checkpoints let you resume rather than restarting from step 0.
- Re-run `eval.py` against the fine-tuned model (via the env var above) and
  compare its report to a baseline run against the stock model — that's
  your signal for whether the fine-tune actually changed behavior or just
  latency.
