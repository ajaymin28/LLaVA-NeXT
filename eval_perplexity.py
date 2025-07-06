import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer (e.g., GPT-2)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Assign the EOS token as the padding token
tokenizer.pad_token = tokenizer.eos_token

def calculate_batch_perplexity(input_texts):
    """
    Calculate perplexity for a batch of input texts using a pretrained language model.

    Args:
    - input_texts (List[str]): A list of input texts to evaluate.

    Returns:
    - List[float]: A list of perplexity scores, one for each input text.
    """
    # Tokenize the batch of texts with padding for uniform length
    inputs = tokenizer(
        input_texts, return_tensors="pt", padding=True, truncation=True
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Pass the input batch through the model to get logits
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Shift the logits and input_ids to align targets correctly
    # Logits dimensions are: (batch_size, seq_length, vocab_size) 
    shift_logits = logits[:, :-1, :]  # Ignore the last token's logits
    shift_labels = input_ids[:, 1:]   # Skip the first token in the labels

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather the log probabilities for the correct tokens
    target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out positions corresponding to padding tokens
    target_log_probs = target_log_probs * attention_mask[:, 1:].to(log_probs.dtype)

    # Compute the mean negative log-likelihood for each sequence
    negative_log_likelihood = -target_log_probs.sum(dim=-1) / attention_mask[:, 1:].sum(dim=-1)

    # Compute perplexity for each sequence
    perplexities = torch.exp(negative_log_likelihood)
    # Take mean of perplexities of each batch
    mean_perplexity_score = torch.mean(perplexities)

    perplexities = perplexities.tolist()

    return {"perplexities": perplexities, "mean_perplexity": mean_perplexity_score}

# Example usage
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step."
]
print(f"Perplexity scores: {calculate_batch_perplexity(texts)}")