import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt

# =============================================================================
# Utility function to generate the first n prime numbers (naively)
# =============================================================================
def generate_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if num % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
        num += 1
    return primes

# =============================================================================
# Dataset for training: each sample is a sliding window from the text.
# =============================================================================
class TextDataset(Dataset):
    def __init__(self, text, seq_length, char_to_idx):
        self.text = text
        self.seq_length = seq_length  # total length = context length + 1 (target)
        self.char_to_idx = char_to_idx
        
    def __len__(self):
        return len(self.text) - self.seq_length + 1
    
    def __getitem__(self, idx):
        sample = self.text[idx:idx+self.seq_length]
        input_seq = sample[:-1]   # context characters
        target_char = sample[-1]    # target character
        input_indices = torch.tensor([self.char_to_idx[c] for c in input_seq], dtype=torch.long)
        target_index = torch.tensor(self.char_to_idx[target_char], dtype=torch.long)
        return input_indices, target_index

# =============================================================================
# PatternEncoder: Integrates signal encoding, tokenization, and token refinement.
# =============================================================================
class PatternEncoder(nn.Module):
    """
    PatternEncoder:
    ------------------
    This block integrates the signal encoding, tokenization, and token refinement steps.
    
    It consists of three sequential operations:
    
    1. Signal Encoding:
       - Computes a unique sine-based signal for each input character using a prime-associated frequency.
       - Applies positional cyclic shifts and sums the signals.
       - Normalizes the aggregated signal using L2 normalization.
       
    2. Tokenization:
       - Projects the normalized signal to produce logits corresponding to multiple tokens.
       - Applies a softmax to obtain a probability distribution over a token vocabulary.
       - Computes token embeddings as a weighted average over a learnable embedding table.
       - Applies multi-head self-attention with residual connection, dropout, and layer normalization to refine token embeddings.
       
    3. Token Refinement:
       - Applies an additional self-attention layer with residual connection, dropout, and normalization to further refine token embeddings.
       - Flattens the refined token embeddings for downstream processing.
       
    The output is a compact token representation (token_flat) that serves as the input for pattern decoding.
    """
    def __init__(self, signal_length, m_tokens, token_vocab_size, token_embedding_dim, dropout_prob, device):
        super(PatternEncoder, self).__init__()
        self.signal_length = signal_length
        self.m_tokens = m_tokens
        self.token_vocab_size = token_vocab_size
        self.token_embedding_dim = token_embedding_dim
        self.device = device
        # Signal Encoder: Create time vector.
        t = torch.linspace(0, 1, steps=signal_length, device=device)
        self.register_buffer("time_vector", t)
        
        # Tokenizer:
        self.linear = nn.Linear(signal_length, m_tokens * token_vocab_size)
        self.token_embedding = nn.Embedding(token_vocab_size, token_embedding_dim)
        self.token_attn = nn.MultiheadAttention(embed_dim=token_embedding_dim, num_heads=4, batch_first=True)
        self.token_dropout = nn.Dropout(0.1)
        self.token_norm = nn.LayerNorm(token_embedding_dim)
        
        # TokenRefiner:
        self.refiner_attn = nn.MultiheadAttention(embed_dim=token_embedding_dim, num_heads=4, batch_first=True)
        self.refiner_dropout = nn.Dropout(dropout_prob)
        self.refiner_norm = nn.LayerNorm(token_embedding_dim)
        
    def forward(self, input_indices, prime_tensor):
        # Signal Encoding:
        # input_indices: (batch, seq_length)
        freqs = prime_tensor[input_indices]  # (batch, seq_length)
        t = self.time_vector.unsqueeze(0).unsqueeze(0)  # (1,1,signal_length)
        freqs_expanded = freqs.unsqueeze(-1)  # (batch, seq_length, 1)
        signals = torch.sin(2 * math.pi * freqs_expanded * t)  # (batch, seq_length, signal_length)
        
        shifted_signals = []
        seq_length = input_indices.size(1)
        for i in range(seq_length):
            signal_i = signals[:, i, :]  # (batch, signal_length)
            shifted = torch.roll(signal_i, shifts=i, dims=1)
            shifted_signals.append(shifted)
        context_signal = torch.stack(shifted_signals, dim=1).sum(dim=1)  # (batch, signal_length)
        norm = context_signal.norm(p=2, dim=1, keepdim=True) + 1e-8
        normalized_signal = context_signal / norm  # (batch, signal_length)
        
        # Tokenization:
        batch_size = normalized_signal.size(0)
        logits = self.linear(normalized_signal)  # (batch, m_tokens * token_vocab_size)
        logits = logits.view(batch_size, self.m_tokens, self.token_vocab_size)  # (batch, m_tokens, token_vocab_size)
        probs = F.softmax(logits, dim=-1)
        token_embeds = torch.matmul(probs, self.token_embedding.weight)  # (batch, m_tokens, token_embedding_dim)
        attn_out, _ = self.token_attn(token_embeds, token_embeds, token_embeds)
        attn_out = self.token_dropout(attn_out)
        token_embeds = self.token_norm(token_embeds + attn_out)
        
        # Token Refinement:
        attn_out2, _ = self.refiner_attn(token_embeds, token_embeds, token_embeds)
        attn_out2 = self.refiner_dropout(attn_out2)
        refined = self.refiner_norm(token_embeds + attn_out2)
        token_flat = refined.reshape(batch_size, -1)  # (batch, m_tokens * token_embedding_dim)
        
        return token_flat, logits, token_embeds, normalized_signal

# =============================================================================
# PatternDecoder: Projects token representations to learned patterns and decodes them into characters.
# =============================================================================
class PatternDecoder(nn.Module):
    """
    PatternDecoder:
    ------------------
    This block takes the flattened token representation and projects it into a latent pattern space.

    It begins with a linear projection that transforms the token representation into pattern logits.
    Then, a pre-decoder multi-head self-attention layer is applied to refine these pattern logits.
    A residual connection adds the original pattern logits to the attention output, followed by dropout
    and a layer normalization step. This refined pattern is then decoded into character logits through a final linear layer,
    which will be used to predict the next character in the sequence.
    """
    def __init__(self, m_tokens, token_embedding_dim, token_vocab_size, char_vocab_size):
        super(PatternDecoder, self).__init__()
        # Project flattened token embeddings to pattern logits.
        self.pattern_prediction = nn.Linear(m_tokens * token_embedding_dim, token_vocab_size)
        # Pre-decoder attention to refine patterns.
        self.pre_decoder_attn = nn.MultiheadAttention(embed_dim=token_vocab_size, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(token_vocab_size)
        # Final decoder: map refined patterns to character logits.
        self.char_decoder = nn.Linear(token_vocab_size, char_vocab_size)
        
    def forward(self, token_flat):
        # Decoding all the way to character logits.
        pattern_logits = self.pattern_prediction(token_flat)
        pattern_logits_seq = pattern_logits.unsqueeze(1)  # (batch, 1, token_vocab_size)
        attn_out, _ = self.pre_decoder_attn(pattern_logits_seq, pattern_logits_seq, pattern_logits_seq)
        attn_out = self.dropout(attn_out)
        refined = self.norm(pattern_logits_seq + attn_out).squeeze(1)  # (batch, token_vocab_size)
        char_logits = self.char_decoder(refined)
        return char_logits

    def decode_debug(self, token_flat):
        """
        Runs the full decoding process and returns intermediate outputs as a dictionary.
        """
        outputs = {}
        pattern_logits = self.pattern_prediction(token_flat)
        outputs["pattern_logits"] = pattern_logits

        pattern_logits_seq = pattern_logits.unsqueeze(1)  # (batch, 1, token_vocab_size)
        attn_out, pre_decoder_attn_weights = self.pre_decoder_attn(pattern_logits_seq, pattern_logits_seq, pattern_logits_seq)
        attn_out = self.dropout(attn_out)
        refined = self.norm(pattern_logits_seq + attn_out).squeeze(1)  # (batch, token_vocab_size)
        outputs["refined_pattern"] = refined
        outputs["pre_decoder_attn_weights"] = pre_decoder_attn_weights

        char_logits = self.char_decoder(refined)
        outputs["char_logits"] = char_logits

        return outputs

# =============================================================================
# IntermediateTransformer: Applies a series of transformer encoder layers on token embeddings.
# =============================================================================
class IntermediateTransformer(nn.Module):
    """
    IntermediateTransformer:
    ---------------------------
    This block applies a series of transformer encoder layers (using PyTorch's TransformerEncoder)
    to refine the token sequence obtained from the PatternEncoder.

    The input is expected to have the shape: [batch, m_tokens, token_embedding_dim].
    The output is also of the same shape.
    """
    def __init__(self, token_embedding_dim, num_layers=2, num_heads=4, dropout=0.1):
        super(IntermediateTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_embedding_dim, nhead=num_heads, dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: [batch, m_tokens, token_embedding_dim]
        # Transformer encoder expects input of shape (sequence, batch, embedding)
        x = x.transpose(0,1)  # Now shape: [m_tokens, batch, token_embedding_dim]
        x = self.transformer_encoder(x)
        x = x.transpose(0,1)  # Back to [batch, m_tokens, token_embedding_dim]
        return x

# =============================================================================
# End2EndModel: Composes the PatternEncoder and PatternDecoder to form the full model.
# =============================================================================
class End2EndModel(nn.Module):
    """
    End2EndModel:
    -----------------
    This is the main model that composes two high-level blocks:
    
    1. PatternEncoder: Integrates signal encoding, tokenization, and token refinement.
       It transforms the input character indices (with associated prime frequencies) into a compact token representation.
       
    2. PatternDecoder: Projects the flattened token representation to a latent pattern space,
       applies pre-decoder attention to refine this pattern (using residual connections, dropout, and normalization),
       and finally decodes the refined pattern into character logits.

    Residual connections, dropout, and layer normalization are used in critical submodules to improve training stability.
    """
    def __init__(self, signal_length, m_tokens, token_vocab_size, token_embedding_dim, char_vocab_size, prime_tensor, device):
        super(End2EndModel, self).__init__()
        self.signal_length = signal_length
        self.m_tokens = m_tokens
        self.token_embedding_dim = token_embedding_dim
        self.char_vocab_size = char_vocab_size
        self.device = device
        
        # Register the prime tensor as a buffer.
        self.register_buffer("prime_tensor", prime_tensor)
        
        # Compose the model components.
        self.pattern_encoder = PatternEncoder(signal_length, m_tokens, token_vocab_size, token_embedding_dim, dropout_prob=0.2, device=device)
        self.intermediate_transformer = IntermediateTransformer(token_embedding_dim, num_layers=2, num_heads=4, dropout=0.1)
        self.pattern_decoder = PatternDecoder(m_tokens, token_embedding_dim, token_vocab_size, char_vocab_size)
    
    def forward(self, input_indices):
        # input_indices: (batch, context_length)
        # Obtain token representations (flattened, as well as token-level outputs) from the encoder.
        token_flat, token_logits, token_embeds, normalized_signal = self.pattern_encoder(input_indices, self.prime_tensor)
        batch_size = token_flat.size(0)
        # Reshape flattened tokens back into sequence form: [batch, m_tokens, token_embedding_dim]
        refined_tokens = token_flat.view(batch_size, self.m_tokens, self.token_embedding_dim)
        # Pass through intermediate transformer blocks.
        transformer_out = self.intermediate_transformer(refined_tokens)
        # Flatten back to [batch, m_tokens * token_embedding_dim] for the decoder.
        transformer_flat = transformer_out.reshape(batch_size, -1)
        # Decode into character logits.
        char_logits = self.pattern_decoder(transformer_flat)
        return char_logits, token_logits
    
    def forward_debug(self, input_indices):
        outputs = {}
        token_flat, token_logits, token_embeds, normalized_signal = self.pattern_encoder(input_indices, self.prime_tensor)
        outputs["context_signal_normalized"] = normalized_signal.detach().cpu().numpy()[0]
        outputs["token_embeds"] = token_embeds.detach().cpu().numpy()[0]
        outputs["token_logits"] = token_logits.detach().cpu().numpy()[0]
        outputs["token_flat"] = token_flat.detach().cpu().numpy()[0]

        batch_size = token_flat.size(0)
        refined_tokens = token_flat.view(batch_size, self.m_tokens, self.token_embedding_dim)
        transformer_out = self.intermediate_transformer(refined_tokens)
        transformer_flat = transformer_out.reshape(batch_size, -1)

        # Use PatternDecoder's decode_debug method on the transformer-enhanced tokens.
        dec_outputs = self.pattern_decoder.decode_debug(transformer_flat)
        outputs["pattern_logits"] = dec_outputs["pattern_logits"].detach().cpu().numpy()[0]
        outputs["refined_pattern"] = dec_outputs["refined_pattern"].detach().cpu().numpy()[0]
        outputs["pre_decoder_attn_weights"] = dec_outputs["pre_decoder_attn_weights"].detach().cpu().numpy()[0]
        outputs["char_logits"] = dec_outputs["char_logits"].detach().cpu().numpy()[0]
        
        return outputs

# =============================================================================
# Inference function: Generate text given a seed.
# =============================================================================
def generate_text(model, seed, idx_to_char, char_to_idx, seq_length, device, generation_length=100, temperature=0.3):
    model.eval()
    generated = seed
    context = list(seed)
    for i in range(generation_length):
        # Ensure the context is of length (seq_length-1) by padding with spaces if necessary.
        if len(context) < seq_length - 1:
            context_window = [' '] * ((seq_length - 1) - len(context)) + context
        else:
            context_window = context[-(seq_length - 1):]
        # Convert characters to indices.
        input_indices = torch.tensor([[char_to_idx[ch] for ch in context_window]], dtype=torch.long, device=device)
        logits, _ = model(input_indices)
        # Scale logits by temperature.
        scaled_logits = logits / temperature
        prob = F.softmax(scaled_logits, dim=-1)
        next_idx = torch.multinomial(prob, num_samples=1).item()
        next_char = idx_to_char[next_idx]
        generated += next_char
        context.append(next_char)
    return generated

# =============================================================================
# Chat mode: Interactively chat with the model.
# =============================================================================
def chat_mode(model, idx_to_char, char_to_idx, seq_length, device, generation_length=100):
    model.eval()
    print("Entering chat mode. Type 'quit' or 'exit' to stop.")
    while True:
        prompt = input("User: ")
        if prompt.lower() in ['quit', 'exit']:
            break
        full_text = generate_text(model, prompt, idx_to_char, char_to_idx, seq_length, device, generation_length)
        # Extract the generated portion.
        response = full_text[len(prompt):].strip()
        print("Model:", response)

# =============================================================================
# Debugging: Plot intermediate layer outputs.
# =============================================================================
def plot_debug_outputs(outputs):
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    
    # Plot raw context signal (if available; here we only have normalized).
    axs[0,0].plot(outputs["context_signal_normalized"])
    axs[0,0].set_title("Context Signal (Normalized)")
    axs[0,0].set_xlabel("Signal Index")
    axs[0,0].set_ylabel("Amplitude")
    
    # Plot token embeddings.
    im1 = axs[1,0].imshow(outputs["token_embeds"], aspect='auto', cmap='viridis')
    axs[1,0].set_title("Token Embeddings")
    plt.colorbar(im1, ax=axs[1,0])
    
    # Plot token logits.
    im2 = axs[1,1].imshow(outputs["token_logits"], aspect='auto', cmap='viridis')
    axs[1,1].set_title("Token Logits")
    plt.colorbar(im2, ax=axs[1,1])
    
    # Plot flattened token representation.
    axs[2,0].plot(outputs["token_flat"])
    axs[2,0].set_title("Flattened Token Representation")
    axs[2,0].set_xlabel("Index")
    axs[2,0].set_ylabel("Value")
    
    # Plot pattern logits.
    axs[2,1].plot(outputs["pattern_logits"])
    axs[2,1].set_title("Pattern Logits")
    axs[2,1].set_xlabel("Pattern Index")
    axs[2,1].set_ylabel("Logit")
    
    # Plot refined pattern.
    axs[3,0].plot(outputs["refined_pattern"])
    axs[3,0].set_title("Refined Pattern")
    axs[3,0].set_xlabel("Pattern Index")
    axs[3,0].set_ylabel("Value")
    
    # Plot character logits.
    axs[3,1].plot(outputs["char_logits"])
    axs[3,1].set_title("Character Logits")
    axs[3,1].set_xlabel("Character Index")
    axs[3,1].set_ylabel("Logit")
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# Main training loop
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=64, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--seq_length", type=int, default=12, help="sequence length for training examples (context+target)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--signal_length", type=int, default=512, help="number of points in the signal")
    parser.add_argument("--m_tokens", type=int, default=8, help="number of tokens to produce from the context signal")
    parser.add_argument("--token_vocab_size", type=int, default=256, help="vocabulary size for the tokenization layer")
    parser.add_argument("--token_embedding_dim", type=int, default=64, help="embedding dimension for tokens")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Read the training text from presentation.txt.
    with open("presentation.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()
    
    # Build the character vocabulary.
    vocab = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}
    char_vocab_size = len(vocab)
    print(f"Vocabulary size: {char_vocab_size}")
    
    # Generate prime numbers (one per character).
    primes = generate_primes(char_vocab_size)
    prime_tensor = torch.tensor(primes, dtype=torch.float, device=device)
    
    # Build the dataset and dataloader.
    dataset = TextDataset(text, seq_length=args.seq_length, char_to_idx=char_to_idx)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize the model.
    model = End2EndModel(
        signal_length=args.signal_length,
        m_tokens=args.m_tokens,
        token_vocab_size=args.token_vocab_size,
        token_embedding_dim=args.token_embedding_dim,
        char_vocab_size=char_vocab_size,
        prime_tensor=prime_tensor,
        device=device,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop.
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_inputs.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs} Loss: {avg_loss:.4f}")
    
    # Inference after training.
    seed = "My name is "
    generated_text = generate_text(model, seed, idx_to_char, char_to_idx, args.seq_length, device, generation_length=200)
    print("Generated text:\n", generated_text)
    
    # Debug: Plot intermediate layer outputs.
    print("Plotting model intermediate outputs for the input: 'My name is '")
    debug_prompt = "My name is "
    debug_context = list(debug_prompt)
    if len(debug_context) < args.seq_length - 1:
        debug_context = [' '] * ((args.seq_length - 1) - len(debug_context)) + debug_context
    else:
        debug_context = debug_context[-(args.seq_length - 1):]
    debug_input = torch.tensor([[char_to_idx[ch] for ch in debug_context]], dtype=torch.long, device=device)
    debug_outputs = model.forward_debug(debug_input)
    plot_debug_outputs(debug_outputs)
    
    # Enter chat mode.
    chat_mode(model, idx_to_char, char_to_idx, args.seq_length, device, generation_length=200)

if __name__ == "__main__":
    main()
