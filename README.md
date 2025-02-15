# End-to-End Text Generation Model

This project implements an end-to-end text generation model in PyTorch that innovatively integrates a tokenization process with signal encoding directly into the training pipeline. The model leverages unique sine-based signals for each input character—using prime numbers to associate different frequencies—before further processing the signals through tokenization and transformer layers. This approach is designed to capture richer representations of text inputs by combining ideas from signal processing with modern attention mechanisms.

## Overview

The architecture is composed of three primary components:

1. **PatternEncoder**
   - **Signal Encoding:**  
     Each character is transformed into a sine-based signal. The signal's frequency is associated with a unique prime number per character, ensuring distinct representations. In addition, a positional cyclic shift is applied to each signal before summing.
   - **Tokenization:**  
     The aggregated sine signals are normalized and sent through a linear projection to produce token logits. These logits are then converted into a probability distribution via softmax, which is used to compute weighted token embeddings from a learnable embedding table.
   - **Token Refinement:**  
     Multi-head self-attention is applied with residual connections, dropout, and layer normalization to further refine the token embeddings. The refined tokens are flattened for downstream processing.

2. **IntermediateTransformer**
   - A series of transformer encoder layers are applied to the sequence of token embeddings. This step refines the tokens further using self-attention, helping the model to capture complex dependencies within the tokenized representation.

3. **PatternDecoder**
   - **Pattern Projection and Refinement:**  
     The flattened token embeddings are projected into a latent pattern space. A pre-decoder multi-head self-attention layer refines the projected pattern with residual connections, dropout, and normalization.
   - **Decoding into Characters:**  
     Finally, the refined pattern is decoded into logits representing the character vocabulary, which are used to predict the next character in the sequence.

## The Original Tokenization Idea

The core innovation of this project is the idea of integrating the tokenization process into the model training, rather than using a conventional static embedding layer. Here’s how it works:

- **Sine-Based Signal Representation:**  
  Instead of mapping characters directly to learned embeddings, each character is first converted to a sine wave signal. The frequency of this sine wave is determined by unique prime numbers generated for the character set. This means that each character has a distinct oscillatory pattern.
  
- **Cyclic Shifts and Summation:**  
  To incorporate positional information, each sine signal is cyclically shifted based on its position in the input sequence, and then the signals are summed. The resulting aggregate is then L2-normalized.

- **Dynamic Tokenization:**  
  The normalized signal is passed through a linear projection to generate logits that represent a distribution over a token vocabulary. The softmax function is then applied, and token embeddings are computed as a weighted sum over a learnable embedding table. This creates dynamic tokens that capture both signal properties and learned semantic features.

This method seamlessly merges signal processing ideas into the text processing pipeline, offering a potentially richer and more discriminative representation for model training.

## Training and Inference

- **Dataset Preparation:**  
  The model uses a sliding window approach on text data to generate training samples. Each sample consists of a context and a target character.

- **Training Loop:**  
  The training procedure optimizes the network using cross-entropy loss over the predicted and actual character indices. The optimizer updates both the transformer and tokenization parameters end-to-end.

- **Interactive Chat Mode:**  
  Post-training, the model can be utilized in an interactive chat mode, generating text responses based on a user prompt.

- **Debugging and Visualization:**  
  The code includes debugging functions that plot intermediate outputs such as normalized signal, token embeddings, logits, and refined patterns to provide insights into the processing at various stages.

## Usage

After installing the necessary dependencies (Python 3.6+, PyTorch, Matplotlib, and NumPy), you can start training the model using:
bash
```
python End2EndModel.py --epochs 64 --batch_size 32 --seq_length 12 --lr 0.001 --signal_length 512 --m_tokens 8 --token_vocab_size 256 --token_embedding_dim 64
```

You can also enter a chat mode after training to interact with the model.

## File Structure

- **End2EndModel.py**  
  Contains the full implementation of the model including:
  - Utility Functions: Prime number generation and dataset creation.
  - Model Architecture: `PatternEncoder`, `IntermediateTransformer`, and `PatternDecoder`.
  - Training and Inference Pipelines.
  - Debugging and visualization functions.

## Conclusion

This project explores an innovative approach to tokenization where the input text is first embedded into a unique signal space via sine-based encoding. By combining signal processing techniques (using prime numbers for unique frequency assignments) with modern attention-based mechanisms, the model aims to capture richer representations of text data. This integrated tokenization process presents an alternative paradigm to traditional fixed embeddings and opens up new possibilities in the field of text generation.

Enjoy exploring and experimenting with this model!
