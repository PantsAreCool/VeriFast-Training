# Task 2: Build a Full Transformer Encoder Layer
# Using the self-attention, feed-forward, layer normalization, and residual connection functions from this lesson, 
# compose a complete Transformer encoder layer. Process a random input of shape (4, 16) through the layer and verify the output shape is correct. 
# Then stack 3 such layers and process the input through all 3. Print the output shape and the L2 norm of the output after each layer to verify 
# the residual connections are preventing the values from vanishing or exploding.

import numpy as np


def softmax(x, axis=-1):
	exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
	return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def self_attention(x, params):
	q = x @ params["w_q"]
	k = x @ params["w_k"]
	v = x @ params["w_v"]

	scores = (q @ k.T) / np.sqrt(x.shape[-1])
	weights = softmax(scores)
	return weights @ v


def feed_forward(x, params):
	hidden = np.maximum(0, x @ params["w_1"] + params["b_1"])
	return hidden @ params["w_2"] + params["b_2"]


def layer_norm(x, epsilon=1e-6):
	mean = np.mean(x, axis=-1, keepdims=True)
	std = np.std(x, axis=-1, keepdims=True)
	return (x - mean) / (std + epsilon)


def residual_connection(x, sublayer_output):
	return x + sublayer_output


def build_encoder_layer(d_model=16, d_ff=64):
	return {
		"attn": {
			"w_q": np.random.randn(d_model, d_model) * 0.1,
			"w_k": np.random.randn(d_model, d_model) * 0.1,
			"w_v": np.random.randn(d_model, d_model) * 0.1,
		},
		"ff": {
			"w_1": np.random.randn(d_model, d_ff) * 0.1,
			"b_1": np.zeros(d_ff),
			"w_2": np.random.randn(d_ff, d_model) * 0.1,
			"b_2": np.zeros(d_model),
		},
	}


def transformer_encoder_layer(x, layer):
	attn_output = self_attention(x, layer["attn"])
	x = layer_norm(residual_connection(x, attn_output))

	ff_output = feed_forward(x, layer["ff"])
	x = layer_norm(residual_connection(x, ff_output))

	return x


def main():
	np.random.seed(42)

	x = np.random.randn(4, 16)
	layers = [build_encoder_layer() for _ in range(3)]

	first_output = transformer_encoder_layer(x, layers[0])
	print("Single encoder layer output shape:", first_output.shape)

	current = x
	for i, layer in enumerate(layers, 1):
		current = transformer_encoder_layer(current, layer)
		print(f"After layer {i}: shape = {current.shape}, L2 norm = {np.linalg.norm(current):.4f}")


if __name__ == "__main__":
	main()

