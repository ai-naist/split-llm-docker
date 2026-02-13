import argparse
import os
import pickle
import socket
import struct
import time

import torch
from transformers import AutoConfig, AutoTokenizer, DynamicCache

from model_loader import load_partial_model
from settings import load_settings

settings = load_settings()
MODEL_NAME = settings.common.model_name
SPLIT_LAYER = settings.common.split_layer
MODEL_DTYPE = settings.common.dtype
SERVER_HOST = settings.client.server_host
PORT = settings.client.port
MAX_NEW_TOKENS = settings.client.max_new_tokens
USE_CHAT_TEMPLATE = settings.client.use_chat_template
SYSTEM_PROMPT = settings.client.system_prompt


def build_stop_token_ids() -> set[int]:
    stop_ids: set[int] = set()

    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))

    for token_text in ["<|im_end|>", "<|endoftext|>"]:
        token_id = tokenizer.convert_tokens_to_ids(token_text)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            stop_ids.add(int(token_id))

    return stop_ids


def send_msg(sock, data):
    packet = pickle.dumps(data)
    length = struct.pack(">I", len(packet))
    sock.sendall(length + packet)


def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack(">I", raw_msglen)[0]
    raw_data = recvall(sock, msglen)
    if not raw_data:
        return None
    return pickle.loads(raw_data)


def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


print("Configuring Client (Qwen 2.5)...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
num_layers = config.num_hidden_layers
STOP_TOKEN_IDS = build_stop_token_ids()

print(f"Loading Layers 0-{SPLIT_LAYER-1}...", flush=True)

model = load_partial_model(
    MODEL_NAME, SPLIT_LAYER, role="client", dtype_name=MODEL_DTYPE
)

embeddings = model.model.embed_tokens
layers = model.model.layers[:SPLIT_LAYER]
rotary_emb = model.model.rotary_emb
model.eval()


def build_input_ids(prompt: str) -> torch.Tensor:
    if USE_CHAT_TEMPLATE and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, torch.Tensor):
            return encoded
        if hasattr(encoded, "input_ids"):
            return encoded.input_ids
        if isinstance(encoded, dict) and "input_ids" in encoded:
            return encoded["input_ids"]
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

    return tokenizer(prompt, return_tensors="pt").input_ids


def generate(prompt):
    print(f"\nPrompt: {prompt}")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((SERVER_HOST, PORT))
    except ConnectionRefusedError:
        print("\nError: Could not connect to server.")
        return

    input_ids = build_input_ids(prompt)
    past_key_values = DynamicCache()
    generated_ids = input_ids
    next_input_ids = input_ids
    seq_len_so_far = 0

    for i in range(MAX_NEW_TOKENS):
        t0 = time.time()

        with torch.no_grad():
            curr_seq_len = next_input_ids.shape[1]
            total_seq_len = seq_len_so_far + curr_seq_len

            position_ids = torch.arange(seq_len_so_far, total_seq_len).unsqueeze(0)
            cache_position = torch.arange(seq_len_so_far, total_seq_len)

            if seq_len_so_far == 0:
                attention_mask = torch.ones((1, 1, total_seq_len, total_seq_len))
                attention_mask = torch.triu(attention_mask, diagonal=1)
                attention_mask = attention_mask.masked_fill(
                    attention_mask == 1, float("-inf")
                )
                attention_mask = attention_mask.masked_fill(
                    attention_mask == 0, float(0.0)
                )
            else:
                attention_mask = torch.zeros((1, 1, 1, total_seq_len))

            hidden = embeddings(next_input_ids)
            # --- 修正: 位置情報(RoPE)の計算 ---
            cos, sin = rotary_emb(hidden, position_ids)
            position_embeddings = (cos, sin)

            for layer in layers:
                hidden = layer(
                    hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,  # <--- 渡す
                )
            t1 = time.time()

            payload = (hidden, position_ids, attention_mask, cache_position)
            send_msg(s, payload)

            logits = recv_msg(s)
            t2 = time.time()

            if logits is None:
                break

            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
            token_id_int = int(next_token_id.item())
            token = tokenizer.decode(next_token_id[0])
            print(
                f"\n[Token {i+1}] '{token}' | Client: {t1-t0:.3f}s | Net+Server: {t2-t1:.3f}s",
                end="",
                flush=True,
            )

            next_input_ids = next_token_id
            seq_len_so_far = total_seq_len
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            if token_id_int in STOP_TOKEN_IDS:
                break

    s.close()
    print("\nDone!")


def resolve_prompt() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    prompt = args.prompt or os.getenv("PROMPT") or os.getenv("DEFAULT_PROMPT")
    if prompt:
        return prompt

    prompt = input("Enter prompt: ").strip()
    if not prompt:
        raise ValueError(
            "Prompt is required. Provide --prompt, PROMPT/DEFAULT_PROMPT env, or interactive input."
        )
    return prompt


if __name__ == "__main__":
    generate(resolve_prompt())
