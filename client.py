import socket
import pickle
import struct
import torch
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SPLIT_LAYER = 11
SERVER_HOST = os.getenv("SERVER_HOST", "localhost")
PORT = 65432


# --- 通信関数 ---
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


# ----------------

print("Configuring Client Model (Memory Split)...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)

# メモリ配置マップ作成
device_map = {}
for name, _ in AutoModelForCausalLM.from_config(config).named_parameters():
    device_map[name] = "meta"  # 基本はロードしない

    # EmbeddingsはClient必須
    if "embed_tokens" in name:
        device_map[name] = "cpu"

    # 前半レイヤー (0〜10) は CPU
    if "layers" in name:
        layer_idx = int(name.split("layers.")[1].split(".")[0])
        if layer_idx < SPLIT_LAYER:
            device_map[name] = "cpu"

print(f"Loading only Layers 0-{SPLIT_LAYER-1}...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map=device_map, low_cpu_mem_usage=True, torch_dtype=torch.float32
)

embeddings = model.model.embed_tokens
layers = model.model.layers[:SPLIT_LAYER]


def generate(prompt):
    print(f"\nPrompt: {prompt}")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((SERVER_HOST, PORT))
    except ConnectionRefusedError:
        print("\nError: Could not connect to server.")
        return

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    past_key_values = None
    generated_ids = input_ids
    next_input_ids = input_ids
    seq_len_so_far = 0

    for i in range(30):
        t0 = time.time()

        with torch.no_grad():
            curr_seq_len = next_input_ids.shape[1]
            total_seq_len = seq_len_so_far + curr_seq_len

            position_ids = torch.arange(seq_len_so_far, total_seq_len).unsqueeze(0)

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
                attention_mask = torch.ones((1, 1, 1, total_seq_len))

            # Client計算
            hidden = embeddings(next_input_ids)
            current_key_values = []

            for k, layer in enumerate(layers):
                layer_past = past_key_values[k] if past_key_values else None
                outputs = layer(
                    hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                hidden = outputs[0]
                current_key_values.append(outputs[1])

            past_key_values = tuple(current_key_values)
            t1 = time.time()

            # Serverへ送信
            payload = (hidden, position_ids, attention_mask)
            send_msg(s, payload)

            logits = recv_msg(s)
            t2 = time.time()

            if logits is None:
                break

            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
            token = tokenizer.decode(next_token_id[0])
            print(
                f"\n[Token {i+1}] '{token}' | Client: {t1-t0:.3f}s | Net+Server: {t2-t1:.3f}s",
                end="",
                flush=True,
            )

            next_input_ids = next_token_id
            seq_len_so_far = total_seq_len
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

    s.close()
    print("\nDone!")


if __name__ == "__main__":
    generate("The capital of France is")
