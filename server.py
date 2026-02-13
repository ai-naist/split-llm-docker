import socket
import pickle
import struct
import torch
import time
from transformers import AutoModelForCausalLM, AutoConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SPLIT_LAYER = 11
HOST = "0.0.0.0"
PORT = 65432

SPLIT_LAYER = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Server using device: {DEVICE}", flush=True)  # cuda と表示されるはず


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

# モデルロード部分
print("Configuring Server Model...", flush=True)
config = AutoConfig.from_pretrained(MODEL_NAME)

# device_mapを修正
device_map = {}
for name, _ in AutoModelForCausalLM.from_config(config).named_parameters():
    device_map[name] = "meta"

    # SPLIT_LAYER以降をGPUへ
    if "layers" in name:
        layer_idx = int(name.split("layers.")[1].split(".")[0])
        if layer_idx >= SPLIT_LAYER:
            device_map[name] = DEVICE  # "cuda"

    if "norm" in name or "lm_head" in name:
        device_map[name] = DEVICE  # "cuda"

# 3. マップに従って部分ロード
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device_map,
    low_cpu_mem_usage=True,  # これが必須
    torch_dtype=torch.float32,
)

# 使うレイヤーだけを取り出す（他はmetaデバイスにあるので触るとエラーになる）
layers = model.model.layers[SPLIT_LAYER:]
norm = model.model.norm
head = model.lm_head
model.eval()

print(f"Server ready on {PORT}", flush=True)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen()
    while True:
        conn, addr = s.accept()
        with conn:
            print(f"Connected: {addr}", flush=True)
            past_key_values = None

            # ... (ソケット待受部分) ...

            while True:
                try:
                    payload = recv_msg(conn)
                    if payload is None:
                        break

                    hidden_states, position_ids, attention_mask = payload

                    # ★ 受信したデータをGPUに転送
                    hidden_states = hidden_states.to(DEVICE)
                    position_ids = position_ids.to(DEVICE)
                    attention_mask = attention_mask.to(DEVICE)

                    # --- 計測開始 ---
                    start_time = time.time()

                    current_key_values = []

                    with torch.no_grad():
                        # 重要: 自分の担当レイヤーだけループ
                        for i, layer in enumerate(layers):
                            layer_past = past_key_values[i] if past_key_values else None

                            # metaデバイスにあるレイヤーはスキップされるので安全
                            outputs = layer(
                                hidden_states,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_value=layer_past,
                                use_cache=True,
                            )
                            hidden_states = outputs[0]
                            current_key_values.append(outputs[1])

                        past_key_values = tuple(current_key_values)
                        hidden_states = norm(hidden_states)
                        logits = head(hidden_states)
                        next_token = logits[:, -1, :]
                        next_token = next_token.cpu()

                    end_time = time.time()
                    # print(f"Server Compute: {end_time - start_time:.4f}s", flush=True)

                    send_msg(conn, next_token)

                except Exception as e:
                    print(f"Error: {e}", flush=True)
                    break
