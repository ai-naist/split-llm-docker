import pickle
import socket
import struct

import torch
from transformers import AutoConfig, DynamicCache

from model_loader import load_partial_model
from settings import load_settings

settings = load_settings()
MODEL_NAME = settings.common.model_name
SPLIT_LAYER = settings.common.split_layer
MODEL_DTYPE = settings.common.dtype
HOST = settings.server.host
PORT = settings.server.port


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


print("Configuring Server (Qwen 2.5)...", flush=True)

config = AutoConfig.from_pretrained(MODEL_NAME)
num_layers = config.num_hidden_layers

print(f"Loading Layers {SPLIT_LAYER}-{num_layers-1}...", flush=True)

model = load_partial_model(
    MODEL_NAME, SPLIT_LAYER, role="server", dtype_name=MODEL_DTYPE
)

layers = model.model.layers[SPLIT_LAYER:]
norm = model.model.norm
head = model.lm_head
rotary_emb = model.model.rotary_emb
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
            past_key_values = DynamicCache()

            while True:
                try:
                    payload = recv_msg(conn)
                    if payload is None:
                        break

                    hidden_states, position_ids, attention_mask, cache_position = (
                        payload
                    )

                    # --- 修正: 位置情報(RoPE)の計算 ---
                    # hidden_statesを使ってデバイスなどを合わせつつ計算
                    cos, sin = rotary_emb(hidden_states, position_ids)
                    position_embeddings = (cos, sin)

                    with torch.no_grad():
                        for layer in layers:
                            hidden_states = layer(
                                hidden_states,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                use_cache=True,
                                cache_position=cache_position,
                                position_embeddings=position_embeddings,  # <--- これを渡す！
                            )
                        hidden_states = norm(hidden_states)
                        logits = head(hidden_states)
                        next_token = logits[:, -1, :]

                    send_msg(conn, next_token)

                except Exception as e:
                    print(f"Error: {e}", flush=True)
                    break
