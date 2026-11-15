"""Join a sharded safetensors checkpoint into one file, byte for byte.

Reads the shards' headers, lays every tensor out again in key order in a single file and copies its raw bytes
unchanged (no torch, no dtype conversion). The folder's config.json, when present, goes into the metadata as
"iris_z_config".

    python join_shards.py <folder with transformer.safetensors.index.json> <output.safetensors>
"""

import json
import os
import struct
import sys

CHUNK = 64 * 1024 * 1024


def read_header(path):
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    return header, 8 + length


def main(folder, output):
    index = json.load(open(os.path.join(folder, "transformer.safetensors.index.json")))
    weight_map = index["weight_map"]
    shards = {}
    for shard in sorted(set(weight_map.values())):
        header, data_start = read_header(os.path.join(folder, shard))
        header.pop("__metadata__", None)
        shards[shard] = (header, data_start)

    layout = {}
    offset = 0
    sources = []
    for key in sorted(weight_map):
        header, data_start = shards[weight_map[key]]
        entry = header[key]
        start, end = entry["data_offsets"]
        layout[key] = {"dtype": entry["dtype"], "shape": entry["shape"], "data_offsets": [offset, offset + end - start]}
        sources.append((weight_map[key], data_start + start, end - start))
        offset += end - start

    every_key = {k for header, _ in shards.values() for k in header}
    if every_key != set(weight_map):
        raise SystemExit(f"index and shards disagree: {sorted(every_key ^ set(weight_map))[:10]}")

    metadata = {"format": "pt", "source": "speridlabs/iris-z " + os.path.basename(os.path.normpath(folder))}
    config_path = os.path.join(folder, "config.json")
    if os.path.exists(config_path):
        metadata["iris_z_config"] = json.dumps(json.load(open(config_path)), separators=(",", ":"))
    header_bytes = json.dumps({"__metadata__": metadata, **layout}, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)

    partial = output + ".partial"
    handles = {shard: open(os.path.join(folder, shard), "rb") for shard in shards}
    try:
        with open(partial, "wb") as out:
            out.write(struct.pack("<Q", len(header_bytes)))
            out.write(header_bytes)
            for i, (shard, position, size) in enumerate(sources):
                src = handles[shard]
                src.seek(position)
                left = size
                while left:
                    block = src.read(min(CHUNK, left))
                    if not block:
                        raise SystemExit(f"{shard} ended early")
                    out.write(block)
                    left -= len(block)
                if i % 25 == 0:
                    print(f"{i + 1}/{len(sources)} tensors", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    expected = 8 + len(header_bytes) + offset
    if os.path.getsize(partial) != expected:
        raise SystemExit(f"size {os.path.getsize(partial)} != {expected}")
    os.replace(partial, output)
    print(f"wrote {output}: {len(layout)} tensors, {offset / 1e9:.2f} GB of data")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
