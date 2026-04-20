#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import pathlib
import struct

import aiohttp

MSG_VIDEO_INIT = 0x01
MSG_VIDEO_NAL = 0x02
MSG_AUDIO_OPUS = 0x03
MSG_SYNC_META = 0x04
MSG_AUDIO_INIT = 0x05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture WS stream payloads for offline decode checks.")
    parser.add_argument("--url", required=True, help="ws:// or wss:// stream URL")
    parser.add_argument("--output-dir", default="ws_capture", help="Directory to write payload files into")
    parser.add_argument("--seconds", type=float, default=4.0, help="Capture duration in seconds")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = out_dir / "stream.h264"
    audio_path = out_dir / "stream.opus"
    meta_path = out_dir / "meta.jsonl"
    init_path = out_dir / "video_init.bin"
    audio_init_path = out_dir / "audio_init.bin"
    deadline = asyncio.get_running_loop().time() + max(float(args.seconds), 0.1)
    counts = {MSG_VIDEO_INIT: 0, MSG_VIDEO_NAL: 0, MSG_AUDIO_OPUS: 0, MSG_SYNC_META: 0, MSG_AUDIO_INIT: 0}

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(args.url, max_msg_size=8 * 1024 * 1024) as ws, \
            video_path.open("wb") as h264_out, \
            audio_path.open("wb") as opus_out, \
            meta_path.open("w", encoding="utf-8") as meta_out:
            while asyncio.get_running_loop().time() < deadline:
                timeout = max(deadline - asyncio.get_running_loop().time(), 0.05)
                msg = await ws.receive(timeout=timeout)
                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                    break
                if msg.type != aiohttp.WSMsgType.BINARY:
                    continue

                blob = msg.data
                if len(blob) < 13:
                    continue
                kind, length, _timestamp_us = struct.unpack(">BIQ", blob[:13])
                payload = blob[13:13 + length]
                counts[kind] = counts.get(kind, 0) + 1

                if kind == MSG_VIDEO_INIT:
                    init_path.write_bytes(payload)
                elif kind == MSG_VIDEO_NAL:
                    h264_out.write(payload)
                elif kind == MSG_AUDIO_OPUS:
                    opus_out.write(payload)
                elif kind == MSG_SYNC_META:
                    meta_out.write(payload.decode("utf-8", errors="replace") + "\n")
                elif kind == MSG_AUDIO_INIT:
                    audio_init_path.write_bytes(payload)

    print("capture complete")
    print(f"  url:        {args.url}")
    print(f"  output dir: {out_dir}")
    for kind, count in sorted(counts.items()):
        print(f"  kind {kind}:  {count}")


if __name__ == "__main__":
    asyncio.run(main())
