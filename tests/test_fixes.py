from __future__ import annotations

import io
import json
import struct
import threading
import unittest
import urllib.request
import zipfile
import zlib
from http.server import ThreadingHTTPServer

from cts2.bin import Reader
from cts2.converter import convert_bytes
from cts2.ctimage import decode_bmp, decompress_rle
from cts2.events_exe import normalize_exe_opcodes
from cts2.mfa import (
    Action,
    Condition,
    EventGroup,
    Frame,
    FrameInstance,
    ImageItem,
    MFA,
    ObjectData,
    Parameter,
    _decode_parameter_payload,
    _read_condition,
    _read_frame_events,
)
from cts2.progress import Reporter
from cts2.scratch import TargetBuilder, build_project
from cts2.transpile import (
    ACT_HIDE,
    ACT_SET_POSITION,
    ACT_SET_X,
    ACT_SHOW,
    COND_ALWAYS,
    COND_REPEAT,
    COND_START_OF_FRAME,
    SYS,
    transpile_frame_blocks,
)
from cts2.web import _H
from tests.exebuilder import (
    build_exe,
    build_game_data,
    frame_chunk,
    frame_instance,
    image_item_normal,
    object_common_25,
    object_common_284,
    object_header_25,
    object_names_25,
    object_props_25,
)


def _rle_compress(raw: bytes) -> bytes:
    """Encode raw bytes using Clickteam byte-RLE (1..128 repeat, >128 literal)."""
    out = bytearray()
    i = 0
    while i < len(raw):
        run = 1
        while i + run < len(raw) and raw[i + run] == raw[i] and run < 128:
            run += 1
        if run >= 3:
            out.append(run)
            out.append(raw[i])
            i += run
        else:
            lit_start = i
            while i < len(raw) and (i - lit_start) < 127:
                if i + 2 < len(raw) and raw[i] == raw[i + 1] == raw[i + 2]:
                    break
                i += 1
            lit = raw[lit_start:i]
            out.append(128 + len(lit))
            out.extend(lit)
    out.append(0)
    return bytes(out)


def _decode_png_rgba(png: bytes):
    """Minimal PNG RGBA reader for verifying decoded pixel colors in tests."""
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    pos = 8
    w = h = 0
    idat = bytearray()
    while pos < len(png):
        length = struct.unpack(">I", png[pos:pos + 4])[0]
        ctype = png[pos + 4:pos + 8]
        data = png[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            w, h = struct.unpack(">II", data[:8])
        elif ctype == b"IDAT":
            idat.extend(data)
    raw = zlib.decompress(bytes(idat))
    stride = w * 4 + 1
    out = bytearray()
    for y in range(h):
        assert raw[y * stride] == 0  # filter None
        out.extend(raw[y * stride + 1:(y + 1) * stride])
    return w, h, bytes(out)


def _validate_sb3_graph(pj: dict) -> None:
    for target in pj["targets"]:
        blocks = target["blocks"]
        ids = set(blocks)
        for bid, blk in blocks.items():
            nxt = blk.get("next")
            if nxt:
                assert nxt in ids, f"{target['name']} {bid} missing next {nxt}"
            par = blk.get("parent")
            if par:
                assert par in ids, f"{target['name']} {bid} missing parent {par}"
            if blk.get("topLevel"):
                assert par is None, f"{target['name']} topLevel block {bid} has parent {par}"
            for key, inp in blk.get("inputs", {}).items():
                if isinstance(inp, list) and len(inp) >= 2:
                    if isinstance(inp[1], str):
                        assert inp[1] in ids, f"{target['name']} {bid}.{key} missing {inp[1]}"
                        assert blocks[inp[1]].get("parent") == bid, (
                            f"{target['name']} {bid}.{key} child {inp[1]} parent mismatch"
                        )


class TestImageAndTextureFixes(unittest.TestCase):
    def test_rle_compressed_24bpp_with_alpha_decodes_real_pixels(self):
        # 3x2 odd-width 24bpp image (width*3 = 9, 4-byte padded row = 12 bytes)
        row0 = bytes([10, 20, 30, 40, 50, 60, 70, 80, 90, 0, 0, 0])
        row1 = bytes([15, 25, 35, 45, 55, 65, 75, 85, 95, 0, 0, 0])
        alpha0 = bytes([200, 150, 100, 0])
        alpha1 = bytes([255, 128, 64, 0])
        rle_payload = _rle_compress(row0 + row1) + _rle_compress(alpha0 + alpha1)

        png = decode_bmp(3, 2, mode=4, flags=0x01 | 0x10, body=rle_payload)
        self.assertIsNotNone(png)
        w, h, rgba = _decode_png_rgba(png)
        self.assertEqual((w, h), (3, 2))
        # BGR -> RGB + alpha plane
        self.assertEqual(list(rgba[:4]), [30, 20, 10, 200])
        self.assertEqual(list(rgba[4:8]), [60, 50, 40, 150])
        self.assertEqual(list(rgba[8:12]), [90, 80, 70, 100])
        self.assertEqual(list(rgba[12:16]), [35, 25, 15, 255])

    def test_24bpp_odd_width_2byte_padded_stride(self):
        # width=3, height=2, 2-byte aligned stride = 10 bytes/row (not 12!)
        row0 = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        row1 = bytes([11, 12, 13, 14, 15, 16, 17, 18, 19, 0])
        png = decode_bmp(3, 2, mode=4, flags=0, body=row0 + row1)
        self.assertIsNotNone(png)
        w, h, rgba = _decode_png_rgba(png)
        self.assertEqual((w, h), (3, 2))
        self.assertEqual(list(rgba[12:16]), [13, 12, 11, 255])

    def test_32bpp_all_zero_alpha_rescued_with_color_key(self):
        # 2x1 32bpp BGRA where stored alpha bytes are 0 (opaque RGB with black key)
        body = bytes([0, 0, 255, 0, 0, 0, 0, 0])  # Red pixel, Black pixel
        png = decode_bmp(2, 1, mode=8, flags=0, body=body, transparent=(0, 0, 0, 255))
        self.assertIsNotNone(png)
        _, _, rgba = _decode_png_rgba(png)
        self.assertEqual(list(rgba[:4]), [255, 0, 0, 255])
        self.assertEqual(list(rgba[4:8]), [0, 0, 0, 0])

    def test_rlew_and_rlet_decompression(self):
        words = b"\x34\x12" * 5
        rlew = bytes([5]) + b"\x34\x12" + bytes([0])
        out, _ = decompress_rle(rlew, 2, 10)
        self.assertEqual(out, words)

        triplets = b"\x01\x02\x03" * 4
        rlet = bytes([4]) + b"\x01\x02\x03" + bytes([0])
        out3, _ = decompress_rle(rlet, 3, 12)
        self.assertEqual(out3, triplets)


class TestGameDataAndMFAFixes(unittest.TestCase):
    def test_build_284_static_active_keeps_animations(self):
        # Build 284 Active object with mov_off==0, version==0 (check==0) and an anim frame
        pixels = bytes([0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0])
        img = image_item_normal(0, 2, 2, pixels, mode=4, flags=0, build=284)
        # Patch object_common_284(check0=False) to have version=0 (so check==0!)
        raw_common = bytearray(object_common_284(frames_per_anim=((0,),), check0=False))
        struct.pack_into("<i", raw_common, 8, 0)  # version=0 -> check==0, first_off=anim_off != 0
        game = build_game_data(
            name="Static284", unicode=False, build=284, images=[img],
            old_objects=[(0, 2, "StaticActive", bytes(raw_common))],
            frames=[frame_chunk(name="Menu", instances=(frame_instance(0, 0, 100, 100),),
                                layers=(("L1", 1.0, 1.0),), unicode=False)],
            frame_handles=(0,),
        )
        exe = build_exe(game, pack_files=[], unicode=False)
        out = convert_bytes(exe, "game.exe")
        with zipfile.ZipFile(io.BytesIO(out["project"])) as z:
            pj = json.loads(z.read("project.json"))
            sprites = [t for t in pj["targets"] if t["name"] == "Menu-StaticActive"]
            self.assertEqual(len(sprites), 1)
            # Must decode the green 2x2 texture, NOT fall back to magenta placeholder
            self.assertFalse(any("placeholder" in w for w in out["report"]["warnings"]))
            png_bytes = z.read(sprites[0]["costumes"][0]["md5ext"])
            w, h, rgba = _decode_png_rgba(png_bytes)
            self.assertEqual((w, h), (2, 2))
            self.assertEqual(list(rgba[:4]), [0, 255, 0, 255])

    def test_parameter_payload_reads_exact_size_and_utf16_and_position(self):
        # Parameter: p_size=12, code=41 (string), payload="hi" in UTF-16LE (6 bytes) + 2 bytes pad
        raw_param = struct.pack("<hh", 12, 41) + b"h\x00i\x00\x00\x00" + b"\x99\x88"
        cond_hdr = struct.pack("<HhhHhbbBBh", 16 + len(raw_param), -1, -1, 0, 0, 0, 0, 1, 0, 0)
        cond = _read_condition(Reader(cond_hdr + raw_param))
        self.assertIsNotNone(cond)
        self.assertEqual(len(cond.parameters[0].raw), 8)
        self.assertEqual(cond.parameters[0].value, {"string": "hi"})

        pos_payload = struct.pack("<HHhh", 7, 0, 320, 240)
        self.assertEqual(
            _decode_parameter_payload(16, pos_payload),
            {"x": 320, "y": 240, "parent": 7, "flags": 0},
        )

    def test_exe_always_normalizes_separately_from_storyboard_start_of_frame(self):
        g = EventGroup(
            0, 0, 0, 0, 0, 0, 0, 0,
            conditions=[Condition(-1, -1, 0, 0, 0, 0, 0, 0, 0, 12, [])],
            actions=[],
        )
        normalize_exe_opcodes(g)
        self.assertEqual(g.conditions[0].num, COND_ALWAYS)

    def test_param_comparison_and_zero_size_expression_never_hang(self):
        # ParamComparison (code=23): 2-byte operator followed by 10-byte int expression (value=42)
        expr = struct.pack("<hhHi", -1, 0, 10, 42)
        cmp_payload = struct.pack("<h", 1) + expr
        decoded = _decode_parameter_payload(23, cmp_payload)
        self.assertEqual(decoded["operator"], 1)
        self.assertEqual(decoded["expressions"][0]["value"], 42)

        # Zero-size or negative-size expression must terminate instead of spinning
        bad_expr = struct.pack("<hhh", 1, -1, 0) + b"\x00" * 10
        self.assertEqual(_decode_parameter_payload(22, bad_expr), {"expressions": []})

        # Zero-size EventGroup inside Evts must not infinite-loop _read_frame_events
        evts = b"\x01\x00\x00\x00" + b"Evts" + struct.pack("<I", 16) + b"\x00" * 16 + b"!DNE"
        self.assertEqual(_read_frame_events(Reader(evts)), [])


def _make_mfa(w: int, h: int, images: dict, frames: list) -> MFA:
    return MFA(
        name="G", description="", path="", author="", copyright="", company="",
        version=0, window_x=w, window_y=h, frame_rate=60,
        mfa_build=294, build_version=0, build_type=0,
        sounds={}, fonts={}, images=images,
        global_values=[], global_strings=[], frames=frames, file_data={},
    )


class TestScratchAndTranspileFixes(unittest.TestCase):
    def test_duplicate_instances_and_uniform_scaling_and_hidden_logic_notes(self):
        png = decode_bmp(2, 2, mode=4, flags=0, body=bytes([255, 0, 0] * 4))
        item = ObjectData(2, 0, "Wall", images=[0], width=2, height=2)
        inst1 = FrameInstance(100, 100, 1, 0, 0, 0, 0, 0)
        inst2 = FrameInstance(500, 200, 0, 1, 0, 0, 0, 0)
        frame = Frame(0, "Level", 1280, 720, (0, 0, 0, 255), 0, 0, "",
                      items=[item], instances=[inst1, inst2],
                      event_groups=[EventGroup(0, 0, 0, 0, 0, 0, 0, 0, [], [])])
        mfa = _make_mfa(1280, 720, {0: ImageItem(0, 0, 1, 0, 2, 2, 4, 0, 1, 1, 0, 0, (0, 0, 0, 0), png, b"")}, [frame])
        sb3, _ = build_project(mfa)
        with zipfile.ZipFile(io.BytesIO(sb3)) as z:
            pj = json.loads(z.read("project.json"))
            _validate_sb3_graph(pj)
            walls = [t for t in pj["targets"] if t["name"].startswith("Level-Wall")]
            self.assertEqual(len(walls), 2)
            # Layer 0 (inst2) must come before Layer 1 (inst1)
            self.assertLess(walls[0]["layerOrder"], walls[1]["layerOrder"])
            # Uniform scale for 1280x720 -> 480x360 is 37.5%
            self.assertAlmostEqual(walls[0]["size"], 37.5, places=1)
            notes = [t for t in pj["targets"] if t["name"] == "Logic-Notes"]
            self.assertEqual(len(notes), 1)
            self.assertFalse(notes[0]["visible"])

    def test_transpile_every_repeat_globals_multi_cond_and_active_coords(self):
        png = decode_bmp(2, 2, mode=4, flags=0, body=bytes([0, 0, 255] * 4))
        item = ObjectData(2, 5, "Hero", images=[0], width=2, height=2)
        frame = Frame(
            0, "F1", 640, 480, (0, 0, 0, 255), 0, 0, "",
            items=[item],
            instances=[FrameInstance(320, 240, 0, 0, 0, 0, 5, 0)],
            event_groups=[
                # 1) Every 100 ticks -> Set Global 1 = 7
                EventGroup(0, 0, 0, 0, 0, 0, 0, 0,
                           [Condition(SYS, -4, 0, 0, 0, 0, 1, 0, 0, 12, [Parameter(13, 8, {"every": 100})])],
                           [Action(SYS, 2004, 0, 0, 0, 0, 2, 0, 12, [Parameter(49, 8, {"global": 1}), Parameter(22, 8, {"expressions": [{"value": 7}]})])]),
                # 2) Repeat 3 -> Set Hero Position (320, 240) -> Scratch (0, 0)
                EventGroup(0, 0, 0, 0, 0, 0, 0, 0,
                           [Condition(SYS, COND_REPEAT, 0, 0, 0, 0, 1, 0, 0, 12, [Parameter(5, 8, {"int": 3})])],
                           [Action(2, ACT_SET_POSITION, 5, 0, 0, 0, 1, 0, 12, [Parameter(16, 12, {"x": 320, "y": 240})])]),
                # 3) Multi-condition: Start of Frame + Compare Global 1 == 7 -> Show Hero
                EventGroup(0, 0, 0, 0, 0, 0, 0, 0,
                           [Condition(SYS, COND_START_OF_FRAME, 0, 0, 0, 0, 0, 0, 0, 12, []),
                            Condition(SYS, -9, 0, 0, 0, 0, 2, 0, 0, 12, [Parameter(49, 8, {"global": 1}), Parameter(22, 8, {"expressions": [{"value": 7}]})])],
                           [Action(2, ACT_SHOW, 5, 0, 0, 0, 0, 0, 12, [])]),
            ],
        )
        mfa = _make_mfa(640, 480, {0: ImageItem(0, 0, 1, 0, 2, 2, 4, 0, 0, 0, 0, 0, (0, 0, 0, 0), png, b"")}, [frame])
        sb3, report = build_project(mfa)
        self.assertEqual(report["events_mapped"], 3)
        with zipfile.ZipFile(io.BytesIO(sb3)) as z:
            pj = json.loads(z.read("project.json"))
            _validate_sb3_graph(pj)
            ev = [t for t in pj["targets"] if t["name"] == "F1-Events"][0]
            # Global 1 is shared (exactly 1 variable created across Set + Compare)
            self.assertEqual(len(ev["variables"]), 1)
            # Every block's control_forever has a non-empty SUBSTACK containing control_wait
            forevers = [b for b in ev["blocks"].values() if b["opcode"] == "control_forever"]
            self.assertTrue(any(
                b["inputs"].get("SUBSTACK") and ev["blocks"][b["inputs"]["SUBSTACK"][1]]["opcode"] == "control_wait"
                for b in forevers
            ))
            # Hero received unique broadcasts for SetPosition (0, 0) and Show
            hero = [t for t in pj["targets"] if t["name"] == "F1-Hero"][0]
            ops = [b["opcode"] for b in hero["blocks"].values()]
            self.assertIn("motion_gotoxy", ops)
            self.assertIn("looks_show", ops)


class TestProgressAndWebFixes(unittest.TestCase):
    def test_reporter_seq_beyond_max_history(self):
        rep = Reporter(max_history=5, min_interval=0.0)
        for i in range(20):
            rep.tick(i + 1, 20, step=f"step {i}")
        seqs = [ev["_seq"] for ev in rep.history]
        self.assertEqual(len(rep.history), 5)
        self.assertEqual(seqs, [17, 18, 19, 20, 21] if seqs[0] == 17 else sorted(seqs))
        self.assertGreater(seqs[-1], 15)

    def test_web_multipart_preserves_crlf_dash_dash_in_binary(self):
        from unittest.mock import patch
        captured = {}

        def fake_start(data: bytes, name: str) -> str:
            captured["data"] = data
            captured["name"] = name
            return "job-test"

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            payload = b"MZ\x90\x00" + b"\r\n--FAKE_BOUNDARY_BYTES\r\n" + b"TAIL"
            boundary = b"----TestBoundary123"
            body = (
                b"--" + boundary + b"\r\n"
                b'Content-Disposition: form-data; name="file"; filename="test.bin"\r\n'
                b"Content-Type: application/octet-stream\r\n\r\n"
                + payload + b"\r\n--" + boundary + b"--\r\n"
            )
            with patch("cts2.web._start_job", side_effect=fake_start):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/convert",
                    data=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    js = json.loads(resp.read())
                    self.assertEqual(js["job"], "job-test")
            self.assertEqual(captured["data"], payload)
            self.assertEqual(captured["name"], "test.bin")
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
