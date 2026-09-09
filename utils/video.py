"""
Memory-safe PyAV video reader (no persistent container / no cache).

Notes:
- Random access is GOP-limited: seek lands on (previous) keyframe then decodes forward.
- Indexing uses PTS->time->index via fps; for VFR sources this can be approximate.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import os
import av
import math
import imageio
import subprocess
import numpy as np
from os.path import dirname, abspath

from PIL import Image
from utils.console import log
from utils.console import yellow
from utils.console import red
from utils.console import print
from utils.console import tqdm
from utils.timer import timer

os.environ['SVT_LOG'] = '1'  # avoid verbose av1 logging

import struct


def read_metadata_from_moov(video_path):
    """Extract video metadata + keyframe PTS from the mp4 moov atom.

    Reads only the moov index (a few MB), not the video payload. Sub-second
    even on multi-GB files; replaces multiple av.open probes for CFR videos.

    Returns dict with: fps, n_frames, height, width, duration, time_base,
    timescale, keyframe_pts (numpy array, or None if all frames are keyframes).
    `n_frames` is the DECODABLE frame count: trailing 0-duration stts samples
    (a muxer end-of-stream artifact, not emitted by the decoder) are excluded.
    Raises ValueError for non-mp4 or missing boxes.
    """

    def _read_header(f):
        # Parse one ISO-BMFF box header: 4-byte big-endian size, 4-byte type.
        # `size` counts the whole box INCLUDING the header, so `start + size`
        # is the next box. `hdr` is the header length so callers can skip to
        # the box payload. Two special size encodings per the spec:
        #   size == 1 -> a 64-bit "largesize" follows (header grows to 16 B);
        #               used by boxes >4 GiB (e.g. a huge mdat).
        #   size == 0 -> box runs to end-of-file (only legal for the last box);
        #               compute its length from the current EOF offset.
        data = f.read(8)
        if len(data) < 8:
            return None, 0, 0
        size, = struct.unpack('>I', data[:4])
        btype = data[4:8]
        hdr = 8
        if size == 1:
            size, = struct.unpack('>Q', f.read(8))
            hdr = 16
        elif size == 0:
            cur = f.tell()
            f.seek(0, 2)
            size = f.tell() - cur + hdr
            f.seek(cur)
        return btype, size, hdr

    def _find(f, target, end):
        # Linear scan of sibling boxes from the current offset up to `end`,
        # returning (payload_start, box_end) of the first box whose type ==
        # `target`, else (None, None). Boxes are walked by `start + size`, so
        # this descends into a container by calling _find again with the
        # returned (payload_start, box_end) as the new search window.
        while f.tell() < end:
            start = f.tell()
            btype, size, hdr = _read_header(f)
            if btype is None:
                break
            if btype == target:
                return f.tell(), start + size
            f.seek(start + size)
        return None, None

    with open(video_path, 'rb') as f:
        fsize = f.seek(0, 2)
        f.seek(0)

        moov_s, moov_e = _find(f, b'moov', fsize)
        if moov_s is None:
            raise ValueError(f'No moov box in {video_path}')

        # Find video trak (hdlr handler_type == 'vide')
        f.seek(moov_s)
        trak_s = trak_e = None
        while f.tell() < moov_e:
            ts, te = _find(f, b'trak', moov_e)
            if ts is None:
                break
            f.seek(ts)
            mdia_s, mdia_e = _find(f, b'mdia', te)
            if mdia_s:
                f.seek(mdia_s)
                hdlr_s, _ = _find(f, b'hdlr', mdia_e)
                if hdlr_s:
                    f.seek(hdlr_s + 4)
                    if f.read(8)[4:8] == b'vide':
                        trak_s, trak_e = ts, te
                        break
            f.seek(te if te else moov_e)

        if trak_s is None:
            raise ValueError(f'No video trak in {video_path}')

        # trak → tkhd (width/height as 16.16 fixed-point)
        f.seek(trak_s)
        tkhd_s, _ = _find(f, b'tkhd', trak_e)
        f.seek(tkhd_s)
        # tkhd is a FullBox: 1-byte version + 3-byte flags lead every such box.
        # The version selects 32- vs 64-bit time fields below.
        ver = struct.unpack('B', f.read(1))[0]
        f.read(3)
        # skip creation/modification/track_ID/reserved/duration
        # v0: 4+4+4+4+4 = 20 bytes; v1: 8+8+4+4+8 = 32 bytes.
        f.read(20 if ver == 0 else 32)
        f.read(8 + 2 + 2 + 2 + 2 + 36)  # reserved + layer + group + volume + reserved + matrix
        # Track display width/height are 16.16 fixed-point; the integer pixel
        # size is the high 16 bits (the fractional part is unused here).
        w_fixed, h_fixed = struct.unpack('>II', f.read(8))
        width = w_fixed >> 16
        height = h_fixed >> 16

        # trak → mdia → mdhd (timescale + duration).
        # timescale = media ticks per second; every PTS/stts delta in this trak
        # is expressed in these ticks, so time_base = 1 / timescale (seconds).
        f.seek(trak_s)
        mdia_s, mdia_e = _find(f, b'mdia', trak_e)
        f.seek(mdia_s)
        mdhd_s, _ = _find(f, b'mdhd', mdia_e)
        f.seek(mdhd_s)
        ver = struct.unpack('B', f.read(1))[0]
        f.read(3)
        if ver == 0:
            f.read(8)  # creation + modification
            timescale, mdhd_duration = struct.unpack('>II', f.read(8))
        else:
            f.read(16)
            timescale, = struct.unpack('>I', f.read(4))
            mdhd_duration, = struct.unpack('>Q', f.read(8))

        # mdia → minf → stbl
        f.seek(mdia_s)
        minf_s, minf_e = _find(f, b'minf', mdia_e)
        f.seek(minf_s)
        stbl_s, stbl_e = _find(f, b'stbl', minf_e)

        # stts → sample durations (gives frame count + fps)
        f.seek(stbl_s)
        stts_s, _ = _find(f, b'stts', stbl_e)
        f.seek(stts_s + 4)  # +4: skip the FullBox 1-byte version + 3-byte flags
        n_stts, = struct.unpack('>I', f.read(4))
        # stts is run-length encoded: n_stts entries of (sample_count,
        # sample_delta), each a big-endian u32 (8 bytes/entry). counts[i] frames
        # share duration deltas[i] ticks. For strict CFR there is usually a
        # single entry, e.g. (n_frames, 512).
        raw = np.frombuffer(f.read(n_stts * 8), dtype='>u4').reshape(-1, 2)
        counts = raw[:, 0].astype(np.int64)
        deltas = raw[:, 1].astype(np.int64)
        # Strip TRAILING zero-duration stts samples before counting frames. A
        # 0-duration final sample is a common muxer/trim artifact (ffmpeg
        # end-of-stream padding): it appears in the stts AND as a packet, but the
        # decoder never emits it as a frame. raw mvgame GT renders all carry stts
        # [(499, 512), (1, 0)] — 500 samples but only 499 decodable frames (last
        # real PTS 254976, while the phantom sits at 255488 == stream.duration).
        # Counting it makes the reader report len == 500 while get_batch can only
        # decode 499, so any request for the last index raises "Expected 1 frames,
        # only got 0" — which is exactly what crashed compute_sequence_gamma (it
        # samples the window's last frame via a linspace endpoint). Drop trailing
        # zero-delta entries so n_frames matches the decodable frame count. Keep
        # at least one entry; a mid-stream 0-delta is left intact on purpose so a
        # genuine anomaly still surfaces downstream rather than being masked.
        while len(deltas) > 1 and deltas[-1] == 0:
            counts = counts[:-1]
            deltas = deltas[:-1]
        n_stts = len(deltas)
        n_frames = int(counts.sum())
        # FPS from the DOMINANT (most-common-by-count) sample delta — i.e. the
        # mode, which equals r_frame_rate / PyAV's stream.base_rate. A
        # count-weighted average is wrong for CFR files whose final sample
        # carries a 0- or odd-duration (common muxer behavior): raw mvgame's
        # stts is [(499, 512), (1, 0)] → weighted avg 510.976 → fps 25.0501
        # instead of the true 25.0. The skewed fps then makes the per-frame
        # target PTS (round(idx / fps / time_base)) miss the real 512-tick frame
        # grid, so get_batch's exact `frame.pts in group` match fails and raises
        # "Expected N frames, only got 1". The mode delta is robust to such tail
        # samples; clean single-entry CFR (aug mvgame / nymeria / epic) is
        # unaffected (mode == the sole delta).
        dominant_delta = int(deltas[int(np.argmax(counts))])
        fps = timescale / dominant_delta if dominant_delta > 0 else 0.0

        # stss → keyframe sample numbers (1-indexed). Absent → all keyframes.
        f.seek(stbl_s)
        stss_s, _ = _find(f, b'stss', stbl_e)
        if stss_s is None:
            kf_pts = None
        else:
            f.seek(stss_s + 4)  # +4: skip FullBox version + flags
            n_kf, = struct.unpack('>I', f.read(4))
            # stss lists keyframe SAMPLE NUMBERS (1-indexed); -1 -> 0-indexed.
            kf_samples = np.frombuffer(f.read(n_kf * 4), dtype='>u4').astype(np.int64) - 1
            # Convert each keyframe sample index to its PTS by walking the stts
            # run-length table (stts gives durations, not absolute PTS).
            if n_stts == 1:
                # Single run: every frame has the same delta -> PTS = index * delta.
                kf_pts = kf_samples * deltas[0]
            else:
                # Multi-run: locate the run each keyframe falls in, then add the
                # intra-run offset. boundaries[i] = sample count through run i;
                # pts_starts[i] = PTS at the first sample of run i (sum of
                # count*delta over preceding runs). searchsorted(side='right')
                # picks the run index `idx`; prev[idx] is the first sample number
                # of that run, so (kf_samples - prev[idx]) is the offset within it.
                boundaries = np.cumsum(counts)
                pts_starts = np.concatenate([[0], np.cumsum(counts[:-1] * deltas[:-1])])
                idx = np.searchsorted(boundaries, kf_samples, side='right')
                prev = np.concatenate([[0], boundaries[:-1]])
                kf_pts = pts_starts[idx] + (kf_samples - prev[idx]) * deltas[idx]

        return dict(
            fps=fps,
            n_frames=n_frames,
            height=height,
            width=width,
            duration=mdhd_duration / timescale if timescale else 0.0,
            time_base=1.0 / timescale if timescale else 0.0,
            timescale=timescale,
            keyframe_pts=kf_pts,
        )


class CFRVideoReader:
    """
    Stateless Video Reader optimized for large-scale datasets.
    This should be used in conjunction with the following FFMPEGVideoWriter due to excessive priors.
    Assumptions:
    - Exactly aligned FPS and time base
    - Strictly constant frame rate
    - Monotonically increasing PTS
    - Starts at PTS 0
    - No B-frames

    Logic:
    - Does NOT hold file handles in __init__.
    - Opens and closes the container within get_batch/get_shape.
    - Uses robust seeking (backward seek + flush + margin) to prevent artifacts.
    - Only works on CFR videos (constant frame rate).
    """

    # PID of the process that imported this module (the parent). Captured at
    # class-definition time so DataLoader worker subprocesses, which fork after
    # import, see a different os.getpid() and are detected as forked below.
    INITIAL_PID = os.getpid()

    @classmethod
    def is_forked(cls):
        return os.getpid() != cls.INITIAL_PID

    def __init__(
        self,
        video_path: str,
        *,
        format: str = "rgb24",
        thread_type: str = "NONE",
        **kwargs  # used for initialization
    ):
        self.video_path = video_path
        self.format = format

        # FFmpeg multi-threaded decode ('AUTO') deadlocks in ANY process tied to
        # fork-based multiprocessing — NOT just forked children but also the MAIN
        # process that forked the DataLoader workers. visualize()/sample_val_batch
        # builds the val sample in the main process and decodes directly (e.g.
        # mvgame_raw CFR + gamma -> compute_sequence_gamma -> get_batch), so AUTO
        # there hangs. is_forked() only catches the forked-child case, so the safe
        # default is 'NONE'; 'AUTO' is opt-in for verified single-process / no-fork
        # contexts. An explicit 'AUTO' is still downgraded inside a forked child as
        # a backstop. thread_type only affects decode parallelism, never the pixels.
        if thread_type == 'AUTO' and self.is_forked():
            thread_type = 'NONE'
        self.thread_type = thread_type

        for key, value in kwargs.items():
            setattr(self, key, value)

        # Single moov-atom parse pulls fps + shape + duration + time_base + keyframe_pts
        # in one file open. No av.open in __init__ — moov index is tiny (a few MB)
        # so this is sub-second even on multi-GB videos. Skip when caller passed
        # pre-computed values via **kwargs (e.g. cached metadata).
        if not (hasattr(self, 'fps') and hasattr(self, 'shape') and hasattr(self, 'keyframe_pts')):
            try:
                meta = read_metadata_from_moov(self.video_path)
                if self.format in ("rgb24", "bgr24"):
                    c = 3
                elif "gray" in self.format:
                    c = 1
                else:
                    c = 3
                self.fps = meta['fps']
                self.duration = meta['duration']
                self.time_base = meta['time_base']
                self.shape = (meta['n_frames'], meta['height'], meta['width'], c)
                kf = meta['keyframe_pts']
                # keyframe_pts is None when stss is absent, i.e. EVERY frame is a
                # keyframe. Synthesize one PTS per frame on the CFR grid: ticks
                # per frame = round(1 / (time_base * fps)) = timescale / fps =
                # the stts delta. max(..., 1e-9) just guards against div-by-zero
                # if metadata was degenerate.
                self.keyframe_pts = kf if kf is not None else np.arange(meta['n_frames'], dtype=np.int64) * int(round(1.0 / max(meta['time_base'] * meta['fps'], 1e-9)))
            except (ValueError, struct.error, OSError):
                # Non-mp4 / unsupported container: fall back to av-based probes.
                self.get_fps()
                self.get_shape()
                self.get_keyframe_pts()

        self.keyframe_pts = np.asarray(self.keyframe_pts)

    def get_fps(self) -> float:
        if hasattr(self, 'fps'):
            return self.fps
        """Robustly determine FPS from stream metadata."""
        with av.open(self.video_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = self.thread_type

            # if stream.base_rate is not None:
            #     try:
            #         v = float(stream.base_rate)
            #         if v > 0:
            #             self.fps = v
            #             return
            #     except Exception:
            #         pass

            # if stream.average_rate is not None:
            #     try:
            #         v = float(stream.average_rate)
            #         if v > 0:
            #             self.fps = v
            #             return
            #     except Exception:
            #         pass

            # self.fps = 30.0
            # base_rate = r_frame_rate = the exact CFR rate (== timescale /
            # mode-delta). Preferred over average_rate, which a trailing
            # 0-duration sample skews (same reasoning as the moov-parse fps mode).
            self.fps = float(stream.base_rate)

    @property
    def f(self): return self.shape[0]
    @property
    def h(self): return self.shape[1]
    @property
    def w(self): return self.shape[2]
    @property
    def c(self): return self.shape[3]

    def get_video_meta(self):
        return {
            'fps': self.fps,
            'duration': self.duration,
            'time_base': self.time_base,
            'shape': self.shape,
            'keyframe_pts': self.keyframe_pts.tolist(),
        }

    def get_shape(self) -> Tuple[int, int, int, int]:
        """
        Returns (Frames, Height, Width, Channels).
        Opens the file briefly to probe metadata.
        """
        if hasattr(self, 'shape'):
            return self.shape

        # Open briefly just to probe
        with av.open(self.video_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = self.thread_type

            h = int(stream.codec_context.height)
            w = int(stream.codec_context.width)

            # Determine channels
            if self.format in ("rgb24", "bgr24"):
                c = 3
            elif "gray" in self.format:
                c = 1
            else:
                c = 3

            # Determine frame count.
            # frames = duration_sec * base_rate, where base_rate is the CFR
            # nominal rate (r_frame_rate), NOT average_rate. The commented-out
            # stream.frames path is avoided because containers often report a
            # wrong/zero header value ("pollution"); deriving from duration is
            # the reliable count for the strictly-CFR videos this reader targets.
            # Sometimes we get pollution from stream.frames
            # if stream.frames > 0:
            #     f = int(stream.frames)
            # elif container.duration is not None:
            #     fps = self.get_fps(stream)
            #     dur_sec = container.duration / av.time_base
            #     f = int(round(dur_sec * fps))
            # elif stream.duration is not None:
            #     fps = self.get_fps(stream)
            #     dur_sec = stream.duration * stream.time_base
            #     f = int(round(dur_sec * fps))
            # else:
            #     f = 0
            f = int(stream.duration * stream.time_base * stream.base_rate)

            self.duration = float(stream.duration * stream.time_base)
            self.time_base = float(stream.time_base)  # convert Fraction to float
            self.shape = (f, h, w, c)

    def get_keyframe_pts(self) -> List[int]:
        """Extract keyframe PTS. Tries fast moov atom parse first, falls back
        to av demux for non-mp4 containers."""
        if hasattr(self, 'keyframe_pts'):
            return self.keyframe_pts

        try:
            meta = read_metadata_from_moov(self.video_path)
            if meta['keyframe_pts'] is not None:
                self.keyframe_pts = meta['keyframe_pts']
                return self.keyframe_pts
        except (ValueError, struct.error, OSError):
            pass

        with av.open(self.video_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = self.thread_type
            keyframe_pts = []
            for packet in container.demux(stream):
                if packet.is_keyframe:
                    keyframe_pts.append(packet.pts)
            self.keyframe_pts = np.asarray(keyframe_pts)

    def __len__(self):
        return self.f

    def get_batch(self,
                  frame_inds: Union[np.ndarray, Iterable[int]],
                  unique_and_sorted: bool = False,
                  return_unstacked: bool = False,
                  return_channel_first: bool = False,
                  return_tensor: bool = False,
                  ratio: float = 1.0,
                  size: List[int] = None) -> np.ndarray:
        """
        Main decoding function.
        Opens file -> Seeks -> Decodes -> Closes file.
        Returns np.ndarray of shape (N, H, W, C).
        """
        if isinstance(frame_inds, list):
            frame_inds = np.asarray(frame_inds)
        if not isinstance(frame_inds, np.ndarray):
            frame_inds = np.asarray(list(frame_inds))
        if not len(frame_inds):
            return np.array([])

        # Get sorted unique indices and reconstruction reverse indices.
        # clip to [-len, len-1] then mod len so out-of-range and negative
        # indices wrap Python-style; np.unique sorts AND dedups so we decode
        # each distinct frame once and later scatter back via inverse_inds.
        frame_inds = np.clip(frame_inds, -len(self), len(self) - 1)
        frame_inds = frame_inds % len(self)  # handle negative indices
        if not unique_and_sorted:
            unique_frame_inds, inverse_inds = np.unique(frame_inds, return_inverse=True)
        else:
            unique_frame_inds = frame_inds
        # Frame index -> PTS on the CFR grid. fps*time_base = 1/delta, so this
        # is round(idx * delta); the round() lands exactly on the integer tick
        # grid (e.g. multiples of 512) that the `frame.pts in group` test below
        # depends on — which is why fps must be the stts MODE, not an average.
        unique_frame_pts = np.round(unique_frame_inds / self.fps / self.time_base).astype(np.int64)  # get the pts values for target frames

        # Group target pts by the keyframe that precedes them, so we seek once
        # per keyframe and decode forward through all wanted frames in that GOP.
        # searchsorted(side='right')-1 = index of the last keyframe at/ before
        # each pts. Because unique_frame_pts is sorted, the keyframe indices are
        # non-decreasing, so np.split at the per-keyframe counts partitions the
        # sorted pts into contiguous groups (G groups, g frames each).
        keyframe_inds = np.searchsorted(self.keyframe_pts, unique_frame_pts, side='right') - 1  # the keyframe corresponding to the indices we want
        unique_keyframe_inds, counts = np.unique(keyframe_inds, return_counts=True)  # G
        unique_keyframe_pts = self.keyframe_pts[unique_keyframe_inds]  # G
        unique_frame_pts_groups = np.split(unique_frame_pts, np.cumsum(counts)[:-1])  # list of pts in the same group, G: g

        unique_frames = []
        loaded = 0  # frames committed by COMPLETED groups; len(unique_frames)-loaded = progress in current group

        with av.open(self.video_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = self.thread_type

            for anchor, group in zip(unique_keyframe_pts, unique_frame_pts_groups):
                container.seek(int(anchor), stream=stream)  # will seek to this keyframe

                # Decode forward from the keyframe, keeping only frames whose PTS
                # is in this group (exact integer match — relies on CFR + PTS-0
                # start + no B-frames so decode order == display order).
                for frame in container.decode(stream):
                    if frame.pts in group:
                        if ratio != 1.0:
                            # +1e-6 before int() truncation absorbs float error so
                            # an exact ratio (e.g. H*0.5) doesn't fall to H/2 - 1.
                            Ho, Wo = (np.asarray(self.shape[1:3]) * ratio + 1e-6).astype(np.int64)
                            Ho, Wo = int(Ho), int(Wo)
                            frame = frame.reformat(height=Ho, width=Wo)
                        if size is not None:
                            frame = frame.reformat(height=size[0], width=size[1])
                        np_frame = frame.to_ndarray(format=self.format)
                        if return_channel_first:
                            np_frame = np_frame.transpose(2, 0, 1)  # 3, H, W
                        unique_frames.append(np_frame)  # H, W, 3
                    if len(unique_frames) - loaded == len(group):
                        break  # got every frame in this group; stop decoding the GOP
                    if frame.pts >= group[-1]:
                        # Passed the last wanted PTS without collecting all of them
                        # (e.g. a requested PTS isn't an actual frame). Stop so the
                        # count check below raises instead of decoding to EOF.
                        break  # safety break

                if len(unique_frames) - loaded != len(group):
                    raise RuntimeError(f'Expected {len(group)} frames to be decoded, only got {len(unique_frames) - loaded}')

                loaded += len(group)

        if not return_unstacked:
            unique_frames = np.asarray(unique_frames)

        if not unique_and_sorted:
            # Scatter the deduped/sorted decode back to the caller's original
            # index order (and duplicate any repeated indices) via inverse_inds.
            unique_frames = np.asarray(unique_frames)
            frames = unique_frames[inverse_inds]  # N, H, W, 3
        else:
            frames = unique_frames

        if return_tensor:
            import torch
            if isinstance(frames, np.ndarray):
                frames = torch.from_numpy(frames)
            else:
                frames = [torch.from_numpy(f) for f in frames]

        return frames

    def __getitem__(self, item):
        """
        Interface wrapper to support slicing and list indexing.
        """
        if isinstance(item, int):
            return self.get_batch([item])[0]

        if isinstance(item, (list, tuple, np.ndarray)):
            return self.get_batch(item)

        if isinstance(item, slice):
            start = 0 if item.start is None else item.start
            stop = item.stop
            if stop is None:
                # We need shape to determine end if not provided,
                # but better to raise error or probe.
                f, _, _, _ = self.shape
                stop = f
            step = 1 if item.step is None else item.step
            return self.get_batch(range(start, stop, step))

        raise TypeError(f"Invalid index type: {type(item)}")

    def decode(self, start_frame: int, end_frame: int) -> np.ndarray:
        """Helper for range decoding."""
        return self.get_batch(range(start_frame, end_frame))


class TorchCodecVideoReader:
    """
    Video Reader using Meta's torchcodec.
    Provides the exact same API as CFRVideoReader but handles VFR, B-frames,
    and exact PTS matching natively without memory leaks.
    """

    def __init__(
        self,
        video_path: str,
        **kwargs
    ):
        import torchcodec

        self.video_path = video_path

        for key, value in kwargs.items():
            setattr(self, key, value)

        # Initialize decoder (this scans the file to build an exact index).
        # The exact index is why this class needs none of CFRVideoReader's
        # CFR / PTS-0 / no-B-frame assumptions: get_frames_at() below indexes
        # by frame number directly instead of computing target PTS values.
        self.decoder = torchcodec.decoders.VideoDecoder(self.video_path)

        # Extract metadata to match CFRVideoReader. average_fps is fine here
        # because frames are addressed by index, not by a PTS derived from fps.
        self.fps = float(self.decoder.metadata.average_fps)
        self.duration = float(self.decoder.metadata.duration_seconds)

        # torchcodec natively decodes to RGB
        c = 3
        f = len(self.decoder)
        h = self.decoder.metadata.height
        w = self.decoder.metadata.width

        self.shape = (f, h, w, c)

    @property
    def f(self): return self.shape[0]
    @property
    def h(self): return self.shape[1]
    @property
    def w(self): return self.shape[2]
    @property
    def c(self): return self.shape[3]

    def __len__(self):
        return self.f

    def get_batch(
        self,
        frame_inds: Union[np.ndarray, Iterable[int]],
        unique_and_sorted: bool = False,  # no need for this
        return_unstacked: bool = False,
        return_channel_first: bool = False,
        return_tensor: bool = False,
        ratio: float = 1.0,
        size: List[int] = None
    ) -> Union[np.ndarray, list]:

        import torch.nn.functional as F

        if isinstance(frame_inds, list):
            frame_inds = np.asarray(frame_inds)
        if not isinstance(frame_inds, np.ndarray):
            frame_inds = np.asarray(list(frame_inds))
        if not len(frame_inds):
            return np.array([])

        # Handle negative indices and clipping just like the original
        frame_inds = np.clip(frame_inds, -len(self), len(self) - 1)
        frame_inds = frame_inds % len(self)

        # torchcodec requires a list of python ints
        query_inds = frame_inds.tolist()

        # torchcodec returns a FrameBatch, .data is a Tensor of shape [N, C, H, W] of dtype uint8
        frames_tensor = self.decoder.get_frames_at(query_inds).data

        # Handle resizing (ratio or size)
        if ratio != 1.0 or size is not None:
            # F.interpolate requires float tensor
            frames_tensor = frames_tensor.float()
            if size is not None:
                frames_tensor = F.interpolate(frames_tensor, size=(size[0], size[1]), mode='bilinear', align_corners=False)
            elif ratio != 1.0:
                Ho = int(self.h * ratio + 1e-6)
                Wo = int(self.w * ratio + 1e-6)
                frames_tensor = F.interpolate(frames_tensor, size=(Ho, Wo), mode='bilinear', align_corners=False)
            frames_tensor = frames_tensor.byte()

        # Handle channel dimension (torchcodec is N, C, H, W by default)
        if not return_channel_first:
            frames_tensor = frames_tensor.permute(0, 2, 3, 1)  # N, H, W, C

        # Handle return types
        if not return_tensor:
            frames = frames_tensor.numpy()
        else:
            frames = frames_tensor

        if return_unstacked:
            frames = list(frames)

        return frames

    def __getitem__(self, item):
        """
        Interface wrapper to support slicing and list indexing.
        """
        if isinstance(item, int):
            return self.get_batch([item])[0]

        if isinstance(item, (list, tuple, np.ndarray)):
            return self.get_batch(item)

        if isinstance(item, slice):
            start = 0 if item.start is None else item.start
            stop = item.stop
            if stop is None:
                stop = self.f
            step = 1 if item.step is None else item.step
            return self.get_batch(range(start, stop, step))

        raise TypeError(f"Invalid index type: {type(item)}")

    def decode(self, start_frame: int, end_frame: int):
        """Helper for range decoding."""
        return self.get_batch(range(start_frame, end_frame))


def write_video(filename: str, frames: np.ndarray, **kwargs):
    """
    Optimized bulk video writer.
    """
    # 1. Handle Torch Tensors without importing torch globally
    if hasattr(frames, 'cpu'):
        frames = frames.cpu().numpy()

    # 2. Vectorized Pre-processing (Faster than per-frame check)
    # Ensure frames are uint8. If float (0.0-1.0), scale and cast.
    if frames.dtype != np.uint8:
        print(f"Converting video data from {frames.dtype} to uint8...")
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)

    F, H, W, C = frames.shape

    # 3. Instantiate Writer
    vw = FFMPEGVideoWriter(filename, n_frames=F, height=H, width=W, **kwargs)

    # 4. Bulk Write (Zero Python loop overhead)
    # .tofile() writes the raw buffer directly to the file object
    vw.write_batch(frames)

    vw.close()


class FFMPEGVideoWriter:
    """
    Multi-process video writer that uses pipe to communicate with an ffmpeg instance.

    - Avoids strange memory leaks in regular PyAV writer.
    """

    def __init__(
        self,
        filename='default.mp4',
        n_frames=500,  # stub argument, not used or required
        height=720,
        width=1280,
        fps=25,
        crf=20,
        maxrate=None,  # None = pure CRF (no VBV cap). A cap starves giant frames.
        encoding="libx264",
        pix_fmt="yuv420p",
        preset="veryslow",  # high quality
        # tune="zerolatency",  # would actually be easier to decode
        tune=None,
        color_range='tv',  # use the default tv range
        movflags="+faststart",
        hwaccel='none',
        loglevel="error",
        # Small GOP so a CFRVideoReader random seek decodes few frames forward
        # from the keyframe (every ~12 frames is an I-frame). Trades file size
        # for fast random access — the whole point of the CFR reader/writer pair.
        gop_size=12,  # at most seek past 1s?
        threads=0,
    ):
        self.filename = filename
        self.height = height
        self.width = width
        os.makedirs(dirname(abspath(filename)), exist_ok=True)

        # Base Command
        command = [
            "ffmpeg",
            "-threads", str(threads),
            "-y",
            "-hide_banner",
            "-loglevel", loglevel,

            # Input Settings.
            # Input is fixed rawvideo rgb24 over stdin: the writer's contract is
            # that frames piped in are contiguous uint8 H*W*3 RGB (what
            # write_video produces and submit_frame validates). -s gives ffmpeg
            # the frame geometry since rawvideo carries no header.
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",  # Stdin

            # Output Settings
            "-an",  # no audio
            "-vcodec", encoding,
            # This occurrence is output-scoped; the earlier one is input-scoped.
            "-threads", str(threads),
            "-pix_fmt", pix_fmt,
            "-color_range", color_range,
            "-movflags", movflags,
            "-g", str(gop_size),  # small GOP for ultrafast decoding
        ]

        # HW Acceleration Injection
        if hwaccel != 'none':
            command.insert(1, "-hwaccel")
            command.insert(2, hwaccel)

        # Quality Control — pure CRF by default. A VBV cap (-maxrate/-bufsize)
        # starves very large frames: at maxrate/fps bits-per-frame, a giant
        # (e.g. 11648x1294 multi-view) frame can't reach CRF quality, and the
        # FINAL frame is hit worst — a forced end-of-stream I-frame that can't
        # borrow bits from future frames craters to a blocky mess. That was the
        # real cause of the "last frame is blurry" reports. Only pass maxrate when
        # a hard bitrate ceiling is actually required.
        if maxrate:
            command += ["-maxrate", str(maxrate), "-bufsize", str(maxrate * 2)]
        command += ["-crf", str(crf)]

        # Speed/Tune presets
        if preset:
            command += ["-preset", preset]
        if tune:
            command += ["-tune", tune]

        self.ffmpeg_proc = subprocess.Popen(
            command + [filename],
            stdin=subprocess.PIPE,
        )

    def write_batch(self, frames: np.ndarray):
        """Writes the entire array to ffmpeg in one go."""
        # Sanity check
        if frames.ndim != 4:
            raise ValueError(f"Expected (F, H, W, 3), got {frames.shape}")

        frames = np.ascontiguousarray(frames)

        # This is the fastest way to write numpy data to a file pipe
        # It avoids the memory copy of .tobytes() if the array is contiguous
        try:
            # frames.tofile(self.ffmpeg_proc.stdin)
            self.ffmpeg_proc.stdin.write(memoryview(frames))
        except BrokenPipeError:
            print("FFmpeg Error: Broken Pipe. Check if dimensions match or if disk is full.")

        # DO NOT FLUSH HERE. Let the OS buffer the pipe.

    def submit_frame(self, frame: np.ndarray):
        """Legacy support for single frame."""
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Expected frame shape {(self.height, self.width, 3)}, got {frame.shape}")
        self.ffmpeg_proc.stdin.write(frame.tobytes())

    def close(self):
        if self.ffmpeg_proc.stdin:
            self.ffmpeg_proc.stdin.close()
        return_code = self.ffmpeg_proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {return_code} while writing {self.filename}"
            )
