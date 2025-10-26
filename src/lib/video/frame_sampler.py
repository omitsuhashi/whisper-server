from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class SampledFrame:
    """抽出したフレーム情報。

    Attributes:
        index: 元動画におけるフレーム番号。
        timestamp: 秒単位のタイムスタンプ。
        image: BGR 配列 (height, width, 3)。
    """

    index: int
    timestamp: float
    image: np.ndarray


class FrameSamplingError(RuntimeError):
    """フレーム抽出時のエラーを表す例外。"""


def sample_key_frames(
    video_path: str | Path,
    *,
    min_scene_span: float = 1.0,
    diff_threshold: float = 0.3,
    max_frames: Optional[int] = None,
) -> List[SampledFrame]:
    """画面切り替えを検出しつつ主要フレームを抽出する。

    画面共有やスライド切り替えなど、映像の内容が大きく変わったタイミングだけを
    取得できるようヒストグラムのバッタチャリヤ距離を用いて差分を評価する。

    Args:
        video_path: 対象動画へのパス。
        min_scene_span: 連続抽出を避けるための最小間隔（秒）。
        diff_threshold: ヒストグラム距離の閾値。0 に近いほど同一。
        max_frames: 指定すると最大抽出数を制限。

    Returns:
        SampledFrame のリスト。BGR の画像配列を含む。
    """

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FrameSamplingError(f"動画を開けませんでした: {path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0  # FPS を取得できない場合は既定値

        frame_gap = max(int(fps * min_scene_span), 1)

        sampled: List[SampledFrame] = []
        last_hist: Optional[np.ndarray] = None
        last_saved_index = -frame_gap

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            if frame_index == 0:
                sampled.append(
                    SampledFrame(
                        index=frame_index,
                        timestamp=_frame_index_to_time(frame_index, fps),
                        image=frame.copy(),
                    )
                )
                last_hist = _calc_histogram(frame)
                last_saved_index = frame_index
            else:
                if max_frames is not None and len(sampled) >= max_frames:
                    break

                if frame_index - last_saved_index < frame_gap:
                    frame_index += 1
                    continue

                hist = _calc_histogram(frame)
                diff = cv2.compareHist(
                    last_hist if last_hist is not None else hist,
                    hist,
                    cv2.HISTCMP_BHATTACHARYYA,
                )
                if diff >= diff_threshold:
                    sampled.append(
                        SampledFrame(
                            index=frame_index,
                            timestamp=_frame_index_to_time(frame_index, fps),
                            image=frame.copy(),
                        )
                    )
                    last_hist = hist
                    last_saved_index = frame_index

            frame_index += 1

        if max_frames is not None:
            sampled = sampled[:max_frames]

        return sampled
    finally:
        capture.release()
        cv2.destroyAllWindows()


def _calc_histogram(frame: np.ndarray) -> np.ndarray:
    """HSV ヒストグラム（正規化済み）を生成する。"""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _frame_index_to_time(frame_index: int, fps: float) -> float:
    return frame_index / fps if fps > 0 else 0.0
