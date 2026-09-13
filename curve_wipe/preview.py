"""Live RealSense view for manual positioning. No robot connection or commands."""
import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', default='243722072895')
    parser.add_argument('--output', type=Path, default=Path('data/preview'))
    args = parser.parse_args()
    import pyrealsense2 as rs
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    title = 'RealSense live - RGB | Depth - Q: quit, S: save, R: red overlay'
    started = False
    red_overlay = False
    last = time.monotonic()
    fps = 0.0
    try:
        pipeline.start(config)
        started = True
        align = rs.align(rs.stream.color)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 1280, 480)
        print('Live camera only. Q/ESC: close; S: save screenshot; R: toggle red overlay.', flush=True)
        while True:
            frames = align.process(pipeline.wait_for_frames(5000))
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            bgr = np.asanyarray(color.get_data()).copy()
            z = np.asanyarray(depth.get_data()).astype(np.float32) * depth.get_units()
            display = bgr.copy()
            if red_overlay:
                from .geometry import segment_red
                mask = segment_red(bgr, np.ones(bgr.shape[:2], dtype=bool))
                display[mask] = (0, 255, 255)
            colored = cv2.applyColorMap(np.clip(z / 1.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            colored[z <= 0] = 0
            now = time.monotonic()
            fps = .9 * fps + .1 / max(now-last, 1e-6)
            last = now
            cv2.putText(display, f'RGB  {fps:.0f} FPS   R: red overlay', (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            center_depth = z[z.shape[0]//2, z.shape[1]//2]
            label = f'Center: {center_depth*1000:.0f} mm' if center_depth > 0 else 'Center: no depth'
            cv2.putText(colored, 'Depth scale: 0..1 m  ' + label, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.drawMarker(display, (320, 240), (0, 255, 0), cv2.MARKER_CROSS, 18, 1)
            view = np.hstack([display, colored])
            cv2.imshow(title, view)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('r'), ord('R')):
                red_overlay = not red_overlay
            if key in (ord('s'), ord('S')):
                args.output.mkdir(parents=True, exist_ok=True)
                name = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                cv2.imwrite(str(args.output / f'{name}_color.png'), bgr)
                cv2.imwrite(str(args.output / f'{name}_view.png'), view)
                print(f'Saved {args.output / name} (preview only, no robot pose)', flush=True)
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
