import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import RectangleSelector


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def normalized(self, w: int, h: int) -> Tuple[float, float, float, float]:
        x1 = max(0, min(self.x1, w - 1))
        y1 = max(0, min(self.y1, h - 1))
        x2 = max(0, min(self.x2, w - 1))
        y2 = max(0, min(self.y2, h - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0
        return cx / w, cy / h, bw / w, bh / h


@dataclass
class State:
    boxes: List[Box] = field(default_factory=list)
    candidate: Optional[Box] = None
    dirty: bool = False


def load_existing_labels(label_path: str, w: int, h: int) -> List[Box]:
    if not os.path.exists(label_path):
        return []
    boxes: List[Box] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = parts
            try:
                cx = float(cx)
                cy = float(cy)
                bw = float(bw)
                bh = float(bh)
            except ValueError:
                continue
            x1 = int((cx - bw / 2.0) * w)
            y1 = int((cy - bh / 2.0) * h)
            x2 = int((cx + bw / 2.0) * w)
            y2 = int((cy + bh / 2.0) * h)
            boxes.append(Box(x1, y1, x2, y2))
    return boxes


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_labels_yolo(boxes: List[Box], label_path: str, w: int, h: int) -> None:
    lines: List[str] = []
    for b in boxes:
        cx, cy, bw, bh = b.normalized(w, h)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def copy_image(src: str, dst: str) -> None:
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def build_image_list(root: str) -> List[Tuple[str, str]]:
    image_list: List[Tuple[str, str]] = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(root, "images", split)
        if not os.path.isdir(split_dir):
            continue
        for name in sorted(os.listdir(split_dir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_list.append((split, os.path.join(split_dir, name)))
    return image_list


def main():
    src_root = "/home/ubuntu/project/deduibi/yolo/dataset/datasetm6"
    dst_root = "/home/ubuntu/project/deduibi/yolo/dataset/datasetm6c"

    plt.rcParams["keymap.quit"] = []

    images = build_image_list(src_root)
    if not images:
        print("No images found.")
        return

    ensure_dir(dst_root)

    state = State()
    idx = 0
    running = True

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("yolo-check")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    selector = None
    img = None
    img_path = ""
    split = ""
    h = w = 0

    def redraw():
        ax.clear()
        ax.imshow(img)
        for b in state.boxes:
            rect = plt.Rectangle((b.x1, b.y1), b.x2 - b.x1, b.y2 - b.y1,
                                 fill=False, edgecolor="lime", linewidth=2)
            ax.add_patch(rect)
        if state.candidate is not None:
            b = state.candidate
            rect = plt.Rectangle((b.x1, b.y1), b.x2 - b.x1, b.y2 - b.y1,
                                 fill=False, edgecolor="orange", linewidth=2)
            ax.add_patch(rect)
        ax.axis("off")
        rel_path = img_path.replace(src_root + os.sep, "", 1)
        text_lines = [
            f"{idx + 1}/{len(images)}  q:prev e:next r:clear b:commit u:save j:jump esc:quit",
            f"boxes:{len(state.boxes)}  status:{'unsaved' if state.dirty else 'saved'}",
            rel_path,
        ]
        y = 0.98
        for line in text_lines:
            t = ax.text(0.01, y, line, transform=ax.transAxes, fontsize=10,
                        color="white", va="top")
            t.set_path_effects([pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
            y -= 0.035
        fig.canvas.draw_idle()

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        state.candidate = Box(x1, y1, x2, y2)
        redraw()

    def attach_selector():
        nonlocal selector
        if selector is not None:
            selector.set_active(False)
        selector = RectangleSelector(
            ax,
            onselect,
            useblit=True,
            button=[1],
            interactive=True,
        )

    def load_current():
        nonlocal img, img_path, split, h, w
        split, img_path = images[idx]
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise RuntimeError(f"Failed to read {img_path}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        label_src = os.path.join(src_root, "labels", split,
                                 os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        state.boxes = load_existing_labels(label_src, w, h)
        state.candidate = None
        state.dirty = False
        redraw()
        attach_selector()

    def on_key(event):
        nonlocal idx, running
        key = event.key
        if key == "q":
            idx = max(0, idx - 1)
            load_current()
        elif key == "e":
            idx = min(len(images) - 1, idx + 1)
            load_current()
        elif key == "r":
            state.boxes = []
            state.candidate = None
            state.dirty = True
            redraw()
        elif key == "b":
            if state.candidate is not None:
                state.boxes.append(state.candidate)
                state.candidate = None
                state.dirty = True
                redraw()
        elif key == "u":
            dst_img_dir = os.path.join(dst_root, "images", split)
            dst_label_dir = os.path.join(dst_root, "labels", split)
            ensure_dir(dst_img_dir)
            ensure_dir(dst_label_dir)
            dst_img_path = os.path.join(dst_img_dir, os.path.basename(img_path))
            dst_label_path = os.path.join(
                dst_label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            )
            save_labels_yolo(state.boxes, dst_label_path, w, h)
            copy_image(img_path, dst_img_path)
            state.dirty = False
            print(f"Saved: {dst_label_path}")
            redraw()
        elif key == "j":
            target = input("Jump to image path (relative to dataset1, e.g. images/val/xxx.jpg): ").strip()
            if target:
                candidate = os.path.normpath(os.path.join(src_root, target))
                for i, (_, p) in enumerate(images):
                    if os.path.normpath(p) == candidate:
                        idx = i
                        load_current()
                        break
        elif key == "escape":
            running = False
            plt.close(fig)

    def on_close(_event):
        nonlocal running
        running = False

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    load_current()

    while running:
        plt.pause(0.05)


if __name__ == "__main__":
    main()
