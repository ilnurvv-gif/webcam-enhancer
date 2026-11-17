"""
Webcam Enhancer — улучшение картинки с веб-камеры в реальном времени (Windows).

Что делает:
  * открывает камеру в максимальном разрешении (MJPG, по умолчанию 1080p) —
    многие камеры по умолчанию отдают только 640x480;
  * временное шумоподавление (усредняет кадры, но не размывает движение);
  * автобаланс белого, автоосвещение лица/теней (CLAHE), гамма, контраст,
    яркость, насыщенность, резкость;
  * отдаёт результат в виртуальную камеру (OBS Virtual Camera), которую
    можно выбрать в Zoom / Telegram / Discord / Teams / браузере.

Клавиши в окне предпросмотра:
  C — сравнение до/после   S — настройки драйвера камеры
  V — вкл/выкл виртуальную камеру   R — сброс настроек   Q / Esc — выход
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

try:
    import pyvirtualcam
except ImportError:  # программа работает и без виртуальной камеры
    pyvirtualcam = None

APP_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
CONFIG_PATH = os.path.join(APP_DIR, "webcam_enhancer.json")

PREVIEW_WIN = "Webcam Enhancer"
SETTINGS_WIN = "Settings"

DEFAULTS = {
    "camera": 0,
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "brightness": 100,   # 100 = без изменений (0..200)
    "contrast": 100,     # % (20..200)
    "gamma": 100,        # 100 = 1.0, больше — светлее тени
    "auto_light": 35,    # сила CLAHE (0..100)
    "white_balance": 1,  # автобаланс белого вкл/выкл
    "denoise": 45,       # шумоподавление (0..100)
    "sharpen": 30,       # резкость (0..100)
    "saturation": 110,   # % (0..200)
    "mirror": 0,         # зеркалить картинку
    "virtual_cam": 1,
}

# (ключ в конфиге, подпись ползунка, максимум)
TRACKBARS = [
    ("brightness", "Brightness", 200),
    ("contrast", "Contrast", 200),
    ("gamma", "Gamma", 200),
    ("auto_light", "Auto light", 100),
    ("white_balance", "White bal", 1),
    ("denoise", "Denoise", 100),
    ("sharpen", "Sharpen", 100),
    ("saturation", "Saturation", 200),
    ("mirror", "Mirror", 1),
]


# ---------------------------------------------------------------- настройки
def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Не удалось сохранить настройки: {e}")


# ------------------------------------------------------------------ камера
def open_camera(index, width, height, fps):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        # MJPG нужен, чтобы USB-камера отдавала 1080p/720p с нормальным FPS
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            real_fps = cap.get(cv2.CAP_PROP_FPS) or fps
            if real_fps < 5 or real_fps > 120:
                real_fps = fps
            return cap, w, h, int(round(real_fps))
        cap.release()
    return None, 0, 0, 0


# ---------------------------------------------------------------- обработка
class Enhancer:
    def __init__(self):
        self.acc = None
        self.acc_small = None
        self.wb_gains = np.ones(3, np.float32)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.x = np.arange(256, dtype=np.float32) / 255.0

    def reset(self):
        self.acc = None
        self.acc_small = None
        self.wb_gains[:] = 1.0

    def denoise(self, frame, strength):
        """Временное шумоподавление с защитой движущихся участков от шлейфа."""
        if strength <= 0:
            self.acc = None
            return frame
        h, w = frame.shape[:2]
        sw, sh = max(w // 4, 1), max(h // 4, 1)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        f = frame.astype(np.float32)
        if self.acc is None or self.acc.shape != f.shape:
            self.acc = f
            self.acc_small = small
            return frame

        # Ищем движение на уменьшенном кадре: уменьшение усредняет шум,
        # а настоящее движение остаётся — и это в ~16 раз быстрее.
        diff = cv2.absdiff(small, self.acc_small)
        b, g, r = cv2.split(diff)
        diff = cv2.max(cv2.max(b, g), r).astype(np.float32)
        thr = 3.0 + strength * 0.25            # порог «это движение»
        motion = np.clip((diff - thr * 0.5) / (thr * 0.5), 0.0, 1.0)
        motion = cv2.dilate(motion, np.ones((3, 3), np.uint8))
        base = 1.0 - 0.8 * strength / 100.0    # доля нового кадра в статике
        alpha = cv2.resize(base + (1.0 - base) * motion, (w, h),
                           interpolation=cv2.INTER_LINEAR)
        alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)

        delta = cv2.subtract(f, self.acc)
        cv2.multiply(delta, alpha, dst=delta)
        cv2.add(self.acc, delta, dst=self.acc)
        out = cv2.convertScaleAbs(self.acc)
        self.acc_small = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA)
        return out

    def tone(self, img, s):
        """Баланс белого + гамма + контраст + яркость одной таблицей на канал."""
        if s["white_balance"]:
            means = np.array(cv2.mean(img)[:3], np.float32)
            target = means.mean() / np.maximum(means, 1.0)
            target = 1.0 + (np.clip(target, 0.6, 1.6) - 1.0) * 0.7
            self.wb_gains = 0.92 * self.wb_gains + 0.08 * target
            gains = self.wb_gains
        else:
            gains = np.ones(3, np.float32)

        gamma = max(s["gamma"], 20) / 100.0
        contrast = max(s["contrast"], 20) / 100.0
        bright = (s["brightness"] - 100) / 255.0
        if (not s["white_balance"] and gamma == 1.0 and contrast == 1.0
                and bright == 0.0):
            return img

        chans = cv2.split(img)
        out = []
        for ch, g in zip(chans, gains):
            y = np.clip(self.x * g, 0.0, 1.0) ** (1.0 / gamma)
            y = (y - 0.5) * contrast + 0.5 + bright
            lut = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)
            out.append(cv2.LUT(ch, lut))
        return cv2.merge(out)

    def light_and_color(self, img, s):
        """Автоосвещение (CLAHE по яркости) и насыщенность в пространстве LAB."""
        k = s["auto_light"] / 100.0
        sat = s["saturation"] / 100.0
        if k <= 0 and sat == 1.0:
            return img
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        if k > 0:
            l = cv2.addWeighted(self.clahe.apply(l), k, l, 1.0 - k, 0)
        if sat != 1.0:
            a = cv2.addWeighted(a, sat, a, 0, 128 * (1.0 - sat))
            b = cv2.addWeighted(b, sat, b, 0, 128 * (1.0 - sat))
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def sharpen(img, strength):
        if strength <= 0:
            return img
        amount = strength / 100.0 * 1.2
        sigma = max(1.0, img.shape[1] / 1280.0)
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)

    def process(self, frame, s):
        img = self.denoise(frame, s["denoise"])
        img = self.tone(img, s)
        img = self.light_and_color(img, s)
        img = self.sharpen(img, s["sharpen"])
        if s["mirror"]:
            img = cv2.flip(img, 1)
        return img


# ---------------------------------------------------------- виртуальная камера
def start_virtual_cam(w, h, fps):
    if pyvirtualcam is None:
        print("pyvirtualcam не установлен — работаю только в режиме предпросмотра.")
        return None
    try:
        cam = pyvirtualcam.Camera(width=w, height=h, fps=fps,
                                  fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"Виртуальная камера запущена: {cam.device}. "
              f"Выберите её в Zoom/Telegram/Discord.")
        return cam
    except Exception as e:  # noqa: BLE001
        print("Не удалось запустить виртуальную камеру.\n"
              "Установите OBS Studio (https://obsproject.com) — в нём есть драйвер "
              "OBS Virtual Camera.\n"
              f"Подробности: {e}")
        return None


# ---------------------------------------------------------------- интерфейс
def build_ui(cfg):
    cv2.namedWindow(PREVIEW_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW_WIN, 960, 540)
    cv2.namedWindow(SETTINGS_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(SETTINGS_WIN, 420, 420)
    for key, label, mx in TRACKBARS:
        cv2.createTrackbar(label, SETTINGS_WIN, int(cfg[key]), mx, lambda _v: None)


def read_ui(cfg):
    for key, label, _mx in TRACKBARS:
        pos = cv2.getTrackbarPos(label, SETTINGS_WIN)
        if pos >= 0:
            cfg[key] = pos


def set_ui(cfg):
    for key, label, _mx in TRACKBARS:
        cv2.setTrackbarPos(label, SETTINGS_WIN, int(cfg[key]))


def make_preview(original, processed, compare, info):
    if compare:
        view = processed.copy()
        if original.shape == processed.shape:
            half = view.shape[1] // 2
            view[:, :half] = original[:, :half]
            cv2.line(view, (half, 0), (half, view.shape[0]), (0, 255, 255), 2)
    else:
        view = processed
    if view.shape[1] > 1280:
        scale = 1280 / view.shape[1]
        view = cv2.resize(view, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    else:
        view = view.copy()
    for i, line in enumerate(info):
        y = 24 + i * 22
        cv2.putText(view, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return view


def window_closed(name):
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


# ---------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Улучшение картинки веб-камеры")
    parser.add_argument("--camera", type=int, help="номер камеры (0, 1, 2…)")
    parser.add_argument("--width", type=int, help="ширина, напр. 1280")
    parser.add_argument("--height", type=int, help="высота, напр. 720")
    parser.add_argument("--fps", type=int, help="кадров в секунду")
    args = parser.parse_args()

    cfg = load_config()
    for key in ("camera", "width", "height", "fps"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    print(f"Открываю камеру {cfg['camera']} "
          f"({cfg['width']}x{cfg['height']} @ {cfg['fps']})…")
    cap, w, h, fps = open_camera(cfg["camera"], cfg["width"], cfg["height"],
                                 cfg["fps"])
    if cap is None:
        print("Камера не найдена или занята другой программой.\n"
              "Закройте Zoom/Teams/браузер или попробуйте --camera 1")
        input("Нажмите Enter для выхода…")
        return
    print(f"Камера работает: {w}x{h} @ {fps} fps")

    enhancer = Enhancer()
    build_ui(cfg)
    vcam = start_virtual_cam(w, h, fps) if cfg["virtual_cam"] else None

    compare = False
    fails = 0
    t_prev, fps_meas = time.perf_counter(), 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 60:
                    print("Камера перестала отдавать кадры.")
                    break
                time.sleep(0.01)
                continue
            fails = 0

            read_ui(cfg)
            out = enhancer.process(frame, cfg)

            if vcam is not None and out.shape[1] == w and out.shape[0] == h:
                vcam.send(out)

            now = time.perf_counter()
            fps_meas = 0.9 * fps_meas + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            src = cv2.flip(frame, 1) if cfg["mirror"] else frame
            info = [
                f"{w}x{h}  {fps_meas:4.1f} fps   virtual cam: "
                f"{'ON' if vcam else 'OFF'}" + ("   [before | after]" if compare else ""),
                "C compare  S camera settings  V virtual cam  R reset  Q quit",
            ]
            cv2.imshow(PREVIEW_WIN, make_preview(src, out, compare, info))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")):
                compare = not compare
            elif key in (ord("s"), ord("S")):
                cap.set(cv2.CAP_PROP_SETTINGS, 1)  # окно драйвера (DirectShow)
            elif key in (ord("r"), ord("R")):
                for k, _label, _mx in TRACKBARS:
                    cfg[k] = DEFAULTS[k]
                set_ui(cfg)
                enhancer.reset()
            elif key in (ord("v"), ord("V")):
                if vcam is not None:
                    vcam.close()
                    vcam = None
                    cfg["virtual_cam"] = 0
                else:
                    vcam = start_virtual_cam(w, h, fps)
                    cfg["virtual_cam"] = 1 if vcam else 0

            if window_closed(PREVIEW_WIN):
                break
    finally:
        save_config(cfg)
        if vcam is not None:
            vcam.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
