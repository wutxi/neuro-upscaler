"""Нейро-апскейлер изображений (локальная модель ESRGAN / Real-ESRGAN и др. через spandrel).
Оценивает время задачи по железу: калибровочный прогон модели на твоей видеокарте/процессоре.
by wutxi
"""
import json
import math
import os
import platform
import queue
import subprocess
import threading
import time
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import psutil
import torch
from PIL import Image, ImageTk

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
CACHE_FILE = APP_DIR / "calibration.json"
COMFY_MODELS = Path(os.environ.get("LOCALAPPDATA", "")) / "Comfy-Desktop/ComfyUI-Installs/ComfyUI/ComfyUI/models/upscale_models"
MODELS_DIR.mkdir(exist_ok=True)

DOWNLOADS = {
    "RealESRGAN_x4plus.pth (фото, x4, 64 МБ)":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x2plus.pth (фото, x2, 64 МБ)":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth (арт/аниме, x4, 17 МБ)":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
}
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PAD = 16

BG, PANEL, FG, MUTED, ACCENT = "#14161a", "#1d2026", "#e8e8ea", "#8a8f98", "#7c5cff"


# ---------------- железо ----------------
def detect_hardware():
    cpu = platform.processor()
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=10, creationflags=0x08000000)
        cpu = out.stdout.strip() or cpu
    except Exception:
        pass
    hw = {
        "cpu": cpu,
        "cores": f"{psutil.cpu_count(logical=False)} ядер / {psutil.cpu_count()} потоков",
        "ram": f"{psutil.virtual_memory().total / 1024**3:.0f} ГБ",
        "gpu": "нет (CUDA недоступна)",
        "vram": 0.0,
        "device": "cpu",
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        hw.update(gpu=p.name, vram=p.total_memory / 1024**3, device="cuda")
    return hw


def pick_tile(hw):
    if hw["device"] == "cpu":
        return 256
    v = hw["vram"]
    return 512 if v >= 10 else 384 if v >= 6 else 256


def list_models():
    found = {}
    for d in (MODELS_DIR, COMFY_MODELS):
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".pth", ".safetensors", ".pt", ".ckpt") and f.name not in found:
                    found[f.name] = f
    return found


def load_cache():
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(c):
    CACHE_FILE.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")


def fmt_time(s):
    if s < 1:
        return "< 1 сек"
    s = int(round(s))
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    if h:
        return f"{h} ч {m} мин"
    if m:
        return f"{m} мин {sec} сек"
    return f"{sec} сек"


def tile_grid(w, h, tile):
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            yield x, y, min(tile, w - x), min(tile, h - y)


# ---------------- движок ----------------
class Engine:
    def __init__(self, hw):
        self.hw = hw
        self.device = torch.device(hw["device"])
        self.model = None
        self.model_name = None
        self.scale = 4
        self.half = False
        self.tile = pick_tile(hw)
        if hw["device"] == "cpu":
            torch.set_num_threads(psutil.cpu_count())

    def load(self, path: Path):
        from spandrel import ModelLoader
        desc = ModelLoader().load_from_file(str(path))
        self.half = self.device.type == "cuda" and desc.supports_half
        desc.model.eval().to(self.device)
        if self.half:
            desc.model.half()
        self.model, self.scale, self.model_name = desc.model, desc.scale, path.name

    def cache_key(self):
        return f"{self.hw['gpu']}|{self.hw['cpu']}|{self.model_name}|tile{self.tile}|half{int(self.half)}"

    @torch.inference_mode()
    def _run(self, arr):  # arr: HxWx3 float32 0..1
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t = t.half() if self.half else t.float()
        out = self.model(t)
        return out.squeeze(0).float().clamp_(0, 1).permute(1, 2, 0).cpu().numpy()

    def calibrate(self):
        """Замер скорости модели на этом железе: сек на один тайл полного размера."""
        cache = load_cache()
        key = self.cache_key()
        if key in cache:
            return cache[key]
        n = self.tile + 2 * PAD
        sample = np.random.rand(n, n, 3).astype(np.float32)
        if self.device.type == "cpu":  # на CPU меряем меньший тайл и масштабируем по площади
            m = 128 + 2 * PAD
            sample = sample[:m, :m]
        self._run(sample)  # прогрев
        self._sync()
        runs = 2 if self.device.type == "cuda" else 1
        t0 = time.perf_counter()
        for _ in range(runs):
            self._run(sample)
        self._sync()
        sec = (time.perf_counter() - t0) / runs
        if self.device.type == "cpu":
            sec *= (n * n) / (sample.shape[0] * sample.shape[1])
        cache[key] = {"sec_per_tile": sec, "correction": 1.0}
        save_cache(cache)
        return cache[key]

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def estimate(self, w, h, calib):
        full = (self.tile + 2 * PAD) ** 2
        work = sum((tw + 2 * PAD) * (th + 2 * PAD) / full for _, _, tw, th in tile_grid(w, h, self.tile))
        io = w * h * self.scale ** 2 * 4e-8  # загрузка/сохранение PNG
        return (calib["sec_per_tile"] * work + io) * calib.get("correction", 1.0)

    def learn(self, estimated, actual):
        cache = load_cache()
        c = cache.get(self.cache_key())
        if c and estimated > 0.5:
            c["correction"] = round(0.5 * c.get("correction", 1.0) + 0.5 * c.get("correction", 1.0) * actual / estimated, 3)
            save_cache(cache)

    def upscale(self, img: Image.Image, progress, stop):
        has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
        alpha = img.convert("RGBA").getchannel("A") if has_alpha else None
        src = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        h, w = src.shape[:2]
        s = self.scale
        out = np.zeros((h * s, w * s, 3), dtype=np.uint8)
        tiles = list(tile_grid(w, h, self.tile))
        for i, (x, y, tw, th) in enumerate(tiles):
            if stop.is_set():
                return None
            x0, y0 = max(0, x - PAD), max(0, y - PAD)
            x1, y1 = min(w, x + tw + PAD), min(h, y + th + PAD)
            try:
                res = self._run(src[y0:y1, x0:x1])
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError("Не хватило видеопамяти — уменьши размер тайла (справа внизу).")
            ox, oy = (x - x0) * s, (y - y0) * s
            out[y * s:(y + th) * s, x * s:(x + tw) * s] = (res[oy:oy + th * s, ox:ox + tw * s] * 255 + 0.5).astype(np.uint8)
            progress(i + 1, len(tiles))
        result = Image.fromarray(out)
        if alpha is not None:
            result.putalpha(alpha.resize(result.size, Image.LANCZOS))
        return result


# ---------------- интерфейс ----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Нейро-апскейлер — by wutxi")
        self.geometry("900x640")
        self.minsize(820, 600)
        self.configure(bg=BG)
        self.q = queue.Queue()
        self.files = []
        self.calib = None
        self.busy = False
        self.stop = threading.Event()
        self.hw = None
        self.engine = None
        self._style()
        self._build()
        self.after(50, self._poll)
        self._bg(self._init_hw)

    # --- оформление
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=PANEL, font=("Segoe UI", 10))
        s.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG, arrowcolor=FG)
        s.map("TCombobox", fieldbackground=[("readonly", PANEL), ("disabled", BG)],
              foreground=[("readonly", FG), ("disabled", MUTED)],
              selectbackground=[("readonly", PANEL)], selectforeground=[("readonly", FG)])
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        s.configure("Horizontal.TProgressbar", troughcolor=PANEL, background=ACCENT, bordercolor=PANEL, thickness=10)

    def _btn(self, parent, text, cmd, primary=False):
        b = tk.Button(parent, text=text, command=cmd, relief="flat", cursor="hand2", padx=14, pady=6,
                      bg=ACCENT if primary else PANEL, fg="white" if primary else FG,
                      activebackground="#9a80ff" if primary else "#2a2e36", activeforeground="white",
                      font=("Segoe UI", 10, "bold" if primary else "normal"), bd=0)
        return b

    def _build(self):
        # верхняя панель + вотермарка
        top = tk.Frame(self, bg=PANEL, height=44)
        top.pack(fill="x")
        tk.Label(top, text="⬆  Нейро-апскейлер", bg=PANEL, fg=FG, font=("Segoe UI", 13, "bold")).pack(side="left", padx=14, pady=8)
        tk.Label(top, text="by wutxi", bg=PANEL, fg=ACCENT, font=("Segoe UI", 11, "bold italic")).pack(side="right", padx=16)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=12)
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(body, bg=PANEL, width=300)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        # файлы
        row = tk.Frame(left, bg=BG)
        row.pack(fill="x")
        self._btn(row, "Выбрать изображения…", self.pick_files).pack(side="left")
        self._btn(row, "Папка сохранения…", self.pick_out).pack(side="left", padx=8)
        self.out_dir = None
        self.out_lbl = tk.Label(left, text="Сохранение: рядом с оригиналом", bg=BG, fg=MUTED, anchor="w")
        self.out_lbl.pack(fill="x", pady=(6, 0))

        self.preview = tk.Label(left, bg=PANEL, fg=MUTED, text="Нет изображения\n\nВыбери файл(ы) для апскейла",
                                font=("Segoe UI", 11))
        self.preview.pack(fill="both", expand=True, pady=10)
        self.info = tk.Label(left, text="", bg=BG, fg=FG, anchor="w", justify="left")
        self.info.pack(fill="x")

        # модель
        mrow = tk.Frame(left, bg=BG)
        mrow.pack(fill="x", pady=(8, 0))
        tk.Label(mrow, text="Модель:", bg=BG, fg=MUTED).pack(side="left")
        self.model_var = tk.StringVar()
        self.model_cb = ttk.Combobox(mrow, textvariable=self.model_var, state="readonly", width=38)
        self.model_cb.pack(side="left", padx=6)
        self.model_cb.bind("<<ComboboxSelected>>", lambda e: self.select_model())
        self._btn(mrow, "Скачать модель…", self.download_dialog).pack(side="left")
        tk.Label(mrow, text="Итог:", bg=BG, fg=MUTED).pack(side="left", padx=(12, 4))
        self.out_scale = tk.StringVar(value="как у модели")
        ttk.Combobox(mrow, textvariable=self.out_scale, state="readonly", width=12,
                     values=["как у модели", "x2", "x3", "x4"]).pack(side="left")

        # прогресс
        self.pb = ttk.Progressbar(left, mode="determinate")
        self.pb.pack(fill="x", pady=(12, 4))
        self.status = tk.Label(left, text="Определяю железо…", bg=BG, fg=MUTED, anchor="w")
        self.status.pack(fill="x")
        brow = tk.Frame(left, bg=BG)
        brow.pack(fill="x", pady=(8, 0))
        self.run_btn = self._btn(brow, "Апскейл", self.start, primary=True)
        self.run_btn.pack(side="left")
        self.stop_btn = self._btn(brow, "Стоп", self.stop.set)
        self.stop_btn.pack(side="left", padx=8)

        # правая панель: железо и оценка
        tk.Label(right, text="ТВОЁ ЖЕЛЕЗО", bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=14, pady=(14, 4))
        self.hw_lbl = tk.Label(right, text="…", bg=PANEL, fg=FG, justify="left", anchor="w", wraplength=270)
        self.hw_lbl.pack(fill="x", padx=14)
        tk.Frame(right, bg="#2a2e36", height=1).pack(fill="x", padx=14, pady=14)
        tk.Label(right, text="ОЦЕНКА ВРЕМЕНИ", bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=14)
        self.est_big = tk.Label(right, text="—", bg=PANEL, fg=ACCENT, font=("Segoe UI", 24, "bold"), anchor="w")
        self.est_big.pack(fill="x", padx=14, pady=(4, 0))
        self.est_lbl = tk.Label(right, text="", bg=PANEL, fg=MUTED, justify="left", anchor="w", wraplength=270)
        self.est_lbl.pack(fill="x", padx=14, pady=(4, 0))

        tile_row = tk.Frame(right, bg=PANEL)
        tile_row.pack(side="bottom", fill="x", padx=14, pady=14)
        tk.Label(tile_row, text="Тайл, px:", bg=PANEL, fg=MUTED).pack(side="left")
        self.tile_var = tk.StringVar()
        cb = ttk.Combobox(tile_row, textvariable=self.tile_var, state="readonly", width=8,
                          values=["128", "192", "256", "384", "512", "768"])
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self.change_tile())

    # --- фоновые задачи
    def _bg(self, fn, *a):
        threading.Thread(target=fn, args=a, daemon=True).start()

    def _ui(self, fn, *a):
        self.q.put((fn, a))

    def _poll(self):
        try:
            while True:
                fn, a = self.q.get_nowait()
                fn(*a)
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def _init_hw(self):
        hw = detect_hardware()
        self.hw = hw
        self.engine = Engine(hw)
        self._ui(self._show_hw)

    def _show_hw(self):
        hw = self.hw
        vram = f"\nВидеопамять: {hw['vram']:.0f} ГБ" if hw["vram"] else ""
        mode = "GPU (CUDA" + (", FP16)" if hw["device"] == "cuda" else ")") if hw["device"] == "cuda" else "CPU — будет медленно"
        self.hw_lbl.config(text=f"Видеокарта: {hw['gpu']}{vram}\nПроцессор: {hw['cpu']}\n{hw['cores']}\n"
                                f"ОЗУ: {hw['ram']}\nРежим: {mode}")
        self.tile_var.set(str(self.engine.tile))
        self.refresh_models()

    def refresh_models(self, select=None):
        self.models = list_models()
        names = list(self.models)
        self.model_cb.config(values=names)
        if not names:
            self.status.config(text="Нет моделей. Нажми «Скачать модель…» или положи .pth в папку models.")
            return
        pick = select if select in names else (self.model_var.get() if self.model_var.get() in names else names[0])
        self.model_var.set(pick)
        self.select_model()

    def select_model(self):
        name = self.model_var.get()
        if not name or self.busy:
            return
        self.calib = None
        self._set_busy(True)
        self.status.config(text=f"Загружаю {name} и меряю скорость на твоём железе…")
        self._bg(self._load_model, self.models[name])

    def _load_model(self, path):
        try:
            self.engine.load(path)
            calib = self.engine.calibrate()
            self._ui(self._model_ready, calib)
        except Exception as e:
            self._ui(self._error, f"Не удалось загрузить модель:\n{e}")

    def _model_ready(self, calib):
        self.calib = calib
        self._set_busy(False)
        self.status.config(text=f"Готово. Модель x{self.engine.scale}, калибровка: "
                                f"{calib['sec_per_tile']:.2f} сек на тайл {self.engine.tile}px")
        self.update_estimate()

    def change_tile(self):
        if self.engine and not self.busy:
            self.engine.tile = int(self.tile_var.get())
            if self.engine.model is not None:
                self.calib = None
                self._set_busy(True)
                self.status.config(text="Перекалибровка под новый размер тайла…")
                self._bg(lambda: self._ui(self._model_ready, self.engine.calibrate()))

    # --- файлы
    def pick_files(self):
        fs = filedialog.askopenfilenames(title="Изображения",
                                         filetypes=[("Изображения", " ".join("*" + e for e in IMG_EXT)), ("Все", "*.*")])
        if not fs:
            return
        self.files = []
        for f in fs:
            try:
                with Image.open(f) as im:
                    self.files.append((Path(f), im.size))
            except Exception:
                pass
        if not self.files:
            return
        p, (w, h) = self.files[0]
        im = Image.open(p)
        im.thumbnail((520, 340))
        self._thumb = ImageTk.PhotoImage(im)
        self.preview.config(image=self._thumb, text="")
        more = f"  (+ ещё {len(self.files) - 1})" if len(self.files) > 1 else ""
        self.info.config(text=f"{p.name} — {w}×{h}{more}")
        self.update_estimate()

    def pick_out(self):
        d = filedialog.askdirectory(title="Куда сохранять")
        if d:
            self.out_dir = Path(d)
            self.out_lbl.config(text=f"Сохранение: {d}")

    def update_estimate(self):
        if not self.files or not self.calib:
            self.est_big.config(text="—")
            self.est_lbl.config(text="Выбери изображение и модель" if not self.files else "Идёт калибровка…")
            return
        e = self.engine
        total = sum(e.estimate(w, h, self.calib) for _, (w, h) in self.files)
        px = sum(w * h for _, (w, h) in self.files)
        w, h = self.files[0][1]
        corr = self.calib.get("correction", 1.0)
        self.est_big.config(text="≈ " + fmt_time(total))
        self.est_lbl.config(text=f"Файлов: {len(self.files)}, {px / 1e6:.1f} Мп\n"
                                 f"Результат (1-й): {w * e.scale}×{h * e.scale}\n"
                                 f"Скорость замерена на твоей {'видеокарте' if e.device.type == 'cuda' else 'CPU'}"
                                 + (f"\nПоправка по прошлым запускам: ×{corr:.2f}" if abs(corr - 1) > 0.01 else ""))

    # --- запуск
    def _set_busy(self, b):
        self.busy = b
        self.run_btn.config(state="disabled" if b else "normal")
        self.model_cb.config(state="disabled" if b else "readonly")

    def start(self):
        if self.busy:
            return
        if not self.files:
            return messagebox.showinfo("Апскейлер", "Сначала выбери изображения.")
        if self.engine is None or self.engine.model is None:
            return messagebox.showinfo("Апскейлер", "Сначала выбери или скачай модель.")
        self.stop.clear()
        self._set_busy(True)
        self._bg(self._work)

    def _work(self):
        e = self.engine
        est_total = sum(e.estimate(w, h, self.calib) for _, (w, h) in self.files)
        t0 = time.perf_counter()
        done_est = 0.0
        saved = []
        try:
            for idx, (path, (w, h)) in enumerate(self.files):
                est_this = e.estimate(w, h, self.calib)
                t_file = time.perf_counter()

                def prog(i, n, idx=idx, est_this=est_this, done_est=done_est, t_file=t_file):
                    frac = (done_est + est_this * i / n) / est_total
                    elapsed = time.perf_counter() - t0
                    left = elapsed / frac - elapsed if frac > 0.02 else est_total - elapsed
                    self._ui(self._progress, frac * 100,
                             f"Файл {idx + 1}/{len(self.files)}: тайл {i}/{n} · прошло {fmt_time(elapsed)} · осталось ≈ {fmt_time(max(0, left))}")

                img = Image.open(path)
                img.load()
                res = e.upscale(img, prog, self.stop)
                if res is None:
                    self._ui(self._finish, "Остановлено.", saved)
                    return
                target = self.out_scale.get()
                if target != "как у модели" and int(target[1:]) != e.scale:
                    k = int(target[1:])
                    res = res.resize((w * k, h * k), Image.LANCZOS)
                    tag = target
                else:
                    tag = f"x{e.scale}"
                out_dir = self.out_dir or path.parent
                out = out_dir / f"{path.stem}_{tag}.png"
                res.save(out)
                saved.append(out)
                e.learn(est_this, time.perf_counter() - t_file)
                done_est += est_this
            el = time.perf_counter() - t0
            self._ui(self._finish, f"Готово за {fmt_time(el)} (оценка была ≈ {fmt_time(est_total)})", saved)
        except Exception as ex:
            self._ui(self._error, str(ex))

    def _progress(self, pct, text):
        self.pb["value"] = pct
        self.status.config(text=text)

    def _finish(self, text, saved):
        self._set_busy(False)
        self.pb["value"] = 100 if saved and "Готово" in text else self.pb["value"]
        self.status.config(text=text)
        self.calib = load_cache().get(self.engine.cache_key(), self.calib)
        self.update_estimate()
        if saved and "Готово" in text:
            if messagebox.askyesno("Готово", f"{text}\n\nСохранено файлов: {len(saved)}\n{saved[0]}\n\nОткрыть папку?"):
                os.startfile(saved[0].parent)

    def _error(self, msg):
        self._set_busy(False)
        self.status.config(text="Ошибка")
        messagebox.showerror("Ошибка", msg)

    # --- загрузка моделей
    def download_dialog(self):
        win = tk.Toplevel(self, bg=BG)
        win.title("Скачать модель")
        win.geometry("460x220")
        win.transient(self)
        tk.Label(win, text="Модели качаются с GitHub (Real-ESRGAN, xinntao)\nв папку models рядом с программой.",
                 bg=BG, fg=MUTED, justify="left").pack(anchor="w", padx=14, pady=(12, 8))
        var = tk.StringVar(value=list(DOWNLOADS)[0])
        for k in DOWNLOADS:
            tk.Radiobutton(win, text=k, variable=var, value=k, bg=BG, fg=FG, selectcolor=PANEL,
                           activebackground=BG, activeforeground=FG).pack(anchor="w", padx=14)

        def go():
            win.destroy()
            self._set_busy(True)
            self._bg(self._download, var.get())
        self._btn(win, "Скачать", go, primary=True).pack(pady=12)

    def _download(self, key):
        url = DOWNLOADS[key]
        name = url.rsplit("/", 1)[1]
        dst = MODELS_DIR / name
        tmp = dst.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=30) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length", 0))
                got = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        self._ui(self._progress, got / total * 100, f"Скачиваю {name}: {got >> 20} / {total >> 20} МБ")
            tmp.replace(dst)
            self._ui(self._downloaded, name)
        except Exception as ex:
            self._ui(self._error, f"Не удалось скачать модель:\n{ex}")

    def _downloaded(self, name):
        self._set_busy(False)
        self.pb["value"] = 0
        self.refresh_models(select=name)


if __name__ == "__main__":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()
