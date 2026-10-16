# Neuro Upscaler — by wutxi

Увеличивает изображения нейросетью **локально, без интернета**. Перед запуском показывает, **сколько займёт задача на твоём железе**.

- Real-ESRGAN: x4 для фото, x4 для аниме/арта, x2/x3/x4 animevideo
- считает на любой видеокарте через Vulkan (NVIDIA / AMD / Intel), если видеокарты нет — на процессоре
- оценка времени: при выборе модели программа делает замер скорости на твоей видеокарте/CPU, а после каждого запуска уточняет поправку
- можно обработать сразу несколько файлов, перетащить их в окно, выбрать итоговый масштаб x2/x3/x4; прозрачность сохраняется

## Скачать

Готовая сборка лежит в [Releases](../../releases): распакуй zip и запусти `Upscaler.exe`. Устанавливать ничего не нужно, папка `models` должна лежать рядом с exe.

## Две версии

| Папка | Что это |
|---|---|
| [`cpp/`](cpp) | Основная версия: C++ / WinAPI + [ncnn](https://github.com/Tencent/ncnn) (Vulkan). Собирается в один exe |
| [`python/`](python) | Python / tkinter + PyTorch + [spandrel](https://github.com/chaiNNer-org/spandrel). Загружает любые `.pth`/`.safetensors`-модели апскейла |

### Сборка C++ (MinGW-w64, CMake, Ninja)

```powershell
cd cpp
# 1. ncnn: распаковать ncnn-<версия>-full-source.zip в third_party/ncnn
cmake -S third_party/ncnn -B third_party/ncnn/build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DNCNN_VULKAN=ON -DNCNN_SHARED_LIB=OFF -DNCNN_OPENMP=ON -DNCNN_SIMPLEOCV=OFF `
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF `
  -DCMAKE_INSTALL_PREFIX=third_party/ncnn-install
cmake --build third_party/ncnn/build ; cmake --install third_party/ncnn/build
# 2. модели: файлы .param/.bin из realesrgan-ncnn-vulkan (xinntao/Real-ESRGAN, Releases) положить в cpp/models
# 3. программа
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ; cmake --build build
# результат: cpp/bin/Upscaler.exe
```

### Запуск Python-версии

Нужны `torch`, `spandrel`, `pillow`, `numpy`, `psutil`. Запуск: `python upscaler.py`. Модели можно скачать прямо из окна программы (кнопка «Скачать модель…») или положить в `python/models`.

Сторонние компоненты и их лицензии перечислены в [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
