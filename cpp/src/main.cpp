// Нейро-апскейлер на C++: Real-ESRGAN через ncnn + Vulkan (любая видеокарта) или CPU.
// Оценка времени — калибровочный прогон модели на твоём железе + поправка по прошлым запускам.
// by wutxi
#define UNICODE
#define _UNICODE
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>
#include <shobjidl.h>
#include <shellapi.h>
#include <wincodec.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "net.h"
#include "gpu.h"

using namespace std;

// ---------------- оформление ----------------
static const COLORREF C_BG = RGB(0x14, 0x16, 0x1a), C_PANEL = RGB(0x1d, 0x20, 0x26), C_FG = RGB(0xe8, 0xe8, 0xea),
                      C_MUTED = RGB(0x8a, 0x8f, 0x98), C_ACCENT = RGB(0x7c, 0x5c, 0xff), C_LINE = RGB(0x2a, 0x2e, 0x36);
static const int PAD = 16;

// ---------------- утилиты ----------------
static wstring exeDir() {
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    wstring p = buf;
    return p.substr(0, p.find_last_of(L"\\/"));
}
static string toUtf8(const wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    string s(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}
static wstring fromUtf8(const string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
    return w;
}
static wstring fmtTime(double s) {
    if (s < 1) return L"1 сек";
    long t = lround(s);
    wchar_t b[64];
    if (t >= 3600) swprintf(b, 64, L"%ld ч %ld мин", t / 3600, t % 3600 / 60);
    else if (t >= 60) swprintf(b, 64, L"%ld мин %ld сек", t / 60, t % 60);
    else swprintf(b, 64, L"%ld сек", t);
    return b;
}
static bool readFile(const wstring& path, vector<unsigned char>& out) {
    FILE* f = _wfopen(path.c_str(), L"rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize(n);
    size_t r = fread(out.data(), 1, n, f);
    fclose(f);
    return r == (size_t)n;
}
static double now() {
    return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------- железо ----------------
struct Device { wstring name; int gpu; size_t vramMB; };  // gpu = -1 → CPU
struct Hardware {
    wstring cpu, cores, ram;
    vector<Device> devices;
};

static Hardware detectHardware() {
    Hardware hw;
    wchar_t name[256] = L"";
    DWORD sz = sizeof(name);
    RegGetValueW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", L"ProcessorNameString",
                 RRF_RT_REG_SZ, nullptr, name, &sz);
    hw.cpu = name;
    while (!hw.cpu.empty() && hw.cpu.back() == L' ') hw.cpu.pop_back();

    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> info(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    int cores = 0;
    if (GetLogicalProcessorInformation(info.data(), &len))
        for (auto& i : info) cores += i.Relationship == RelationProcessorCore;
    hw.cores = to_wstring(cores) + L" ядер / " + to_wstring(thread::hardware_concurrency()) + L" потоков";

    MEMORYSTATUSEX ms{sizeof(ms)};
    GlobalMemoryStatusEx(&ms);
    hw.ram = to_wstring((int)lround(ms.ullTotalPhys / 1073741824.0)) + L" ГБ";

    ncnn::create_gpu_instance();
    int n = ncnn::get_gpu_count();
    vector<Device> integrated;
    for (int i = 0; i < n; i++) {
        const ncnn::GpuInfo& gi = ncnn::get_gpu_info(i);
        if (gi.type() == 3) continue;  // программный Vulkan (CPU) — не нужен
        Device d{fromUtf8(gi.device_name()), i, ncnn::get_gpu_device(i)->get_heap_budget()};
        (gi.type() == 0 ? hw.devices : integrated).push_back(d);
    }
    hw.devices.insert(hw.devices.end(), integrated.begin(), integrated.end());
    hw.devices.push_back({L"CPU — " + hw.cpu, -1, 0});
    return hw;
}

static int autoTile(const Device& d) {
    if (d.gpu < 0) return 128;
    if (d.vramMB > 8000) return 512;
    if (d.vramMB > 4000) return 384;
    if (d.vramMB > 1900) return 256;
    if (d.vramMB > 550) return 128;
    return 64;
}

// ---------------- калибровка (кэш) ----------------
struct Calib { double secPerTile = 0, corr = 1.0; };
static wstring calibPath() { return exeDir() + L"\\calibration.txt"; }
static map<string, Calib> loadCalib() {
    map<string, Calib> m;
    ifstream f(calibPath().c_str());
    string line;
    while (getline(f, line)) {
        auto a = line.find('\t'), b = line.rfind('\t');
        if (a == string::npos || a == b) continue;
        m[line.substr(0, a)] = {atof(line.substr(a + 1, b - a - 1).c_str()), atof(line.substr(b + 1).c_str())};
    }
    return m;
}
static void saveCalib(const map<string, Calib>& m) {
    ofstream f(calibPath().c_str());
    for (auto& [k, v] : m) f << k << '\t' << v.secPerTile << '\t' << v.corr << '\n';
}

// ---------------- движок ----------------
struct Engine {
    unique_ptr<ncnn::Net> net;
    vector<unsigned char> param, bin;
    int gpu = -1, scale = 4, tile = 256;
    wstring model;

    bool load(const wstring& base, const wstring& modelName, int gpuId, wstring& err) {
        net.reset(new ncnn::Net);
        gpu = gpuId;
        model = modelName;
        net->opt.use_vulkan_compute = gpu >= 0;
        net->opt.num_threads = (int)thread::hardware_concurrency();
        if (gpu >= 0) net->set_vulkan_device(gpu);
        if (!readFile(base + L".param", param) || !readFile(base + L".bin", bin)) {
            err = L"Не найдены файлы модели " + base + L".param/.bin";
            return false;
        }
        param.push_back(0);
        if (net->load_param_mem((const char*)param.data()) != 0 || net->load_model(bin.data()) <= 0) {
            err = L"Не удалось загрузить модель " + modelName;
            return false;
        }
        scale = 4;
        for (int s = 2; s <= 4; s++)
            if (modelName.find(L"x" + to_wstring(s)) != wstring::npos) scale = s;
        return true;
    }

    // rgb: w×h, stride → out: (w*scale)×(h*scale), ostride
    void run(const unsigned char* rgb, int w, int h, int stride, unsigned char* out, int ostride) {
        ncnn::Mat in = ncnn::Mat::from_pixels(rgb, ncnn::Mat::PIXEL_RGB, w, h, stride);
        const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f}, den[3] = {255.f, 255.f, 255.f};
        in.substract_mean_normalize(nullptr, norm);
        ncnn::Extractor ex = net->create_extractor();
        ex.input("data", in);
        ncnn::Mat o;
        ex.extract("output", o);
        o.substract_mean_normalize(nullptr, den);
        o.to_pixels(out, ncnn::Mat::PIXEL_RGB, ostride);
    }

    string key(const wstring& devName) const {
        return toUtf8(devName + L"|" + model + L"|tile" + to_wstring(tile));
    }

    Calib calibrate(const wstring& devName) {
        auto m = loadCalib();
        string k = key(devName);
        if (m.count(k)) return m[k];
        int n = (gpu >= 0 ? tile : 64) + 2 * PAD;  // на CPU меряем маленький тайл и масштабируем по площади
        vector<unsigned char> src(n * n * 3), dst((size_t)n * scale * n * scale * 3);
        mt19937 rng(1);
        for (auto& v : src) v = rng() & 255;
        run(src.data(), n, n, n * 3, dst.data(), n * scale * 3);  // прогрев: компиляция шейдеров
        int runs = gpu >= 0 ? 2 : 1;
        double t0 = now();
        for (int i = 0; i < runs; i++) run(src.data(), n, n, n * 3, dst.data(), n * scale * 3);
        double sec = (now() - t0) / runs;
        int full = tile + 2 * PAD;
        if (gpu < 0) sec *= double(full) * full / (double(n) * n);
        Calib c{sec, 1.0};
        m[k] = c;
        saveCalib(m);
        return c;
    }

    double estimate(int w, int h, const Calib& c) const {
        double full = double(tile + 2 * PAD) * (tile + 2 * PAD), work = 0;
        for (int y = 0; y < h; y += tile)
            for (int x = 0; x < w; x += tile) {
                int tw = min(tile, w - x) + 2 * PAD, th = min(tile, h - y) + 2 * PAD;
                work += tw * th / full;
            }
        double io = double(w) * h * scale * scale * 5e-8;  // декод/кодирование PNG
        return (c.secPerTile * work + io) * c.corr;
    }

    void learn(const wstring& devName, double est, double actual) {
        auto m = loadCalib();
        string k = key(devName);
        if (!m.count(k) || est < 0.5) return;
        double c = m[k].corr;
        m[k].corr = 0.5 * c + 0.5 * c * actual / est;
        saveCalib(m);
    }
};

// ---------------- изображения (WIC) ----------------
static IWICImagingFactory* wic = nullptr;

struct Image {
    int w = 0, h = 0;
    vector<unsigned char> rgb, alpha;  // alpha пуст, если прозрачности нет
};

static bool loadImage(const wstring& path, Image& img) {
    IWICBitmapDecoder* dec = nullptr;
    IWICBitmapFrameDecode* fr = nullptr;
    IWICFormatConverter* cv = nullptr;
    bool ok = false;
    if (SUCCEEDED(wic->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec)) &&
        SUCCEEDED(dec->GetFrame(0, &fr)) && SUCCEEDED(wic->CreateFormatConverter(&cv)) &&
        SUCCEEDED(cv->Initialize(fr, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom))) {
        UINT w, h;
        cv->GetSize(&w, &h);
        vector<unsigned char> rgba((size_t)w * h * 4);
        if (SUCCEEDED(cv->CopyPixels(nullptr, w * 4, (UINT)rgba.size(), rgba.data()))) {
            img.w = w;
            img.h = h;
            img.rgb.resize((size_t)w * h * 3);
            bool hasA = false;
            for (size_t i = 0, n = (size_t)w * h; i < n; i++) {
                memcpy(&img.rgb[i * 3], &rgba[i * 4], 3);
                hasA |= rgba[i * 4 + 3] != 255;
            }
            img.alpha.clear();
            if (hasA) {
                img.alpha.resize((size_t)w * h);
                for (size_t i = 0, n = (size_t)w * h; i < n; i++) img.alpha[i] = rgba[i * 4 + 3];
            }
            ok = true;
        }
    }
    if (cv) cv->Release();
    if (fr) fr->Release();
    if (dec) dec->Release();
    return ok;
}

static bool imageSize(const wstring& path, int& w, int& h) {
    IWICBitmapDecoder* dec = nullptr;
    IWICBitmapFrameDecode* fr = nullptr;
    bool ok = false;
    if (SUCCEEDED(wic->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec)) &&
        SUCCEEDED(dec->GetFrame(0, &fr))) {
        UINT uw, uh;
        ok = SUCCEEDED(fr->GetSize(&uw, &uh));
        w = uw, h = uh;
    }
    if (fr) fr->Release();
    if (dec) dec->Release();
    return ok;
}

// Масштабирование буфера через WIC (Fant ≈ качественный даунскейл)
static vector<unsigned char> wicResize(const unsigned char* data, int w, int h, int bpp, REFWICPixelFormatGUID fmt, int nw, int nh) {
    vector<unsigned char> out((size_t)nw * nh * bpp);
    IWICBitmap* bmp = nullptr;
    IWICBitmapScaler* sc = nullptr;
    if (SUCCEEDED(wic->CreateBitmapFromMemory(w, h, fmt, w * bpp, (UINT)((size_t)w * h * bpp), (BYTE*)data, &bmp)) &&
        SUCCEEDED(wic->CreateBitmapScaler(&sc)) && SUCCEEDED(sc->Initialize(bmp, nw, nh, (WICBitmapInterpolationMode)4 /* HighQualityCubic, Win10+ */)))
        sc->CopyPixels(nullptr, nw * bpp, (UINT)out.size(), out.data());
    if (sc) sc->Release();
    if (bmp) bmp->Release();
    return out;
}

static bool savePng(const wstring& path, const unsigned char* data, int w, int h, bool rgba) {
    int bpp = rgba ? 4 : 3;
    IWICStream* st = nullptr;
    IWICBitmapEncoder* enc = nullptr;
    IWICBitmapFrameEncode* fr = nullptr;
    IWICBitmap* bmp = nullptr;
    bool ok = false;
    WICPixelFormatGUID f = rgba ? GUID_WICPixelFormat32bppBGRA : GUID_WICPixelFormat24bppBGR;
    if (SUCCEEDED(wic->CreateBitmapFromMemory(w, h, rgba ? GUID_WICPixelFormat32bppRGBA : GUID_WICPixelFormat24bppRGB, w * bpp,
                                              (UINT)((size_t)w * h * bpp), (BYTE*)data, &bmp)) &&
        SUCCEEDED(wic->CreateStream(&st)) && SUCCEEDED(st->InitializeFromFilename(path.c_str(), GENERIC_WRITE)) &&
        SUCCEEDED(wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc)) && SUCCEEDED(enc->Initialize(st, WICBitmapEncoderNoCache)) &&
        SUCCEEDED(enc->CreateNewFrame(&fr, nullptr)) && SUCCEEDED(fr->Initialize(nullptr)) && SUCCEEDED(fr->SetSize(w, h)) &&
        SUCCEEDED(fr->SetPixelFormat(&f)) && SUCCEEDED(fr->WriteSource(bmp, nullptr)) && SUCCEEDED(fr->Commit()) && SUCCEEDED(enc->Commit()))
        ok = true;
    if (fr) fr->Release();
    if (enc) enc->Release();
    if (st) st->Release();
    if (bmp) bmp->Release();
    return ok;
}

static HBITMAP makeThumb(const wstring& path, int maxW, int maxH, int& tw, int& th) {
    Image img;
    if (!loadImage(path, img)) return nullptr;
    double k = min({double(maxW) / img.w, double(maxH) / img.h, 1.0});
    tw = max(1, (int)(img.w * k));
    th = max(1, (int)(img.h * k));
    auto small = wicResize(img.rgb.data(), img.w, img.h, 3, GUID_WICPixelFormat24bppRGB, tw, th);
    BITMAPINFO bi{};
    bi.bmiHeader = {sizeof(BITMAPINFOHEADER), tw, -th, 1, 32, BI_RGB};
    void* bits;
    HBITMAP hb = CreateDIBSection(nullptr, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    auto* p = (unsigned char*)bits;
    for (int i = 0; i < tw * th; i++) {
        p[i * 4] = small[i * 3 + 2];
        p[i * 4 + 1] = small[i * 3 + 1];
        p[i * 4 + 2] = small[i * 3];
        p[i * 4 + 3] = 255;
    }
    return hb;
}

// ---------------- приложение ----------------
enum { ID_FILES = 101, ID_OUTDIR, ID_RUN, ID_STOP, ID_MODEL, ID_SCALE, ID_DEVICE, ID_TILE };
enum { WM_HW = WM_APP + 1, WM_READY, WM_PROG, WM_DONE, WM_FAIL };

struct FileItem { wstring path; int w, h; };

static struct State {
    HWND wnd = nullptr;
    HFONT fNorm, fBold, fTitle, fMark, fBig, fSmall;
    HBRUSH brPanel;
    int dpi = 96;
    Hardware hw;
    bool hwReady = false;
    Engine eng;
    Calib calib;
    bool calibReady = false;
    vector<wstring> models;  // базовые имена без расширения
    vector<FileItem> files;
    wstring outDir;
    HBITMAP thumb = nullptr;
    int thumbW = 0, thumbH = 0;
    atomic<bool> busy{false}, stop{false};
    mutex mx;
    wstring status = L"Определяю железо…";
    double progress = 0;
    wstring lastSaved;
    HWND hFiles, hOut, hRun, hStop, hModel, hScale, hDevice, hTile;
} G;

static int S(int v) { return MulDiv(v, G.dpi, 96); }
static void setStatus(const wstring& s) {
    lock_guard<mutex> l(G.mx);
    G.status = s;
}
static int curDevice() { return max(0, (int)SendMessageW(G.hDevice, CB_GETCURSEL, 0, 0)); }
static const Device& device() { return G.hw.devices[curDevice()]; }

static void enableUi(bool on) {
    for (HWND h : {G.hFiles, G.hOut, G.hRun, G.hModel, G.hScale, G.hDevice, G.hTile}) EnableWindow(h, on);
    EnableWindow(G.hStop, !on);
}

static void scanModels() {
    G.models.clear();
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW((exeDir() + L"\\models\\*.param").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        wstring n = fd.cFileName;
        n = n.substr(0, n.size() - 6);
        if (GetFileAttributesW((exeDir() + L"\\models\\" + n + L".bin").c_str()) != INVALID_FILE_ATTRIBUTES) G.models.push_back(n);
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    // фото-модель первой
    for (size_t i = 0; i < G.models.size(); i++)
        if (G.models[i] == L"realesrgan-x4plus") swap(G.models[0], G.models[i]);
}

// загрузка модели и калибровка в фоне
static void prepareModel() {
    int mi = (int)SendMessageW(G.hModel, CB_GETCURSEL, 0, 0);
    if (mi < 0 || mi >= (int)G.models.size()) return;
    G.busy = true;
    G.calibReady = false;
    enableUi(false);
    EnableWindow(G.hStop, FALSE);
    Device dev = device();
    wstring model = G.models[mi];
    int tile = 64 << SendMessageW(G.hTile, CB_GETCURSEL, 0, 0);
    setStatus(L"Загружаю " + model + L" и замеряю скорость на «" + dev.name + L"»…");
    InvalidateRect(G.wnd, nullptr, FALSE);
    thread([dev, model, tile] {
        wstring err;
        G.eng.tile = tile;
        if (!G.eng.load(exeDir() + L"\\models\\" + model, model, dev.gpu, err)) {
            setStatus(err);
            PostMessageW(G.wnd, WM_FAIL, 0, 0);
            return;
        }
        G.calib = G.eng.calibrate(dev.name);
        PostMessageW(G.wnd, WM_READY, 0, 0);
    }).detach();
}

static double totalEstimate() {
    double t = 0;
    for (auto& f : G.files) t += G.eng.estimate(f.w, f.h, G.calib);
    return t;
}

static int outScale() {
    int i = (int)SendMessageW(G.hScale, CB_GETCURSEL, 0, 0);
    return i <= 0 ? G.eng.scale : i + 1;  // 0: как у модели, 1: x2, 2: x3, 3: x4
}

static void worker() {
    const wstring devName = device().name;
    const int target = outScale();
    double estTotal = totalEstimate(), t0 = now(), doneEst = 0;
    int saved = 0;
    for (size_t fi = 0; fi < G.files.size() && !G.stop; fi++) {
        const FileItem& f = G.files[fi];
        double estThis = G.eng.estimate(f.w, f.h, G.calib), tf = now();
        Image img;
        if (!loadImage(f.path, img)) {
            setStatus(L"Не удалось открыть " + f.path);
            continue;
        }
        const int s = G.eng.scale, W = img.w, H = img.h, T = G.eng.tile;
        vector<unsigned char> out((size_t)W * s * H * s * 3), tmp;
        int nTiles = ((W + T - 1) / T) * ((H + T - 1) / T), ti = 0;
        for (int y = 0; y < H && !G.stop; y += T)
            for (int x = 0; x < W && !G.stop; x += T) {
                int tw = min(T, W - x), th = min(T, H - y);
                int x0 = max(0, x - PAD), y0 = max(0, y - PAD), x1 = min(W, x + tw + PAD), y1 = min(H, y + th + PAD);
                int iw = x1 - x0, ih = y1 - y0;
                tmp.resize((size_t)iw * s * ih * s * 3);
                G.eng.run(&img.rgb[((size_t)y0 * W + x0) * 3], iw, ih, W * 3, tmp.data(), iw * s * 3);
                int ox = (x - x0) * s, oy = (y - y0) * s;
                for (int r = 0; r < th * s; r++)
                    memcpy(&out[(((size_t)y * s + r) * W * s + (size_t)x * s) * 3], &tmp[(((size_t)oy + r) * iw * s + ox) * 3], (size_t)tw * s * 3);
                ti++;
                double frac = (doneEst + estThis * ti / nTiles) / max(estTotal, 1e-6), el = now() - t0;
                double left = frac > 0.02 ? el / frac - el : estTotal - el;
                wchar_t b[256];
                swprintf(b, 256, L"Файл %zu/%zu: тайл %d/%d · прошло %ls · осталось ≈ %ls", fi + 1, G.files.size(), ti, nTiles,
                         fmtTime(el).c_str(), fmtTime(max(0.0, left)).c_str());
                {
                    lock_guard<mutex> l(G.mx);
                    G.status = b;
                    G.progress = min(1.0, frac);
                }
                PostMessageW(G.wnd, WM_PROG, 0, 0);
            }
        if (G.stop) break;

        int OW = W * s, OH = H * s;
        if (target != s) {
            int nw = W * target, nh = H * target;
            out = wicResize(out.data(), OW, OH, 3, GUID_WICPixelFormat24bppRGB, nw, nh);
            OW = nw, OH = nh;
        }
        vector<unsigned char> rgba;
        if (!img.alpha.empty()) {
            auto a = wicResize(img.alpha.data(), W, H, 1, GUID_WICPixelFormat8bppGray, OW, OH);
            rgba.resize((size_t)OW * OH * 4);
            for (size_t i = 0, n = (size_t)OW * OH; i < n; i++) {
                memcpy(&rgba[i * 4], &out[i * 3], 3);
                rgba[i * 4 + 3] = a[i];
            }
        }
        wstring dir = G.outDir.empty() ? f.path.substr(0, f.path.find_last_of(L"\\/")) : G.outDir;
        wstring name = f.path.substr(f.path.find_last_of(L"\\/") + 1);
        name = name.substr(0, name.find_last_of(L'.'));
        wstring dst = dir + L"\\" + name + L"_x" + to_wstring(target) + L".png";
        bool ok = rgba.empty() ? savePng(dst, out.data(), OW, OH, false) : savePng(dst, rgba.data(), OW, OH, true);
        if (ok) {
            saved++;
            lock_guard<mutex> l(G.mx);
            G.lastSaved = dst;
        }
        G.eng.learn(devName, estThis, now() - tf);
        doneEst += estThis;
    }
    if (G.stop) setStatus(L"Остановлено.");
    else setStatus(L"Готово за " + fmtTime(now() - t0) + L" (оценка была ≈ " + fmtTime(estTotal) + L"), сохранено: " + to_wstring(saved));
    PostMessageW(G.wnd, WM_DONE, saved, 0);
}

// ---------------- выбор файлов ----------------
static void setFiles(const vector<wstring>& paths);
static void pickFiles() {
    vector<wchar_t> buf(65536);
    OPENFILENAMEW ofn{sizeof(ofn)};
    ofn.hwndOwner = G.wnd;
    ofn.lpstrFilter = L"Изображения\0*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff\0Все файлы\0*.*\0";
    ofn.lpstrFile = buf.data();
    ofn.nMaxFile = (DWORD)buf.size();
    ofn.Flags = OFN_ALLOWMULTISELECT | OFN_EXPLORER | OFN_FILEMUSTEXIST;
    if (!GetOpenFileNameW(&ofn)) return;
    vector<wstring> parts;
    for (wchar_t* p = buf.data(); *p; p += wcslen(p) + 1) parts.push_back(p);
    vector<wstring> paths;
    if (parts.size() == 1) paths = parts;
    else
        for (size_t i = 1; i < parts.size(); i++) paths.push_back(parts[0] + L"\\" + parts[i]);
    setFiles(paths);
}

static void setFiles(const vector<wstring>& paths) {
    if (G.busy) return;
    G.files.clear();
    for (auto& p : paths) {
        int w, h;
        if (imageSize(p, w, h)) G.files.push_back({p, w, h});
    }
    if (G.thumb) DeleteObject(G.thumb), G.thumb = nullptr;
    if (!G.files.empty()) G.thumb = makeThumb(G.files[0].path, S(560), S(300), G.thumbW, G.thumbH);
    InvalidateRect(G.wnd, nullptr, FALSE);
}

static void pickOutDir() {
    IFileOpenDialog* d = nullptr;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL, IID_PPV_ARGS(&d)))) return;
    DWORD o;
    d->GetOptions(&o);
    d->SetOptions(o | FOS_PICKFOLDERS);
    d->SetTitle(L"Куда сохранять");
    if (SUCCEEDED(d->Show(G.wnd))) {
        IShellItem* it;
        if (SUCCEEDED(d->GetResult(&it))) {
            PWSTR p;
            if (SUCCEEDED(it->GetDisplayName(SIGDN_FILESYSPATH, &p))) G.outDir = p, CoTaskMemFree(p);
            it->Release();
        }
    }
    d->Release();
    InvalidateRect(G.wnd, nullptr, FALSE);
}

// ---------------- отрисовка ----------------
static void text(HDC dc, HFONT f, COLORREF c, int x, int y, const wstring& s, int w = 0, UINT fmt = DT_LEFT | DT_SINGLELINE | DT_NOPREFIX) {
    SelectObject(dc, f);
    SetTextColor(dc, c);
    RECT r{x, y, x + (w ? w : 4000), y + 4000};
    DrawTextW(dc, s.c_str(), -1, &r, fmt);
}
static void fill(HDC dc, int x, int y, int w, int h, COLORREF c) {
    HBRUSH b = CreateSolidBrush(c);
    RECT r{x, y, x + w, y + h};
    FillRect(dc, &r, b);
    DeleteObject(b);
}

static void paint(HDC dc, int W, int H) {
    fill(dc, 0, 0, W, H, C_BG);
    SetBkMode(dc, TRANSPARENT);
    // шапка + вотермарка справа сверху
    fill(dc, 0, 0, W, S(48), C_PANEL);
    text(dc, G.fTitle, C_FG, S(18), S(12), L"⬆  Нейро-апскейлер");
    RECT rm{0, 0, W - S(18), S(48)};
    SelectObject(dc, G.fMark);
    SetTextColor(dc, C_ACCENT);
    DrawTextW(dc, L"by wutxi", -1, &rm, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);

    wstring status;
    double prog;
    {
        lock_guard<mutex> l(G.mx);
        status = G.status;
        prog = G.progress;
    }
    // левая часть
    text(dc, G.fSmall, C_MUTED, S(16), S(102), L"Сохранение: " + (G.outDir.empty() ? L"рядом с оригиналом" : G.outDir), S(580),
         DT_LEFT | DT_SINGLELINE | DT_PATH_ELLIPSIS | DT_NOPREFIX);
    int px = S(16), py = S(126), pw = S(580), ph = S(310);
    fill(dc, px, py, pw, ph, C_PANEL);
    if (G.thumb) {
        HDC m = CreateCompatibleDC(dc);
        SelectObject(m, G.thumb);
        BitBlt(dc, px + (pw - G.thumbW) / 2, py + (ph - G.thumbH) / 2, G.thumbW, G.thumbH, m, 0, 0, SRCCOPY);
        DeleteDC(m);
    } else {
        RECT r{px, py, px + pw, py + ph};
        SelectObject(dc, G.fNorm);
        SetTextColor(dc, C_MUTED);
        DrawTextW(dc, L"Перетащи сюда изображения или нажми «Выбрать изображения…»", -1, &r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    }
    if (!G.files.empty()) {
        auto& f = G.files[0];
        wstring n = f.path.substr(f.path.find_last_of(L"\\/") + 1) + L" — " + to_wstring(f.w) + L"×" + to_wstring(f.h);
        if (G.files.size() > 1) n += L"   (+ ещё " + to_wstring(G.files.size() - 1) + L")";
        text(dc, G.fNorm, C_FG, S(16), S(444), n, S(580), DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS | DT_NOPREFIX);
    }
    text(dc, G.fNorm, C_MUTED, S(16), S(482), L"Модель:");
    text(dc, G.fNorm, C_MUTED, S(410), S(482), L"Итог:");
    // прогресс
    fill(dc, S(16), S(522), S(580), S(8), C_PANEL);
    fill(dc, S(16), S(522), (int)(S(580) * prog), S(8), C_ACCENT);
    text(dc, G.fSmall, C_MUTED, S(16), S(536), status, S(580), DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS | DT_NOPREFIX);

    // правая панель
    int rx = S(612), rw = W - rx - S(16);
    fill(dc, rx, S(64), rw, H - S(80), C_PANEL);
    int tx = rx + S(16), tw = rw - S(32);
    text(dc, G.fBold, C_MUTED, tx, S(78), L"ТВОЁ ЖЕЛЕЗО");
    if (G.hwReady) {
        wstring gpu, vram;
        for (auto& d : G.hw.devices)
            if (d.gpu >= 0) {
                gpu = d.name;
                vram = to_wstring((d.vramMB + 512) / 1024) + L" ГБ";
                break;
            }
        wstring hw = L"Видеокарта: " + (gpu.empty() ? L"нет (Vulkan)" : gpu) + (vram.empty() ? L"" : L"\nВидеопамять: " + vram) +
                     L"\nПроцессор: " + G.hw.cpu + L"\n" + G.hw.cores + L"\nОЗУ: " + G.hw.ram + L"\nСчитает: " +
                     (device().gpu >= 0 ? L"GPU (Vulkan, FP16)" : L"CPU — будет медленно");
        text(dc, G.fNorm, C_FG, tx, S(102), hw, tw, DT_LEFT | DT_WORDBREAK | DT_NOPREFIX);
    }
    fill(dc, tx, S(262), tw, 1, C_LINE);
    text(dc, G.fBold, C_MUTED, tx, S(276), L"ОЦЕНКА ВРЕМЕНИ");
    wstring big = L"—", det;
    if (G.files.empty()) det = L"Выбери изображение";
    else if (!G.calibReady) det = L"Идёт замер скорости…";
    else {
        double px2 = 0;
        for (auto& f : G.files) px2 += double(f.w) * f.h;
        big = L"≈ " + fmtTime(totalEstimate());
        int sc = outScale();
        wchar_t b[512];
        swprintf(b, 512, L"Файлов: %zu, %.1f Мп\nРезультат (1-й): %d×%d\nЗамер: %.2f сек на тайл %dpx%ls", G.files.size(), px2 / 1e6,
                 G.files[0].w * sc, G.files[0].h * sc, G.calib.secPerTile, G.eng.tile,
                 fabs(G.calib.corr - 1) > 0.01 ? (L"\nПоправка по прошлым запускам: ×" + to_wstring(G.calib.corr).substr(0, 4)).c_str() : L"");
        det = b;
    }
    text(dc, G.fBig, C_ACCENT, tx, S(298), big, tw);
    text(dc, G.fSmall, C_MUTED, tx, S(346), det, tw, DT_LEFT | DT_WORDBREAK | DT_NOPREFIX);
    text(dc, G.fSmall, C_MUTED, tx, S(478), L"Устройство:");
    text(dc, G.fSmall, C_MUTED, tx, S(530), L"Тайл, px (меньше = меньше VRAM):", tw);
}

// ---------------- контролы ----------------
static HWND mkButton(const wchar_t* t, int id, int x, int y, int w, int h) {
    HWND b = CreateWindowW(L"BUTTON", t, WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, S(x), S(y), S(w), S(h), G.wnd, (HMENU)(INT_PTR)id, nullptr, nullptr);
    return b;
}
static HWND mkCombo(int id, int x, int y, int w) {
    HWND c = CreateWindowW(L"COMBOBOX", L"", WS_CHILD | WS_VISIBLE | WS_VSCROLL | CBS_DROPDOWNLIST | CBS_OWNERDRAWFIXED | CBS_HASSTRINGS,
                           S(x), S(y), S(w), S(300), G.wnd, (HMENU)(INT_PTR)id, nullptr, nullptr);
    SendMessageW(c, WM_SETFONT, (WPARAM)G.fNorm, TRUE);
    return c;
}
static void comboAdd(HWND c, const wstring& s) { SendMessageW(c, CB_ADDSTRING, 0, (LPARAM)s.c_str()); }

static void drawItem(DRAWITEMSTRUCT* d) {
    HDC dc = d->hDC;
    RECT r = d->rcItem;
    SetBkMode(dc, TRANSPARENT);
    if (d->CtlType == ODT_BUTTON) {
        bool primary = d->CtlID == ID_RUN, dis = d->itemState & ODS_DISABLED, down = d->itemState & ODS_SELECTED;
        COLORREF bg = primary ? (dis ? RGB(0x4a, 0x3d, 0x8a) : down ? RGB(0x65, 0x48, 0xe0) : C_ACCENT) : (down ? C_LINE : C_PANEL);
        fill(dc, r.left, r.top, r.right - r.left, r.bottom - r.top, bg);
        wchar_t t[128];
        GetWindowTextW(d->hwndItem, t, 128);
        SelectObject(dc, primary ? G.fBold : G.fNorm);
        SetTextColor(dc, dis ? C_MUTED : (primary ? RGB(255, 255, 255) : C_FG));
        DrawTextW(dc, t, -1, &r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    } else if (d->CtlType == ODT_COMBOBOX) {
        bool sel = (d->itemState & ODS_SELECTED) && !(d->itemState & ODS_COMBOBOXEDIT);
        fill(dc, r.left, r.top, r.right - r.left, r.bottom - r.top, sel ? C_ACCENT : C_PANEL);
        if (d->itemID == (UINT)-1) return;
        wchar_t t[512];
        SendMessageW(d->hwndItem, CB_GETLBTEXT, d->itemID, (LPARAM)t);
        r.left += S(6);
        SelectObject(dc, G.fNorm);
        SetTextColor(dc, (d->itemState & ODS_DISABLED) ? C_MUTED : C_FG);
        DrawTextW(dc, t, -1, &r, DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS | DT_NOPREFIX);
    }
}

static void onHardware() {
    G.hwReady = true;
    for (auto& d : G.hw.devices) comboAdd(G.hDevice, d.name);
    SendMessageW(G.hDevice, CB_SETCURSEL, 0, 0);
    int t = autoTile(G.hw.devices[0]);
    for (int i = 0; i < 5; i++)
        if ((64 << i) == t) SendMessageW(G.hTile, CB_SETCURSEL, i, 0);
    scanModels();
    for (auto& m : G.models) comboAdd(G.hModel, m);
    if (G.models.empty()) {
        setStatus(L"Нет моделей: положи .param и .bin в папку models рядом с программой.");
        enableUi(true);
        InvalidateRect(G.wnd, nullptr, FALSE);
        return;
    }
    SendMessageW(G.hModel, CB_SETCURSEL, 0, 0);
    prepareModel();
}

static LRESULT CALLBACK WndProc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE: {
        G.wnd = h;
        G.fNorm = CreateFontW(-S(14), 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
        G.fSmall = CreateFontW(-S(13), 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
        G.fBold = CreateFontW(-S(13), 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
        G.fTitle = CreateFontW(-S(18), 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
        G.fMark = CreateFontW(-S(16), 0, 0, 0, FW_BOLD, 1, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
        G.fBig = CreateFontW(-S(32), 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
        G.brPanel = CreateSolidBrush(C_PANEL);
        G.hFiles = mkButton(L"Выбрать изображения…", ID_FILES, 16, 62, 200, 32);
        G.hOut = mkButton(L"Папка сохранения…", ID_OUTDIR, 224, 62, 180, 32);
        G.hModel = mkCombo(ID_MODEL, 80, 476, 316);
        G.hScale = mkCombo(ID_SCALE, 450, 476, 146);
        for (auto s : {L"как у модели", L"x2", L"x3", L"x4"}) comboAdd(G.hScale, s);
        SendMessageW(G.hScale, CB_SETCURSEL, 0, 0);
        G.hRun = mkButton(L"Апскейл", ID_RUN, 16, 566, 140, 36);
        G.hStop = mkButton(L"Стоп", ID_STOP, 164, 566, 100, 36);
        G.hDevice = mkCombo(ID_DEVICE, 628, 498, 268);
        G.hTile = mkCombo(ID_TILE, 628, 550, 120);
        for (int i = 0; i < 5; i++) comboAdd(G.hTile, to_wstring(64 << i));
        enableUi(false);
        EnableWindow(G.hStop, FALSE);
        DragAcceptFiles(h, TRUE);
        {  // файлы из командной строки / «Открыть с помощью»
            int argc;
            LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
            vector<wstring> paths(argv + 1, argv + argc);
            LocalFree(argv);
            if (!paths.empty()) setFiles(paths);
        }
        thread([] {
            G.hw = detectHardware();
            PostMessageW(G.wnd, WM_HW, 0, 0);
        }).detach();
        return 0;
    }
    case WM_MEASUREITEM: {
        auto* m = (MEASUREITEMSTRUCT*)lp;
        m->itemHeight = S(24);
        return TRUE;
    }
    case WM_DRAWITEM:
        drawItem((DRAWITEMSTRUCT*)lp);
        return TRUE;
    case WM_CTLCOLORLISTBOX:
        SetBkColor((HDC)wp, C_PANEL);
        SetTextColor((HDC)wp, C_FG);
        return (LRESULT)G.brPanel;
    case WM_COMMAND: {
        int id = LOWORD(wp), code = HIWORD(wp);
        if (id == ID_FILES) pickFiles();
        else if (id == ID_OUTDIR) pickOutDir();
        else if (id == ID_STOP) G.stop = true;
        else if (id == ID_RUN) {
            if (G.files.empty()) MessageBoxW(h, L"Сначала выбери изображения.", L"Апскейлер", MB_ICONINFORMATION);
            else if (G.calibReady && !G.busy) {
                G.busy = true;
                G.stop = false;
                enableUi(false);
                thread(worker).detach();
            }
        } else if ((id == ID_MODEL || id == ID_DEVICE || id == ID_TILE) && code == CBN_SELCHANGE) {
            if (id == ID_DEVICE) {  // для нового устройства — свой автотайл
                int t = autoTile(device());
                for (int i = 0; i < 5; i++)
                    if ((64 << i) == t) SendMessageW(G.hTile, CB_SETCURSEL, i, 0);
            }
            prepareModel();
        } else if (id == ID_SCALE && code == CBN_SELCHANGE)
            InvalidateRect(h, nullptr, FALSE);
        return 0;
    }
    case WM_DROPFILES: {
        HDROP d = (HDROP)wp;
        vector<wstring> paths;
        wchar_t p[MAX_PATH * 4];
        for (UINT i = 0, n = DragQueryFileW(d, 0xFFFFFFFF, nullptr, 0); i < n; i++)
            if (DragQueryFileW(d, i, p, MAX_PATH * 4)) paths.push_back(p);
        DragFinish(d);
        setFiles(paths);
        return 0;
    }
    case WM_HW:
        onHardware();
        return 0;
    case WM_READY: {
        G.busy = false;
        G.calibReady = true;
        enableUi(true);
        wchar_t b[256];
        swprintf(b, 256, L"Готово. Модель x%d, замер: %.2f сек на тайл %dpx", G.eng.scale, G.calib.secPerTile, G.eng.tile);
        setStatus(b);
        InvalidateRect(h, nullptr, FALSE);
        return 0;
    }
    case WM_PROG:
        InvalidateRect(h, nullptr, FALSE);
        return 0;
    case WM_DONE: {
        G.busy = false;
        enableUi(true);
        auto m = loadCalib();
        auto k = G.eng.key(device().name);
        if (m.count(k)) G.calib = m[k];
        {
            lock_guard<mutex> l(G.mx);
            if (!G.stop) G.progress = 1;
        }
        InvalidateRect(h, nullptr, FALSE);
        if (wp > 0 && !G.stop) {
            wstring st, last;
            {
                lock_guard<mutex> l(G.mx);
                st = G.status;
                last = G.lastSaved;
            }
            if (MessageBoxW(h, (st + L"\n\n" + last + L"\n\nОткрыть папку?").c_str(), L"Готово", MB_YESNO | MB_ICONINFORMATION) == IDYES)
                ShellExecuteW(nullptr, L"open", L"explorer.exe", (L"/select,\"" + last + L"\"").c_str(), nullptr, SW_SHOWNORMAL);
        }
        return 0;
    }
    case WM_FAIL: {
        G.busy = false;
        enableUi(true);
        wstring st;
        {
            lock_guard<mutex> l(G.mx);
            st = G.status;
        }
        InvalidateRect(h, nullptr, FALSE);
        MessageBoxW(h, st.c_str(), L"Ошибка", MB_ICONERROR);
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(h, &ps);
        RECT rc;
        GetClientRect(h, &rc);
        HDC m = CreateCompatibleDC(dc);
        HBITMAP bmp = CreateCompatibleBitmap(dc, rc.right, rc.bottom);
        HGDIOBJ old = SelectObject(m, bmp);
        paint(m, rc.right, rc.bottom);
        BitBlt(dc, 0, 0, rc.right, rc.bottom, m, 0, 0, SRCCOPY);
        SelectObject(m, old);
        DeleteObject(bmp);
        DeleteDC(m);
        EndPaint(h, &ps);
        return 0;
    }
    case WM_CLOSE:
        if (G.busy && MessageBoxW(h, L"Идёт обработка. Закрыть?", L"Апскейлер", MB_YESNO | MB_ICONQUESTION) != IDYES) return 0;
        G.stop = true;
        DestroyWindow(h);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE, PWSTR, int show) {
    SetProcessDPIAware();
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic));
    HDC sdc = GetDC(nullptr);
    G.dpi = GetDeviceCaps(sdc, LOGPIXELSY);
    ReleaseDC(nullptr, sdc);

    WNDCLASSW wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hIcon = LoadIcon(inst, MAKEINTRESOURCE(1));
    wc.lpszClassName = L"WutxiUpscaler";
    RegisterClassW(&wc);
    DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    RECT r{0, 0, S(912), S(620)};
    AdjustWindowRect(&r, style, FALSE);
    HWND h = CreateWindowW(wc.lpszClassName, L"Нейро-апскейлер — by wutxi", style, CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left,
                           r.bottom - r.top, nullptr, nullptr, inst, nullptr);
    ShowWindow(h, show);
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    // ждём завершения фоновой работы, чтобы корректно освободить Vulkan
    for (int i = 0; i < 100 && G.busy; i++) Sleep(50);
    G.eng.net.reset();
    ncnn::destroy_gpu_instance();
    return 0;
}
