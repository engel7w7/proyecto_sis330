import os
import sys
import json
import time
import argparse
import numpy as np
from PIL import Image
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import cv2

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATA_PROCESSED_DIR,
    DATASET_ASVSPOOF_2019_DIR,
    DATASET_FACEFORENSICS_DIR,
    DATASET_CIFAKE_DIR,
    IMG_SIZE
)

HAAR_CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

def ensure_directories():
    dirs = [
        os.path.join(DATA_PROCESSED_DIR, 'audio', 'real'),
        os.path.join(DATA_PROCESSED_DIR, 'audio', 'fake'),
        os.path.join(DATA_PROCESSED_DIR, 'vision', 'real'),
        os.path.join(DATA_PROCESSED_DIR, 'vision', 'fake'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return {
        'audio_real': dirs[0],
        'audio_fake': dirs[1],
        'vision_real': dirs[2],
        'vision_fake': dirs[3]
    }

def audio_to_spectrogram(audio_data, sr=16000, target_size=IMG_SIZE):
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    if sr != 16000:
        num_samples = int(len(audio_data) * 16000 / sr)
        audio_data = np.interp(np.linspace(0, len(audio_data), num_samples), np.arange(len(audio_data)), audio_data)
        sr = 16000

    if audio_data.dtype != np.float32:
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data.astype(np.float32) / max_val
        else:
            audio_data = audio_data.astype(np.float32)

    target_samples = 48000
    if len(audio_data) < target_samples:
        audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:target_samples]

    f, t, Sxx = signal.spectrogram(audio_data, fs=16000, nperseg=512, noverlap=256)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)

    s_min, s_max = Sxx_log.min(), Sxx_log.max()
    s_norm = (Sxx_log - s_min) / (s_max - s_min + 1e-6)
    s_img = (s_norm * 255.0).astype(np.uint8)

    s_color_bgr = cv2.applyColorMap(s_img, cv2.COLORMAP_MAGMA)
    s_color_rgb = cv2.cvtColor(s_color_bgr, cv2.COLOR_BGR2RGB)
    pil_spec = Image.fromarray(s_color_rgb).resize((target_size, target_size), Image.Resampling.BILINEAR)
    return pil_spec

def process_asvspoof_audio(output_dirs, max_samples=300):
    print("\n--- [ETL AUDIO] Procesando ASVspoof 2019 ---")
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile no instalado, omitiendo lectura de FLAC")
        return 0, 0

    proto_path = os.path.join(DATASET_ASVSPOOF_2019_DIR, 'LA', 'LA', 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.dev.trl.txt')
    flac_dir = os.path.join(DATASET_ASVSPOOF_2019_DIR, 'LA', 'LA', 'ASVspoof2019_LA_dev', 'flac')

    if not os.path.exists(proto_path) or not os.path.exists(flac_dir):
        print(f"Protocolo o directorio FLAC no encontrado: {proto_path}")
        return 0, 0

    target_per_class = max_samples // 2
    count_real = 0
    count_fake = 0

    with open(proto_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) >= 5:
                fid = parts[1]
                label = parts[4].lower()
                file_path = os.path.join(flac_dir, f"{fid}.flac")

                if not os.path.exists(file_path):
                    continue

                if label == 'bonafide' and count_real < target_per_class:
                    data, sr = sf.read(file_path)
                    spec_img = audio_to_spectrogram(data, sr)
                    out_path = os.path.join(output_dirs['audio_real'], f"spec_real_{fid}.png")
                    spec_img.save(out_path, format="PNG")
                    count_real += 1
                elif label == 'spoof' and count_fake < target_per_class:
                    data, sr = sf.read(file_path)
                    spec_img = audio_to_spectrogram(data, sr)
                    out_path = os.path.join(output_dirs['audio_fake'], f"spec_fake_{fid}.png")
                    spec_img.save(out_path, format="PNG")
                    count_fake += 1

            if count_real >= target_per_class and count_fake >= target_per_class:
                break

    print(f" -> Audio Procesado Exitosamente: {count_real} reales (bonafide) y {count_fake} clonados (spoof)")
    return count_real, count_fake

def extract_and_crop_face(img_bgr, target_size=IMG_SIZE):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    if len(faces) == 0:
        return None

    fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
    h, w = img_bgr.shape[:2]

    margin_x = int(fw * 0.15)
    margin_y = int(fh * 0.15)
    x1 = max(0, fx - margin_x)
    y1 = max(0, fy - margin_y)
    x2 = min(w, fx + fw + margin_x)
    y2 = min(h, fy + fh + margin_y)

    face_crop = img_bgr[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return face_resized

def process_faceforensics_videos(output_dirs, max_samples=300):
    print("\n--- [ETL VISION] Procesando FaceForensics & Retratos ---")
    orig_dir = os.path.join(DATASET_FACEFORENSICS_DIR, 'original_sequences', 'actors', 'c23', 'videos')
    manip_dir = os.path.join(DATASET_FACEFORENSICS_DIR, 'manipulated_sequences', 'DeepFakeDetection', 'c23', 'videos')

    target_per_class = max_samples // 2
    count_real = 0
    count_fake = 0

    if os.path.exists(orig_dir):
        mp4_files = [os.path.join(orig_dir, f) for f in os.listdir(orig_dir) if f.endswith('.mp4')]
        for vid_path in mp4_files:
            if count_real >= target_per_class:
                break
            cap = cv2.VideoCapture(vid_path)
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 60
            keyframes = np.linspace(10, max(11, total_f - 10), 3, dtype=int)
            for idx in keyframes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    face = extract_and_crop_face(frame)
                    if face is not None and count_real < target_per_class:
                        out_name = f"face_real_ff_{os.path.splitext(os.path.basename(vid_path))[0]}_f{idx}.jpg"
                        cv2.imwrite(os.path.join(output_dirs['vision_real'], out_name), face)
                        count_real += 1
            cap.release()

    if os.path.exists(manip_dir):
        mp4_files = [os.path.join(manip_dir, f) for f in os.listdir(manip_dir) if f.endswith('.mp4')]
        for vid_path in mp4_files:
            if count_fake >= target_per_class:
                break
            cap = cv2.VideoCapture(vid_path)
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 60
            keyframes = np.linspace(10, max(11, total_f - 10), 3, dtype=int)
            for idx in keyframes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    face = extract_and_crop_face(frame)
                    if face is not None and count_fake < target_per_class:
                        out_name = f"face_fake_ff_{os.path.splitext(os.path.basename(vid_path))[0]}_f{idx}.jpg"
                        cv2.imwrite(os.path.join(output_dirs['vision_fake'], out_name), face)
                        count_fake += 1
            cap.release()

    test_img_dir = os.path.join(PROJECT_ROOT, "archivos_de_prueba", "02_imagenes")
    if os.path.exists(test_img_dir):
        for f in os.listdir(test_img_dir):
            if not f.lower().endswith(('.jpg', '.png')):
                continue
            is_fake = "deepfake" in f.lower() or "faceswap" in f.lower()
            p = os.path.join(test_img_dir, f)
            img = cv2.imread(p)
            if img is not None:
                face = extract_and_crop_face(img)
                if face is None:
                    face = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                if is_fake and count_fake < target_per_class:
                    cv2.imwrite(os.path.join(output_dirs['vision_fake'], f"sample_{f}"), face)
                    count_fake += 1
                elif not is_fake and count_real < target_per_class:
                    cv2.imwrite(os.path.join(output_dirs['vision_real'], f"sample_{f}"), face)
                    count_real += 1

    print(f" -> Vision Procesada Exitosamente: {count_real} rostros reales y {count_fake} deepfakes")
    return count_real, count_fake

def main():
    parser = argparse.ArgumentParser(description="Pipeline ETL de Preprocesamiento de Datasets SIS-330")
    parser.add_argument("--samples_per_modality", type=int, default=300, help="Cantidad de muestras a procesar por modalidad (150 real / 150 fake)")
    args = parser.parse_args()

    print("INICIANDO PREPROCESAMIENTO DE DATASETS CRUDOS -> ARCHIVOS PROCESADOS (SIS-330)")
    print(f"Origen Datasets: {os.path.join(PROJECT_ROOT, 'datasets')}")
    print(f"Destino Procesados: {DATA_PROCESSED_DIR}")

    output_dirs = ensure_directories()
    t0 = time.time()

    a_real, a_fake = process_asvspoof_audio(output_dirs, max_samples=args.samples_per_modality)
    v_real, v_fake = process_faceforensics_videos(output_dirs, max_samples=args.samples_per_modality)

    elapsed = time.time() - t0

    manifest = {
        "fecha_procesamiento": time.strftime("%Y-%m-%d %H:%M:%S"),
        "origen_datasets": {
            "audio": "datasets/ASVspoof_2019_Dataset/LA",
            "vision": "datasets/FaceForensics"
        },
        "destino_procesados": "detector_multimodal_edge/data/processed",
        "conteo_muestras": {
            "audio_real": a_real,
            "audio_fake": a_fake,
            "audio_total": a_real + a_fake,
            "vision_real": v_real,
            "vision_fake": v_fake,
            "vision_total": v_real + v_fake,
            "total_muestras_procesadas": a_real + a_fake + v_real + v_fake
        },
        "parametros_transformacion": {
            "img_size": [IMG_SIZE, IMG_SIZE],
            "audio_sr": 16000,
            "audio_window_sec": 3.0,
            "audio_stft": {"nperseg": 512, "noverlap": 256},
            "audio_colormap": "COLORMAP_MAGMA",
            "vision_face_detector": "OpenCV Haar Cascade Frontal Face",
            "vision_face_margin": 0.15
        },
        "compatibilidad_dataloaders": "src.models.dataloaders.MultimodalDeepfakeDataset",
        "tiempo_procesamiento_seg": round(elapsed, 2)
    }

    manifest_path = os.path.join(DATA_PROCESSED_DIR, "manifest_resumen.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)

    print(f"Audio Reales (Espectrogramas PNG Color):   {a_real:>4} archivos en data/processed/audio/real")
    print(f"Audio Clonados (Espectrogramas PNG Color): {a_fake:>4} archivos en data/processed/audio/fake")
    print(f"Vision Reales (Rostros Recortados):        {v_real:>4} archivos en data/processed/vision/real")
    print(f"Vision Deepfake (Rostros Recortados):      {v_fake:>4} archivos en data/processed/vision/fake")
    print(f"TOTAL ARCHIVOS PROCESADOS:                 {a_real + a_fake + v_real + v_fake:>4} archivos")
    print(f"Manifest generado en: {manifest_path}")
    print(f"Tiempo total transcurrido: {elapsed:.2f} s")

if __name__ == "__main__":
    main()
