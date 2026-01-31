"""
SAM 2.1 + DINOv2 Pipeline
Автоматическая детекция ВСЕХ объектов и извлечение эмбеддингов

Workflow:
1. SAM 2.1 (pipeline) - автоматически находит ВСЕ объекты → маски + bbox
2. DINOv2 - извлекает мощные 768D эмбеддинги для каждого объекта
"""

import torch
from transformers import (
    pipeline,
    AutoImageProcessor,
    AutoModel
)
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import json

# Настройки
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FRAMES_DIR = DATA_DIR / "scand_spot_cafe-2" / "output_frames"
NUM_FRAMES = 8
OUTPUT_DIR = BASE_DIR / "sam2_dinov2_output"
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = 0 if torch.cuda.is_available() else -1  # для pipeline: 0=cuda, -1=cpu

print(f"🔧 Устройство: {'cuda' if DEVICE == 0 else 'cpu'}")
print("="*70)

# ============================================================================
# ЭТАП 1: Загрузка моделей
# ============================================================================
print("\n📦 ЗАГРУЗКА МОДЕЛЕЙ")
print("-"*70)

# SAM 2.1 для автоматической сегментации через pipeline
print("1️⃣ Загрузка SAM 2.1 (Automatic Mask Generation)...")
mask_generator = pipeline(
    "mask-generation",
    model="facebook/sam2.1-hiera-base-plus",
    device=DEVICE,
    points_per_batch=64
)
print(f"   ✅ SAM 2.1 загружен: facebook/sam2.1-hiera-base-plus")
print(f"   └─ Режим: Automatic mask generation (все объекты)")

# DINOv2 для извлечения эмбеддингов
print("2️⃣ Загрузка DINOv2...")
dino_model_name = "facebook/dinov2-base"
dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
dino_device = "cuda" if torch.cuda.is_available() else "cpu"
dino_model = AutoModel.from_pretrained(dino_model_name).to(dino_device)
dino_model.eval()
print(f"   ✅ DINOv2 загружен: {dino_model_name}")
print(f"   └─ Размер эмбеддингов: {dino_model.config.hidden_size}D")

# ============================================================================
# ЭТАП 2: Загрузка кадров
# ============================================================================
print("\n" + "="*70)
print("📸 ЗАГРУЗКА КАДРОВ")
print("-"*70)

frame_files = sorted(FRAMES_DIR.glob("*.png"))[:NUM_FRAMES]
print(f"Загружено {len(frame_files)} кадров:")
for i, f in enumerate(frame_files):
    print(f"   [{i}] {f.name}")

images = [Image.open(f).convert("RGB") for f in frame_files]
print(f"   Размер кадра: {images[0].size}")

# ============================================================================
# Вспомогательные функции
# ============================================================================

def mask_to_bbox(mask):
    """Конвертируем маску в bounding box"""
    # Если маска это PIL Image, конвертируем
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    # Бинаризуем если нужно
    if mask.dtype != bool:
        mask = mask > 0.5

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return [int(x1), int(y1), int(x2), int(y2)]

# ============================================================================
# ЭТАП 3: Обработка через SAM 2.1 (автоматическая генерация масок)
# ============================================================================
print("\n" + "="*70)
print("🎭 ЭТАП 1: SAM 2.1 - Автоматическая детекция ВСЕХ объектов")
print("-"*70)

all_frame_results = []

for frame_idx, image in enumerate(images[:2]):  # Пока только 2 кадра для теста
    print(f"\n   Кадр {frame_idx}...")

    # SAM 2.1 автоматически находит ВСЕ объекты
    outputs = mask_generator(image, points_per_batch=64)

    num_objects = len(outputs["masks"])
    print(f"      → Найдено {num_objects} объектов")

    # Конвертируем маски в удобный формат
    masks_data = []
    for obj_idx in range(num_objects):
        # Маска уже в формате PIL или numpy
        mask = np.array(outputs["masks"][obj_idx])

        # Вычисляем bounding box из маски
        bbox = mask_to_bbox(mask)

        if bbox is not None:
            masks_data.append({
                'mask': mask,
                'bbox': bbox,
                'score': outputs["scores"][obj_idx] if "scores" in outputs else 1.0
            })

    print(f"      → После фильтрации: {len(masks_data)} объектов с валидными bbox")

    all_frame_results.append({
        'frame_idx': frame_idx,
        'num_objects': len(masks_data),
        'masks': masks_data
    })

# ============================================================================
# ЭТАП 4: Извлечение эмбеддингов через DINOv2
# ============================================================================
print("\n" + "="*70)
print("🧬 ЭТАП 2: DINOv2 - Извлечение эмбеддингов объектов")
print("-"*70)

for result in all_frame_results:
    frame_idx = result['frame_idx']
    image = images[frame_idx]

    print(f"\n   Кадр {frame_idx}: обработка {result['num_objects']} объектов...")

    for obj_idx, mask_data in enumerate(result['masks']):
        bbox = mask_data['bbox']

        # Вырезаем объект по bounding box
        x1, y1, x2, y2 = bbox
        object_crop = image.crop((x1, y1, x2, y2))

        # Извлекаем эмбеддинг через DINOv2
        inputs = dino_processor(images=object_crop, return_tensors="pt").to(dino_device)

        with torch.no_grad():
            outputs = dino_model(**inputs)

        # Берем CLS token (глобальный признак объекта)
        object_embedding = outputs.last_hidden_state[0, 0].cpu().numpy()  # [768]

        # Сохраняем эмбеддинг
        mask_data['embedding'] = object_embedding
        mask_data['embedding_dim'] = len(object_embedding)

        # Вычисляем площадь объекта
        mask_area = mask_data['mask'].sum()
        mask_data['area'] = int(mask_area)

        print(f"      Obj #{obj_idx}: bbox={bbox}, area={mask_area:.0f}px, "
              f"emb={len(object_embedding)}D, score={mask_data['score']:.3f}")

# ============================================================================
# ЭТАП 5: Визуализация результатов
# ============================================================================
print("\n" + "="*70)
print("🎨 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("-"*70)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

for result in all_frame_results:
    frame_idx = result['frame_idx']
    image = images[frame_idx]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(f'Кадр {frame_idx}: SAM 2.1 + DINOv2 Pipeline',
                 fontsize=16, fontweight='bold')

    # 1. Оригинальное изображение
    axes[0].imshow(image)
    axes[0].set_title('Оригинал', fontsize=12)
    axes[0].axis('off')

    # 2. Маски SAM 2.1 (все объекты)
    axes[1].imshow(image)
    axes[1].set_title(f'SAM 2.1 Masks ({result["num_objects"]} объектов)', fontsize=12)
    axes[1].axis('off')

    # Накладываем маски разными цветами
    if result['num_objects'] > 0:
        overlay = np.zeros((*image.size[::-1], 3))
        colors = plt.cm.rainbow(np.linspace(0, 1, result['num_objects']))

        for obj_idx, mask_data in enumerate(result['masks']):
            mask = mask_data['mask']
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            color = colors[obj_idx][:3]
            overlay[mask > 0.5] = color

        axes[1].imshow(overlay, alpha=0.6)

    # 3. Bounding boxes + эмбеддинги
    axes[2].imshow(image)
    axes[2].set_title('Bounding Boxes + 768D Эмбеддинги', fontsize=12)
    axes[2].axis('off')

    if result['num_objects'] > 0:
        colors = plt.cm.rainbow(np.linspace(0, 1, result['num_objects']))

        for obj_idx, mask_data in enumerate(result['masks']):
            bbox = mask_data['bbox']
            x1, y1, x2, y2 = bbox

            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2.5, edgecolor=colors[obj_idx], facecolor='none'
            )
            axes[2].add_patch(rect)

            # Текст с информацией
            area_k = mask_data['area'] / 1000
            axes[2].text(
                x1, y1-8,
                f"#{obj_idx} | {mask_data['embedding_dim']}D | {area_k:.1f}K px",
                color='white', fontsize=9, weight='bold',
                bbox=dict(facecolor=colors[obj_idx], alpha=0.8, pad=2)
            )

    plt.tight_layout()
    output_file = OUTPUT_DIR / f"frame_{frame_idx:03d}_sam2_result.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✅ Сохранено: {output_file.name}")
    plt.close()

# ============================================================================
# ЭТАП 6: Сохранение данных
# ============================================================================
print("\n" + "="*70)
print("💾 СОХРАНЕНИЕ ДАННЫХ")
print("-"*70)

# Сохраняем эмбеддинги и метаданные
output_data = {
    'model_info': {
        'sam': 'facebook/sam2.1-hiera-base-plus',
        'dino': 'facebook/dinov2-base',
        'embedding_dim': 768
    },
    'num_frames': len(all_frame_results),
    'frames': []
}

for result in all_frame_results:
    frame_data = {
        'frame_idx': result['frame_idx'],
        'frame_file': frame_files[result['frame_idx']].name,
        'num_objects': result['num_objects'],
        'objects': []
    }

    for obj_idx, mask_data in enumerate(result['masks']):
        obj_data = {
            'object_id': obj_idx,
            'bbox': mask_data['bbox'],
            'area': mask_data['area'],
            'sam_score': float(mask_data['score']),
            'embedding': mask_data['embedding'].tolist(),
            'embedding_dim': mask_data['embedding_dim']
        }
        frame_data['objects'].append(obj_data)

    output_data['frames'].append(frame_data)

# Сохраняем JSON
json_file = OUTPUT_DIR / "sam2_dinov2_results.json"
with open(json_file, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"   ✅ Метаданные: {json_file.name}")

# Сохраняем эмбеддинги отдельно (numpy для удобства)
embeddings_file = OUTPUT_DIR / "object_embeddings.npy"
np.save(embeddings_file, output_data)
print(f"   ✅ Эмбеддинги: {embeddings_file.name}")

# ============================================================================
# ИТОГИ
# ============================================================================
print("\n" + "="*70)
print("✨ ИТОГОВАЯ СТАТИСТИКА")
print("="*70)

total_objects = sum(r['num_objects'] for r in all_frame_results)
avg_objects = total_objects / len(all_frame_results) if all_frame_results else 0

print(f"📊 Обработано кадров: {len(all_frame_results)}")
print(f"🎯 Найдено объектов: {total_objects} (в среднем {avg_objects:.1f} на кадр)")
print(f"📐 Размер эмбеддингов: 768D (DINOv2)")
print(f"💾 Результаты сохранены в: {OUTPUT_DIR}/")
print()
print("📁 Структура выхода для каждого объекта:")
print("   ├─ Маска (SAM 2.1): бинарная маска пикселей")
print("   ├─ BBox: [x1, y1, x2, y2]")
print("   ├─ Area: количество пикселей объекта")
print("   ├─ SAM Score: уверенность сегментации")
print("   └─ Эмбеддинг (DINOv2): 768D вектор признаков")
print()
print("🚀 КЛЮЧЕВОЕ ПРЕИМУЩЕСТВО:")
print("   SAM 2.1 автоматически находит ВСЕ объекты без промптов!")
print("   Не нужно заранее знать что искать - модель найдет всё сама.")
print("="*70)
