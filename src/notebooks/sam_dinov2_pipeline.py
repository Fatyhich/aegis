"""
SAM + DINOv2 Pipeline
Автоматическая детекция объектов и извлечение эмбеддингов

Workflow:
1. SAM - автоматически находит все объекты → маски
2. DINOv2 - извлекает мощные 768D эмбеддинги для каждого объекта
"""

import torch
from transformers import (
    SamModel,
    SamProcessor,
    AutoImageProcessor,
    AutoModel
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import json

# Настройки
BASE_DIR = Path(__file__).parent  # Директория скрипта
DATA_DIR = BASE_DIR / "data"
FRAMES_DIR = DATA_DIR / "scand_spot_cafe-2" / "output_frames"
NUM_FRAMES = 8
OUTPUT_DIR = BASE_DIR / "sam_dinov2_output"
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔧 Устройство: {DEVICE}")
print("="*70)

# ============================================================================
# ЭТАП 1: Загрузка моделей
# ============================================================================
print("\n📦 ЗАГРУЗКА МОДЕЛЕЙ")
print("-"*70)

# SAM для автоматической сегментации
print("1️⃣ Загрузка SAM (Segment Anything Model)...")
sam_model_name = "facebook/sam-vit-base"
sam_processor = SamProcessor.from_pretrained(sam_model_name)
sam_model = SamModel.from_pretrained(sam_model_name).to(DEVICE)
sam_model.eval()
print(f"   ✅ SAM загружен: {sam_model_name}")

# DINOv2 для извлечения эмбеддингов
print("2️⃣ Загрузка DINOv2...")
dino_model_name = "facebook/dinov2-base"
dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
dino_model = AutoModel.from_pretrained(dino_model_name).to(DEVICE)
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
# ЭТАП 3: Обработка через SAM (автоматическая сегментация)
# ============================================================================
print("\n" + "="*70)
print("🎭 ЭТАП 1: SAM - Автоматическая детекция объектов")
print("-"*70)

def generate_automatic_masks(image, model, processor):
    """
    SAM в режиме automatic mask generation - находит все объекты
    """
    # Создаем сетку точек для промптов
    h, w = image.size[1], image.size[0]
    points_per_side = 32  # Плотность сетки

    # Генерируем сетку точек
    x = np.linspace(0, w, points_per_side)
    y = np.linspace(0, h, points_per_side)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Обрабатываем батчами (SAM может быть медленным)
    all_masks = []
    all_scores = []
    batch_size = 64

    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        # Каждая точка - это промпт "объект здесь"
        input_points = [[p.tolist()] for p in batch_points]

        inputs = processor(
            image,
            input_points=input_points,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        # Получаем маски
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]

        # Берем лучшую маску для каждой точки
        scores = outputs.iou_scores.cpu().numpy()
        best_masks = masks[range(len(masks)), scores.argmax(1)]

        all_masks.extend(best_masks)
        all_scores.extend(scores.max(1))

    # Фильтруем по качеству
    quality_threshold = 0.8
    good_masks = [
        (mask, score) for mask, score in zip(all_masks, all_scores)
        if score > quality_threshold
    ]

    # NMS - убираем сильно перекрывающиеся маски
    final_masks = non_max_suppression_masks(good_masks, iou_threshold=0.7)

    return final_masks

def non_max_suppression_masks(masks_with_scores, iou_threshold=0.7):
    """Убираем сильно перекрывающиеся маски"""
    if len(masks_with_scores) == 0:
        return []

    # Сортируем по скору
    sorted_masks = sorted(masks_with_scores, key=lambda x: x[1], reverse=True)

    keep = []
    while len(sorted_masks) > 0:
        current = sorted_masks.pop(0)
        keep.append(current)

        # Убираем маски с большим перекрытием
        sorted_masks = [
            m for m in sorted_masks
            if mask_iou(current[0], m[0]) < iou_threshold
        ]

    return keep

def mask_iou(mask1, mask2):
    """Вычисляем IoU между двумя масками"""
    mask1 = mask1.squeeze().numpy() if torch.is_tensor(mask1) else mask1
    mask2 = mask2.squeeze().numpy() if torch.is_tensor(mask2) else mask2

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    return intersection / (union + 1e-6)

def mask_to_bbox(mask):
    """Конвертируем маску в bounding box"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return [int(x1), int(y1), int(x2), int(y2)]

print("🔍 Обработка кадров через SAM...")
print("   Это может занять время (генерация масок для всех объектов)...")

all_frame_results = []

for frame_idx, image in enumerate(images[:2]):  # Пока только 2 кадра для теста
    print(f"\n   Кадр {frame_idx}...")

    # Упрощенный подход: используем один центральный промпт
    # для демонстрации (полная automatic mask generation очень медленная)
    h, w = image.size[1], image.size[0]

    # Создаем несколько промптов в разных частях изображения
    # Формат для SAM: [[batch][points_per_prompt][xy]]
    input_points = [[
        [w//4, h//4],      # Верх-лево
        [3*w//4, h//4],    # Верх-право
        [w//2, h//2],      # Центр
        [w//4, 3*h//4],    # Низ-лево
        [3*w//4, 3*h//4],  # Низ-право
    ]]

    inputs = sam_processor(
        image,
        input_points=input_points,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    # Post-process masks
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )[0]  # [batch, num_queries, num_masks, H, W]

    scores = outputs.iou_scores.cpu().numpy()  # [batch, num_queries, num_masks]

    # Берем лучшую маску для каждого промпта
    best_masks = []
    num_prompts = masks.shape[0]
    for i in range(num_prompts):
        best_idx = scores[0, i].argmax()  # [0] - batch dimension
        if scores[0, i, best_idx] > 0.7:  # Порог качества
            mask = masks[i, best_idx].squeeze()
            best_masks.append({
                'mask': mask.numpy(),
                'score': float(scores[0, i, best_idx]),
                'bbox': mask_to_bbox(mask.numpy())
            })

    print(f"      → Найдено {len(best_masks)} объектов")
    all_frame_results.append({
        'frame_idx': frame_idx,
        'num_objects': len(best_masks),
        'masks': best_masks
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
        mask = mask_data['mask']
        bbox = mask_data['bbox']

        if bbox is None:
            continue

        # Вырезаем объект по bounding box
        x1, y1, x2, y2 = bbox
        object_crop = image.crop((x1, y1, x2, y2))

        # Извлекаем эмбеддинг через DINOv2
        inputs = dino_processor(images=object_crop, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = dino_model(**inputs)

        # Берем CLS token (глобальный признак объекта)
        object_embedding = outputs.last_hidden_state[0, 0].cpu().numpy()  # [768]

        # Сохраняем эмбеддинг
        mask_data['embedding'] = object_embedding
        mask_data['embedding_dim'] = len(object_embedding)

        print(f"      Объект {obj_idx}: bbox={bbox}, emb_dim={len(object_embedding)}, score={mask_data['score']:.3f}")

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

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Кадр {frame_idx}: SAM + DINOv2 Pipeline', fontsize=14, fontweight='bold')

    # 1. Оригинальное изображение
    axes[0].imshow(image)
    axes[0].set_title('Оригинал')
    axes[0].axis('off')

    # 2. Маски SAM
    axes[1].imshow(image)
    axes[1].set_title(f'SAM Masks ({result["num_objects"]} объектов)')
    axes[1].axis('off')

    # Накладываем маски разными цветами
    overlay = np.zeros((*result['masks'][0]['mask'].shape, 3))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(result['masks'])))

    for obj_idx, mask_data in enumerate(result['masks']):
        mask = mask_data['mask']
        color = colors[obj_idx][:3]
        overlay[mask > 0.5] = color

    axes[1].imshow(overlay, alpha=0.5)

    # 3. Bounding boxes + метаданные
    axes[2].imshow(image)
    axes[2].set_title('Bounding Boxes + Эмбеддинги')
    axes[2].axis('off')

    for obj_idx, mask_data in enumerate(result['masks']):
        bbox = mask_data['bbox']
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=colors[obj_idx], facecolor='none'
        )
        axes[2].add_patch(rect)

        # Текст с информацией
        axes[2].text(
            x1, y1-5,
            f"#{obj_idx} | {mask_data['embedding_dim']}D",
            color='white', fontsize=8,
            bbox=dict(facecolor=colors[obj_idx], alpha=0.7)
        )

    plt.tight_layout()
    output_file = OUTPUT_DIR / f"frame_{frame_idx:03d}_result.png"
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
    'num_frames': len(all_frame_results),
    'embedding_dim': 768,
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
            'sam_score': mask_data['score'],
            'embedding': mask_data['embedding'].tolist(),
            'embedding_dim': mask_data['embedding_dim']
        }
        frame_data['objects'].append(obj_data)

    output_data['frames'].append(frame_data)

# Сохраняем JSON
json_file = OUTPUT_DIR / "sam_dinov2_results.json"
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
print(f"📊 Обработано кадров: {len(all_frame_results)}")
print(f"🎯 Найдено объектов: {total_objects}")
print(f"📐 Размер эмбеддингов: 768D (DINOv2)")
print(f"💾 Результаты сохранены в: {OUTPUT_DIR}/")
print()
print("📁 Структура выхода для каждого объекта:")
print("   ├─ Маска (SAM): бинарная маска пикселей")
print("   ├─ BBox: [x1, y1, x2, y2]")
print("   ├─ SAM Score: уверенность сегментации")
print("   └─ Эмбеддинг (DINOv2): 768D вектор признаков")
print("="*70)
