import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.utils import make_grid
from PIL import Image, ImageDraw

from dataset import CompCarsDataset, get_dataloaders  # importa dal tuo file

def visualize_batch(batch, denormalize=None):
    """
    Mostra le prime 4 immagini del batch con i bounding box.
    batch: dizionario ritornato dal DataLoader
    denormalize: funzione per riportare i tensori in [0,255] (opzionale)
    """
    images = batch['image']        # [B,3,H,W]
    bboxes = batch['bbox']         # [B,4]
    viewpoints = batch['viewpoint']
    B = images.size(0)

    # Prendi le prime 4 (o meno)
    n = min(4, B)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    for i in range(n):
        img_tensor = images[i]
        # denormalizza se necessario
        img = T.ToPILImage()(img_tensor)
        draw = ImageDraw.Draw(img)
        x1,y1,x2,y2 = bboxes[i].tolist()
        draw.rectangle([x1,y1,x2,y2], outline='red', width=3)
        axes[i].imshow(img)
        axes[i].set_title(f"View: {viewpoints[i].item()}")
        axes[i].axis('off')
    plt.show()


if __name__ == '__main__':
    # Imposta qui la cartella del tuo CompCars
    ROOT = '../0_dataset/data'

    # Ottieni i loader (80% train, seed=42)
    train_loader, test_loader = get_dataloaders(ROOT, train_ratio=0.8, batch_size=8, seed=42)

    # Prendi un batch di training e uno di test
    train_batch = next(iter(train_loader))
    test_batch  = next(iter(test_loader))

    # Stampa forme dei tensori
    print("---- TRAIN BATCH ----")
    print("Images:", train_batch['image'].shape)
    print("Viewpoints:", train_batch['viewpoint'].shape)
    print("BBoxes:", train_batch['bbox'].shape)
    print("All train BBoxes:\n", train_batch['bbox'])

    print("\n---- TEST BATCH ----")
    print("Images:", test_batch['image'].shape)
    print("Viewpoints:", test_batch['viewpoint'].shape)
    print("BBoxes:", test_batch['bbox'].shape)

    # Visualizza le prime immagini del batch di train
    visualize_batch(train_batch)
