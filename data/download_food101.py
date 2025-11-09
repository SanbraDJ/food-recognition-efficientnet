"""
Script para descargar y extraer el dataset Food-101
"""
import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_food101(data_dir='data'):
    """
    Descarga y extrae el dataset Food-101
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    food101_path = data_path / 'food-101'
    
    if food101_path.exists():
        print("✅ El dataset Food-101 ya está descargado.")
        return str(food101_path)
    
    print("📥 Descargando Food-101 dataset (~5GB)...")
    print("⏳ Esto puede tomar varios minutos dependiendo de tu conexión...")
    
    url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    tar_path = data_path / 'food-101.tar.gz'
    
    # Descargar
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='Descargando') as t:
        urllib.request.urlretrieve(url, tar_path, reporthook=t.update_to)
    
    print("📦 Extrayendo archivos...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=data_path)
    
    # Eliminar archivo tar
    os.remove(tar_path)
    
    print(f"✅ Dataset descargado y extraído en: {food101_path}")
    return str(food101_path)

if __name__ == "__main__":
    download_food101()