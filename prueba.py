import os
import glob

# Ruta base donde buscar
base_path = "drive/MyDrive"

# Encuentra todos los .jpg recursivamente
jpg_files = glob.glob(os.path.join(base_path, "**/*.jpg"), recursive=True)

print(f"Encontrados {len(jpg_files)} archivos JPG")

# Borra en lotes de 20
batch_size = 20
for i in range(0, len(jpg_files), batch_size):
    batch = jpg_files[i:i+batch_size]
    print(f"\n🗑️ Borrando lote {i//batch_size + 1}: {len(batch)} archivos")
    for file in batch:
        try:
            os.remove(file)
            print(f"  ✅ {file}")
        except Exception as e:
            print(f"  ⚠️ No se pudo borrar {file}: {e}")
