"""Fase 1: imprime la(s) GPU detectada(s) por CuPy.

Uso:
    python fase_1/gpu_available.py

Además de listar las GPU, ejecuta una operación pequeña en cada una para
confirmar que CuPy no solo la detecta, sino que también puede usarla.
"""

import sys

try:
    import cupy as cp
except ImportError:
    sys.exit("CuPy no está instalado. Instala requirements.txt en el entorno virtual.")


def describir_gpu(indice: int) -> None:
    props = cp.cuda.runtime.getDeviceProperties(indice)
    nombre = props["name"].decode()
    capacidad = f"{props['major']}.{props['minor']}"
    memoria_gb = props["totalGlobalMem"] / 1024**3

    print(f"GPU {indice}: {nombre}")
    print(f"  Compute capability : {capacidad}")
    print(f"  Memoria total      : {memoria_gb:.1f} GB")
    print(f"  Multiprocesadores  : {props['multiProcessorCount']}")

    # Prueba rápida: suma de un arreglo en la GPU y comparación con el valor esperado.
    with cp.cuda.Device(indice):
        n = 1_000_000
        suma = int(cp.arange(n, dtype=cp.int64).sum())
        esperado = n * (n - 1) // 2
        estado = "OK" if suma == esperado else f"ERROR ({suma} != {esperado})"
        print(f"  Prueba de cómputo  : {estado}")


def main() -> int:
    try:
        cantidad = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        print(f"CuPy no pudo acceder a CUDA: {error}")
        print("Revisa que haya una GPU NVIDIA y que el driver esté instalado.")
        return 1

    if cantidad == 0:
        print("CuPy no detectó ninguna GPU.")
        return 1

    version_driver = cp.cuda.runtime.driverGetVersion()
    version_runtime = cp.cuda.runtime.runtimeGetVersion()
    print(f"CuPy {cp.__version__}")
    print(f"CUDA driver {version_driver // 1000}.{version_driver % 1000 // 10}, "
          f"runtime {version_runtime // 1000}.{version_runtime % 1000 // 10}")
    print(f"GPU detectadas: {cantidad}\n")

    for indice in range(cantidad):
        describir_gpu(indice)
    return 0


if __name__ == "__main__":
    sys.exit(main())
