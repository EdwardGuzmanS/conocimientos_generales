"""Fase 2: conectar una cámara, capturar un frame y guardarlo como TIFF de 16 bits.

Funciona con cámaras Thorlabs (Scientific Camera SDK, p. ej. Kiralux CS895MU) y Basler (pylon).

Uso:
    python first_connection.py                          # Thorlabs (por defecto)
    python first_connection.py --marca basler           # Basler
    python first_connection.py --marca basler --emulada # Basler simulada, sin cámara

Opciones útiles:
    --exposicion-us 5000      tiempo de exposición en microsegundos
    --roi 640 480 0 0         ancho alto offset_x offset_y
    --salida captura.tiff     nombre del archivo de salida
    --dll-dir RUTA            (Thorlabs) carpeta con las DLL del SDK
"""

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import tifffile

# Carpetas donde se buscan las DLL del SDK de Thorlabs si no se indica --dll-dir.
RUTAS_DLL_THORLABS = [
    os.environ.get("THORLABS_SDK_DLLS", ""),
    r"C:\Program Files\Thorlabs\Scientific Imaging\ThorCam",
]


@dataclass
class Captura:
    imagen: np.ndarray
    modelo: str
    serie: str
    bits: int            # bits reales del sensor (12 en la CS895MU)
    exposicion_us: float
    roi: tuple           # (ancho, alto, offset_x, offset_y) aplicados por la cámara


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Primera conexión con una cámara Thorlabs o Basler.")
    p.add_argument("--marca", choices=["thorlabs", "basler"], default="thorlabs",
                   help="Marca de la cámara (por defecto thorlabs).")
    p.add_argument("--exposicion-us", type=float, default=5000.0,
                   help="Tiempo de exposición en microsegundos (por defecto 5000).")
    p.add_argument("--roi", type=int, nargs=4, metavar=("ANCHO", "ALTO", "OFFSET_X", "OFFSET_Y"),
                   default=[640, 480, 0, 0], help="Región de interés (por defecto 640 480 0 0).")
    p.add_argument("--salida", default="captura_16bits.tiff", help="Archivo TIFF de salida.")
    p.add_argument("--emulada", action="store_true",
                   help="Solo Basler: usar una cámara simulada (no necesita cámara física).")
    p.add_argument("--dll-dir", default=None,
                   help="Solo Thorlabs: carpeta con las DLL del SDK (thorlabs_tsi_*.dll).")
    args = p.parse_args()
    if args.emulada and args.marca != "basler":
        p.error("--emulada solo existe para Basler; Thorlabs no trae cámara simulada.")
    return args


def limitar(valor, minimo, maximo):
    return max(minimo, min(valor, maximo))


# --------------------------------------------------------------------------- Thorlabs
def preparar_dlls_thorlabs(dll_dir: str | None) -> str | None:
    """Indica a Windows dónde están las DLL del SDK, que el paquete de Python necesita."""
    for ruta in ([dll_dir] if dll_dir else RUTAS_DLL_THORLABS):
        if ruta and os.path.isdir(ruta):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(ruta)
            os.environ["PATH"] = ruta + os.pathsep + os.environ.get("PATH", "")
            return ruta
    return None


def capturar_thorlabs(args) -> Captura:
    ruta = preparar_dlls_thorlabs(args.dll_dir)
    # El paquete thorlabs_tsi_sdk 1.2+ trae sus propias DLL; --dll-dir solo hace falta en versiones antiguas.
    if ruta:
        print(f"DLL del SDK Thorlabs: {ruta}")
    try:
        from thorlabs_tsi_sdk.tl_camera import ROI, TLCameraSDK
    except ImportError:
        sys.exit("Falta el paquete thorlabs_tsi_sdk. Instálalo desde el zip del SDK que trae ThorCam.")

    with TLCameraSDK() as sdk:
        # 1) Inicializar el SDK y abrir la primera cámara disponible.
        series = sdk.discover_available_cameras()
        if not series:
            raise RuntimeError("No se encontró ninguna cámara Thorlabs. Revisa el cable USB 3.0 y cierra ThorCam.")

        with sdk.open_camera(series[0]) as camara:
            print(f"Cámara abierta: {camara.model} (serie {camara.serial_number})")

            # 2) Exposición, dentro del rango que acepta la cámara.
            rango = camara.exposure_time_range_us
            camara.exposure_time_us = int(limitar(args.exposicion_us, rango.min, rango.max))

            # 3) ROI. Thorlabs la define con dos esquinas (incluidas):
            #    superior izquierda (x1, y1) e inferior derecha (x2, y2).
            ancho, alto, x, y = args.roi
            x1, y1, x2, y2 = x, y, x + ancho - 1, y + alto - 1
            try:
                r = camara.roi_range
                x1 = limitar(x1, r.upper_left_x_pixels_min, r.upper_left_x_pixels_max)
                y1 = limitar(y1, r.upper_left_y_pixels_min, r.upper_left_y_pixels_max)
                x2 = limitar(x2, r.lower_right_x_pixels_min, r.lower_right_x_pixels_max)
                y2 = limitar(y2, r.lower_right_y_pixels_min, r.lower_right_y_pixels_max)
            except AttributeError:
                pass  # versión del SDK sin roi_range: la cámara ajusta la ROI por sí sola
            camara.roi = ROI(x1, y1, x2, y2)

            # 4) Capturar un frame con un disparo por software.
            camara.frames_per_trigger_zero_for_unlimited = 1
            camara.image_poll_timeout_ms = 2000 + int(camara.exposure_time_us / 1000)
            camara.arm(2)
            try:
                camara.issue_software_trigger()
                frame = camara.get_pending_frame_or_null()
                if frame is None:
                    raise RuntimeError("Se agotó el tiempo de espera y la cámara no entregó un frame.")
                ancho_real = camara.image_width_pixels
                alto_real = camara.image_height_pixels
                # El SDK reutiliza su memoria: se copia y se da forma (alto, ancho).
                imagen = np.copy(frame.image_buffer).reshape(alto_real, ancho_real)
            finally:
                camara.disarm()

            roi_real = camara.roi
            return Captura(imagen, camara.model, camara.serial_number, camara.bit_depth,
                           camara.exposure_time_us,
                           (ancho_real, alto_real, roi_real.upper_left_x_pixels, roi_real.upper_left_y_pixels))


# --------------------------------------------------------------------------- Basler
def ajustar_a_incremento(valor: int, nodo) -> int:
    """Ajusta un valor al rango y al incremento que acepta la cámara (ej. múltiplos de 4)."""
    valor = limitar(valor, nodo.Min, nodo.Max)
    return nodo.Min + ((valor - nodo.Min) // nodo.Inc) * nodo.Inc


def capturar_basler(args) -> Captura:
    if args.emulada:
        os.environ["PYLON_CAMEMU"] = "1"  # debe definirse antes de importar pypylon
    try:
        from pypylon import pylon
    except ImportError:
        sys.exit("Falta pypylon. Instálalo con: pip install pypylon")

    # 1) Inicializar el SDK y abrir la primera cámara disponible.
    fabrica = pylon.TlFactory.GetInstance()
    dispositivos = fabrica.EnumerateDevices()
    if not dispositivos:
        raise RuntimeError("No se encontró ninguna cámara Basler. Revisa la conexión o usa --emulada.")

    camara = pylon.InstantCamera(fabrica.CreateDevice(dispositivos[0]))
    camara.Open()
    try:
        info = camara.GetDeviceInfo()
        print(f"Cámara abierta: {info.GetModelName()} (serie {info.GetSerialNumber()})")

        # Formato monocromo con más bits disponible.
        formato = next(f for f in ("Mono16", "Mono12", "Mono10", "Mono8")
                       if f in camara.PixelFormat.Symbolics)
        camara.PixelFormat.Value = formato

        # 2) Exposición (ExposureTime en cámaras nuevas, ExposureTimeAbs en algunas ace GigE).
        nodo = camara.ExposureTime if hasattr(camara, "ExposureTime") else camara.ExposureTimeAbs
        if hasattr(camara, "ExposureAuto"):
            camara.ExposureAuto.Value = "Off"
        nodo.Value = limitar(args.exposicion_us, nodo.Min, nodo.Max)

        # 3) ROI: primero offsets en 0 para liberar el tamaño máximo.
        ancho, alto, x, y = args.roi
        camara.OffsetX.Value = 0
        camara.OffsetY.Value = 0
        camara.Width.Value = ajustar_a_incremento(ancho, camara.Width)
        camara.Height.Value = ajustar_a_incremento(alto, camara.Height)
        camara.OffsetX.Value = ajustar_a_incremento(x, camara.OffsetX)
        camara.OffsetY.Value = ajustar_a_incremento(y, camara.OffsetY)

        # 4) Capturar un frame.
        resultado = camara.GrabOne(5000)
        try:
            if not resultado.GrabSucceeded():
                raise RuntimeError(f"Falló la captura: {resultado.ErrorDescription}")
            imagen = resultado.Array.copy()
        finally:
            resultado.Release()

        return Captura(imagen, info.GetModelName(), info.GetSerialNumber(), int(formato[4:]), nodo.Value,
                       (camara.Width.Value, camara.Height.Value, camara.OffsetX.Value, camara.OffsetY.Value))
    finally:
        camara.Close()


# --------------------------------------------------------------------------- TIFF
def verificar_tiff(ruta: str, ancho: int, alto: int) -> bool:
    imagen = tifffile.imread(ruta)
    ok_bits = imagen.dtype == np.uint16
    ok_tamano = imagen.shape == (alto, ancho)
    print(f"\nVerificación de {ruta}:")
    print(f"  Tipo de dato : {imagen.dtype}  -> {'OK' if ok_bits else 'ERROR, se esperaba uint16'}")
    print(f"  Tamaño       : {imagen.shape[1]} x {imagen.shape[0]}  -> "
          f"{'OK' if ok_tamano else f'ERROR, se esperaba {ancho} x {alto}'}")
    print(f"  Rango valores: {imagen.min()} a {imagen.max()}")
    return ok_bits and ok_tamano


def main() -> int:
    args = parse_args()
    try:
        captura = capturar_thorlabs(args) if args.marca == "thorlabs" else capturar_basler(args)
    except RuntimeError as error:
        print(error)
        return 1

    ancho, alto, ox, oy = captura.roi
    print(f"Bits del sensor : {captura.bits} (valores de 0 a {2 ** captura.bits - 1})")
    print(f"Exposición      : {captura.exposicion_us:.0f} us")
    print(f"ROI             : {ancho} x {alto} en ({ox}, {oy})")

    # 5) Guardar como TIFF de 16 bits (uint16) y reabrirlo para verificar.
    tifffile.imwrite(args.salida, captura.imagen.astype(np.uint16))
    print(f"\nGuardado: {args.salida}")
    return 0 if verificar_tiff(args.salida, ancho, alto) else 1


if __name__ == "__main__":
    sys.exit(main())
