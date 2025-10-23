# Utils for fotography
import numpy as np
import cv2 as cv
import itertools
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Optional
from sklearn.cluster import MiniBatchKMeans
from skimage.segmentation import slic


def find_channel_percentiles(image, percentile):
    """
    Calcula el valor de intensidad correspondiente a un percentil dado
    para cada canal de una imagen.

    Parámetros
    ----------
    image : np.ndarray
        Imagen de entrada en formato NumPy. Puede ser en escala de grises (2D) 
        o en color (3D) con cualquier número de canales.
    percentile : float
        Percentil a calcular (0-100).

    Retorna
    -------
    dict
        Diccionario donde las claves son los índices de canal (0, 1, 2, ...)
        y los valores son los niveles de intensidad correspondientes al
        percentil solicitado.
    
    Ejemplo
    -------
    >>> vals = find_channel_percentiles(img, 95)
    >>> print(vals)
    {0: 245, 1: 243, 2: 250}
    """
    if image.ndim == 2:
        # Imagen en escala de grises: convertir a 3D para unificar procesamiento
        image = image[:, :, np.newaxis]

    percentiles = {}

    for c in range(image.shape[2]):
        channel = image[:, :, c]

        # Calcular histograma del canal (256 bins para intensidades 0-255)
        hist = cv.calcHist([channel], [0], None, [256], [0, 256])
        cumulative_hist = np.cumsum(hist)

        # Valor objetivo basado en el percentil
        target_value = percentile * cumulative_hist[-1] / 100

        # Encontrar el primer índice que supere el target_value
        value = np.searchsorted(cumulative_hist, target_value)
        percentiles[c] = int(value)

    return percentiles


def visualize_permutations(image, transformations, channel_names):
    """
    Aplica todas las permutaciones posibles de transformaciones a los tres canales
    de una imagen, las visualiza y devuelve los resultados.

    Parámetros
    ----------
    image : np.ndarray
        Imagen de entrada en formato NumPy (debe tener 3 canales: BGR).
    transformations : list of callable
        Lista de funciones de transformación que reciben un canal (2D array)
        y devuelven un canal transformado del mismo tamaño.
    channel_names : dict
        Diccionario que mapea cada función de `transformations` a un nombre 
        descriptivo (para títulos en las gráficas).

    Retorna
    -------
    list of dict
        Lista de resultados. Cada elemento es un diccionario con:
            - "permutation": tupla de transformaciones aplicadas (B, G, R)
            - "image": imagen resultante en escala de grises después de fusionar y transformar

    Ejemplo
    -------
    >>> results = visualize_permutations(img, [np.copy, cv.bitwise_not], 
                                         {np.copy: "Original", cv.bitwise_not: "Invertido"})
    >>> first = results[0]["image"]
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("La imagen debe tener exactamente 3 canales (BGR).")
    
    b_channel, g_channel, r_channel = cv.split(image)
    permutations = list(itertools.product(transformations, repeat=3))
    num_permutations = len(permutations)

    # Preparar la figura para visualización
    fig, axes = plt.subplots(
        nrows=int(np.ceil(num_permutations / 3)), ncols=3, figsize=(15, num_permutations)
    )
    axes = axes.ravel()

    results = []

    for i, perm in enumerate(permutations):
        # Aplicar transformaciones a cada canal
        b_trans = perm[0](b_channel)
        g_trans = perm[1](g_channel)
        r_trans = perm[2](r_channel)

        # Combinar canales y convertir a escala de grises
        merged_img = cv.merge((b_trans, g_trans, r_trans))
        merged_gray = cv.cvtColor(merged_img, cv.COLOR_BGR2GRAY)

        # Guardar el resultado en la lista
        results.append({
            "permutation": perm,
            "image": merged_gray
        })

        # Mostrar en subplot
        axes[i].imshow(merged_gray, cmap="gray")
        axes[i].set_title(
            f"B: {channel_names[perm[0]]}, "
            f"G: {channel_names[perm[1]]}, "
            f"R: {channel_names[perm[2]]}"
        )
        axes[i].axis("off")

    # Apagar subplots no usados
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    return results


def overlay_images(background_img, overlay_img, 
                   x_shift=0, y_shift=0,
                   y_start=0, y_end=None,
                   z_order="foreground",
                   white_to_transparent=True,
                   resize_overlay=False):
    """
    Superpone una imagen sobre otra, con soporte para desplazamiento,
    redimensionado opcional, orden de apilado y tratamiento de píxeles blancos
    como transparentes.

    Parámetros
    ----------
    background_img : np.ndarray | PIL.Image.Image
        Imagen base (numpy array). Puede ser en escala de grises o RGB.
    overlay_img : PIL.Image.Image | np.ndarray
        Imagen a superponer. Puede incluir canal alfa (RGBA).
    x_shift : int, opcional
        Desplazamiento horizontal de la imagen superpuesta 
        (positivo → derecha, negativo → izquierda).
    y_shift : int, opcional
        Desplazamiento vertical de la imagen superpuesta 
        (positivo → abajo, negativo → arriba).
    y_start : int, opcional
        Coordenada Y inicial donde se coloca la imagen superpuesta.
    y_end : int, opcional
        Coordenada Y final (define el alto de redimensionado si resize_overlay=True). 
        Si es None, se usa la altura completa de la imagen base.
    z_order : str, opcional
        Define el orden de apilado:
        - "foreground": coloca la imagen superpuesta delante.
        - "background": coloca la imagen superpuesta detrás.
    white_to_transparent : bool, opcional
        Si True, convierte los píxeles blancos de la imagen base en transparentes.
    resize_overlay : bool, opcional
        Si True, redimensiona la imagen superpuesta para ajustarse al ancho 
        de la imagen base y a la altura definida por y_start/y_end.

    Retorna
    -------
    np.ndarray
        Imagen final compuesta como array RGBA (convertida de vuelta a numpy).
    """
    # --- Normalizar entrada ---
    if isinstance(background_img, np.ndarray):
        if background_img.ndim == 2:
            background_pil = Image.fromarray(background_img, mode="L").convert("RGBA")
        elif background_img.shape[2] == 3:
            background_pil = Image.fromarray(background_img).convert("RGBA")
        elif background_img.shape[2] == 4:
            background_pil = Image.fromarray(background_img, mode="RGBA")
        else:
            raise ValueError("La imagen de fondo debe ser 2D (grises) o 3/4 canales (RGB/RGBA).")
    elif isinstance(background_img, Image.Image):
        background_pil = background_img.convert("RGBA")
    else:
        raise TypeError("background_img debe ser un np.ndarray o PIL.Image.Image")

    if isinstance(overlay_img, np.ndarray):
        overlay_pil = Image.fromarray(overlay_img)
    elif isinstance(overlay_img, Image.Image):
        overlay_pil = overlay_img
    else:
        raise TypeError("overlay_img debe ser un np.ndarray o PIL.Image.Image")

    overlay_pil = overlay_pil.convert("RGBA")

    base_w, base_h = background_pil.size
    if y_end is None:
        y_end = base_h

    # --- Redimensionar overlay si se indica ---
    if resize_overlay:
        overlay_pil = overlay_pil.resize((base_w, y_end - y_start), Image.LANCZOS)

    # --- Convertir blancos a transparente si se indica ---
    if white_to_transparent:
        bg_data = np.array(background_pil)
        white_mask = np.all(bg_data[:, :, :3] > 240, axis=-1)
        bg_data[white_mask] = [255, 255, 255, 0]
        background_pil = Image.fromarray(bg_data, mode="RGBA")

    # --- Crear lienzo de salida ---
    result = Image.new("RGBA", background_pil.size, (255, 255, 255, 0))

    paste_x = x_shift
    paste_y = y_start + y_shift

    if z_order == "background":
        result.paste(overlay_pil, (paste_x, paste_y), overlay_pil)
        result.alpha_composite(background_pil)
    else:
        result.paste(background_pil, (0, 0))
        result.paste(overlay_pil, (paste_x, paste_y), overlay_pil)

    return np.array(result)



def double_exposure(img1, img2, exposure_factor=0.5, resize_to_match=True):
    """
    Combina dos imágenes simulando una doble exposición analógica.

    Parámetros
    ----------
    img1 : np.ndarray | PIL.Image.Image
        Primera imagen (numpy array o PIL). Actúa como base.
    img2 : np.ndarray | PIL.Image.Image
        Segunda imagen que se mezclará con la primera.
    exposure_factor : float, opcional
        Factor de mezcla entre las imágenes (0.0 - 1.0):
        - 0.0 = solo la primera imagen.
        - 0.5 = mezcla equitativa (efecto típico de doble exposición).
        - 1.0 = solo la segunda imagen.
    resize_to_match : bool, opcional
        Si True, redimensiona img2 para que tenga el mismo tamaño que img1.

    Retorna
    -------
    np.ndarray
        Imagen resultante en formato RGB (uint8, rango 0-255).

    Ejemplo
    -------
    >>> blended = double_exposure(img_a, img_b, exposure_factor=0.6)
    >>> cv2.imshow("Double Exposure", blended)
    """
    # --- Normalizar entrada ---
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)

    # Convertir a 3 canales si están en escala de grises
    if img1.ndim == 2:
        img1 = np.stack([img1] * 3, axis=-1)
    if img2.ndim == 2:
        img2 = np.stack([img2] * 3, axis=-1)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Redimensionar img2 si es necesario
    if resize_to_match and (h1 != h2 or w1 != w2):
        img2 = np.array(Image.fromarray(img2).resize((w1, h1), Image.LANCZOS))

    # Convertir a float y normalizar a [0, 1]
    img1_f = img1.astype(np.float32) / 255.0
    img2_f = img2.astype(np.float32) / 255.0

    # --- Mezcla tipo doble exposición ---
    # Simulación básica: superposición aditiva y normalización
    blended = img1_f * (1 - exposure_factor) + img2_f * exposure_factor
    blended = np.clip(blended, 0, 1)

    # Volver a uint8
    return (blended * 255).astype(np.uint8)


def conditional_overlay(
    base_img,
    overlay_img,
    brightness_threshold=None,
    r_range=None,
    g_range=None,
    b_range=None,
    color_mode="strict"
):
    """
    Superpone una imagen sobre otra de manera condicional, usando 
    brillo y/o rangos de color como máscara.

    Parámetros
    ----------
    base_img : np.ndarray | PIL.Image.Image
        Imagen base. Puede ser en escala de grises o RGB.
    overlay_img : np.ndarray | PIL.Image.Image
        Imagen a superponer. Se convierte a RGBA para manejar transparencia.
    brightness_threshold : int | None, opcional
        Si se especifica, superpone solo donde la luminosidad media de los píxeles 
        sea menor que este valor (0-255).
    r_range, g_range, b_range : tuple[int, int] | None, opcional
        Rangos de intensidad por canal para crear máscaras de color.
        Ejemplo: r_range=(50, 180) -> considera válidos solo píxeles rojos en ese rango.
    color_mode : {"strict", "loose"}, opcional
        - "strict": requiere que todos los canales estén dentro de sus rangos.
        - "loose": requiere que al menos un canal esté dentro de su rango.

    Retorna
    -------
    np.ndarray
        Imagen resultante con la superposición aplicada donde se cumplan las condiciones.

    Ejemplo
    -------
    >>> result = conditional_overlay(
    ...     antena_v_img,
    ...     contour_resized_pil,
    ...     brightness_threshold=145,
    ...     r_range=(80, 200),
    ...     g_range=(50, 220),
    ...     b_range=(30, 180),
    ...     color_mode="strict"
    ... )
    >>> plt.imshow(result); plt.axis("off"); plt.show()
    """
    # --- Convertir imágenes a RGBA ---
    if isinstance(base_img, Image.Image):
        base_pil = base_img.convert("RGBA")
    elif isinstance(base_img, np.ndarray):
        if base_img.ndim == 2:
            base_pil = Image.fromarray(base_img, mode="L").convert("RGBA")
        else:
            base_pil = Image.fromarray(base_img).convert("RGBA")
    else:
        raise TypeError("base_img debe ser np.ndarray o PIL.Image.Image")

    if isinstance(overlay_img, Image.Image):
        overlay_pil = overlay_img.convert("RGBA")
    elif isinstance(overlay_img, np.ndarray):
        overlay_pil = Image.fromarray(overlay_img).convert("RGBA")
    else:
        raise TypeError("overlay_img debe ser np.ndarray o PIL.Image.Image")

    base_np = np.array(base_pil)
    overlay_np = np.array(overlay_pil)

    # --- Crear máscara inicial: todo True ---
    mask = np.ones(base_np.shape[:2], dtype=bool)

    # --- Máscara por brillo ---
    if brightness_threshold is not None:
        avg_brightness = np.mean(base_np[..., :3], axis=2)
        mask &= avg_brightness < brightness_threshold

    # --- Máscara por color ---
    if any(v is not None for v in [r_range, g_range, b_range]):
        r = base_np[..., 0]
        g = base_np[..., 1]
        b = base_np[..., 2]

        masks = []
        if r_range: masks.append((r >= r_range[0]) & (r <= r_range[1]))
        if g_range: masks.append((g >= g_range[0]) & (g <= g_range[1]))
        if b_range: masks.append((b >= b_range[0]) & (b <= b_range[1]))

        if masks:
            color_mask = masks[0]
            for m in masks[1:]:
                color_mask = color_mask & m if color_mode == "strict" else color_mask | m
            mask &= color_mask

    # --- Aplicar overlay donde mask es True ---
    result = base_np.copy()
    alpha = overlay_np[..., 3] / 255.0

    for c in range(3):  # aplicar sobre R, G, B
        result[..., c][mask] = (
            (1 - alpha[mask]) * result[..., c][mask] + alpha[mask] * overlay_np[..., c][mask]
        )

    return result



def segment_and_color(
    image_path: str,
    n_colors: int,
    resize: bool = False,
    target_res: int = 2048,
    use_superpixels: bool = True,
    n_superpixels: int = 500,
    selected_clusters: Optional[List[int]] = None,
    apply_modular_effect: bool = False,
    random_state: Optional[int] = 0,
    show_result: bool = True
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Segment an image into color clusters using MiniBatchKMeans (optionally after
    superpixel preprocessing), extract per-cluster masks and subimages, and
    compose a visualization from selected clusters.

    Each selected cluster can optionally be recolored using a unique random
    modular color transformation before being combined into the final image.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    n_colors : int, default=6
        Number of color clusters for MiniBatchKMeans.
    resize : bool, default=False
        Whether to resize the image to a target resolution before processing.
    target_res : int, default=2048
        Target size (shorter side) when resizing, preserving aspect ratio.
    use_superpixels : bool, default=True
        Whether to preprocess the image with SLIC superpixels before clustering.
    n_superpixels : int, default=500
        Number of superpixels to use if `use_superpixels` is True.
    selected_clusters : list[int], optional
        Indices of clusters to include in the composed output. If None, all
        clusters are included.
    apply_modular_effect : bool, default=False
        Whether to apply a random modular color transform to each selected cluster
        before composition.
    random_state : int, optional
        Random seed for reproducibility.
    show_result : bool, default=True
        If True, displays the final composed image with matplotlib.

    Returns
    -------
    composed_image : np.ndarray
        RGB image composed from the selected (optionally recolored) clusters.
    masks : list[np.ndarray]
        Binary masks (H×W) corresponding to each color cluster.
    subimages : list[np.ndarray]
        RGB images (H×W×3) showing each cluster in isolation.

    Notes
    -----
    - The function performs clustering in the CIELab color space for perceptual
      accuracy, but returns results in RGB.
    - Works for any image resolution or aspect ratio.
    - Each selected cluster gets its own random modular color transform when
      `apply_modular_effect` is enabled.
    """

    image_bgr = cv.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

    if resize:
        h, w = image_rgb.shape[:2]
        scale = target_res / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_rgb = cv.resize(image_rgb, (new_w, new_h), interpolation=cv.INTER_LANCZOS4)

    if use_superpixels:
        segments = slic(
            image_rgb,
            n_segments=n_superpixels,
            compactness=10,
            start_label=0,
            channel_axis=-1
        )
        superpixel_image = np.zeros_like(image_rgb, dtype=np.float32)
        for seg_val in np.unique(segments):
            mask = segments == seg_val
            mean_color = image_rgb[mask].mean(axis=0)
            superpixel_image[mask] = mean_color
        work_image = superpixel_image.astype(np.uint8)
    else:
        work_image = image_rgb

    lab = cv.cvtColor(work_image, cv.COLOR_RGB2LAB)
    data = lab.reshape((-1, 3))

    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=random_state,
        batch_size=4096
    )
    labels = kmeans.fit_predict(data)
    labels_2d = labels.reshape(lab.shape[:2])

    # ---- Generate masks and subimages ----
    masks = []
    subimages = []
    for i in range(n_colors):
        mask = (labels_2d == i).astype(np.uint8)
        masks.append(mask)
        masked_image = image_rgb.copy()
        masked_image[mask == 0] = 0
        subimages.append(masked_image)

    if selected_clusters is None:
        selected_clusters = list(range(n_colors))

    composed_image = np.zeros_like(image_rgb, dtype=np.uint8)

    for idx in selected_clusters:
        mask = masks[idx].astype(bool)
        cluster_img = np.zeros_like(image_rgb)
        cluster_img[mask] = image_rgb[mask]

        if apply_modular_effect:
            modules = np.random.randint(255, size=3)
            offsets = np.random.randint(255, size=3)
            for c in range(3):
                cluster_img[:, :, c] = (offsets[c] * cluster_img[:, :, c]) % modules[c]

        composed_image[mask] = cluster_img[mask]

    if show_result:
        plt.figure(figsize=(6, 6))
        plt.imshow(composed_image)
        plt.axis("off")
        plt.show()

    return composed_image, masks, subimages
