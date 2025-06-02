import numpy as np
from sklearn.feature_extraction.image import PatchExtractor
from skimage.transform import resize

def extract_patches(img,N,scale=1.0,patch_size = (64, 64)):
    """
    Extrae N parches de una imagen
    :param image: Imagen de entrada
    :param N: Cantidad de parches a extraer
    :param scale: Escala de la imagen (1.0 = original)
    :param patch_size: Tamaño del parche
    :return: Parche extraido
    """
    # Calcula el tamaño del parche extraído basado en el factor de escala dado
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))

    # Inicializa un objeto PatchExtractor con el tamaño de parche calculado,
    # el número máximo de parches, y una semilla de estado aleatorio
    extractor = PatchExtractor(patch_size=extracted_patch_size, max_patches=N, random_state=0)

    # Extrae parches de la imagen dada
    # img[np.newaxis] se utiliza la entrada de PatchExtractor es un conjunto de imágenes
    patches = extractor.transform(img[np.newaxis])

    # Si el factor de escala no es 1, redimensiona cada parche extraído
    # al tamaño del parche original
    if scale != 1:
        #patches = np.array([resize(patch, patch_size,preserve_range=True) for patch in patches])
        patches = np.array([resize(patch, patch_size) for patch in patches])

    
    # Devuelve la lista de parches extraídos (y posiblemente redimensionados)
    return patches