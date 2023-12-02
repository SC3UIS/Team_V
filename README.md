# Team V: Procesamiento de imágenes en paralelo
## Contexto del Proyecto
Este proyecto se enfoca en el procesamiento de imágenes en paralelo, realizando tareas como la conversión a escala de grises, ecualización del histograma y detección de bordes mediante técnicas de computación en paralelo, específicamente utilizando el paradigma de memoria compartida con OpenMP. Las pruebas se llevaron a cabo en imágenes de tres resoluciones diferentes: 4K (8,294,000 píxeles), 2K (3,686,400 píxeles) y HD (2,073,600 píxeles).

## Estructura del Proyecto
La estructura del proyecto consta de tres imágenes de distintas resoluciones, un archivo "readme.txt" y el código fuente en C++ denominado "equalizer.cpp," que contiene la implementación de las funciones correspondientes. La biblioteca OpenCV se emplea únicamente para la lectura de imágenes en diversos formatos. Tras la ejecución del programa, se genera una carpeta "output" con dos subdirectorios, "secuencial" y "OpenMP," que contienen las imágenes resultantes de la aplicación de filtros. Además, se crea un archivo de texto llamado "INFORME" que almacena los tiempos de ejecución de cada función.

## Funciones Paralelizadas y Mejoras con OpenMP
### ConvertToGray
*	**convertToGrayManual:** Esta función convierte una imagen a escala de grises de manera secuencial, recorriendo los píxeles mediante dos bucles anidados y aplicando una fórmula de conversión.
*	**convertToGrayOpenMP:** La versión paralela utiliza la directiva #pragma omp parallel for para crear un bloque de código paralelo, generando un equipo de hilos que ejecutan el bucle de manera simultánea. La cláusula collapse(2) combina eficientemente los dos bucles anidados, permitiendo que cada hilo maneje múltiples iteraciones del bucle combinado. La cláusula schedule(static) asigna estáticamente un conjunto de iteraciones a cada hilo, mejorando la eficiencia.

### EqualizeHistogram
*	**equalizeHistogramManual:** Realiza la ecualización del histograma de una imagen de manera secuencial, calculando el histograma, generando una función de transformación acumulativa y aplicándola a cada píxel.
*	**equalizeHistogramOpenMP:** La versión paralela utiliza #pragma omp parallel for collapse(2) para establecer un entorno paralelo y combinar eficientemente los dos bucles anidados. La cláusula schedule(static) asigna estáticamente un conjunto de iteraciones a cada hilo, optimizando la distribución de carga de trabajo.

### DetectEdgesSobelNormalized
*	**detectEdgesSobelNormalizedManual:** Aplica la detección de bordes con el operador Sobel de manera secuencial, calculando la convolución, la magnitud y normalizando los resultados.
*	**detectEdgesSobelNormalizedOpenMP:** La versión paralela utiliza #pragma omp parallel for collapse(2) para crear un entorno paralelo y schedule(static) para asignar estáticamente un conjunto de iteraciones. Ambas cláusulas optimizan la paralelización.

## Entorno de ejecución
Las pruebas se llevaron a cabo en un portátil Intel Core de 3.40 GHz con 8 núcleos x64, ejecutando Ubuntu 22.04.3 LTS. La elección de un sistema Linux se basa en la facilidad de instalación y configuración de la librería OpenCV.

## Análisis de resultados
La paralelización con OpenMP proporciona mejoras notables en tareas intensivas en cómputo, como la conversión a escala de grises y la ecualización de histograma. Sin embargo, la detección de bordes muestra resultados mixtos, indicando que no todas las tareas son igualmente paralelizables. Se observa una mejora general en el rendimiento al aumentar la resolución de la imagen, respaldando la eficacia de la paralelización en situaciones más demandantes.

## Tiempos de Ejecución
### Resolución 4K:
*	convertToGrayManual: 172 ms
*	equalizeHistogramManual: 70 ms
*	detectEdgesSobelNormalizedManual: 625 ms
### Resolución 4K con OpenMP:
*	convertToGrayOpenMP: 161 ms
*	equalizeHistogramOpenMP: 57 ms
*	detectEdgesSobelNormalizedOpenMP: 617 ms
### Resolución 2K:
*	convertToGrayManual: 82 ms
*	equalizeHistogramManual: 29 ms
*	detectEdgesSobelNormalizedManual: 287 ms
### Resolución 2K con OpenMP:
*	convertToGrayOpenMP: 78 ms
*	equalizeHistogramOpenMP: 27 ms
*	detectEdgesSobelNormalizedOpenMP: 292 ms
### Resolución HD:
*	convertToGrayManual: 44 ms
*	equalizeHistogramManual: 16 ms
*	detectEdgesSobelNormalizedManual: 188 ms
### Resolución HD con OpenMP:
*	convertToGrayOpenMP: 46 ms
*	equalizeHistogramOpenMP: 14 ms
*	detectEdgesSobelNormalizedOpenMP: 166 ms

## Recomendaciones
Si se implementa la función de detección de bordes de otra forma, podría paralelizarse de manera más eficaz, potencialmente logrando mejoras significativas en su rendimiento.
