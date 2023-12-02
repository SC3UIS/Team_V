/*
=================================================================================
|                          AUTORES DEL CÓDIGO                                  |
=================================================================================
| - Levir Heladio Hernandez Suarez                                             |
| - Manuel Alejandro Herrera Poveda                                            |
| - Santiago Andres Bolaños Cruz                                               |
|                                                                              |
| Este proyecto se enfoca en el procesamiento de imágenes en paralelo,         |
| realizando tareas como la conversión a escala de grises, ecualización del    |
| histograma y detección de bordes mediante técnicas de computación en         |
| paralelo, específicamente utilizando el paradigma de memoria compartida      |
| con OpenMP.                                                                  |
=================================================================================
*/

include <opencv2/opencv.hpp>
#include <functional>
#include <filesystem>
#include <fstream>
#include <chrono>

namespace fs = std::filesystem;

// Función para cargar la imagen de entrada
cv::Mat loadImage(const std::string& filename)
{
    cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (image.empty())
    {
        std::cerr << "Error cargando la imagen." << std::endl;
        exit(EXIT_FAILURE);
    }
    return image;
}

// Función para crear directorios de salida
void createDirectories()
{
    // Define los directorios de salida
    std::vector<std::string> directories = {
        "output/Secuencial",
        "output/OpenMP"
    };
    // Crea directorios
    for (const auto& dir : directories)
    {
        if (!fs::create_directories(dir))
        {
            std::cerr << "Error al crear el directorio: " << dir << std::endl;
            // Manejar el error según sea necesario
        }
    }
}

// Función para convertir la imagen a escala de grises de forma manual
cv::Mat convertToGrayManual(const cv::Mat& inputImage)
{
    cv::Mat grayImage(inputImage.size(), CV_8UC1);
    for (int i = 0; i < inputImage.rows; ++i)
    {
        for (int j = 0; j < inputImage.cols; ++j)
        {
            uchar b = inputImage.at<cv::Vec3b>(i, j)[0];
            uchar g = inputImage.at<cv::Vec3b>(i, j)[1];
            uchar r = inputImage.at<cv::Vec3b>(i, j)[2];
            grayImage.at<uchar>(i, j) = static_cast<uchar>(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
    return grayImage;
}

// Función para calcular la ecualización del histograma de forma manual
cv::Mat equalizeHistogramManual(const cv::Mat& inputImage)
{
    cv::Mat outputImage(inputImage.size(), CV_8UC1);
    // Calcular el histograma
    std::vector<int> histogram(256, 0);
    for (int i = 0; i < inputImage.rows; ++i)
    {
        const uchar* row = inputImage.ptr<uchar>(i);
        for (int j = 0; j < inputImage.cols; ++j)
        {
            histogram[row[j]]++;
        }
    }
    // Calcular la función de transformación acumulativa
    std::vector<int> cumulativeHistogram(256, 0);
    cumulativeHistogram[0] = histogram[0];
    for (int i = 1; i < 256; ++i)
    {
        cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i];
    }
    // Normalizar la función de transformación acumulativa
    std::vector<uchar> transformation(256, 0);
    for (int i = 0; i < 256; ++i)
    {
        transformation[i] = static_cast<uchar>(255 * cumulativeHistogram[i] / (inputImage.rows * inputImage.cols));
    }
    // Aplicar la transformación a cada píxel
    for (int i = 0; i < inputImage.rows; ++i)
    {
        const uchar* row = inputImage.ptr<uchar>(i);
        uchar* outputRow = outputImage.ptr<uchar>(i);
        for (int j = 0; j < inputImage.cols; ++j)
        {
            outputRow[j] = transformation[row[j]];
        }
    }
    return outputImage;
}

// Función para realizar detección de bordes utilizando el operador Sobel de forma manual
cv::Mat detectEdgesSobelNormalizedManual(const cv::Mat& inputImage)
{
    cv::Mat edgesImage(inputImage.size(), CV_8UC1);
    // Definir el kernel Sobel en dirección x e y
    int kernelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int kernelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    // Aplicar el operador Sobel de forma manual y normalizar
    for (int i = 1; i < inputImage.rows - 1; ++i)
    {
        for (int j = 1; j < inputImage.cols - 1; ++j)
        {
            int gx = 0, gy = 0;
            for (int u = -1; u <= 1; ++u)
            {
                for (int v = -1; v <= 1; ++v)
                {
                    int pixelValue = static_cast<int>(inputImage.at<uchar>(i + u, j + v));
                    gx += kernelX[u + 1][v + 1] * pixelValue;
                    gy += kernelY[u + 1][v + 1] * pixelValue;
                }
            }
            double magnitude = sqrt(gx * gx + gy * gy);
            edgesImage.at<uchar>(i, j) = static_cast<uchar>(magnitude / 4); // Normalización manual
        }
    }
    return edgesImage;
}

// Función para convertir la imagen a escala de grises con OpenMP
cv::Mat convertToGrayOPENMP(const cv::Mat& inputImage)
{
    cv::Mat grayImage(inputImage.size(), CV_8UC1);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < inputImage.rows; ++i)
    {
        for (int j = 0; j < inputImage.cols; ++j)
        {
            uchar b = inputImage.at<cv::Vec3b>(i, j)[0];
            uchar g = inputImage.at<cv::Vec3b>(i, j)[1];
            uchar r = inputImage.at<cv::Vec3b>(i, j)[2];
            grayImage.at<uchar>(i, j) = static_cast<uchar>(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
    return grayImage;
}

// Función para calcular la ecualización del histograma con OpenMP
cv::Mat equalizeHistogramOPENMP(const cv::Mat& inputImage)
{
    cv::Mat outputImage(inputImage.size(), CV_8UC1);
    // Calcular el histograma
    std::vector<int> histogram(256, 0);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < inputImage.rows; ++i)
    {
        const uchar* row = inputImage.ptr<uchar>(i);
        for (int j = 0; j < inputImage.cols; ++j)
        {
            histogram[row[j]]++;
        }
    }
    // Calcular la función de transformación acumulativa
    std::vector<int> cumulativeHistogram(256, 0);
    cumulativeHistogram[0] = histogram[0];
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < 256; ++i)
    {
        cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i];
    }
    // Normalizar la función de transformación acumulativa
    std::vector<uchar> transformation(256, 0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < 256; ++i)
    {
        transformation[i] = static_cast<uchar>(255 * cumulativeHistogram[i] / (inputImage.rows * inputImage.cols));
    }
    // Aplicar la transformación a cada píxel
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < inputImage.rows; ++i)
    {
        const uchar* row = inputImage.ptr<uchar>(i);
        uchar* outputRow = outputImage.ptr<uchar>(i);
        for (int j = 0; j < inputImage.cols; ++j)
        {
            outputRow[j] = transformation[row[j]];
        }
    }
    return outputImage;
}

// Función para realizar detección de bordes utilizando el operador Sobel con normalización con OpenMP
cv::Mat detectEdgesSobelNormalizedOPENMP(const cv::Mat& inputImage)
{
    cv::Mat edgesImage(inputImage.size(), CV_8UC1);
    // Definir el kernel Sobel en dirección x e y
    int kernelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int kernelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < inputImage.rows - 1; ++i)
    {
        for (int j = 1; j < inputImage.cols - 1; ++j)
        {
            int gx = 0, gy = 0;
            for (int u = -1; u <= 1; ++u)
            {
                for (int v = -1; v <= 1; ++v)
                {
                    int pixelValue = static_cast<int>(inputImage.at<uchar>(i + u, j + v));
                    gx += kernelX[u + 1][v + 1] * pixelValue;
                    gy += kernelY[u + 1][v + 1] * pixelValue;
                }
            }
            double magnitude = sqrt(gx * gx + gy * gy);
            edgesImage.at<uchar>(i, j) = static_cast<uchar>(magnitude / 4); // Normalización manual
        }
    }
    return edgesImage;
}

// Función para medir el tiempo de ejecución de una función
cv::Mat measureExecutionTime(const std::function<cv::Mat(const cv::Mat&)>& func, const cv::Mat& inputImage, const std::string& functionName)
{
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result = func(inputImage);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::ofstream informeFile("INFORME.txt", std::ios_base::app); // Modo de agregar
    if (informeFile.is_open())
    {
        // Guardar tiempos en el archivo
        informeFile << "Tiempo " << functionName << ": " << duration.count() << " ms\n";
        informeFile.close();
    }
    else
    {
        std::cerr << "No se pudo abrir el archivo INFORME.txt\n";
    }
    
    return result;
}

// Función para procesar una imagen con un sufijo dado
void processImage(const std::string& fileSuffix)
{
    // Construir el nombre del archivo con el sufijo proporcionado
    std::string inputFileName = "input" + fileSuffix + ".png";

    // CARGAR IMAGEN
    cv::Mat inputImage = loadImage(inputFileName);

    // APLICAR FILTROS
    cv::Mat grayImageManual = measureExecutionTime(convertToGrayManual, inputImage, "convertToGrayManual" + fileSuffix);
    cv::Mat outputImageManual = measureExecutionTime(equalizeHistogramManual, grayImageManual, "equalizeHistogramManual" + fileSuffix);
    cv::Mat edgesImageManual = measureExecutionTime(detectEdgesSobelNormalizedManual, grayImageManual, "detectEdgesSobelNormalizedManual" + fileSuffix);

    cv::Mat grayImageOPENMP = measureExecutionTime(convertToGrayOPENMP, inputImage, "convertToGrayOPENMP" + fileSuffix);
    cv::Mat outputImageOPENMP = measureExecutionTime(equalizeHistogramOPENMP, grayImageOPENMP, "equalizeHistogramOPENMP" + fileSuffix);
    cv::Mat edgesImageOPENMP = measureExecutionTime(detectEdgesSobelNormalizedOPENMP, grayImageOPENMP, "detectEdgesSobelNormalizedOPENMP" + fileSuffix);

    // GUARDAR FILTROS

    cv::imwrite("output/Secuencial/output_gray" + fileSuffix + ".png", grayImageManual);
    cv::imwrite("output/Secuencial/output_equalized" + fileSuffix + ".png", outputImageManual);
    cv::imwrite("output/Secuencial/output_edges" + fileSuffix + ".png", edgesImageManual);

    cv::imwrite("output/OpenMP/output_gray" + fileSuffix + ".png", grayImageOPENMP);
    cv::imwrite("output/OpenMP/output_equalized" + fileSuffix + ".png", outputImageOPENMP);
    cv::imwrite("output/OpenMP/output_edges" + fileSuffix + ".png", edgesImageOPENMP);
}

// Función principal
int main()
{
    createDirectories();
    processImage("_4K");
    processImage("_2K");
    processImage("_HD");
    return 0;
}

