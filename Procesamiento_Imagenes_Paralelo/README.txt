INSTRUCCIONES DE COMPILACION DEL PROYECTO
Para poder compilar y ejecutar este proyecto desde un entorno linux se requiere la instalacion de OpenCV y los compiladores GNU.

VERIFICA CUALES NO TIENES INSTALADAS

g++ --version
gcc --version
dpkg -l | grep libopencv-dev

INSTALA LAS NECESARIAS

sudo apt update
sudo apt install build-essential
sudo apt install libopencv-dev

VERIFICA LA INSTALACION

g++ --version
gcc --version
dpkg -l | grep libopencv-dev

COMPILA EL CODIGO FUENTE Y CORRE EL EJECUTABLE

g++ equalizer.cpp -o equalizer.exe `pkg-config --cflags --libs opencv4`
./equalizer.exe
