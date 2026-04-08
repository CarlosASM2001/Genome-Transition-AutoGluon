#!/bin/bash

# Script para compilar y ejecutar el test de AutoML

echo "================================================================"
echo "  Compilando y ejecutando Test AutoML"
echo "================================================================"
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directorio base
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

echo "Directorio de trabajo: $BASE_DIR"
echo ""

# Verificar si existe el directorio lib
if [ ! -d "lib" ]; then
    echo -e "${RED}❌ Error: No existe el directorio 'lib' con las dependencias${NC}"
    echo "Asegúrate de tener las librerías necesarias (org.json, etc.)"
    exit 1
fi

# Verificar si existe el servicio FastAPI
echo -e "${YELLOW}⚠ Verificando servicio FastAPI...${NC}"
if curl -s http://143.198.77.77:8000 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Servicio FastAPI está corriendo${NC}"
else
    echo -e "${RED}❌ El servicio FastAPI NO está corriendo en http://143.198.77.77:8000${NC}"
    echo ""
    echo "Por favor, inicia el servicio primero:"
    echo "  cd /ruta/al/servicio"
    echo "  python -m uvicorn main:app --reload"
    echo ""
    read -p "¿Deseas continuar de todas formas? (s/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Paso 1: Compilando clases Java"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Crear directorio de salida si no existe
mkdir -p bin

# Compilar las clases necesarias
echo "Compilando AutoMLClasificador.java..."
javac -cp ".:lib/*" -d bin -sourcepath src src/clasificador/AutoMLClasificador.java

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error al compilar AutoMLClasificador${NC}"
    exit 1
fi

echo "Compilando MiddleWare.java..."
javac -cp ".:lib/*:bin" -d bin -sourcepath src src/util/MiddleWare.java

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error al compilar MiddleWare${NC}"
    exit 1
fi

echo "Compilando GeneConstructor.java..."
javac -cp ".:lib/*:bin" -d bin -sourcepath src src/gene/information/GeneConstructor.java

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error al compilar GeneConstructor${NC}"
    exit 1
fi

echo "Compilando Analizer.java..."
javac -cp ".:lib/*:bin" -d bin -sourcepath src src/gene/information/Analizer.java

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error al compilar Analizer${NC}"
    exit 1
fi

echo "Compilando TestAutoML.java..."
javac -cp ".:lib/*:bin" -d bin TestAutoML.java

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error al compilar TestAutoML${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Compilación exitosa${NC}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Paso 2: Ejecutando test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Configurar variables de entorno para SWI-Prolog y JPL
export SWI_HOME_DIR="/opt/homebrew/Cellar/swi-prolog/9.2.9/lib/swipl"
export DYLD_LIBRARY_PATH="/opt/homebrew/Cellar/swi-prolog/9.2.9/lib/swipl/lib/arm64-darwin:$DYLD_LIBRARY_PATH"
export CLASSPATH=".:bin:lib/*"

# Ejecutar el test
java -Djava.library.path="/opt/homebrew/Cellar/swi-prolog/9.2.9/lib/swipl/lib/arm64-darwin" -cp "$CLASSPATH" TestAutoML

java -Djava.library.path="/opt/homebrew/Cellar/swi-prolog/9.2.9/lib/swipl/lib/arm64-darwin" -cp "$CLASSPATH" TestIntegracionAutoML 2>&1


if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Test ejecutado correctamente${NC}"
else
    echo ""
    echo -e "${RED}❌ El test falló${NC}"
    exit 1
fi
