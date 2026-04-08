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

# Detectar separador de classpath segun SO
# Windows (Git Bash/MSYS/Cygwin) usa ';' y Unix usa ':'
if [[ "$OSTYPE" == msys* || "$OSTYPE" == cygwin* || "$OSTYPE" == win32* ]]; then
    CP_SEP=";"
else
    CP_SEP=":"
fi

CP_MAIN=".${CP_SEP}lib/*"
CP_BIN=".${CP_SEP}lib/*${CP_SEP}bin"
API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
API_PROJECT_DIR="${API_PROJECT_DIR:-$(cd "$BASE_DIR/.." && pwd)}"
API_START_CMD="${API_START_CMD:-python -m uvicorn app.main:app --host 127.0.0.1 --port 8000}"
API_PID=""

health_check() {
    curl -fsS "${API_BASE_URL}/health" > /dev/null 2>&1
}



# Verificar si existe el directorio lib
if [ ! -d "lib" ]; then
    echo -e "${RED}❌ Error: No existe el directorio 'lib' con las dependencias${NC}"
    echo "Asegúrate de tener las librerías necesarias (org.json, etc.)"
    exit 1
fi

# Verificar si existe el servicio FastAPI
echo -e "${YELLOW}⚠ Verificando servicio FastAPI...${NC}"
if health_check; then
    echo -e "${GREEN}✓ Servicio FastAPI está corriendo${NC}"
else
    echo -e "${YELLOW}⚠ Servicio FastAPI no detectado en ${API_BASE_URL}. Intentando iniciarlo automáticamente...${NC}"
    (
        cd "$API_PROJECT_DIR" && nohup bash -lc "$API_START_CMD" > /tmp/genome_transition_api.log 2>&1 &
        echo $! > /tmp/genome_transition_api.pid
    )
    API_PID="$(cat /tmp/genome_transition_api.pid 2>/dev/null || true)"
    sleep 5

    if health_check; then
        echo -e "${GREEN}✓ Servicio FastAPI iniciado automáticamente en ${API_BASE_URL}${NC}"
        echo "   Logs: /tmp/genome_transition_api.log"
    else
        echo -e "${RED}❌ El servicio FastAPI NO está corriendo en ${API_BASE_URL}${NC}"
        echo "   Intento de arranque falló. Revisa logs: /tmp/genome_transition_api.log"
        echo ""
        echo "   También puedes iniciarlo manualmente con:"
        echo "   cd \"$API_PROJECT_DIR\""
        echo "   $API_START_CMD"
        if [ -t 0 ]; then
            echo ""
            read -p "¿Deseas continuar de todas formas? (s/n): " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Ss]$ ]]; then
                exit 1
            fi
        else
            echo ""
            echo -e "${RED}❌ Entorno no interactivo detectado; abortando sin prompt.${NC}"
            exit 1
        fi
    fi
fi

# Si el script inició FastAPI, detenerlo al finalizar.
cleanup() {
    if [ -n "$API_PID" ] && ps -p "$API_PID" > /dev/null 2>&1; then
        kill "$API_PID" > /dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if ! health_check; then
    echo ""
    echo -e "${RED}❌ No hay conexión con FastAPI en ${API_BASE_URL}. Abortando tests.${NC}"
    exit 1
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
javac -cp "$CP_MAIN" -d bin -sourcepath src src/clasificador/AutoMLClasificador.java

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error al compilar AutoMLClasificador${NC}"
    exit 1
fi

echo "Compilando MiddleWare.java..."
javac -cp "$CP_BIN" -d bin -sourcepath src src/gene/information/GeneConstructor.java

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error al compilar MiddleWare${NC}"
    exit 1
fi

echo "Compilando GeneConstructor.java..."
javac -cp "$CP_BIN" -d bin -sourcepath src src/gene/information/GeneConstructor.java

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error al compilar GeneConstructor${NC}"
    exit 1
fi

echo "Compilando Analizer.java..."
javac -cp "$CP_BIN" -d bin -sourcepath src src/gene/information/Analizer.java

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error al compilar Analizer${NC}"
    exit 1
fi

echo "Compilando TestAutoML.java..."
javac -cp "$CP_BIN" -d bin TestAutoML.java

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
if [[ "$OSTYPE" == darwin* ]]; then
    export SWI_HOME_DIR="/opt/homebrew/Cellar/swi-prolog/9.2.9/lib/swipl"
    export DYLD_LIBRARY_PATH="/opt/homebrew/Cellar/swi-prolog/9.2.9/lib/swipl/lib/arm64-darwin:$DYLD_LIBRARY_PATH"
    JAVA_LIB_PATH="/opt/homebrew/Cellar/swi-prolog/9.2.9/lib/swipl/lib/arm64-darwin"
elif [[ "$OSTYPE" == linux* ]]; then
    # Ajusta estas rutas si tu instalación de SWI-Prolog difiere.
    export SWI_HOME_DIR="${SWI_HOME_DIR:-/usr/lib/swi-prolog}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/lib/swi-prolog/lib/x86_64-linux}:$LD_LIBRARY_PATH"
    JAVA_LIB_PATH="${JAVA_LIB_PATH:-/usr/lib/swi-prolog/lib/x86_64-linux}"
else
    # En Windows/Git Bash normalmente JPL se resuelve por PATH/Java.library.path del sistema.
    JAVA_LIB_PATH="${JAVA_LIB_PATH:-}"
fi
# Ejecutar el test
export CLASSPATH=".${CP_SEP}bin${CP_SEP}lib/*"

# Ejecutar el test
if [ -n "$JAVA_LIB_PATH" ]; then
    java -Djava.library.path="$JAVA_LIB_PATH" -cp "$CLASSPATH" TestAutoML
    java -Djava.library.path="$JAVA_LIB_PATH" -cp "$CLASSPATH" TestIntegracionAutoML 2>&1
else
    java -cp "$CLASSPATH" TestAutoML
    java -cp "$CLASSPATH" TestIntegracionAutoML 2>&1
fi


if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Test ejecutado correctamente${NC}"
else
    echo ""
    echo -e "${RED}❌ El test falló${NC}"
    exit 1
fi
