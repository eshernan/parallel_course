# Ficha de compilación y ejecución en clúster

Cada ejemplo o ejercicio avanzado incluye esta ficha. Los valores se diligencian para el clúster institucional en el que se realizó la práctica; los nombres que siguen corresponden a campos de información y no a módulos o particiones universales.

## Manifiesto obligatorio

| Campo | Valor que debe registrar el material |
|---|---|
| Sistema | Distribución, versión, kernel y arquitectura de host. |
| Compilador | Nombre, versión, ruta y estándar C++ utilizado. |
| Framework/API | ROCm/HIP, implementación SYCL, Kokkos o RAJA y versión. |
| Dependencias | Versiones y rutas CMake de bibliotecas requeridas. |
| Backend | CPU/OpenMP, HIP, CUDA, SYCL u otro backend probado. |
| GPU | Modelo, arquitectura `gfx*`/`sm_*`, memoria y driver/runtime. |
| Scheduler | Slurm u otro planificador y versión. |
| Cola | Partición/QoS/cuenta autorizadas para el ejercicio. |
| Recursos | Nodos, tareas, CPU por tarea, GPU, memoria y tiempo límite. |
| Módulos | Secuencia exacta de `module load` o activación equivalente. |
| Afinidad | Mapeo y binding de procesos/hilos cuando aplique. |
| Construcción | Preset o comando CMake completo y directorio de build. |
| Ejecución | Comando interactivo y/o script por lotes. |
| Evidencia | Salida de inventario, log, CSV/JSON y versión del commit. |

## Contrato por tecnología

### ROCm/HIP

La práctica requiere un nodo AMD compatible, `amdclang++` o `hipcc`, el runtime HIP, una arquitectura `gfx*` identificada y acceso autorizado a las herramientas de perfilado. El script solicita de manera explícita una GPU AMD, pues `--gres=gpu:1` no selecciona necesariamente fabricante o modelo en todos los clústeres.

### SYCL

El manifiesto identifica la implementación y el backend. La selección de dispositivo se comprueba durante la ejecución y queda registrada en el log. En una práctica que exige GPU, el fallback en CPU indica un error de configuración.

### Kokkos

El manifiesto describe cómo se construyó Kokkos: backend de host, backend de dispositivo, arquitectura y compilador. `Kokkos_DIR` referencia esa instalación específica. Las instalaciones CPU, HIP y CUDA se administran como perfiles distintos.

### RAJA

El manifiesto registra los backends habilitados durante la construcción de RAJA, `RAJA_DIR` y las dependencias asociadas. En GPU también se informa el método de gestión de memoria y las versiones de ROCm/rocPRIM o CUDA/CUB.

## Validación previa

Antes de enviar un trabajo de larga duración se ejecuta una prueba pequeña que:

1. imprima el dispositivo y backend seleccionados;
2. compruebe que el recurso solicitado está visible;
3. ejecute la referencia y la variante paralela;
4. compare resultados con tolerancia declarada;
5. falle con código distinto de cero si alguna comprobación no se cumple.

Los scripts utilizados en el clúster se almacenan junto con el ejercicio. Las plantillas genéricas se identifican como tales y se indican como verificadas cuando han sido probadas en una infraestructura concreta.
