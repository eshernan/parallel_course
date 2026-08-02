# Ficha de compilación y ejecución en clúster

Todo ejemplo o ejercicio avanzado debe completar esta ficha. Los valores dependen del clúster institucional; los nombres siguientes son campos, no módulos o particiones universales.

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

Requiere un nodo AMD compatible, `amdclang++` o `hipcc`, runtime HIP, arquitectura `gfx*` y herramientas de perfilado autorizadas. El script debe solicitar una GPU AMD explícitamente; no se asume que `--gres=gpu:1` seleccione fabricante o modelo en todos los clústeres.

### SYCL

Declara implementación y backend. La selección de dispositivo debe comprobarse en tiempo de ejecución y guardarse en el log. Si el ejercicio exige GPU, un fallback CPU se considera error de configuración, no una ejecución válida.

### Kokkos

Declara cómo fue construido Kokkos: backend host, backend device, arquitectura y compilador. `Kokkos_DIR` debe referirse a esa instalación exacta. Una instalación CPU y una instalación HIP/CUDA se consideran perfiles distintos.

### RAJA

Declara los backends habilitados al construir RAJA, `RAJA_DIR` y las dependencias asociadas. Para GPU se registra además el método de gestión de memoria y las versiones de ROCm/rocPRIM o CUDA/CUB.

## Validación previa

Antes de enviar una carga larga, el ejercicio debe ejecutar una prueba pequeña que:

1. imprima el dispositivo y backend seleccionados;
2. compruebe que el recurso solicitado está visible;
3. ejecute la referencia y la variante paralela;
4. compare resultados con tolerancia declarada;
5. falle con código distinto de cero si alguna comprobación no se cumple.

Los scripts reales del clúster se almacenarán junto al ejercicio. No se publicará una plantilla con nombres ficticios como si estuviera certificada para una infraestructura concreta.
