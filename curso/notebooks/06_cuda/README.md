# Notebooks del tema 06: CUDA C++

El módulo CUDA tiene seis sesiones y un máximo de tres notebooks:

1. `01_modelo_cuda.ipynb`: toolchain CUDA 13, modelo SIMT, grid/bloque/hilo, primer kernel, memoria, transferencias y manejo de errores.
2. `02_memoria_tiling.ipynb`: coalescencia, memoria compartida, bancos, tiles, multiplicación de matrices, reducciones y primitivas de warp.
3. `03_bibliotecas_perfiles.ipynb`: Thrust/CUB, cuBLAS/cuBLASLt, cuFFT, cuSPARSE, cuSOLVER y cuRAND; streams, ocupación, Nsight y decisión biblioteca frente a kernel propio.

Cada notebook deberá ejecutar fuentes de `curso/ejemplos/06_cuda/`, comprobar una referencia CPU y construir sus gráficas desde datos exportados.

## Alcance conceptual obligatorio

Antes de optimizar, el estudiante debe poder explicar:

- diferencia entre host, device, kernel, thread, warp, bloque, grid y stream;
- ejecución SIMT, divergencia y sincronización;
- memoria global, compartida, local, constante, registrada (*pinned*) y unificada;
- coalescencia, latencia, ancho de banda, ocupación e intensidad aritmética;
- relación entre forma del bloque, tile, reutilización de datos, halo y tratamiento de bordes;
- diferencia entre tiempo de kernel y tiempo extremo a extremo;
- asincronía, eventos, errores diferidos y vida útil de memoria, handles, planes y descriptores;
- precisión, orden de operaciones, determinismo y tolerancia numérica.

El tile se presenta como una estrategia de descomposición, no como un tamaño mágico. Se comparará una multiplicación de matrices ingenua con una versión por tiles que use memoria compartida, sincronización y límites correctos para dimensiones no múltiplos del tile. La práctica mostrará cuándo aumenta la reutilización y cuándo el tamaño elegido perjudica ocupación, registros o memoria compartida.

## Manera de trabajar con bibliotecas aceleradas

Todas las bibliotecas seguirán el mismo ciclo para hacer visibles sus costos:

```text
identificar operación → comprobar layout/tipo → crear contexto o plan
     → consultar/asignar workspace → asociar stream → ejecutar
     → comprobar estado → sincronizar donde corresponda → validar → liberar
```

Se medirán por separado:

1. preparación de datos, conversión de formato y transferencias;
2. creación de handle, descriptor, plan o workspace;
3. ejecución repetida con datos residentes en GPU;
4. tiempo extremo a extremo y error frente a la referencia.

Una biblioteca ofrece ventaja cuando el problema coincide con una primitiva optimizada, hay suficiente trabajo o un batch adecuado, los datos permanecen en GPU y se reutilizan planes/workspaces. Puede no ofrecer ventaja para entradas pequeñas, una única llamada rodeada de transferencias, conversiones de formato costosas o una operación especializada que pueda fusionarse en un kernel sencillo.

## Bibliotecas que se cubrirán

| Biblioteca | Función | Uso mínimo que se enseñará | Cuándo suele convenir |
|---|---|---|---|
| Thrust 3.0.1 | Algoritmos paralelos C++ de alto nivel | `device_vector`, política de ejecución, `transform`, `reduce`, `scan` y `sort` | Prototipos y pipelines expresables como algoritmos estándar, cuando claridad y composición pesan más que controlar cada warp. |
| CUB 3.0.1 | Primitivas jerárquicas thread/warp/block/device | `BlockReduce` y `DeviceReduce` con consulta y reutilización de almacenamiento temporal | Reducciones, scans, selección e histogramas que requieren más control y rendimiento que Thrust sin reimplementar primitivas complejas. |
| cuBLAS/cuBLASLt | BLAS denso y multiplicación de matrices | crear handle, fijar stream, revisar layout/leading dimension, ejecutar GEMM, validar y destruir | Matrices medianas/grandes o batches, datos residentes y tipos soportados; cuBLASLt cuando se requieren heurísticas, precisiones o epílogos/fusión. |
| cuFFT | Transformadas de Fourier 1D y batched | crear/reutilizar plan, revisar workspace, ejecutar R2C/C2R y normalizar el resultado | Transformadas medianas/grandes o repetidas; la creación del plan y las transferencias pueden dominar casos pequeños o de una sola ejecución. |
| cuSPARSE | Álgebra lineal dispersa | construir descriptores CSR y vectores, consultar buffer, ejecutar SpMV y liberar recursos | Matrices grandes con suficiente dispersión y estructura favorable; la conversión de formato y el bajo paralelismo pueden anular la ventaja. |
| cuSOLVER | Factorizaciones y solución de sistemas densos | consultar workspace, factorizar/solucionar un sistema y comprobar `info` | Problemas densos de costo alto y datos ya residentes; no sustituye validar condicionamiento, singularidad o precisión. |
| cuRAND | Generación pseudo/cuasi-aleatoria en host o device | crear generador, fijar semilla/offset/stream, generar y destruir; comparar con estado por hilo | Monte Carlo o grandes secuencias consumidas en GPU; inicializar estados o transferir muestras puede dominar trabajos pequeños. |

cuBLAS y CUB/Thrust tendrán práctica obligatoria. cuFFT y cuSPARSE tendrán ejemplos guiados. cuSOLVER y cuRAND se estudiarán mediante casos representativos y criterios de selección. No se pretende memorizar las APIs completas.

En CUDA 13.0 se utilizará la API genérica de cuSPARSE; las APIs legacy no se enseñarán. `cuSOLVERSp` y `cuSOLVERRf` están deprecadas, por lo que el curso no construirá material nuevo sobre ellas. cuDSS se mencionará como ruta moderna para solucionadores dispersos directos, fuera del alcance obligatorio.

## Construcción

Los ejemplos enlazarán targets importados de CMake, según corresponda:

```cmake
target_link_libraries(ejemplo PRIVATE
  course_cuda
  CUDA::cublas CUDA::cufft CUDA::cusparse
  CUDA::cusolver CUDA::curand)
```

Thrust y CUB son cabeceras de CUDA Core Compute Libraries incluidas con el Toolkit. El código debe comprobar tanto errores del runtime como estados devueltos por cada biblioteca.

## Gráficas explicativas

Se usarán gráficas con una sola pregunta y pocas series:

- naïve frente a tiled frente a cuBLAS para GEMM, separando ejecución y extremo a extremo;
- ancho de banda efectivo frente a patrón coalescente/no coalescente;
- CUB/Thrust frente a reducción propia correcta para varios tamaños;
- cuFFT con y sin reutilización de plan;
- SpMV frente a densidad/formato, sin generalizar desde una sola matriz;
- transferencia, preparación y cómputo como componentes separados del tiempo.

Cada comparación utilizará el mismo problema, precisión, tolerancia y dispositivo. Una aceleración de biblioteca no se atribuirá únicamente al algoritmo: se explicarán también layout, precisión, Tensor Cores cuando apliquen, fusión, workspace y residencia de datos.
