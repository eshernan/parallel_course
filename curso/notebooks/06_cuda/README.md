# Notebooks del tema 06: CUDA C++

El módulo CUDA ocupa seis sesiones y se desarrolla en tres notebooks ejecutables:

1. [`01_modelo_cuda.ipynb`](01_modelo_cuda.ipynb): toolchain CUDA 13, modelo SIMT, grid/bloque/hilo, primer kernel, memoria, transferencias y manejo de errores.
2. [`02_memoria_tiling.ipynb`](02_memoria_tiling.ipynb): coalescencia, memoria compartida, bancos, tiles, multiplicación de matrices, reducciones y primitivas de warp.
3. [`03_bibliotecas_perfiles.ipynb`](03_bibliotecas_perfiles.ipynb): Thrust/CUB, cuBLAS/cuBLASLt, cuFFT, cuSPARSE, cuSOLVER y cuRAND; streams, ocupación, Nsight y decisión biblioteca frente a kernel propio.

[Tema anterior: OpenMP target](../05_openmp_target/README.md) · [Volver al índice del curso](../../../INDICE_CURSO.md) · [Siguiente tema: programación híbrida](../07_hibrido/README.md)

Los notebooks ejecutan modelos analíticos portables y definen las prácticas, invariantes y criterios de evidencia del módulo. `curso/ejemplos/06_cuda/` documenta el inventario previsto, pero todavía no contiene sus fuentes compilables; por tanto, esta edición no presenta esos programas ni resultados de GPU como si ya existieran. Cuando las fuentes se incorporen, deberán compararse con una referencia en CPU y exportar los datos con los que se elaboren las gráficas.

## Alcance conceptual obligatorio

La discusión conceptual cubre los siguientes aspectos antes de entrar en optimizaciones específicas:

- diferencia entre host, device, kernel, thread, warp, bloque, grid y stream;
- ejecución SIMT, divergencia y sincronización;
- memoria global, compartida, local, constante, registrada (*pinned*) y unificada;
- coalescencia, latencia, ancho de banda, ocupación e intensidad aritmética;
- relación entre forma del bloque, tile, reutilización de datos, halo y tratamiento de bordes;
- diferencia entre tiempo de kernel y tiempo extremo a extremo;
- asincronía, eventos, errores diferidos y vida útil de memoria, handles, planes y descriptores;
- precisión, orden de operaciones, determinismo y tolerancia numérica.

El trabajo por mosaicos se estudia como una estrategia de descomposición y reutilización de datos. La práctica compara una multiplicación de matrices directa con una versión que emplea memoria compartida, sincronización y manejo correcto de dimensiones que no son múltiplos del tamaño del mosaico. A partir de las mediciones se discute el efecto del tamaño elegido sobre la ocupación, el uso de registros y la memoria compartida.

## Manera de trabajar con bibliotecas aceleradas

Para comparar las bibliotecas con el mismo criterio se utiliza el siguiente ciclo de trabajo:

```text
identificar operación → comprobar layout/tipo → crear contexto o plan
     → consultar/asignar workspace → asociar stream → ejecutar
     → comprobar estado → sincronizar donde corresponda → validar → liberar
```

El informe presenta por separado:

1. preparación de datos, conversión de formato y transferencias;
2. creación de handle, descriptor, plan o workspace;
3. ejecución repetida con datos residentes en GPU;
4. tiempo extremo a extremo y error frente a la referencia.

Las bibliotecas resultan especialmente útiles cuando la operación coincide con una primitiva optimizada, el tamaño o el lote aprovecha el dispositivo y los datos permanecen en la GPU durante varias llamadas. En problemas pequeños, o cuando cada llamada exige transferencias y conversiones de formato, esos costos pueden superar el tiempo de cómputo. Una operación muy específica también puede beneficiarse de la fusión en un kernel propio.

## Bibliotecas incluidas en el alcance académico

| Biblioteca | Función | Uso mínimo previsto para la práctica compilable | Cuándo suele convenir |
|---|---|---|---|
| Thrust 3.0.1 | Algoritmos paralelos C++ de alto nivel | `device_vector`, política de ejecución, `transform`, `reduce`, `scan` y `sort` | Prototipos y pipelines expresables como algoritmos estándar, cuando claridad y composición pesan más que controlar cada warp. |
| CUB 3.0.1 | Primitivas jerárquicas thread/warp/block/device | `BlockReduce` y `DeviceReduce` con consulta y reutilización de almacenamiento temporal | Reducciones, scans, selección e histogramas que requieren más control y rendimiento que Thrust sin reimplementar primitivas complejas. |
| cuBLAS/cuBLASLt | BLAS denso y multiplicación de matrices | crear handle, fijar stream, revisar layout/leading dimension, ejecutar GEMM, validar y destruir | Matrices medianas/grandes o batches, datos residentes y tipos soportados; cuBLASLt cuando se requieren heurísticas, precisiones o epílogos/fusión. |
| cuFFT | Transformadas de Fourier 1D y batched | crear/reutilizar plan, revisar workspace, ejecutar R2C/C2R y normalizar el resultado | Transformadas medianas/grandes o repetidas; la creación del plan y las transferencias pueden dominar casos pequeños o de una sola ejecución. |
| cuSPARSE | Álgebra lineal dispersa | construir descriptores CSR y vectores, consultar buffer, ejecutar SpMV y liberar recursos | Matrices grandes con suficiente dispersión y estructura favorable; la conversión de formato y el bajo paralelismo pueden anular la ventaja. |
| cuSOLVER | Factorizaciones y solución de sistemas densos | consultar workspace, factorizar/solucionar un sistema y comprobar `info` | Problemas densos de costo alto y datos ya residentes; no sustituye validar condicionamiento, singularidad o precisión. |
| cuRAND | Generación pseudo/cuasi-aleatoria en host o device | crear generador, fijar semilla/offset/stream, generar y destruir; comparar con estado por hilo | Monte Carlo o grandes secuencias consumidas en GPU; inicializar estados o transferir muestras puede dominar trabajos pequeños. |

cuBLAS y CUB/Thrust forman parte de la práctica evaluada. cuFFT y cuSPARSE se trabajan mediante ejemplos guiados. Para cuSOLVER y cuRAND se estudian casos representativos y criterios de selección. El propósito es comprender el patrón de integración de estas bibliotecas y consultar con solvencia su documentación, en lugar de memorizar las interfaces completas.

Los ejemplos de cuSPARSE emplean la API genérica disponible en CUDA 13.0. Las interfaces históricas se consultan únicamente para interpretar código existente. Debido a la deprecación de `cuSOLVERSp` y `cuSOLVERRf`, no se desarrollan ejercicios nuevos con esas APIs; cuDSS se presenta como referencia actual para solucionadores dispersos directos, por fuera del alcance evaluado.

## Construcción

Los ejemplos enlazan los targets importados de CMake que correspondan a cada biblioteca:

```cmake
target_link_libraries(ejemplo PRIVATE
  course_cuda
  CUDA::cublas CUDA::cufft CUDA::cusparse
  CUDA::cusolver CUDA::curand)
```

Thrust y CUB forman parte de CUDA Core Compute Libraries y sus cabeceras se distribuyen con el Toolkit. Los programas comprueban tanto los errores del runtime como los estados devueltos por cada biblioteca.

## Gráficas explicativas

Cada gráfica responde una pregunta concreta y contiene únicamente las series necesarias para discutirla:

- naïve frente a tiled frente a cuBLAS para GEMM, separando ejecución y extremo a extremo;
- ancho de banda efectivo frente a patrón coalescente/no coalescente;
- CUB/Thrust frente a reducción propia correcta para varios tamaños;
- cuFFT con y sin reutilización de plan;
- SpMV frente a densidad/formato, sin generalizar desde una sola matriz;
- transferencia, preparación y cómputo como componentes separados del tiempo.

Las comparaciones mantienen el problema, la precisión, la tolerancia y el dispositivo. En la interpretación se consideran el algoritmo, la disposición de los datos, el uso de Tensor Cores cuando corresponda, la fusión de operaciones, el espacio de trabajo y la permanencia de los datos en la GPU.
