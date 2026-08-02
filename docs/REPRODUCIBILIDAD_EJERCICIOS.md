# Protocolo de compilación y reproducibilidad de ejercicios

Fecha de versión: 2 de agosto de 2026.

## 1. Propósito y alcance

Este protocolo establece cómo se acepta un ejercicio antes de utilizarlo en clase. La prueba registra el sistema operativo, la arquitectura, el procesador y las herramientas disponibles; después configura el ejercicio con CMake, construye el objetivo declarado y ejecuta sus pruebas con CTest.

El resultado se informa por ejercicio y por plataforma. Una compilación en `x86_64` no se presenta como evidencia para ARM64, y una ejecución en una máquina virtual sin acelerador no demuestra que un programa CUDA o HIP funcione en una GPU.

A la fecha de este documento, `curso/ejercicios/` y `curso/ejercicios/soluciones/` contienen la organización de los temas, pero todavía no contienen fuentes de ejercicios activos. En consecuencia, el informe de esas dos carpetas indica **0 ejercicios activos, 0 compilados**. El repositorio sí compila dos programas internos, uno C17 y otro C++20, para comprobar que el mecanismo de validación funciona en cada runner. Estas pruebas internas no se contabilizan como ejercicios del curso.

Esta distinción debe conservarse en los informes. Un flujo satisfactorio en el estado actual acredita la integridad del inventario y el funcionamiento del mecanismo; la compatibilidad de un ejercicio comienza a acreditarse cuando aparece en el informe como `compiled` o `tested`.

## 2. Artefactos de validación

| Artefacto | Función |
|---|---|
| `validation/exercises.schema.json` | Contrato formal del archivo `exercise.json`. |
| `validation/templates/exercise.json` | Punto de partida para registrar un ejercicio. |
| `validation/policy.json` | Número mínimo de ejercicios públicos activos, extensiones administradas y exclusiones. |
| `validation/solutions-policy.json` | Política independiente para las soluciones docentes. |
| `validation/validate_exercises.py` | Detecta fuentes sin registrar, valida manifiestos, configura, compila, ejecuta CTest y escribe el informe JSON. |
| `validation/platform_manifest.py` | Registra sistema, kernel, arquitectura, CPU, compiladores, MPI y aceleradores; también comprueba el perfil solicitado. |
| `validation/preflight.py` | Punto de entrada que debe ejecutar el usuario antes de abrir los ejercicios. Coordina el inventario de plataforma, los controles C17/C++20 y la construcción de las actividades compatibles. |
| `validation/validate_navigation.py` | Comprueba los enlaces locales del índice, la presencia de los temas 00–08 y los enlaces de retorno de cada notebook. |
| `validation/fixtures/` | Programas mínimos C17 y C++20 que prueban el propio mecanismo. |
| `.github/workflows/exercise-portability.yml` | Matriz automática en runners administrados por GitHub. |
| `.github/workflows/exercise-hardware.yml` | Ejecución manual en equipos o nodos institucionales conectados como runners *self-hosted*. |

Los informes y las salidas de compilación se escriben en `build/validation/`. En GitHub se conservan como artefactos descargables durante 30 días para la matriz general y 90 días para las pruebas en hardware institucional.

## 3. Contrato de un ejercicio

Cada ejercicio público se ubica en `curso/ejercicios/<tema>/<ejercicio>/`. Su solución utiliza la misma estructura dentro de `curso/ejercicios/soluciones/<tema>/<ejercicio>/`.

```text
<ejercicio>/
├── README.md             # Enunciado o explicación de la solución
├── exercise.json         # Plataforma, lenguaje, dependencias y objetivo
├── CMakeLists.txt        # Construcción autónoma fuera del árbol
├── src/                  # Fuentes compilables
├── data/                 # Entradas pequeñas, cuando se requieran
└── tests/                # Pruebas o scripts invocados por CTest
```

El manifiesto declara:

- identificador, tema, título y estado;
- lenguaje y estándar;
- objetivo de CMake y presencia de pruebas;
- herramientas y capacidades requeridas, por ejemplo `openmp`, `mpi`, `cuda` o `hip`;
- sistemas operativos y arquitecturas admitidos;
- restricciones de fabricante de CPU y acelerador.

Un ejercicio con estado `active` debe tener `CMakeLists.txt`, construir el objetivo indicado y, cuando `build.tests` sea verdadero, registrar al menos las comprobaciones necesarias en CTest. Los estados `planned` y `retired` se informan, pero no se compilan. Ninguna fuente puede quedar por fuera de un directorio que contenga `exercise.json`: el validador termina con error si la encuentra.

Cuando se incorpora un ejercicio activo también se aumenta `minimum_active_exercises` en la política correspondiente y el mínimo de su capacidad en `minimum_by_capability`. Así, la eliminación accidental de un ejercicio no produce un resultado verde con un inventario vacío ni desaparece silenciosamente una familia completa, como CUDA o MPI.

## 4. Ejecución local

La comprobación previa es un requisito de los laboratorios. Desde la raíz del repositorio se ejecuta:

```console
python3 validation/preflight.py
```

El comando termina con estado satisfactorio únicamente cuando el índice conserva rutas válidas, la plataforma puede construir los controles C17/C++20 y todos los ejercicios activos compatibles. Los informes quedan en `build/validation/preflight/`. Si el comando retorna un error, primero se corrige la navegación, la plataforma o el perfil; no se continúa con el ejercicio.

En una máquina institucional se fija el perfil que se desea comprobar:

```console
python3 validation/preflight.py --profile linux-intel
python3 validation/preflight.py --profile linux-amd
python3 validation/preflight.py --profile linux-arm64
python3 validation/preflight.py --profile linux-nvidia-cuda
python3 validation/preflight.py --profile linux-amd-rocm
```

El perfil `local-cpu`, utilizado por defecto, detecta sistema y arquitectura sin exigir un fabricante particular. Los perfiles institucionales no solo seleccionan capacidades: comprueban que la máquina corresponda a la arquitectura, CPU o acelerador solicitado.

Para diagnosticar o mantener por separado cada etapa se pueden ejecutar las órdenes subyacentes:

```console
python3 validation/platform_manifest.py \
  --output build/validation/local/platform.json \
  --require-command cmake

python3 validation/validate_exercises.py \
  --exercises-root validation/fixtures \
  --policy validation/fixtures/policy.json \
  --build-root build/validation/local/harness \
  --report build/validation/local/harness.json \
  --compile --test

python3 validation/validate_exercises.py \
  --exercises-root curso/ejercicios \
  --policy validation/policy.json \
  --build-root build/validation/local/exercises \
  --report build/validation/local/exercises.json \
  --compile --test

python3 validation/validate_exercises.py \
  --exercises-root curso/ejercicios/soluciones \
  --policy validation/solutions-policy.json \
  --build-root build/validation/local/solutions \
  --report build/validation/local/solutions.json \
  --compile --test
```

En Windows se utiliza `python` si ese es el nombre con el que se instaló el intérprete. Los directorios de construcción son independientes por ejercicio; el validador no modifica las fuentes.

Antes de una clase práctica, el docente ejecuta estas órdenes en el ambiente que utilizarán los estudiantes y conserva los tres informes JSON. Para MPI, CUDA, HIP u offload se añade la prueba en el nodo correspondiente, pues la compilación en un portátil no sustituye la ejecución con el runtime y el dispositivo previstos.

## 5. Evidencia de esta revisión

El 2 de agosto de 2026 se ejecutó `preflight.py` en la siguiente plataforma local:

| Sistema | Arquitectura y CPU | Toolchain observado | Resultado |
|---|---|---|---|
| macOS 26.5.2; kernel Darwin 25.5.0 | ARM64; Apple M4, 10 núcleos | Apple Clang 21.0.0; CMake 3.31.5 | Controles C17 y C++20: 2 compilados, 2 pruebas CTest aprobadas. Inventario docente: 0 ejercicios activos. |

La tabla identifica con precisión lo que fue probado y no se utiliza para afirmar compatibilidad con otras arquitecturas. Cada ejecución de GitHub o del clúster produce su propio `platform.json`; los resultados aceptados se consultan en los artefactos del trabajo correspondiente.

## 6. Matriz administrada por GitHub

El flujo `Portabilidad de ejercicios` se ejecuta en cada *pull request* hacia `master`, en los envíos a `master` y a `feature/actualizacion_2026`, y por solicitud manual.

| Sistema | Runner | Arquitectura comprobada | Alcance |
|---|---|---:|---|
| Ubuntu 24.04 | `ubuntu-24.04` | x86-64 | C/C++ y capacidades CPU disponibles en la imagen. |
| Ubuntu 24.04 | `ubuntu-24.04-arm` | ARM64 | Portabilidad Linux/ARM64. |
| macOS 15 | `macos-15-intel` | x86-64 Intel | Portabilidad macOS/Intel. |
| macOS 15 | `macos-15` | ARM64 Apple Silicon | Portabilidad macOS/ARM64. |
| Windows Server 2025 | `windows-2025` | x86-64 | Portabilidad Windows/MSVC. |
| Windows 11 | `windows-11-arm` | ARM64 | Portabilidad Windows/ARM64; imagen en vista previa. |

GitHub documenta esas arquitecturas y etiquetas en la [referencia de runners administrados](https://docs.github.com/en/actions/reference/runners/github-hosted-runners). En Ubuntu y Windows x86-64 la arquitectura sí forma parte del contrato del runner, pero el fabricante concreto de CPU no. El manifiesto registra el procesador que atendió cada trabajo; ese dato no permite prometer que la siguiente ejecución tendrá el mismo fabricante.

Por esa razón, la matriz general demuestra portabilidad por sistema operativo y arquitectura, no una certificación diferenciada Intel/AMD. macOS Intel sí aporta una ejecución Intel; la validación AMD se reserva para una máquina institucional identificada y comprobada.

## 7. Intel, AMD, ARM y aceleradores reales

GitHub admite runners *self-hosted* en x64 y ARM64, además de ARM32 en Linux, según su [referencia de plataformas soportadas](https://docs.github.com/en/actions/reference/runners/self-hosted-runners). El repositorio ofrece el flujo manual `Ejercicios en hardware institucional` con cinco perfiles:

| Perfil | Etiqueta personalizada requerida | Comprobación dentro del trabajo | Ejercicios seleccionados |
|---|---|---|---|
| Intel/Linux | `linux-intel` | x86-64 y CPU Intel | capacidad `cpu` |
| AMD/Linux | `linux-amd` | x86-64 y CPU AMD | capacidad `cpu` |
| ARM/Linux | `linux-arm64` | ARM64 | capacidad `cpu` |
| NVIDIA/Linux | `linux-nvidia-cuda` | x86-64, `nvcc` y GPU visible en `nvidia-smi` | capacidad `cuda` |
| AMD GPU/Linux | `linux-amd-rocm` | x86-64, `hipcc` y GPU visible en `rocminfo` | capacidad `hip` |

El administrador registra cada equipo con las etiquetas predeterminadas `self-hosted` y `linux`, más una de las etiquetas anteriores. GitHub explica el [uso de etiquetas para dirigir trabajos](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow) y la [administración de etiquetas personalizadas](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/apply-labels). Como una etiqueta personalizada no certifica por sí misma la máquina, `platform_manifest.py --profile ...` contrasta la etiqueta con la arquitectura, el fabricante o el acelerador detectado y detiene el trabajo cuando no corresponden.

Los runners institucionales deben ser efímeros o restablecerse entre trabajos, no contener credenciales docentes en el ambiente de compilación y aceptar únicamente flujos de ramas controladas. Si no hay un runner compatible en línea, GitHub deja el trabajo en cola y finalmente lo cancela; por ello este flujo es manual y no bloquea todos los *pull requests*.

Los runners GPU administrados por GitHub existen entre las opciones de mayor capacidad, pero su disponibilidad depende del tipo de cuenta y no reemplaza la configuración del clúster del curso. Para este repositorio se conserva la ruta *self-hosted*, que permite fijar driver, toolkit, modelo de GPU, interconexión y planificador.

## 8. Niveles de evidencia

La afirmación de reproducibilidad debe indicar hasta qué nivel llegó la prueba:

1. **Inventario:** el manifiesto es válido y no hay fuentes sin registrar.
2. **Construcción:** CMake configura y el compilador produce el objetivo.
3. **Prueba funcional:** CTest compara el resultado con una referencia pequeña.
4. **Runtime paralelo:** se ejecuta con el número de hilos, procesos o dispositivo previsto y se verifican errores del runtime.
5. **Experimento docente:** se ejecuta en el clúster o laboratorio definido, con entradas, repeticiones, afinidad y versiones fijadas; se conservan resultados y manifiesto de plataforma.

La integración continua administrada cubre los niveles 1 a 3 para los ejercicios compatibles con CPU. Los niveles 4 y 5 de MPI, offload y GPU requieren el runner institucional y, cuando corresponda, el sistema de colas del clúster.

## 9. Criterio para declarar un ejercicio reproducible

Un ejercicio se acepta para una sesión cuando:

- su estado es `active` y está incluido en la política mínima;
- la matriz obligatoria termina satisfactoriamente en todas las plataformas que declara;
- el informe contiene fabricante y modelo de CPU, arquitectura, sistema operativo, compilador y CMake;
- CTest comprueba una salida conocida y retorna un código distinto de cero ante una falla;
- las dependencias paralelas y los requisitos de hardware están declarados;
- cuando usa MPI o acelerador, existe evidencia reciente del runner o nodo real;
- el README del ejercicio reproduce las órdenes y describe las entradas, la salida esperada y las limitaciones;
- el notebook invoca el mismo objetivo de construcción, sin mantener una copia divergente del código.

Se considera reciente una ejecución realizada después del último cambio del ejercicio, su solución, su `CMakeLists.txt`, su manifiesto o la configuración global que lo afecta. La evidencia se revisa antes de abrir el tema en el aula y se vuelve a generar si cambia el entorno del laboratorio.
