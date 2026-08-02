# Estándares de C y C++: versiones, evolución y soporte

Fecha de corte: 2 de agosto de 2026.

Este panorama se estudia en el tema 00. Su propósito no es recorrer todas las características incorporadas durante cuatro décadas, sino enseñar a distinguir la especificación del lenguaje, la implementación del compilador, la biblioteca estándar y los servicios que aporta el sistema operativo. En programación paralela esa distinción es necesaria: aceptar `-std=c++20`, por ejemplo, no demuestra que estén disponibles todos los componentes de la biblioteca ni que una extensión de OpenMP, MPI o CUDA sea compatible con el mismo compilador.

## Cómo se interpreta una versión

Los nombres C23 y C++23 identifican el ciclo en el que se cerró el trabajo técnico. Las publicaciones vigentes de ISO aparecieron en 2024: [ISO/IEC 9899:2024](https://www.iso.org/standard/82075.html) para C e [ISO/IEC 14882:2024](https://www.iso.org/standard/83626.html) para C++. Una revisión en preparación, como C2y o C++26, no se trata como norma publicada aunque un compilador ya ofrezca una bandera para ensayarla.

En el laboratorio se comprueban por separado cuatro niveles:

1. el modo de lenguaje solicitado al compilador;
2. las características que realmente implementa el analizador del lenguaje;
3. las funciones y tipos disponibles en la biblioteca estándar;
4. las dependencias que introduce el sistema: biblioteca C, ABI, SDK, entorno de ejecución y versión mínima de despliegue.

## Evolución de C

| Revisión | Publicación | Aporte que conviene reconocer en el curso | Situación en 2026 |
|---|---:|---|---|
| C89/C90 | 1989/1990 | Base del lenguaje estandarizado y de gran parte del código científico heredado. | Revisión histórica, ampliamente implementada. |
| C95 | 1995 | Enmienda de internacionalización sobre C90. | Incorporada en revisiones posteriores. |
| C99 | 1999 | Declaraciones en el bloque, `inline`, `restrict`, tipos enteros y complejos, arreglos de longitud variable y mejoras numéricas. | Todavía frecuente en bibliotecas científicas. |
| C11 | 2011 | Modelo de memoria, `_Atomic`, almacenamiento local al hilo, `_Generic` y una biblioteca de hilos opcional. | Base conceptual para discutir concurrencia y operaciones atómicas. |
| C17 | 2018 | Correcciones y aclaraciones, sin un conjunto amplio de características nuevas. | Versión normativa adoptada por el curso. |
| C23 | 2024 | Atributos, `nullptr`, `typeof`, enteros de precisión definida, inferencia con `auto`, `#embed` y ajustes importantes del lenguaje y la biblioteca. | Última revisión publicada. |
| C2y | En preparación | Siguiente revisión; el contenido cambia a medida que WG14 acepta propuestas y corrige C23. | Borrador de trabajo, no norma del curso. |

El [estado de proyectos de WG14](https://www.open-std.org/jtc1/sc22/wg14/www/projects.html) conserva la relación oficial de revisiones. El borrador C2y [N3886](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3886.pdf), vigente a la fecha de corte, sirve para observar el trabajo del comité, no para imponer requisitos a las entregas.

C17 también aparece en algunos textos como C18: el trabajo técnico corresponde a 2017, la publicación de ISO a 2018 y la macro `__STDC_VERSION__` toma el valor `201710L`. En este repositorio se utiliza el nombre C17 de manera consistente.

## Evolución de C++

| Revisión | Publicación | Aporte que conviene reconocer en el curso | Situación en 2026 |
|---|---:|---|---|
| C++98/C++03 | 1998/2003 | Primer estándar y revisión de mantenimiento; consolidación de clases, plantillas y STL. | Relevante para comprender código heredado. |
| C++11 | 2011 | Modelo de memoria, hilos, atomics, lambdas, semántica de movimiento y RAII moderno. | Punto de quiebre para concurrencia portable. |
| C++14 | 2014 | Ajustes a lambdas, plantillas y `constexpr`. | Revisión incremental y ampliamente implementada. |
| C++17 | 2017 | `if constexpr`, `std::filesystem`, tipos de vocabulario y algoritmos paralelos. | Base todavía común en compiladores de clúster. |
| C++20 | 2020 | `std::jthread`, `stop_token`, espera/notificación atómica, conceptos, rangos, corutinas y módulos. | Versión normativa adoptada por el curso; no se requieren módulos. |
| C++23 | 2024 | Mejoras de rangos, `std::expected`, `mdspan`, `print`, explicit object parameter y extensiones de `constexpr`. | Última revisión publicada; el soporte sigue variando por característica. |
| C++26 | Publicación prevista en 2026 | Contratos, reflexión, biblioteca SIMD y otras incorporaciones del ciclo, sujetas al texto final y a su implementación. | Borrador de trabajo; soporte experimental o parcial. |
| C++29 | Ciclo siguiente | Propuestas posteriores a C++26. | Trabajo inicial; no se emplea en el curso. |

WG21 mantiene los [documentos públicos del comité](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/). A la fecha de corte, [N5054](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/n5054.pdf) continúa identificado como borrador de trabajo de C++; por ello el curso no presenta C++26 como estándar publicado.

## Soporte en Linux, macOS y Windows

La siguiente matriz resume los entornos de compilación disponibles y no pretende reemplazar las tablas de conformidad de sus productores. “Parcial” indica que se debe consultar la característica concreta y su biblioteca; no significa que el compilador sea inadecuado para el curso.

| Plataforma | Entorno de compilación en agosto de 2026 | C | C++ | Decisión para la asignatura |
|---|---|---|---|---|
| Linux | GCC 16.1 o Clang 22.1 en instalaciones actuales; GCC 15.3 en la imagen del curso. | GCC declara soporte de C23 y soporte experimental de C2y. Clang clasifica C11, C17, C23 y C2y como parciales y documenta cada característica. | GCC considera C++20 casi completo, con módulos todavía experimentales, y C++23/C++26 experimentales. Clang clasifica C++20, C++23 y C++26 como parciales. | Es la plataforma de ejecución y calificación. Se compila de manera explícita con C17 y C++20. |
| macOS | Apple Clang y libc++ incluidos en Xcode 26.6; el Clang distribuido por LLVM puede instalarse por separado. | C17 se utiliza para las pruebas locales de portabilidad. Como la matriz de Apple se concentra en C++, toda característica de C23 se comprueba de manera individual y Apple Clang no se supone idéntico al Clang distribuido por LLVM. | Apple documenta cobertura amplia de C++20 y adopción progresiva de C++23/C++26. Algunas funciones de libc++ dependen de la versión del SDK y del objetivo mínimo de macOS. | Se admite para desarrollo y pruebas de portabilidad. Los experimentos de referencia se repiten en Linux. |
| Windows | MSVC Build Tools 14.51 en Visual Studio 2026 18.8; Clang 22.1 y GCC mediante MinGW-w64 son alternativas. | MSVC ofrece `/std:c11` y `/std:c17`, pero conserva vacíos en características opcionales y de biblioteca. `/std:clatest` incorpora trabajo posterior de forma experimental. | C++20 es apropiado para el subconjunto docente. Microsoft continúa incorporando características de C++23 y algunas de C++26 mediante los modos más recientes; la conformidad se comprueba por característica. | El código C/C++ portable puede probarse de forma nativa. Pthreads, el entorno MPI/Linux, Slurm y la ejecución institucional se trabajan en WSL2, una máquina Linux o el clúster. |

Fuentes de seguimiento por compilador:

- GCC: [versiones publicadas](https://gcc.gnu.org/releases.html), [estado de C](https://gcc.gnu.org/projects/c-status.html) y [estado de C++](https://gcc.gnu.org/projects/cxx-status.html).
- LLVM/Clang: [notas de Clang 22.1](https://releases.llvm.org/22.1.0/tools/clang/docs/ReleaseNotes.html), [estado de C](https://clang.llvm.org/c_status.html) y [estado de C++](https://clang.llvm.org/cxx_status.html).
- Apple: [soporte de C++ en Apple Clang y libc++](https://developer.apple.com/xcode/cpp/) y [versiones de Xcode compatibles con macOS](https://developer.apple.com/xcode/system-requirements/).
- Microsoft: [opciones `/std`](https://learn.microsoft.com/en-us/cpp/build/reference/std-specify-language-standard-version?view=msvc-180), [matriz de conformidad](https://learn.microsoft.com/en-us/cpp/overview/visual-cpp-language-conformance?view=msvc-180) y [notas de Visual Studio 2026](https://learn.microsoft.com/en-us/visualstudio/releases/2026/release-notes).

## Elección del curso

El curso conserva C17 y C++20 como contrato de portabilidad. No se adopta automáticamente el último modo disponible: los ejemplos deben funcionar con Pthreads, OpenMP, MPI y CUDA dentro de una misma configuración verificable. Aunque GCC 16.1 es la rama principal más reciente a la fecha de corte, CUDA 13.0 admite GCC hasta la rama 15; por esta razón la imagen docente fija GCC 15.3. La [matriz de compiladores host de CUDA 13](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-installation-guide-linux/index.html#host-compiler-support-policy) sustenta esta decisión.

La selección explícita del estándar también evita depender de los valores predeterminados: en GCC 15, C23 es el modo C por defecto, mientras que C++17 sigue siendo el modo C++ por defecto. CMake exige `C_STANDARD 17` y `CXX_STANDARD 20` para que el resultado no cambie por esa diferencia.

La selección no impide discutir C23, C++23, C2y o C++26. Estas revisiones se examinan para comprender la evolución del lenguaje y para aprender a leer una tabla de conformidad, pero las entregas no dependen de una característica experimental.

## Comprobación durante el laboratorio

Cada estudiante registra la versión del compilador y verifica los modos seleccionados:

```console
gcc --version
g++ --version
gcc -std=c17 -dM -E -x c /dev/null | grep __STDC_VERSION__
g++ -std=c++20 -dM -E -x c++ /dev/null | grep __cplusplus
```

En MSVC se registra `cl /Bv` y se compila el mismo programa de diagnóstico con `/std:c17` o `/std:c++20`. Para una característica de C++ se consultan además las macros de prueba de `<version>`; para C se revisan `__STDC_VERSION__` y las macros `__STDC_NO_*`. La presencia de la macro se acompaña de una compilación y una prueba mínima, pues el número del modo de lenguaje no basta para establecer conformidad funcional.
