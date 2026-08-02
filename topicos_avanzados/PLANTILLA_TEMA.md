# Guía para desarrollar un tema avanzado

Cada unidad utiliza entre uno y tres notebooks, además de programas fuente independientes. Esta guía establece el contenido académico y experimental mínimo que se espera antes de ofrecer el tema.

## 1. Pregunta guía y alcance

- Problema técnico al que responde el modelo o framework.
- Casos en los que conviene y casos en los que no.
- Qué queda fuera del tema y qué conocimientos previos se presuponen.

## 2. Conceptos clave

- Glosario corto antes de usar la API.
- Capas visibles: aplicación, abstracción, runtime/backend, controlador y hardware.
- Modelo de ejecución, jerarquía de paralelismo y unidad de planificación.
- Modelo de memoria, propiedad de datos, transferencias y consistencia.
- Sincronización, asincronía, errores y finalización del trabajo.
- Garantías de portabilidad y dependencias específicas del proveedor.

El tema incluye un mapa conceptual o un diagrama de capas con el estilo de boceto técnico a lápiz adoptado para la asignatura. Cada figura se concentra en una relación concreta de ejecución, memoria o software.

## 3. Manera de trabajo

La práctica sigue una secuencia que pueda reproducirse:

```text
inventariar entorno → configurar backend → compilar → validar
        → perfilar → interpretar → ajustar → volver a validar
```

En cada paso se identifica el componente responsable: compilador, runtime, controlador, planificador, CPU o GPU.

## 4. Prerrequisitos reproducibles

La sección de preparación registra:

- sistema operativo y arquitectura;
- compilador, versión y comando;
- estándar de C++ y flags imprescindibles;
- librerías, versiones y raíces CMake;
- backend y arquitectura CPU/GPU;
- controlador/runtime;
- módulos de entorno que deben cargarse;
- variables de entorno relevantes;
- CMake preset o comando completo;
- requisitos del clúster descritos en [`ENTORNOS_CLUSTER.md`](ENTORNOS_CLUSTER.md).

Si falta un prerrequisito, el notebook detiene la ejecución e informa qué componente debe instalarse o configurarse. Cuando la práctica exige GPU, una ejecución por fallback en CPU se reporta como un error de configuración.

## 5. Ejemplo ilustrativo

El primer ejemplo utiliza una referencia serial pequeña y fácil de comprobar. La versión paralela introduce una abstracción principal en cada etapa, conserva el manejo explícito de errores y produce una salida numérica verificable. El programa completo se almacena en `ejemplos/<topico>/`; el notebook se encarga de construirlo y ejecutarlo.

## 6. Experimento y gráficas

El diseño del experimento especifica hipótesis, variables, calentamiento, número de repeticiones, métrica, tolerancia numérica y hardware. Para las gráficas se adoptan los siguientes criterios:

- una pregunta por gráfica;
- preferiblemente entre una y tres series;
- título que formule la comparación y ejes con unidades;
- línea base o techo teórico cuando corresponda;
- incertidumbre o distribución cuando haya variabilidad;
- paleta sobria compatible con el estilo visual del curso;
- dos o tres frases de interpretación inmediatamente después.

Entre las gráficas pertinentes están el tiempo frente al tamaño, la aceleración frente a los recursos, el ancho de banda, la relación entre transferencia y cómputo y la comparación de backends sobre el mismo hardware. Las capturas del perfilador se reservan para ubicar un evento o un cuello de botella concreto.

## 7. Ejercicio y evaluación

El enunciado registra los prerrequisitos conceptuales, archivos iniciales, entorno de compilación, bibliotecas, backend, hardware, recursos del planificador, comandos de ejecución, salida esperada, tolerancia, pruebas públicas y rúbrica. También distingue entre ejecución local, sesión interactiva en un nodo y trabajo por lotes.

La solución explica las decisiones de ejecución y memoria, valida los resultados y adjunta el manifiesto real del entorno. La calificación del componente de rendimiento se realiza sobre una implementación que haya superado las pruebas de corrección.

## 8. Cierre

- conceptos que el estudiante estará en capacidad de explicar sin acudir al código;
- errores frecuentes y cómo detectarlos;
- límites de la abstracción;
- comparación con CUDA, HIP, OpenMP target u otro baseline pertinente;
- especificación, documentación oficial y bibliografía.
