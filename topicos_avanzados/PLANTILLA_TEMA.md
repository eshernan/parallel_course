# Plantilla para un tópico avanzado

Cada cápsula utilizará entre uno y tres notebooks y fuentes independientes. Esta plantilla es un criterio de aceptación, no un índice decorativo.

## 1. Pregunta guía y alcance

- Problema técnico que resuelve el modelo o framework.
- Casos en los que conviene y casos en los que no.
- Qué queda fuera del tema y qué conocimientos previos se presuponen.

## 2. Conceptos clave

- Glosario corto antes de usar la API.
- Capas visibles: aplicación, abstracción, runtime/backend, controlador y hardware.
- Modelo de ejecución, jerarquía de paralelismo y unidad de planificación.
- Modelo de memoria, propiedad de datos, transferencias y consistencia.
- Sincronización, asincronía, errores y finalización del trabajo.
- Garantías de portabilidad y dependencias específicas del proveedor.

Se incluirá un mapa conceptual o diagrama de capas en estilo boceto técnico a lápiz. Una figura debe explicar una relación concreta; no debe intentar resumir toda la API.

## 3. Manera de trabajo

El notebook mostrará un flujo verificable:

```text
inventariar entorno → configurar backend → compilar → validar
        → perfilar → interpretar → ajustar → volver a validar
```

Cada paso debe identificar quién actúa: compilador, runtime, driver, scheduler, CPU o GPU.

## 4. Prerrequisitos reproducibles

Antes de compilar se declara:

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

El notebook debe fallar temprano con un mensaje claro cuando falta un prerrequisito; no debe ejecutar silenciosamente sobre CPU cuando el objetivo era GPU.

## 5. Ejemplo ilustrativo

Se parte de una referencia serial pequeña y comprobable. El ejemplo paralelo introduce una sola abstracción principal por etapa, conserva manejo explícito de errores y produce una salida numérica verificable. El fuente completo vive en `ejemplos/<topico>/` y el notebook lo construye y ejecuta.

## 6. Experimento y gráficas

Cada experimento formula hipótesis, variables, calentamiento, repeticiones, métrica, tolerancia numérica y hardware. Las gráficas cumplen estas reglas:

- una pregunta por gráfica;
- preferiblemente entre una y tres series;
- título que formule la comparación y ejes con unidades;
- línea base o techo teórico cuando corresponda;
- incertidumbre o distribución cuando haya variabilidad;
- paleta sobria compatible con el estilo visual del curso;
- dos o tres frases de interpretación inmediatamente después.

Gráficas sugeridas: tiempo frente a tamaño, aceleración frente a recursos, ancho de banda, transferencia frente a cómputo o rendimiento entre backends en el mismo hardware. Una captura de perfilador solo se usa para localizar un evento o cuello de botella concreto.

## 7. Ejercicio y evaluación

El ejercicio declara prerrequisitos conceptuales, archivos de partida, entorno de compilación, librerías, backend, hardware, recursos del scheduler, comandos exactos, salida esperada, tolerancia, pruebas públicas y rúbrica. Debe distinguir entre ejecución local, nodo interactivo y trabajo por lotes.

La solución explica las decisiones de ejecución y memoria, valida resultados antes de medir y adjunta el manifiesto real del entorno. No se otorga crédito de rendimiento a resultados incorrectos.

## 8. Cierre

- conceptos que deben poder explicar sin código;
- errores frecuentes y cómo detectarlos;
- límites de la abstracción;
- comparación con CUDA, HIP, OpenMP target u otro baseline pertinente;
- especificación, documentación oficial y bibliografía.
