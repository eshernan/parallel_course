# Ejercicios

Los ejercicios se identifican por tema y número; por ejemplo, `03_openmp/02_histograma/`. En cada carpeta se incluyen:

- enunciado y resultados de aprendizaje;
- esqueleto de código;
- entrada pequeña y salida esperada;
- pruebas públicas;
- criterios de rendimiento aplicables después de superar las pruebas de corrección;
- rúbrica breve.

Cada ejercicio cuenta además con `exercise.json` y `CMakeLists.txt`. El manifiesto declara el estándar, el objetivo de construcción, las dependencias y las plataformas admitidas. El código inicial debe compilar aun cuando contenga secciones que el estudiante deba completar; las pruebas pueden fallar hasta completar la actividad, pero esa condición se documenta expresamente y no se confunde con un error de construcción.

Antes de ejecutar una actividad, el estudiante corre `python3 validation/preflight.py` desde la raíz. Solo continúa si la comprobación termina satisfactoriamente. La estructura esperada, los perfiles de hardware y las órdenes de diagnóstico se describen en el [protocolo de compilación y reproducibilidad](../../docs/REPRODUCIBILIDAD_EJERCICIOS.md). Al agregar una actividad se incrementa `minimum_active_exercises` en `validation/policy.json`; de este modo, la integración continua detecta tanto fuentes sin registrar como eliminaciones accidentales del inventario.

Las soluciones de referencia se encuentran en `soluciones/` y se excluyen de la rama que se distribuye a los estudiantes mientras la actividad se encuentre abierta.
