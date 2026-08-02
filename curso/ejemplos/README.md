# Ejemplos

Esta carpeta define la organización prevista para los programas compilables que acompañan los notebooks. En la edición actual no deben suponerse disponibles: cada ejemplo solo existe cuando su subcarpeta contiene las fuentes, la construcción y las pruebas descritas a continuación.

Cuando se incorpore un ejemplo completo, su subcarpeta debe incluir:

- `CMakeLists.txt` o integración con el CMake superior.
- versión serial de referencia;
- una o más versiones paralelas;
- pruebas de corrección;
- manejo explícito de errores;
- opción para exportar métricas a CSV/JSON;
- README con compilación, ejecución y hardware requerido.

Los ejecutables, objetos, perfiles y conjuntos voluminosos de resultados se generan durante la práctica y quedan por fuera del control de versiones.
