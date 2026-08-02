#!/usr/bin/env python3
"""Genera los notebooks canónicos de los temas 00–08."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


@dataclass(frozen=True)
class NotebookSpec:
    topic: str
    directory: str
    filename: str
    title: str
    sessions: tuple[int, ...]
    question: str
    outcomes: tuple[str, ...]
    concepts: tuple[str, ...]
    analyses: tuple[tuple[str, str, str, str], ...]
    practice: tuple[str, ...]
    pitfalls: tuple[str, ...]
    acceptance: tuple[str, ...]
    references: tuple[tuple[str, str], ...]


def spec(
    topic: str,
    directory: str,
    filename: str,
    title: str,
    sessions: tuple[int, ...],
    question: str,
    outcomes: tuple[str, ...],
    concepts: tuple[str, ...],
    analyses: tuple[tuple[str, str, str, str], ...],
    practice: tuple[str, ...],
    pitfalls: tuple[str, ...],
    acceptance: tuple[str, ...],
    references: tuple[tuple[str, str], ...],
) -> NotebookSpec:
    return NotebookSpec(
        topic, directory, filename, title, sessions, question, outcomes, concepts,
        analyses, practice, pitfalls, acceptance, references,
    )


SPECS = (
    spec(
        "00", "00_entorno", "00_entorno_reproducible.ipynb", "Entorno reproducible y diagnóstico", (1, 2),
        "¿Cómo demostrar que una práctica puede construirse y repetirse en el equipo disponible?",
        ("Distinguir plataforma, toolchain, runtime y dependencia.", "Registrar evidencia mínima del sistema sin confundir disponibilidad con compatibilidad.", "Ejecutar el preflight y leer sus informes antes de iniciar una práctica."),
        ("Un entorno reproducible declara versiones, arquitectura y comandos; no se reduce a una lista de paquetes.", "El manifiesto de plataforma describe lo observado. La política del ejercicio determina si ese equipo es compatible.", "CMake configura y CTest verifica corrección; las mediciones de rendimiento se realizan solo después de superar las pruebas."),
        (
            ("Inventario local", "Se inspeccionan datos portables del intérprete y la presencia de herramientas sin instalar ni modificar el sistema.", dedent('''
                import platform, shutil, sys
                inventory = {
                    "python": sys.version.split()[0],
                    "os": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                    "logical_cpus": __import__("os").cpu_count(),
                    "cmake": shutil.which("cmake"),
                    "ctest": shutil.which("ctest"),
                    "cc": shutil.which("cc"),
                    "cxx": shutil.which("c++"),
                }
                assert inventory["logical_cpus"] and inventory["logical_cpus"] > 0
                for key, value in inventory.items():
                    print(f"{key:12}: {value}")
            '''), "Una ruta ausente se reporta como evidencia diagnóstica; no se sustituye por una afirmación de soporte."),
            ("Configuración declarada", "Se extraen las versiones canónicas del toolchain y se comprueba que las claves esenciales estén presentes.", dedent('''
                import re
                toolchain = (ROOT / "config" / "course-toolchain.cmake").read_text(encoding="utf-8")
                pairs = dict(re.findall(r'set\\((COURSE_[A-Z0-9_]+) "([^"]+)"', toolchain))
                required = {"COURSE_GCC_VERSION", "COURSE_CXX_STANDARD", "COURSE_MPI_VERSION", "COURSE_CUDA_VERSION", "COURSE_PYTHON_VERSION"}
                assert required <= pairs.keys(), required - pairs.keys()
                for key in sorted(required):
                    print(f"{key}={pairs[key]}")
            '''), "La versión declarada es un requisito; el manifiesto del equipo permite contrastarla con la versión observada."),
        ),
        ("Ejecutar `python3 validation/preflight.py` desde la raíz.", "Conservar los JSON de `build/validation/preflight/`.", "Explicar qué comprobó cada etapa y qué no demuestra todavía."),
        ("Continuar aunque el preflight falle.", "Confundir arquitectura con fabricante de CPU.", "Afirmar soporte de GPU porque `nvcc` está instalado sin ejecutar en un dispositivo."),
        ("Preflight con código de salida cero.", "Manifiesto de plataforma adjunto al informe.", "Limitaciones del equipo descritas explícitamente."),
        (("Protocolo de reproducibilidad", "../../../docs/REPRODUCIBILIDAD_EJERCICIOS.md"), ("Configuración del curso", "../../../config/README.md")),
    ),
    spec(
        "00", "00_entorno", "01_estandares_compiladores.ipynb", "Estándares, compiladores y macros", (1,),
        "¿Qué puede afirmarse a partir de una macro de versión y qué requiere una prueba de conformidad?",
        ("Interpretar `__STDC_VERSION__` y `__cplusplus`.", "Separar estándar, implementación, biblioteca y bandera de compilación.", "Comparar diagnósticos sin asumir equivalencia total entre GCC, Clang y MSVC."),
        ("La aceptación de `-std` o `/std` selecciona un modo; no certifica toda la especificación.", "Las macros informan el nivel anunciado por la implementación para la unidad de traducción.", "Las extensiones, la biblioteca estándar y el sistema objetivo también afectan portabilidad y comportamiento."),
        (
            ("Interpretación de macros", "Se traduce el valor publicado por el compilador a una etiqueta útil para el informe.", dedent('''
                c_levels = {199901: "C99", 201112: "C11", 201710: "C17", 202311: "C23"}
                cpp_levels = {201103: "C++11", 201402: "C++14", 201703: "C++17", 202002: "C++20", 202302: "C++23"}
                def announced(value, levels):
                    candidates = [(number, label) for number, label in levels.items() if value >= number]
                    return max(candidates, default=(0, "anterior/no declarado"))[1]
                assert announced(201710, c_levels) == "C17"
                assert announced(202002, cpp_levels) == "C++20"
                for value in (199901, 201710, 202311): print(value, announced(value, c_levels))
                for value in (201703, 202002, 202302): print(value, announced(value, cpp_levels))
            '''), "El resultado se redacta como nivel anunciado, no como garantía de conformidad completa."),
            ("Contrato por plataforma", "Se construye una tabla de invocaciones sin ejecutar compiladores que podrían no estar instalados.", dedent('''
                commands = {
                    "GCC C17": "gcc -std=c17 -Wall -Wextra fuente.c",
                    "GCC C++20": "g++ -std=c++20 -Wall -Wextra fuente.cpp",
                    "Clang C17": "clang -std=c17 -Wall -Wextra fuente.c",
                    "Clang C++20": "clang++ -std=c++20 -Wall -Wextra fuente.cpp",
                    "MSVC C17": "cl /std:c17 /W4 fuente.c",
                    "MSVC C++20": "cl /std:c++20 /W4 fuente.cpp",
                }
                assert len(commands) == 6
                for compiler, command in commands.items(): print(f"{compiler:14} | {command}")
            '''), "La comparación mantiene lenguaje, estándar y advertencias; las capacidades reales se verifican con programas mínimos."),
        ),
        ("Compilar un diagnóstico que imprima ambas macros.", "Registrar compilador y biblioteca junto con el resultado.", "Comparar el informe con la matriz documentada, sin extrapolar características no probadas."),
        ("Usar `__cplusplus` para un archivo C.", "Comparar solo el nombre comercial del compilador.", "Tratar una extensión aceptada como parte obligatoria del estándar."),
        ("Macro y comando conservados en el informe.", "Prueba mínima construida con advertencias habilitadas.", "Conclusión limitada a la característica observada."),
        (("Panorama C/C++", "../../../docs/ESTANDARES_C_CPP.md"), ("Planeación", "../../../docs/PLANEACION_CURSO.md")),
    ),
    spec(
        "01", "01_fundamentos", "01_modelos.ipynb", "Modelos de paralelismo y descomposición", (3,),
        "¿Qué parte del trabajo puede ejecutarse simultáneamente y qué dependencias fijan el camino crítico?",
        ("Distinguir concurrencia, paralelismo de datos y de tareas.", "Representar dependencias mediante un DAG.", "Calcular trabajo, span y paralelismo promedio."),
        ("El trabajo T1 suma el costo de todas las tareas; el span T∞ es el camino dependiente más largo.", "La aceleración con p recursos está acotada por min(p, T1/T∞), aun sin costos de comunicación.", "Una descomposición correcta conserva dependencias y evita crear más coordinación que cómputo útil."),
        (
            ("Trabajo y camino crítico", "Se evalúa un DAG pequeño con duraciones explícitas.", dedent('''
                tasks = {"leer": 2, "partir": 1, "A": 5, "B": 4, "combinar": 2}
                predecessors = {"leer": [], "partir": ["leer"], "A": ["partir"], "B": ["partir"], "combinar": ["A", "B"]}
                finish = {}
                for task in tasks:
                    start = max((finish[p] for p in predecessors[task]), default=0)
                    finish[task] = start + tasks[task]
                work = sum(tasks.values())
                span = max(finish.values())
                parallelism = work / span
                assert (work, span) == (14, 10)
                print({"work": work, "span": span, "parallelism": round(parallelism, 2), "finish": finish})
            '''), "A y B son concurrentes, pero lectura, partición y combinación permanecen en el camino crítico."),
            ("Partición balanceada", "Se distribuyen n elementos sin perder ni duplicar índices.", dedent('''
                def ranges(n, workers):
                    q, r = divmod(n, workers)
                    start = 0
                    result = []
                    for worker in range(workers):
                        size = q + (worker < r)
                        result.append((start, start + size))
                        start += size
                    return result
                chunks = ranges(23, 4)
                covered = [i for begin, end in chunks for i in range(begin, end)]
                assert covered == list(range(23))
                print(chunks)
            '''), "El resto se reparte de forma determinista y la cobertura constituye una prueba simple de corrección."),
        ),
        ("Dibujar el DAG de una operación del curso.", "Identificar T1, T∞ y la granularidad de cada nodo.", "Proponer una descomposición y señalar la sincronización necesaria."),
        ("Confundir más tareas con mayor paralelismo.", "Omitir dependencias de datos.", "Evaluar únicamente el tiempo paralelo sin referencia serial."),
        ("DAG acíclico y dependencias justificadas.", "Cobertura de datos sin solapamientos involuntarios.", "Cotas de aceleración calculadas antes de medir."),
        (("Planeación: fundamentos", "../../../docs/PLANEACION_CURSO.md#6-calendario-de-38-sesiones"), ("Índice del curso", "../../../INDICE_CURSO.md")),
    ),
    spec(
        "01", "01_fundamentos", "02_escalabilidad.ipynb", "Amdahl, Gustafson y escalabilidad", (4, 6),
        "¿Por qué una aceleración observada debe interpretarse junto con eficiencia, tamaño y modelo de escalado?",
        ("Calcular límites de Amdahl y Gustafson.", "Distinguir escalado fuerte y débil.", "Reportar eficiencia y overhead con una línea base estable."),
        ("Amdahl mantiene el tamaño fijo y hace visible la fracción serial.", "Gustafson razona sobre problemas cuyo trabajo paralelo crece con los recursos.", "La eficiencia Sp/p cae por serialización, comunicación, desbalance y costos del runtime."),
        (
            ("Límites analíticos", "Se comparan ambos modelos para la misma fracción paralela.", dedent('''
                def amdahl(parallel_fraction, p): return 1 / ((1 - parallel_fraction) + parallel_fraction / p)
                def gustafson(parallel_fraction, p): return p - (1 - parallel_fraction) * (p - 1)
                fraction = 0.95
                rows = []
                for p in (1, 2, 4, 8, 16, 32):
                    rows.append((p, amdahl(fraction, p), gustafson(fraction, p)))
                assert rows[0] == (1, 1.0, 1.0)
                for p, strong, scaled in rows: print(f"p={p:2} Amdahl={strong:6.2f} Gustafson={scaled:6.2f}")
            '''), "Los modelos responden preguntas diferentes; no deben presentarse como predicciones intercambiables."),
            ("Métricas observadas", "Se calcula aceleración, eficiencia y overhead a partir de tiempos sintéticos trazables.", dedent('''
                serial = 12.0
                times = {1: 12.0, 2: 6.4, 4: 3.5, 8: 2.1, 16: 1.65}
                for p, elapsed in times.items():
                    speedup = serial / elapsed
                    efficiency = speedup / p
                    overhead = p * elapsed - serial
                    assert 0 < efficiency <= 1.000001
                    print(f"p={p:2} t={elapsed:4.2f} S={speedup:5.2f} E={efficiency:5.3f} To={overhead:5.2f}")
            '''), "El tiempo de p=1 de la versión paralela y el serial optimizado son referencias distintas y deben etiquetarse."),
        ),
        ("Formular por separado una hipótesis fuerte y una débil.", "Usar al menos cinco repeticiones y reportar dispersión.", "Identificar el primer punto de saturación y una causa verificable."),
        ("Comparar ejecutables con opciones de compilación distintas.", "Usar el mejor tiempo sin justificarlo.", "Omitir tamaño del problema, afinidad o hardware."),
        ("Definición explícita de tamaño fijo o trabajo por recurso.", "Aceleración y eficiencia derivadas de los mismos datos.", "Datos crudos y manifiesto de plataforma conservados."),
        (("Protocolo experimental", "../../../docs/REPRODUCIBILIDAD_EJERCICIOS.md#8-niveles-de-evidencia"), ("Planeación", "../../../docs/PLANEACION_CURSO.md")),
    ),
    spec(
        "01", "01_fundamentos", "03_memoria_roofline.ipynb", "Jerarquía de memoria y Roofline", (5, 6),
        "¿La ejecución está limitada por cómputo, ancho de banda, latencia o localidad?",
        ("Calcular intensidad aritmética.", "Aplicar el límite Roofline sin confundirlo con una medición.", "Reconocer localidad, NUMA y false sharing como causas observables."),
        ("Roofline acota rendimiento por min(pico, ancho de banda × intensidad).", "Una línea de caché compartida por escrituras de varios núcleos puede invalidarse aunque las variables sean distintas.", "En NUMA, ubicación de memoria, first-touch y afinidad forman parte del experimento."),
        (
            ("Límite Roofline", "Se calcula el techo para varias intensidades con unidades coherentes.", dedent('''
                peak_gflops = 800.0
                bandwidth_gbs = 120.0
                intensities = (0.125, 0.5, 1, 2, 4, 8, 16)
                ridge = peak_gflops / bandwidth_gbs
                assert abs(ridge - 20/3) < 1e-12
                print(f"punto de transición = {ridge:.2f} FLOP/byte")
                for intensity in intensities:
                    ceiling = min(peak_gflops, bandwidth_gbs * intensity)
                    assert 0 < ceiling <= peak_gflops
                    regime = "memoria" if bandwidth_gbs * intensity < peak_gflops else "cómputo"
                    print(f"I={intensity:6.3f} techo={ceiling:7.1f} GFLOP/s régimen={regime}")
            '''), "El techo no es rendimiento obtenido: se compara con mediciones de la misma precisión y operación."),
            ("Líneas de caché", "Se observa cuándo contadores adyacentes comparten una línea de 64 bytes.", dedent('''
                line_size = 64
                element_size = 8
                addresses = [i * element_size for i in range(16)]
                mapping = {i: address // line_size for i, address in enumerate(addresses)}
                assert len({mapping[i] for i in range(8)}) == 1
                for index, line in mapping.items(): print(f"contador {index:2} -> línea {line}")
                print("separación mínima en elementos:", line_size // element_size)
            '''), "Separar o alinear contadores puede reducir false sharing, pero aumenta memoria y debe medirse."),
        ),
        ("Estimar bytes transferidos y FLOP de un kernel.", "Medir una referencia con tamaño que exceda caché cuando la pregunta sea ancho de banda.", "Registrar afinidad y política NUMA junto con la curva."),
        ("Usar FLOP/s de pico de otra precisión.", "Confundir misses con prueba automática de false sharing.", "Comparar tamaños que realizan cantidades distintas de trabajo."),
        ("Unidades y precisión explícitas.", "Punto Roofline calculado y medición diferenciada.", "Hipótesis de memoria contrastada con al menos un contador o experimento controlado."),
        (("Planeación: memoria", "../../../docs/PLANEACION_CURSO.md"), ("Protocolo", "../../../docs/REPRODUCIBILIDAD_EJERCICIOS.md")),
    ),
    spec(
        "02", "02_memoria_compartida", "01_pthreads.ipynb", "Pthreads: ciclo de vida y partición", (7,),
        "¿Cómo crear trabajo concurrente sin perder argumentos, errores ni cobertura de datos?",
        ("Explicar create/join y la vida útil de argumentos.", "Particionar datos con cobertura comprobable.", "Comparar la salida paralela con una referencia serial."),
        ("`pthread_create` inicia una función con un argumento cuya vida útil debe abarcar el acceso del hilo.", "`pthread_join` establece finalización y permite recuperar estado; ignorar códigos de retorno oculta fallos.", "La partición debe especificar rangos semiabiertos y funcionar cuando n no es múltiplo del número de hilos."),
        (
            ("Rangos de trabajo", "Se prueba una partición por bloques para casos irregulares.", dedent('''
                def partition(n, workers):
                    q, r = divmod(n, workers)
                    starts = [worker * q + min(worker, r) for worker in range(workers)]
                    return [(start, start + q + (worker < r)) for worker, start in enumerate(starts)]
                for n, workers in ((3, 5), (17, 4), (32, 8)):
                    chunks = partition(n, workers)
                    flattened = [i for begin, end in chunks for i in range(begin, end)]
                    assert flattened == list(range(n))
                    print(n, workers, chunks)
            '''), "Los hilos sin elementos reciben un rango vacío válido; el programa no debe leer fuera de límites."),
            ("Referencia y reducción", "Se simula la suma de parciales y se compara con una referencia única.", dedent('''
                values = [((i * 17) % 23) - 11 for i in range(101)]
                chunks = partition(len(values), 6)
                partials = [sum(values[begin:end]) for begin, end in chunks]
                parallel_result = sum(partials)
                reference = sum(values)
                assert parallel_result == reference
                print({"partials": partials, "result": parallel_result})
            '''), "En C, cada parcial debe tener almacenamiento independiente y la combinación ocurre después de join."),
        ),
        ("Compilar y revisar `pthreads/thread_creation.c`.", "Agregar comprobación de cada retorno de la API.", "Probar n<threads, n no divisible y n grande."),
        ("Pasar la dirección de una variable de bucle compartida.", "Salir de `main` antes de join.", "Medir una versión paralela incorrecta."),
        ("Todos los retornos se comprueban.", "Cobertura de índices demostrada.", "Resultado igual a la referencia serial."),
        (("Creación de hilos", "../../../pthreads/thread_creation.c"), ("Ejemplo de mutex", "../../../pthreads/thread_mutex.c")),
    ),
    spec(
        "02", "02_memoria_compartida", "02_sincronizacion.ipynb", "Sincronización, condición y deadlock", (8, 10),
        "¿Qué invariante protege cada primitiva y cómo se detecta un ciclo de espera?",
        ("Relacionar mutex, condición y barrera con invariantes.", "Explicar happens-before sin usar tiempo como sincronización.", "Detectar un ciclo en un grafo wait-for."),
        ("Un mutex protege un invariante, no una línea aislada.", "Una variable de condición se espera dentro de un bucle que reevalúa el predicado.", "Un deadlock requiere exclusión, retención y espera, no expropiación y espera circular."),
        (
            ("Grafo de espera", "Se detecta un ciclo mediante búsqueda en profundidad.", dedent('''
                def has_cycle(graph):
                    visiting, done = set(), set()
                    def visit(node):
                        if node in visiting: return True
                        if node in done: return False
                        visiting.add(node)
                        if any(visit(next_node) for next_node in graph.get(node, ())): return True
                        visiting.remove(node); done.add(node); return False
                    return any(visit(node) for node in graph)
                safe = {"T0": ["T1"], "T1": []}
                deadlock = {"T0": ["T1"], "T1": ["T0"]}
                assert not has_cycle(safe) and has_cycle(deadlock)
                print("seguro:", has_cycle(safe), "deadlock:", has_cycle(deadlock))
            '''), "Ordenar globalmente la adquisición de recursos rompe la condición de espera circular."),
            ("Buffer acotado", "Se comprueban invariantes de ocupación para una secuencia productor-consumidor.", dedent('''
                from collections import deque
                capacity, buffer = 3, deque()
                operations = [("put", 4), ("put", 7), ("get", None), ("put", 9), ("get", None), ("get", None)]
                consumed = []
                for operation, value in operations:
                    if operation == "put":
                        assert len(buffer) < capacity
                        buffer.append(value)
                    else:
                        assert buffer
                        consumed.append(buffer.popleft())
                    assert 0 <= len(buffer) <= capacity
                assert consumed == [4, 7, 9]
                print(consumed)
            '''), "En Pthreads, el predicado sería `count>0` o `count<capacity` protegido por el mismo mutex."),
        ),
        ("Anotar el invariante al lado de cada estado compartido.", "Construir un caso que fuerce intercalaciones distintas.", "Ejecutar ThreadSanitizer cuando el toolchain lo soporte."),
        ("Usar `sleep` para ordenar hilos.", "Esperar condición con `if` en lugar de `while`.", "Bloquear recursos en órdenes diferentes."),
        ("Invariantes escritos y comprobados.", "Ausencia de ciclos en el orden de locks.", "Pruebas repetidas y detector de carreras documentado."),
        (("Mutex", "../../../pthreads/thread_mutex.c"), ("Deadlock", "../../../pthreads/thread_deadlock.c")),
    ),
    spec(
        "02", "02_memoria_compartida", "03_cpp20_atomics.ipynb", "C++20: RAII, jthread y atomics", (9, 10),
        "¿Cuándo una operación atómica es suficiente y qué relación de memoria necesita el algoritmo?",
        ("Diferenciar atomicidad, orden y exclusión mutua.", "Usar RAII y `std::jthread` para administrar vida útil.", "Elegir órdenes de memoria a partir del protocolo, no por intuición."),
        ("Una variable atómica evita carreras sobre esa variable, pero no vuelve atómico un invariante compuesto.", "Release publica escrituras anteriores y acquire permite observarlas cuando lee el valor publicado.", "`memory_order_relaxed` sirve para contadores sin relación de publicación; seq_cst ofrece el modelo global más fuerte y costoso."),
        (
            ("Separación de contadores", "Se calcula padding para evitar compartir líneas de caché cuando cada hilo escribe su contador.", dedent('''
                def padded_stride(value_bytes, cache_line=64):
                    return ((value_bytes + cache_line - 1) // cache_line) * cache_line
                assert padded_stride(8) == 64
                assert padded_stride(72) == 128
                for size in (4, 8, 16, 64, 72): print(size, padded_stride(size))
            '''), "En C++ se prefiere `std::hardware_destructive_interference_size` cuando está disponible, verificando la implementación."),
            ("Selección razonada", "Una tabla relaciona patrones mínimos con el orden que debe justificarse.", dedent('''
                protocols = {
                    "contador independiente": ("relaxed", "no publica otros datos"),
                    "bandera de publicación": ("release/acquire", "publica y consume estado previo"),
                    "algoritmo sin prueba formal": ("seq_cst", "punto de partida conservador"),
                    "invariante compuesto": ("mutex", "varias ubicaciones cambian juntas"),
                }
                assert protocols["invariante compuesto"][0] == "mutex"
                for pattern, decision in protocols.items(): print(f"{pattern:27} -> {decision[0]:15} | {decision[1]}")
            '''), "La tabla no reemplaza la prueba de happens-before del algoritmo concreto."),
        ),
        ("Reescribir un contador con relaxed y justificar por qué no publica datos.", "Modelar una bandera release/acquire con productor y consumidor.", "Ejecutar sanitizadores y comparar con una versión protegida por mutex."),
        ("Usar `volatile` como sincronización.", "Aplicar relaxed a una publicación sin relación de memoria.", "Crear hilos sin una política clara de cancelación y join."),
        ("No hay carreras en el detector.", "Cada orden de memoria tiene justificación escrita.", "La referencia protegida por mutex produce el mismo resultado."),
        (("Ejemplos Pthreads relacionados", "../../../pthreads/"), ("Planeación C++20", "../../../docs/PLANEACION_CURSO.md")),
    ),
    spec(
        "03", "03_openmp", "01_modelo_datos.ipynb", "OpenMP: fork-join y entorno de datos", (11, 12),
        "¿Qué variables comparte cada hilo y cuáles deben ser privadas para conservar corrección?",
        ("Explicar fork-join y equipos de hilos.", "Auditar `shared`, `private`, `firstprivate` y reducciones.", "Usar `default(none)` como herramienta de revisión."),
        ("OpenMP crea equipos alrededor de regiones paralelas y sincroniza implícitamente salvo cláusula contraria.", "El alcance léxico de C/C++ no basta para deducir el atributo de datos de OpenMP.", "Afinidad y schedule afectan localidad, pero no deben cambiar el resultado correcto."),
        (
            ("Auditoría de variables", "Se documenta el rol de cada dato en una suma paralela.", dedent('''
                variables = {
                    "entrada": ("shared", "solo lectura"),
                    "n": ("shared", "límite inmutable"),
                    "i": ("private", "índice de iteración"),
                    "parcial": ("private", "acumulador por hilo"),
                    "total": ("reduction", "combinación asociativa definida"),
                }
                assert variables["i"][0] == "private"
                for name, (scope, reason) in variables.items(): print(f"{name:8} {scope:10} {reason}")
            '''), "La auditoría convierte `default(none)` en una explicación del diseño y no solo en una exigencia sintáctica."),
            ("Schedule estático", "Se visualiza la asignación de iteraciones por bloques contiguos.", dedent('''
                n, threads = 19, 4
                owner = {}
                q, r = divmod(n, threads)
                start = 0
                for thread in range(threads):
                    end = start + q + (thread < r)
                    for i in range(start, end): owner[i] = thread
                    start = end
                assert sorted(owner) == list(range(n))
                for thread in range(threads): print(thread, [i for i in owner if owner[i] == thread])
            '''), "La implementación de `schedule(static)` puede distribuir chunks según la cláusula; se documenta la forma usada en el experimento."),
        ),
        ("Compilar `openmp/hello.cc` y observar identificadores de hilo.", "Agregar `default(none)` a un bucle y clasificar todas las variables.", "Registrar `OMP_NUM_THREADS`, afinidad y schedule."),
        ("Asumir que toda variable local es private.", "Escribir salida concurrente y usar su orden como evidencia.", "Cambiar schedule y tamaño a la vez."),
        ("Clasificación explícita de datos.", "Salida igual a referencia serial.", "Entorno OpenMP conservado en el informe."),
        (("Hello OpenMP", "../../../openmp/hello.cc"), ("Data sharing", "../../../openmp/data_sharing.c")),
    ),
    spec(
        "03", "03_openmp", "02_bucles_reducciones.ipynb", "OpenMP: bucles, reducciones y SIMD", (12, 13),
        "¿Cómo distribuir iteraciones y combinar resultados sin introducir desbalance ni error numérico injustificado?",
        ("Comparar schedules con una carga conocida.", "Distinguir reduction, atomic y critical.", "Medir error numérico además del tiempo."),
        ("Una reducción crea acumuladores privados y una combinación definida por el operador.", "Atomic protege una actualización compatible; critical serializa una región arbitraria.", "La suma en punto flotante no es asociativa y el orden paralelo puede cambiar el redondeo."),
        (
            ("Balance de carga", "Se compara una asignación estática contigua con round-robin para costos crecientes.", dedent('''
                costs = [1 + (i % 7) ** 2 for i in range(32)]
                threads = 4
                contiguous = [sum(costs[t*8:(t+1)*8]) for t in range(threads)]
                cyclic = [sum(costs[t::threads]) for t in range(threads)]
                def imbalance(loads): return max(loads) / (sum(loads) / len(loads))
                assert sum(contiguous) == sum(costs) == sum(cyclic)
                print("contiguo", contiguous, "desbalance", round(imbalance(contiguous), 3))
                print("cíclico ", cyclic, "desbalance", round(imbalance(cyclic), 3))
            '''), "El mejor schedule depende del costo por iteración, localidad y overhead; la tabla formula una hipótesis, no una regla universal."),
            ("Precisión de la reducción", "Se contrasta suma ingenua con `math.fsum` como referencia numérica más estable.", dedent('''
                import math
                values = [1e16, 1.0, -1e16] * 1000
                naive = sum(values)
                stable = math.fsum(values)
                print({"sum": naive, "fsum": stable, "error_absoluto": abs(naive - stable)})
                assert stable == 1000.0
            '''), "El resultado esperado y la tolerancia deben definirse antes de comparar estrategias paralelas."),
        ),
        ("Comparar `schedule(static)`, dinámico y guiado con la misma entrada.", "Validar una integral/reducción frente a referencia.", "Reportar tiempo, eficiencia y error para cada configuración."),
        ("Usar critical para toda la iteración.", "Aceptar igualdad exacta de flotantes sin análisis.", "Vectorizar un bucle con dependencias."),
        ("Cláusula de reducción correcta.", "Tolerancia y referencia documentadas.", "Schedule y chunk registrados con el resultado."),
        (("Integral OpenMP", "../../../openmp/integral.cc"), ("Reducción", "../../../openmp/reduction/integral.cc")),
    ),
    spec(
        "03", "03_openmp", "03_tareas_rendimiento.ipynb", "OpenMP: tareas, dependencias y granularidad", (14, 15),
        "¿Cuándo un DAG de tareas expone paralelismo suficiente para compensar el costo de creación y sincronización?",
        ("Representar tareas y dependencias.", "Distinguir task, taskgroup y taskwait.", "Elegir un corte de granularidad mediante medición."),
        ("Las tareas expresan trabajo potencialmente diferido y ejecutado por cualquier hilo del equipo.", "Las dependencias se asocian a regiones de almacenamiento y forman un DAG.", "Una recursión fina puede producir más overhead que trabajo; un cutoff conserva trabajo secuencial en hojas pequeñas."),
        (
            ("Camino crítico de tareas", "Se calcula el tiempo mínimo ideal de un DAG de bloques.", dedent('''
                duration = {"A": 3, "B": 5, "C": 2, "D": 4, "E": 1}
                deps = {"A": [], "B": [], "C": ["A"], "D": ["A", "B"], "E": ["C", "D"]}
                finish = {}
                for task in duration:
                    finish[task] = duration[task] + max((finish[p] for p in deps[task]), default=0)
                work, span = sum(duration.values()), max(finish.values())
                assert (work, span) == (15, 10)
                print({"work": work, "span": span, "ideal_parallelism": work/span})
            '''), "El span revela si más hilos pueden ayudar antes de considerar overhead y ancho de banda."),
            ("Modelo de cutoff", "Se estima cuándo el trabajo por tarea supera un overhead de creación medido.", dedent('''
                overhead_us = 3.2
                cost_per_item_us = 0.08
                candidates = (8, 16, 32, 64, 128, 256)
                for items in candidates:
                    useful = items * cost_per_item_us
                    ratio = useful / overhead_us
                    print(f"items={items:3} trabajo={useful:5.2f}us trabajo/overhead={ratio:4.1f}")
                cutoff = next(items for items in candidates if items * cost_per_item_us >= 5 * overhead_us)
                assert cutoff == 256
                print("cutoff inicial:", cutoff)
            '''), "El factor cinco es una hipótesis de partida; el cutoff final se obtiene en el hardware objetivo."),
        ),
        ("Dibujar dependencias de mergesort o stencil.", "Medir número de tareas y tiempo para varios cutoffs.", "Comparar con una versión `parallel for` cuando la estructura lo permita."),
        ("Crear una región paralela por llamada recursiva.", "Omitir `single` al generar el DAG.", "Medir solo un tamaño y declarar un cutoff universal."),
        ("DAG sin carreras ni dependencias faltantes.", "Cutoff justificado con curva.", "Resultado comparado con versión serial."),
        (("Planeación OpenMP", "../../../docs/PLANEACION_CURSO.md"), ("Ejemplos OpenMP", "../../../openmp/")),
    ),
    spec(
        "04", "04_mpi", "01_punto_a_punto.ipynb", "MPI punto a punto y progreso", (16, 17, 18),
        "¿Cómo hacer coincidir mensajes sin depender del orden accidental ni introducir interbloqueos?",
        ("Identificar comunicador, rango, tag, datatype y count.", "Comparar bloqueo, no bloqueo y `MPI_Sendrecv`.", "Validar emparejamiento y vida útil de buffers."),
        ("Un mensaje coincide por comunicador, origen permitido y tag; datatype/count determinan interpretación y capacidad.", "`MPI_Isend/Irecv` inicia operaciones cuyos buffers no pueden reutilizarse hasta completar la solicitud.", "El progreso y el buffering no deben usarse para justificar un patrón potencialmente bloqueante."),
        (
            ("Anillo determinista", "Se genera el contrato de mensajes para cada rango.", dedent('''
                def ring_contract(size, tag=17):
                    return [{"rank": r, "send_to": (r+1)%size, "recv_from": (r-1)%size, "tag": tag} for r in range(size)]
                contract = ring_contract(6)
                for row in contract:
                    sender = row["recv_from"]
                    assert (sender + 1) % 6 == row["rank"]
                    print(row)
            '''), "El mismo tag es seguro dentro del contrato del anillo; fases distintas deben distinguirse o sincronizarse."),
            ("Descomposición de halo", "Se calculan vecinos y rangos de un dominio 1D, incluidos extremos físicos.", dedent('''
                n, size = 25, 4
                q, r = divmod(n, size)
                start = 0
                rows = []
                for rank in range(size):
                    local = q + (rank < r)
                    rows.append((rank, start, start+local, rank-1 if rank else None, rank+1 if rank+1<size else None))
                    start += local
                assert rows[-1][2] == n
                for row in rows: print(row)
            '''), "Los procesos extremos usan condiciones de frontera o `MPI_PROC_NULL`; no reciben un halo inexistente."),
        ),
        ("Compilar `mpi/hello_mpi.c` y `mpi/ring_pass.c`.", "Construir una variante `Sendrecv` y otra no bloqueante.", "Probar tamaños de proceso 1, 2 y mayores que dos."),
        ("Asumir que `MPI_Send` siempre bufferiza.", "Reutilizar un buffer antes de Wait.", "Ignorar status y tamaño recibido."),
        ("Todos los mensajes tienen contrato compatible.", "No hay deadlock para tamaños admitidos.", "Errores MPI y códigos de salida se comprueban."),
        (("Hello MPI", "../../../mpi/hello_mpi.c"), ("Anillo", "../../../mpi/ring_pass.c")),
    ),
    spec(
        "04", "04_mpi", "02_colectivas_topologias.ipynb", "MPI colectivas, datatypes y topologías", (19, 20),
        "¿Qué patrón colectivo expresa la comunicación y cómo cambia su costo con procesos y datos?",
        ("Seleccionar broadcast, scatter/gather, reduce/allreduce.", "Modelar costo con latencia y ancho de banda.", "Calcular vecinos de una topología cartesiana."),
        ("Las colectivas deben invocarse en orden compatible por todos los procesos del comunicador.", "Un datatype derivado describe layout; no convierte automáticamente tipos ni corrige extensiones erróneas.", "Las topologías asocian estructura lógica y pueden facilitar mapeo, sin garantizar colocación física."),
        (
            ("Costo de colectivas", "Se compara un modelo lineal con uno arbóreo para broadcast.", dedent('''
                import math
                latency_us, bandwidth_gbs, bytes_ = 2.0, 12.0, 8_000_000
                transfer_us = bytes_ / (bandwidth_gbs * 1e9) * 1e6
                assert transfer_us > latency_us
                for p in (2, 4, 8, 16, 32):
                    linear = (p-1) * (latency_us + transfer_us)
                    tree = math.ceil(math.log2(p)) * (latency_us + transfer_us)
                    assert tree <= linear
                    print(f"p={p:2} lineal={linear:9.1f}us árbol={tree:9.1f}us")
            '''), "El modelo orienta la hipótesis; la biblioteca puede segmentar, usar árboles distintos y adaptar el algoritmo al tamaño."),
            ("Vecinos cartesianos", "Se enumeran coordenadas y vecinos sin periodicidad en una malla 3×4.", dedent('''
                rows, cols = 3, 4
                def rank(r, c): return r*cols+c if 0 <= r < rows and 0 <= c < cols else None
                assert rank(0, 0) == 0 and rank(2, 3) == 11 and rank(-1, 0) is None
                for r in range(rows):
                    for c in range(cols):
                        neighbors = {"N": rank(r-1,c), "S": rank(r+1,c), "W": rank(r,c-1), "E": rank(r,c+1)}
                        print(rank(r,c), (r,c), neighbors)
            '''), "En MPI, `MPI_Cart_shift` obtiene vecinos de acuerdo con dimensiones, periodicidad y posible reordenamiento."),
        ),
        ("Reemplazar una secuencia manual por la colectiva equivalente.", "Verificar counts y desplazamientos para tamaños irregulares.", "Medir colectiva por tamaño de mensaje y número de procesos."),
        ("Invocar colectivas en órdenes distintos.", "Suponer que reduce entrega resultado a todos.", "Crear datatype sin revisar extent."),
        ("Orden colectivo compatible.", "Counts, tipos y buffers válidos en cada rango.", "Modelo de costo contrastado con datos."),
        (("Ejemplos MPI", "../../../mpi/"), ("Planeación MPI", "../../../docs/PLANEACION_CURSO.md")),
    ),
    spec(
        "04", "04_mpi", "03_escalabilidad_slurm.ipynb", "MPI en clúster: Slurm y escalabilidad", (21,),
        "¿Cómo convertir una ejecución distribuida en un experimento repetible y atribuible a recursos concretos?",
        ("Distinguir asignación Slurm de lanzamiento MPI.", "Calcular escalado fuerte y débil.", "Registrar nodos, tareas, afinidad, versiones y repeticiones."),
        ("La reserva define recursos; `srun` o `mpiexec` inicia procesos según la integración del clúster.", "El mapeo proceso–núcleo–NUMA debe registrarse porque cambia comunicación y memoria.", "Los resultados válidos incluyen script, job id, módulos, entrada, tiempos crudos y manifiesto."),
        (
            ("Escalado fuerte", "Se calculan speedup y eficiencia de una serie sintética.", dedent('''
                times = {1: 84.0, 2: 44.5, 4: 24.0, 8: 14.2, 16: 10.1}
                baseline = times[1]
                assert list(times) == [1, 2, 4, 8, 16]
                for p, elapsed in times.items():
                    speedup = baseline / elapsed
                    efficiency = speedup / p
                    assert 0 < efficiency <= 1
                    print(f"p={p:2} tiempo={elapsed:5.1f}s speedup={speedup:5.2f} eficiencia={efficiency:5.3f}")
            '''), "La pérdida de eficiencia debe relacionarse con comunicación, desbalance o saturación mediante mediciones adicionales."),
            ("Plan de trabajos", "Se construye una matriz de recursos sin ejecutar el planificador.", dedent('''
                plan = []
                for nodes in (1, 2, 4):
                    for tasks_per_node in (1, 2, 4):
                        plan.append({"nodes": nodes, "ntasks_per_node": tasks_per_node, "total_ranks": nodes*tasks_per_node, "repetitions": 5})
                assert len(plan) == 9
                for row in plan: print(row)
            '''), "Cada punto debe ejecutarse con la misma entrada, política de afinidad y versión del binario."),
        ),
        ("Preparar un script `sbatch` con salida que incluya job id y hostnames.", "Ejecutar repeticiones evitando mezclar calentamiento.", "Conservar CSV/JSON y manifiesto junto con la gráfica."),
        ("Usar nodos asignados de manera interactiva sin registrar opciones.", "Comparar jobs con frecuencias o afinidades distintas.", "Presentar solo speedup sin tiempos crudos."),
        ("Script y configuración Slurm versionados.", "Cinco o más repeticiones por punto.", "Escalado acompañado de dispersión y explicación."),
        (("Scripts MPI", "../../../mpi/"), ("Protocolo de evidencia", "../../../docs/REPRODUCIBILIDAD_EJERCICIOS.md#8-niveles-de-evidencia")),
    ),
    spec(
        "05", "05_openmp_target", "01_modelo_offload.ipynb", "Modelo de offload y portabilidad", (22,),
        "¿Cuándo desplazar cómputo a un dispositivo compensa transferencia, inicialización y sincronización?",
        ("Separar host, device y runtime de offload.", "Modelar tiempo extremo a extremo.", "Diseñar fallback CPU con corrección equivalente."),
        ("Offload portable conserva una interfaz, no rendimiento idéntico entre dispositivos.", "El costo total incluye descubrimiento, asignación, mapeo, transferencia, kernel y sincronización.", "Una ruta CPU válida permite probar corrección sin afirmar evidencia de acelerador."),
        (
            ("Punto de equilibrio", "Se compara CPU con dispositivo incluyendo transferencias.", dedent('''
                cpu_rate = 40e9
                device_rate = 600e9
                link_rate = 24e9
                startup = 80e-6
                assert min(cpu_rate, device_rate, link_rate, startup) > 0
                for bytes_ in (1e5, 1e6, 1e7, 1e8, 1e9):
                    work = 4 * bytes_
                    cpu = work / cpu_rate
                    device = startup + 2*bytes_/link_rate + work/device_rate
                    print(f"bytes={bytes_:10.0f} cpu={cpu*1e3:8.3f}ms offload={device*1e3:8.3f}ms conviene={device<cpu}")
            '''), "El modelo debe recalibrarse con el enlace, dispositivo y kernel reales; no incluye aún solapamiento ni reutilización de datos."),
            ("Selección de ruta", "Se hace explícita una política de fallback verificable.", dedent('''
                def execution_path(devices, requested=True):
                    if requested and devices > 0: return "device"
                    return "host-fallback"
                assert execution_path(0) == "host-fallback"
                assert execution_path(1) == "device"
                for devices in (0, 1, 2): print(devices, execution_path(devices))
            '''), "El informe registra la ruta usada; ejecutar fallback no demuestra soporte ni rendimiento del dispositivo."),
        ),
        ("Registrar número y tipo de dispositivos visibles.", "Medir por separado transferencia y cómputo.", "Comparar resultado con la misma referencia serial."),
        ("Cronometrar solo el kernel y llamarlo tiempo total.", "Suponer que portabilidad implica ausencia de ajustes.", "Ocultar que se ejecutó fallback CPU."),
        ("Ruta host/device registrada.", "Tiempo extremo a extremo y tiempo de kernel separados.", "Tolerancia de corrección idéntica."),
        (("Toolchain", "../../../config/course-toolchain.cmake"), ("Planeación offload", "../../../docs/PLANEACION_CURSO.md")),
    ),
    spec(
        "05", "05_openmp_target", "02_openmp_target.ipynb", "OpenMP target: datos, reducción y fallback", (23, 24),
        "¿Cómo minimizar mapeos sin perder coherencia entre host y dispositivo?",
        ("Explicar `target`, `teams` y `distribute parallel for`.", "Elegir map/to/from/alloc según el flujo de datos.", "Validar reducción en dispositivo y fallback."),
        ("Las regiones de datos persistentes evitan transferencias repetidas cuando varias operaciones reutilizan arreglos.", "`map(to:)` inicializa en dispositivo, `from:` recupera y `tofrom:` realiza ambas direcciones.", "Una reducción requiere soporte del compilador/runtime y se valida con una referencia numérica."),
        (
            ("Contrato de mapeo", "Se deriva la dirección mínima para entradas y salidas de vector add.", dedent('''
                arrays = {"a": {"read": True, "write": False}, "b": {"read": True, "write": False}, "c": {"read": False, "write": True}}
                def clause(access):
                    if access["read"] and access["write"]: return "tofrom"
                    if access["read"]: return "to"
                    if access["write"]: return "from"
                    return "alloc"
                mapping = {name: clause(access) for name, access in arrays.items()}
                assert mapping == {"a": "to", "b": "to", "c": "from"}
                print(mapping)
            '''), "El contrato se revisa cuando un arreglo persiste entre kernels o se inicializa en el dispositivo."),
            ("Fallback correcto", "Se ejecuta una referencia portable de vector add y se valida elemento a elemento.", dedent('''
                n = 1000
                a = [i * 0.5 for i in range(n)]
                b = [1.0 - i * 0.25 for i in range(n)]
                c = [x + y for x, y in zip(a, b)]
                expected = [1.0 + i * 0.25 for i in range(n)]
                error = max(abs(x-y) for x, y in zip(c, expected))
                assert error < 1e-12
                print({"n": n, "max_error": error, "path": "modelo CPU para validar el kernel target"})
            '''), "La misma entrada y tolerancia se reutilizan cuando el kernel OpenMP target se ejecuta en hardware real."),
        ),
        ("Implementar vector add con región de datos explícita.", "Reutilizar datos durante varias operaciones y medir ambas variantes.", "Registrar compilador, plugin de offload y dispositivo."),
        ("Mapear `tofrom` todo por comodidad.", "Acceder en host antes de sincronizar.", "Declarar éxito GPU sin comprobar `omp_is_initial_device`."),
        ("Mapeo mínimo justificado.", "Fallback y dispositivo producen resultados equivalentes.", "Informe identifica inequívocamente dónde se ejecutó."),
        (("Planeación target", "../../../docs/PLANEACION_CURSO.md"), ("Protocolo", "../../../docs/REPRODUCIBILIDAD_EJERCICIOS.md")),
    ),
    spec(
        "06", "06_cuda", "01_modelo_cuda.ipynb", "CUDA: modelo SIMT, grid y memoria", (25, 26),
        "¿Cómo mapear un dominio a grid/bloque/hilo y medir el costo completo con errores comprobados?",
        ("Explicar host, device, kernel, warp, bloque y grid.", "Calcular cobertura con guardas de borde.", "Separar transferencia, kernel y tiempo extremo a extremo."),
        ("Los hilos de un warp siguen un modelo SIMT; divergencia serializa caminos dentro del warp.", "Los bloques deben ser independientes salvo coordinación mediante lanzamientos separados o mecanismos específicos.", "Los errores de lanzamiento y los errores asíncronos se comprueban en puntos distintos."),
        (
            ("Cobertura del grid", "Se calcula el número de bloques y se prueban tamaños no múltiplos.", dedent('''
                def launch_shape(n, block):
                    grid = (n + block - 1) // block
                    launched = grid * block
                    return {"n": n, "block": block, "grid": grid, "launched": launched, "guarded": launched-n}
                for n in (1, 255, 256, 257, 1000, 1_000_003):
                    row = launch_shape(n, 256)
                    assert row["launched"] >= n and row["guarded"] < 256
                    print(row)
            '''), "Cada kernel usa `if (i<n)` cuando la geometría lanza hilos adicionales."),
            ("Descomposición temporal", "Se evita atribuir al kernel los costos de preparación y copia.", dedent('''
                runs = {"alloc": 0.18, "h2d": 1.45, "kernel": 0.62, "d2h": 1.10, "sync": 0.08}
                total = sum(runs.values())
                for phase, elapsed in runs.items(): print(f"{phase:8} {elapsed:5.2f} ms {100*elapsed/total:5.1f}%")
                print("total", round(total, 3), "ms")
                assert total > runs["kernel"]
            '''), "Eventos CUDA miden trabajo en streams; un reloj de host delimita el tiempo extremo a extremo con sincronización explícita."),
        ),
        ("Construir vector add y comparar contra CPU.", "Probar tamaños 0/1, no múltiplos y grandes.", "Ejecutar Compute Sanitizer y conservar dispositivo/toolkit."),
        ("Omitir la guarda de borde.", "Leer resultados antes de sincronizar.", "Comprobar solo `cudaGetLastError` y no el trabajo asíncrono."),
        ("Máximo error dentro de tolerancia.", "Todos los estados CUDA comprobados.", "Kernel y total reportados por separado."),
        (("Guía CUDA", "README.md"), ("Ejemplos CUDA", "../../ejemplos/06_cuda/README.md")),
    ),
    spec(
        "06", "06_cuda", "02_memoria_tiling.ipynb", "CUDA: coalescencia, memoria compartida y tiling", (27, 28),
        "¿Cómo aumentar reutilización sin exceder recursos ni romper bordes y sincronización?",
        ("Relacionar coalescencia, bancos y memoria compartida.", "Calcular recursos de un tile.", "Validar GEMM tiled frente a CPU para dimensiones irregulares."),
        ("Tiling carga datos reutilizados en memoria compartida y sincroniza antes de consumirlos.", "Un tile mayor puede aumentar reutilización pero también registros, memoria compartida y presión de ocupación.", "Las dimensiones no múltiplos requieren cargas condicionadas y ceros fuera del dominio."),
        (
            ("Recursos por tile", "Se calculan hilos, bloques, memoria compartida y tiles de K para GEMM.", dedent('''
                def tile_resources(m, n, k, tile, bytes_per_value=4):
                    return {
                        "grid": ((n+tile-1)//tile, (m+tile-1)//tile),
                        "threads_per_block": tile*tile,
                        "shared_bytes": 2*tile*tile*bytes_per_value,
                        "k_tiles": (k+tile-1)//tile,
                    }
                for tile in (8, 16, 32):
                    row = tile_resources(1000, 777, 513, tile)
                    assert row["threads_per_block"] <= 1024
                    print(tile, row)
            '''), "La validez geométrica no garantiza buena ocupación; se consulta el límite real y el perfil del kernel."),
            ("Intensidad aproximada", "Se compara reutilización ideal de una GEMM ingenua y una tiled.", dedent('''
                def intensity(tile, bytes_per_value=4):
                    flops = 2 * tile * tile * tile
                    bytes_loaded = 2 * tile * tile * bytes_per_value
                    return flops / bytes_loaded
                for tile in (8, 16, 32): print(tile, f"{intensity(tile):.2f} FLOP/byte por fase ideal")
                assert intensity(32) > intensity(8)
            '''), "El cálculo ideal omite escrituras, cachés, bordes y recargas; sirve para formular la hipótesis de reutilización."),
        ),
        ("Comparar GEMM ingenua, tiled y cuBLAS con la misma precisión.", "Incluir dimensiones no divisibles por tile.", "Perfilar coalescencia, bancos, ocupación y tiempo total."),
        ("Sincronizar fuera de una rama que no alcanza todo el bloque.", "Comparar operaciones diferentes.", "Elegir tile solo por tiempo de un caso."),
        ("Bordes comprobados con referencia CPU.", "Recursos por bloque dentro de límites.", "Interpretación sustentada en métricas del perfil."),
        (("Guía CUDA", "README.md"), ("Fuentes CUDA heredadas", "../../../cuda/03_cuda_thread_programming/")),
    ),
    spec(
        "06", "06_cuda", "03_bibliotecas_perfiles.ipynb", "CUDA: bibliotecas, streams y perfiles", (28, 29, 30),
        "¿Cuándo usar una biblioteca acelerada y cómo separar preparación, transferencia y ejecución repetida?",
        ("Seleccionar Thrust/CUB o una biblioteca de dominio.", "Administrar handle, plan, descriptor, workspace y stream.", "Interpretar Nsight y tiempo extremo a extremo sin ocultar preparación."),
        ("Las bibliotecas aprovechan algoritmos y kernels especializados cuando layout, tipo y tamaño coinciden con su contrato.", "Planes y workspaces se reutilizan; crearlos dentro de cada repetición distorsiona la medición.", "Una comparación justa mantiene operación matemática, precisión, tolerancia, datos residentes y costos incluidos."),
        (
            ("Selección inicial", "Una función explícita evita recomendar una biblioteca sin considerar la operación.", dedent('''
                def candidate(operation):
                    return {
                        "transform": "Thrust",
                        "reduce": "CUB/Thrust",
                        "gemm": "cuBLAS/cuBLASLt",
                        "fft": "cuFFT",
                        "spmv": "cuSPARSE",
                        "dense_solve": "cuSOLVER",
                        "random": "cuRAND",
                    }.get(operation, "kernel propio o composición")
                operations = ("transform", "reduce", "gemm", "fft", "spmv", "dense_solve", "random", "stencil_fused")
                assert candidate("gemm") == "cuBLAS/cuBLASLt"
                assert candidate("stencil_fused") == "kernel propio o composición"
                for operation in operations: print(f"{operation:14} -> {candidate(operation)}")
            '''), "La selección final incorpora tamaños, layout, residencia, precisión, repetición y posibilidad de fusión."),
            ("Perfil de fases", "Se resume una serie repetida mediante mediana y se separa preparación.", dedent('''
                import statistics
                phases = {
                    "plan_once": [3.4],
                    "h2d": [1.1, 1.0, 1.2, 1.1, 1.0],
                    "compute": [0.42, 0.40, 0.41, 0.43, 0.40],
                    "d2h": [0.8, 0.82, 0.79, 0.81, 0.8],
                }
                medians = {name: statistics.median(values) for name, values in phases.items()}
                repeated_total = medians["h2d"] + medians["compute"] + medians["d2h"]
                print(medians, "total_repetido_ms", repeated_total)
                assert repeated_total > medians["compute"]
            '''), "Se publican tanto ejecución con datos residentes como extremo a extremo; el plan se amortiza solo cuando se reutiliza."),
        ),
        ("Implementar el ciclo crear–configurar–ejecutar–validar–destruir.", "Comparar una biblioteca con referencia y kernel correcto.", "Conservar comandos Nsight y métricas que respondan la pregunta."),
        ("Incluir creación de plan solo en una variante.", "Comparar column-major con row-major sin ajustar layout.", "Usar ocupación alta como sinónimo de rendimiento."),
        ("Estados de biblioteca comprobados.", "Recursos liberados y streams documentados.", "Preparación, ejecución residente y total separados."),
        (("Guía y bibliotecas CUDA", "README.md"), ("Ejemplos CUDA", "../../ejemplos/06_cuda/README.md")),
    ),
    spec(
        "07", "07_hibrido", "01_mpi_openmp.ipynb", "Híbrido MPI + OpenMP", (31, 34),
        "¿Cómo mapear procesos e hilos a nodos, NUMA y núcleos sin sobresuscripción?",
        ("Relacionar ranks, hilos, núcleos y dominios NUMA.", "Interpretar niveles de `MPI_Init_thread`.", "Diseñar afinidad y first-touch reproducibles."),
        ("MPI distribuye memoria entre procesos; OpenMP explota memoria compartida dentro del proceso.", "El nivel de soporte de hilos limita qué hilos pueden invocar MPI.", "ranks × threads debe corresponder a la asignación y la afinidad debe evitar migraciones no controladas."),
        (
            ("Mapa de recursos", "Se genera una asignación simple por nodo y se comprueba que no exceda núcleos.", dedent('''
                nodes, cores_per_node, ranks_per_node, threads_per_rank = 2, 32, 4, 8
                assert ranks_per_node * threads_per_rank <= cores_per_node
                mapping = []
                for node in range(nodes):
                    for local_rank in range(ranks_per_node):
                        first_core = local_rank * threads_per_rank
                        mapping.append((node, local_rank, tuple(range(first_core, first_core+threads_per_rank))))
                for row in mapping: print(row)
            '''), "En hardware con SMT o NUMA, la política se adapta y se registra mediante herramientas del runtime/planificador."),
            ("Soporte de hilos MPI", "Se ordenan los niveles y se verifica una solicitud.", dedent('''
                levels = {"SINGLE": 0, "FUNNELED": 1, "SERIALIZED": 2, "MULTIPLE": 3}
                requested, provided = "FUNNELED", "SERIALIZED"
                assert levels[provided] >= levels[requested]
                for name, value in levels.items(): print(value, name)
                print("solicitado", requested, "provisto", provided)
            '''), "El programa aborta o cambia de estrategia si el nivel provisto es inferior al solicitado."),
        ),
        ("Elegir ranks por NUMA y hilos por rank.", "Registrar `OMP_PLACES`, `OMP_PROC_BIND` y opciones Slurm.", "Comparar MPI puro, OpenMP puro e híbrido con igual total de núcleos."),
        ("Usar `MPI_THREAD_MULTIPLE` sin necesitarlo ni medir costo.", "Olvidar que bibliotecas internas pueden crear hilos.", "Comparar configuraciones con distinto total de recursos."),
        ("No hay sobresuscripción involuntaria.", "Nivel MPI provisto comprobado.", "Afinidad y first-touch documentados."),
        (("Entornos de clúster", "../../../topicos_avanzados/ENTORNOS_CLUSTER.md"), ("Planeación híbrida", "../../../docs/PLANEACION_CURSO.md")),
    ),
    spec(
        "07", "07_hibrido", "02_mpi_gpu.ipynb", "Híbrido MPI + GPU", (32, 34),
        "¿Cómo asignar procesos a dispositivos y solapar halos sin ocultar transferencias?",
        ("Construir un mapeo rank local–GPU.", "Descomponer halo, transferencia y comunicación.", "Distinguir MPI GPU-aware de staging por host."),
        ("El rank local, no el global, suele determinar la GPU dentro de un nodo.", "Cuando hay más ranks que GPUs aparece compartición que debe ser intencional.", "GPU-aware MPI puede aceptar buffers de dispositivo, pero soporte, ruta y sincronización deben verificarse."),
        (
            ("Asignación de dispositivos", "Se detecta sobresuscripción a partir de ranks locales y GPUs visibles.", dedent('''
                local_ranks, gpus = 6, 4
                assignment = {rank: rank % gpus for rank in range(local_ranks)}
                users = {gpu: [rank for rank, selected in assignment.items() if selected == gpu] for gpu in range(gpus)}
                print(assignment)
                for gpu, ranks in users.items(): print("GPU", gpu, "ranks", ranks, "compartida", len(ranks)>1)
                assert set(assignment.values()) == set(range(gpus))
            '''), "Si la política exige un rank por GPU, la asignación debe fallar en lugar de compartir silenciosamente."),
            ("Volumen de halo", "Se calcula comunicación por paso para una malla 3D descompuesta en una dimensión.", dedent('''
                ny, nz, layers, bytes_per_value = 512, 256, 2, 8
                one_face = ny * nz * layers * bytes_per_value
                interior_rank = 2 * one_face
                print({"one_face_MiB": one_face/2**20, "interior_rank_MiB": interior_rank/2**20})
                assert interior_rank == 2 * one_face
            '''), "El modelo se combina con ancho de banda PCIe/NVLink y red para decidir staging, empaquetado y solapamiento."),
        ),
        ("Registrar rank global/local, bus id y modelo de GPU.", "Comparar staging host y GPU-aware cuando ambos existan.", "Medir interior, halo, red y sincronización con la misma entrada."),
        ("Seleccionar GPU con rank global.", "Asumir GPU-aware por aceptar un puntero.", "Sincronizar todo el dispositivo y eliminar el solapamiento."),
        ("Mapeo proceso–GPU inequívoco.", "Halos validados contra referencia.", "Ruta de comunicación y sincronización documentadas."),
        (("Entornos de clúster", "../../../topicos_avanzados/ENTORNOS_CLUSTER.md"), ("Protocolo hardware", "../../../docs/REPRODUCIBILIDAD_EJERCICIOS.md")),
    ),
    spec(
        "07", "07_hibrido", "03_perfilado_reproducible.ipynb", "Perfilado integral y reproducibilidad", (33, 34),
        "¿Qué fracción del tiempo corresponde a cómputo, comunicación, transferencia, E/S y espera?",
        ("Diseñar una línea temporal de extremo a extremo.", "Resumir repeticiones con mediana y dispersión.", "Relacionar cuellos de botella con evidencia de herramientas."),
        ("Un perfil útil comienza con una pregunta y una región delimitada.", "Mediana y MAD son robustas frente a algunos outliers, pero los datos crudos se conservan.", "Las herramientas cambian el tiempo; la corrida instrumentada explica comportamiento y la corrida ligera cuantifica rendimiento."),
        (
            ("Mediana y MAD", "Se resume una muestra sin eliminar observaciones silenciosamente.", dedent('''
                import statistics
                samples = [10.2, 10.0, 10.1, 10.3, 10.05, 14.8, 10.15]
                median = statistics.median(samples)
                mad = statistics.median(abs(value-median) for value in samples)
                print({"samples": samples, "median": median, "MAD": mad})
                assert mad > 0
            '''), "El valor 14.8 se investiga con logs y sistema; no se borra solo porque sea incómodo."),
            ("Descomposición del total", "Se verifica que las fases y el tiempo no atribuido concuerden con el total.", dedent('''
                phases = {"compute": 48.0, "mpi": 21.0, "h2d_d2h": 12.0, "io": 7.0, "barriers": 6.0}
                total = 100.0
                unattributed = total - sum(phases.values())
                assert unattributed >= 0
                for name, percent in phases.items(): print(f"{name:10} {percent:5.1f}%")
                print("no atribuido", unattributed, "%")
            '''), "La suma inferior a 100 % hace visible instrumentación incompleta; una suma superior indica doble conteo."),
        ),
        ("Definir una pregunta para cada herramienta de perfil.", "Ejecutar serie ligera y corrida instrumentada separadas.", "Conservar comando, versiones, trazas y resumen derivado."),
        ("Perfilar todo sin región ni hipótesis.", "Comparar tiempos instrumentados con no instrumentados como equivalentes.", "Presentar porcentajes que no reconcilian con el total."),
        ("Pregunta y región declaradas.", "Datos crudos trazables al resumen.", "Conclusión respaldada por métrica y no por una captura aislada."),
        (("Protocolo de evidencia", "../../../docs/REPRODUCIBILIDAD_EJERCICIOS.md"), ("Planeación híbrida", "../../../docs/PLANEACION_CURSO.md")),
    ),
    spec(
        "08", "08_proyecto", "01_guia_proyecto.ipynb", "Guía del proyecto final reproducible", (35, 36, 37, 38),
        "¿Cómo demostrar corrección, rendimiento y reproducibilidad de una solución paralela completa?",
        ("Convertir una pregunta en hipótesis y matriz experimental.", "Definir referencia serial, pruebas y tolerancia antes de optimizar.", "Preparar evidencia, informe y defensa reproducibles."),
        ("El proyecto comienza con una referencia correcta y entradas representativas.", "Cada optimización cambia una variable controlada y conserva pruebas.", "La entrega incluye código, ambiente, comandos, datos, análisis de incertidumbre, limitaciones y atribución."),
        (
            ("Matriz experimental", "Se enumeran configuraciones sin ejecutar experimentos implícitos.", dedent('''
                from itertools import product
                sizes = (100_000, 1_000_000)
                resources = (1, 2, 4, 8)
                variants = ("serial", "parallel")
                matrix = [dict(size=n, resources=p, variant=v, repetitions=7) for n, p, v in product(sizes, resources, variants) if v == "parallel" or p == 1]
                assert all(row["repetitions"] >= 5 for row in matrix)
                print("configuraciones:", len(matrix))
                for row in matrix: print(row)
            '''), "La matriz se reduce si el costo es excesivo, pero la decisión se documenta antes de observar resultados."),
            ("Puerta de aceptación", "Se modelan condiciones que deben cumplirse antes de interpretar speedup.", dedent('''
                evidence = {
                    "serial_reference": True,
                    "correctness_tests": True,
                    "platform_manifest": True,
                    "raw_measurements": True,
                    "repetitions": 7,
                    "error_within_tolerance": True,
                }
                required_true = ("serial_reference", "correctness_tests", "platform_manifest", "raw_measurements", "error_within_tolerance")
                accepted = all(evidence[key] for key in required_true) and evidence["repetitions"] >= 5
                assert accepted
                print("habilitado para análisis de rendimiento:", accepted)
            '''), "Si la corrección falla, se documenta el diseño pero no se califican aceleración ni eficiencia."),
        ),
        ("Redactar pregunta, hipótesis, variables y riesgos.", "Automatizar construcción, pruebas y exportación de métricas.", "Ensayar reproducción desde un checkout limpio y preparar la defensa."),
        ("Optimizar antes de fijar referencia.", "Escoger solo resultados favorables.", "Entregar gráficas sin datos, comandos o unidades."),
        ("Checkout limpio reproduce pruebas y figuras.", "Resultados incluyen incertidumbre y limitaciones.", "Licencias, contribuciones y uso de fuentes están declarados."),
        (("Rúbrica y calendario", "../../../docs/PLANEACION_CURSO.md#7-evaluaciones"), ("Protocolo reproducible", "../../../docs/REPRODUCIBILIDAD_EJERCICIOS.md")),
    ),
)

TOPIC_TITLES = {
    "00": "Entorno y lenguajes",
    "01": "Fundamentos, arquitectura y rendimiento",
    "02": "Memoria compartida",
    "03": "OpenMP",
    "04": "MPI",
    "05": "OpenMP target",
    "06": "CUDA C++",
    "07": "Programación híbrida y perfilado",
    "08": "Proyecto final",
}


def markdown_cell(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict[str, object]:
    normalized = dedent(source).strip() + "\n"
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": normalized.splitlines(keepends=True)}


def bullets(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items)


def notebook(specification: NotebookSpec) -> dict[str, object]:
    session_text = ", ".join(str(value) for value in specification.sessions)
    cells: list[dict[str, object]] = [
        markdown_cell(
            f"[← Volver al índice del curso](../../../INDICE_CURSO.md) · [Guía del tema {specification.topic}](README.md)\n\n"
            f"# {specification.title}\n\n"
            f"**Tema:** {specification.topic} · **Sesiones:** {session_text} · **Edición:** 1.0.2026\n\n"
            f"**Pregunta guía:** {specification.question}\n"
        ),
        markdown_cell("## Resultados de aprendizaje\n\n" + bullets(specification.outcomes) + "\n"),
        markdown_cell("## Modelo conceptual\n\n" + "\n\n".join(specification.concepts) + "\n"),
        code_cell(dedent(f'''
            from pathlib import Path

            def find_repository(start: Path) -> Path:
                for candidate in (start.resolve(), *start.resolve().parents):
                    if (candidate / "INDICE_CURSO.md").is_file():
                        return candidate
                raise RuntimeError("No se encontró la raíz del repositorio")

            ROOT = find_repository(Path.cwd())
            TOPIC = "{specification.topic}"
            NOTEBOOK = "{specification.directory}/{specification.filename}"
            assert (ROOT / "curso" / "notebooks" / "{specification.directory}" / "README.md").is_file()
            print(f"Repositorio: {{ROOT}}")
            print(f"Notebook: {{NOTEBOOK}}")
        ''')),
    ]
    for heading, introduction, source, interpretation in specification.analyses:
        cells.extend(
            [
                markdown_cell(f"## {heading}\n\n{introduction}\n"),
                code_cell(source),
                markdown_cell(f"**Interpretación.** {interpretation}\n"),
            ]
        )
    cells.extend(
        [
            markdown_cell("## Práctica reproducible\n\n" + "\n".join(f"{index}. {item}" for index, item in enumerate(specification.practice, 1)) + "\n"),
            markdown_cell("## Errores frecuentes\n\n" + bullets(specification.pitfalls) + "\n\n## Criterios de aceptación\n\n" + bullets(specification.acceptance) + "\n"),
            markdown_cell("## Referencias y material relacionado\n\n" + "\n".join(f"- [{label}]({target})" for label, target in specification.references) + "\n"),
            markdown_cell(f"[← Volver al índice del curso](../../../INDICE_CURSO.md) · [Continuar desde la guía del tema {specification.topic}](README.md)\n"),
        ]
    )
    for index, cell in enumerate(cells):
        cell["id"] = f"{str(cell['cell_type'])[0]}-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "course": {
                "edition": "1.0.2026",
                "topic": specification.topic,
                "directory": specification.directory,
                "sessions": list(specification.sessions),
                "title": specification.title,
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="No escribe; falla si un notebook difiere del canónico")
    return parser.parse_args()


def manifest() -> dict[str, object]:
    topics: list[dict[str, object]] = []
    for topic in TOPIC_TITLES:
        selected = [item for item in SPECS if item.topic == topic]
        directories = {item.directory for item in selected}
        if len(directories) != 1:
            raise ValueError(f"El tema {topic} no tiene un directorio único")
        topics.append(
            {
                "topic": topic,
                "title": TOPIC_TITLES[topic],
                "directory": selected[0].directory,
                "notebooks": [
                    {
                        "filename": item.filename,
                        "title": item.title,
                        "sessions": list(item.sessions),
                    }
                    for item in selected
                ],
            }
        )
    return {"schema_version": "1.0", "edition": "1.0.2026", "total_notebooks": len(SPECS), "topics": topics}


def main() -> int:
    args = parse_args()
    repository = Path(__file__).resolve().parent.parent
    failures: list[str] = []
    for specification in SPECS:
        target = repository / "curso" / "notebooks" / specification.directory / specification.filename
        rendered = json.dumps(notebook(specification), ensure_ascii=False, indent=1) + "\n"
        if args.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != rendered:
                failures.append(str(target.relative_to(repository)))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
    manifest_target = repository / "validation" / "notebooks-manifest.json"
    rendered_manifest = json.dumps(manifest(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not manifest_target.is_file() or manifest_target.read_text(encoding="utf-8") != rendered_manifest:
            failures.append(str(manifest_target.relative_to(repository)))
    else:
        manifest_target.write_text(rendered_manifest, encoding="utf-8")
    if failures:
        print("Notebooks ausentes o diferentes del generador:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"Notebooks canónicos verificados: {len(SPECS)}" if args.check else f"Notebooks generados: {len(SPECS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
