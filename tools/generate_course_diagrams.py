#!/usr/bin/env python3
"""Genera los diagramas SVG compartidos por los notebooks del curso."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


STYLE = """
  .title { font: 700 28px system-ui, sans-serif; fill: #243447; }
  .label { font: 600 18px system-ui, sans-serif; fill: #243447; }
  .small { font: 15px system-ui, sans-serif; fill: #334155; }
  .tiny { font: 13px system-ui, sans-serif; fill: #475569; }
  .box { fill: #fdfbf5; stroke: #50677a; stroke-width: 2; rx: 12; }
  .blue { fill: #dce9ef; stroke: #506f83; stroke-width: 2; rx: 12; }
  .green { fill: #dfeadf; stroke: #5d7d65; stroke-width: 2; rx: 12; }
  .accent { fill: #f3ddd4; stroke: #a95642; stroke-width: 2.5; rx: 12; }
  .line { fill: none; stroke: #50677a; stroke-width: 2.5; marker-end: url(#arrow); }
  .critical { fill: none; stroke: #a95642; stroke-width: 4; marker-end: url(#arrow-red); }
  .dash { fill: none; stroke: #718096; stroke-width: 2; stroke-dasharray: 8 7; marker-end: url(#arrow); }
"""


def svg(title: str, description: str, body: str, height: int = 440) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" viewBox="0 0 960 {height}" role="img" aria-labelledby="title desc">
<title id="title">{title}</title>
<desc id="desc">{description}</desc>
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#50677a"/></marker>
  <marker id="arrow-red" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#a95642"/></marker>
</defs>
<style>{STYLE}</style>
<rect width="960" height="{height}" fill="#fbf7ed"/>
<text x="40" y="46" class="title">{title}</text>
{body}
</svg>
'''


ASSETS = {
    "ruta-reproducible.svg": svg(
        "Ruta de trabajo reproducible",
        "Secuencia desde la pregunta y la referencia serial hasta la validación, medición e informe.",
        '''
<rect x="35" y="135" width="135" height="80" class="blue"/><text x="102" y="168" text-anchor="middle" class="label">Pregunta</text><text x="102" y="192" text-anchor="middle" class="small">e hipótesis</text>
<rect x="195" y="135" width="135" height="80" class="box"/><text x="262" y="168" text-anchor="middle" class="label">Referencia</text><text x="262" y="192" text-anchor="middle" class="small">serial correcta</text>
<rect x="355" y="135" width="135" height="80" class="box"/><text x="422" y="168" text-anchor="middle" class="label">Versión</text><text x="422" y="192" text-anchor="middle" class="small">paralela</text>
<rect x="515" y="135" width="135" height="80" class="accent"/><text x="582" y="168" text-anchor="middle" class="label">Validar</text><text x="582" y="192" text-anchor="middle" class="small">salida y errores</text>
<rect x="675" y="135" width="115" height="80" class="green"/><text x="732" y="180" text-anchor="middle" class="label">Medir</text>
<rect x="815" y="135" width="110" height="80" class="blue"/><text x="870" y="168" text-anchor="middle" class="label">Informar</text><text x="870" y="192" text-anchor="middle" class="small">con evidencia</text>
<path d="M170 175 H195 M330 175 H355 M490 175 H515 M650 175 H675 M790 175 H815" class="line"/>
<path d="M870 235 C870 340 420 340 420 235" class="dash"/><text x="640" y="330" text-anchor="middle" class="small">una conclusión nueva genera otra hipótesis</text>
<text x="480" y="95" text-anchor="middle" class="small">La velocidad se interpreta solo después de demostrar corrección.</text>
'''),
    "capas-toolchain.svg": svg(
        "Capas de una ejecución portable",
        "Capas desde el código fuente y el estándar hasta el compilador, runtime, sistema operativo y hardware.",
        '''
<rect x="90" y="90" width="780" height="55" class="blue"/><text x="480" y="125" text-anchor="middle" class="label">Código fuente + estándar seleccionado</text>
<rect x="140" y="165" width="680" height="55" class="box"/><text x="480" y="200" text-anchor="middle" class="label">Compilador + biblioteca estándar</text>
<rect x="190" y="240" width="580" height="55" class="green"/><text x="480" y="275" text-anchor="middle" class="label">Runtime y dependencias</text>
<rect x="240" y="315" width="480" height="55" class="box"/><text x="480" y="350" text-anchor="middle" class="label">Sistema operativo + arquitectura + hardware</text>
<path d="M480 145 V165 M480 220 V240 M480 295 V315" class="line"/>
<text x="480" y="410" text-anchor="middle" class="small">Una macro describe una capa; una prueba reproducible registra todas.</text>
'''),
    "dag-camino-critico.svg": svg(
        "DAG, trabajo y camino crítico",
        "Grafo acíclico leer, partir, tareas A y B, y combinar; el camino crítico pasa por A.",
        '''
<rect x="55" y="170" width="120" height="70" class="accent"/><text x="115" y="200" text-anchor="middle" class="label">leer</text><text x="115" y="222" text-anchor="middle" class="small">2 unidades</text>
<rect x="225" y="170" width="120" height="70" class="accent"/><text x="285" y="200" text-anchor="middle" class="label">partir</text><text x="285" y="222" text-anchor="middle" class="small">1 unidad</text>
<rect x="420" y="95" width="120" height="70" class="accent"/><text x="480" y="125" text-anchor="middle" class="label">A</text><text x="480" y="147" text-anchor="middle" class="small">5 unidades</text>
<rect x="420" y="250" width="120" height="70" class="box"/><text x="480" y="280" text-anchor="middle" class="label">B</text><text x="480" y="302" text-anchor="middle" class="small">4 unidades</text>
<rect x="665" y="170" width="145" height="70" class="accent"/><text x="737" y="200" text-anchor="middle" class="label">combinar</text><text x="737" y="222" text-anchor="middle" class="small">2 unidades</text>
<path d="M175 205 H225 M345 205 L420 130 M540 130 L665 200 M810 205 H895" class="critical"/>
<path d="M345 205 L420 285 M540 285 L665 215" class="line"/>
<text x="480" y="380" text-anchor="middle" class="small">Trabajo T₁=14 · span T∞=10 · paralelismo medio=1,4</text>
'''),
    "escalabilidad.svg": svg(
        "Cómo leer una curva de escalabilidad",
        "Ejes de recursos y aceleración con líneas ideal, limitada por Amdahl y observada.",
        '''
<path d="M110 350 H860 M110 350 V85" stroke="#334155" stroke-width="2.5" fill="none"/>
<text x="485" y="405" text-anchor="middle" class="label">recursos p</text><text x="35" y="220" transform="rotate(-90 35 220)" text-anchor="middle" class="label">aceleración Sₚ</text>
<polyline points="110,350 220,310 340,260 480,205 650,145 820,90" fill="none" stroke="#718096" stroke-width="2.5" stroke-dasharray="8 7"/>
<polyline points="110,350 220,312 340,275 480,245 650,225 820,215" fill="none" stroke="#a95642" stroke-width="4"/>
<polyline points="110,350 220,320 340,290 480,268 650,260 820,262" fill="none" stroke="#506f83" stroke-width="4"/>
<text x="720" y="120" class="small">ideal S=p</text><text x="700" y="205" class="small">límite de Amdahl</text><text x="700" y="288" class="small">medición</text>
<rect x="120" y="95" width="285" height="85" class="green"/><text x="140" y="125" class="small">Preguntas obligatorias:</text><text x="140" y="150" class="tiny">¿tamaño fijo o carga por recurso?</text><text x="140" y="170" class="tiny">¿qué explica la pérdida de eficiencia?</text>
'''),
    "jerarquia-memoria.svg": svg(
        "Jerarquía de memoria y costo de movimiento",
        "Niveles de registros, cachés, memoria principal y almacenamiento con capacidad y latencia crecientes.",
        '''
<rect x="330" y="80" width="300" height="55" class="accent"/><text x="480" y="115" text-anchor="middle" class="label">Registros</text>
<rect x="275" y="150" width="410" height="55" class="blue"/><text x="480" y="185" text-anchor="middle" class="label">Cachés L1 / L2 / L3</text>
<rect x="210" y="220" width="540" height="55" class="green"/><text x="480" y="255" text-anchor="middle" class="label">Memoria principal / NUMA</text>
<rect x="140" y="290" width="680" height="55" class="box"/><text x="480" y="325" text-anchor="middle" class="label">Almacenamiento y memoria remota</text>
<path d="M95 90 V335" class="line"/><text x="65" y="220" transform="rotate(-90 65 220)" text-anchor="middle" class="small">más capacidad y latencia</text>
<path d="M865 335 V90" class="line"/><text x="900" y="220" transform="rotate(-90 900 220)" text-anchor="middle" class="small">más ancho de banda</text>
<text x="480" y="400" text-anchor="middle" class="small">Optimizar suele significar reutilizar datos antes de bajar al siguiente nivel.</text>
'''),
    "fork-join.svg": svg(
        "Patrón fork–join",
        "Una región serial crea trabajadores, distribuye trabajo y espera su finalización antes de continuar.",
        '''
<rect x="45" y="175" width="130" height="70" class="blue"/><text x="110" y="215" text-anchor="middle" class="label">serial</text>
<circle cx="245" cy="210" r="34" fill="#f3ddd4" stroke="#a95642" stroke-width="2.5"/><text x="245" y="216" text-anchor="middle" class="label">fork</text>
<rect x="340" y="80" width="190" height="55" class="box"/><text x="435" y="115" text-anchor="middle" class="label">trabajador 0</text>
<rect x="340" y="180" width="190" height="55" class="box"/><text x="435" y="215" text-anchor="middle" class="label">trabajador 1</text>
<rect x="340" y="280" width="190" height="55" class="box"/><text x="435" y="315" text-anchor="middle" class="label">trabajador n−1</text>
<circle cx="625" cy="210" r="34" fill="#dfeadf" stroke="#5d7d65" stroke-width="2.5"/><text x="625" y="216" text-anchor="middle" class="label">join</text>
<rect x="720" y="175" width="170" height="70" class="green"/><text x="805" y="203" text-anchor="middle" class="label">resultado</text><text x="805" y="227" text-anchor="middle" class="small">validado</text>
<path d="M175 210 H211 M279 200 L340 108 M279 210 H340 M279 220 L340 308 M530 108 L591 200 M530 208 H591 M530 308 L591 220 M659 210 H720" class="line"/>
'''),
    "happens-before.svg": svg(
        "Sincronización y happens-before",
        "El productor publica datos bajo sincronización y el consumidor los observa después de esperar la condición.",
        '''
<rect x="70" y="100" width="210" height="65" class="blue"/><text x="175" y="138" text-anchor="middle" class="label">productor escribe</text>
<rect x="70" y="220" width="210" height="65" class="accent"/><text x="175" y="258" text-anchor="middle" class="label">signal / unlock</text>
<rect x="620" y="220" width="210" height="65" class="accent"/><text x="725" y="258" text-anchor="middle" class="label">wait / lock</text>
<rect x="620" y="100" width="210" height="65" class="green"/><text x="725" y="138" text-anchor="middle" class="label">consumidor lee</text>
<path d="M175 165 V220 M280 252 H620 M725 220 V165" class="critical"/>
<text x="450" y="235" text-anchor="middle" class="small">relación de sincronización</text>
<rect x="330" y="330" width="240" height="60" class="box"/><text x="450" y="355" text-anchor="middle" class="small">La condición se comprueba</text><text x="450" y="376" text-anchor="middle" class="small">siempre dentro de un bucle.</text>
'''),
    "distribucion-trabajo.svg": svg(
        "Distribución, reducción y balance",
        "Iteraciones repartidas entre cuatro trabajadores y combinación final de resultados parciales.",
        '''
<text x="70" y="95" class="label">iteraciones</text>
<g transform="translate(190 65)"><rect width="70" height="45" class="blue"/><rect x="75" width="70" height="45" class="green"/><rect x="150" width="70" height="45" class="accent"/><rect x="225" width="70" height="45" class="box"/><rect x="300" width="70" height="45" class="blue"/><rect x="375" width="70" height="45" class="green"/><rect x="450" width="70" height="45" class="accent"/><rect x="525" width="70" height="45" class="box"/></g>
<rect x="90" y="180" width="150" height="60" class="blue"/><text x="165" y="217" text-anchor="middle" class="label">trabajador 0</text>
<rect x="300" y="180" width="150" height="60" class="green"/><text x="375" y="217" text-anchor="middle" class="label">trabajador 1</text>
<rect x="510" y="180" width="150" height="60" class="accent"/><text x="585" y="217" text-anchor="middle" class="label">trabajador 2</text>
<rect x="720" y="180" width="150" height="60" class="box"/><text x="795" y="217" text-anchor="middle" class="label">trabajador 3</text>
<path d="M165 240 L420 330 M375 240 L445 330 M585 240 L475 330 M795 240 L500 330" class="line"/>
<rect x="370" y="330" width="220" height="65" class="green"/><text x="480" y="370" text-anchor="middle" class="label">reducción verificada</text>
'''),
    "mpi-comunicacion.svg": svg(
        "Comunicación entre procesos MPI",
        "Cuatro rangos intercambian mensajes punto a punto y participan en una operación colectiva.",
        '''
<rect x="90" y="100" width="150" height="70" class="blue"/><text x="165" y="142" text-anchor="middle" class="label">rank 0</text>
<rect x="720" y="100" width="150" height="70" class="green"/><text x="795" y="142" text-anchor="middle" class="label">rank 1</text>
<rect x="720" y="290" width="150" height="70" class="accent"/><text x="795" y="332" text-anchor="middle" class="label">rank 2</text>
<rect x="90" y="290" width="150" height="70" class="box"/><text x="165" y="332" text-anchor="middle" class="label">rank 3</text>
<path d="M240 135 H720 M795 170 V290 M720 325 H240 M165 290 V170" class="line"/>
<circle cx="480" cy="225" r="70" fill="#fdfbf5" stroke="#50677a" stroke-width="2.5"/><text x="480" y="218" text-anchor="middle" class="label">colectiva</text><text x="480" y="244" text-anchor="middle" class="small">mismo comunicador</text>
<path d="M240 150 L420 200 M720 150 L540 200 M720 310 L540 250 M240 310 L420 250" class="dash"/>
'''),
    "offload-host-device.svg": svg(
        "Flujo host–dispositivo",
        "El host prepara datos, transfiere al dispositivo, ejecuta un kernel, recupera resultados y valida.",
        '''
<rect x="45" y="145" width="175" height="95" class="blue"/><text x="132" y="180" text-anchor="middle" class="label">Host</text><text x="132" y="205" text-anchor="middle" class="small">prepara + referencia</text>
<rect x="330" y="95" width="300" height="190" class="box"/><text x="480" y="130" text-anchor="middle" class="label">Dispositivo</text><rect x="370" y="160" width="80" height="55" class="accent"/><rect x="460" y="160" width="80" height="55" class="accent"/><rect x="550" y="160" width="45" height="55" class="accent"/><text x="480" y="250" text-anchor="middle" class="small">kernel sobre grid de trabajo</text>
<rect x="740" y="145" width="175" height="95" class="green"/><text x="827" y="180" text-anchor="middle" class="label">Validación</text><text x="827" y="205" text-anchor="middle" class="small">error + tiempos</text>
<path d="M220 170 H330" class="line"/><text x="275" y="155" text-anchor="middle" class="tiny">H2D</text><path d="M630 215 H740" class="line"/><text x="685" y="205" text-anchor="middle" class="tiny">D2H</text>
<path d="M740 265 C650 365 300 365 220 265" class="dash"/><text x="480" y="365" text-anchor="middle" class="small">comparar con la misma operación y tolerancia</text>
'''),
    "cuda-grid-tiling.svg": svg(
        "Jerarquía CUDA y trabajo por tiles",
        "Un grid contiene bloques, cada bloque contiene hilos y coopera sobre un tile con memoria compartida.",
        '''
<rect x="55" y="90" width="500" height="285" class="blue"/><text x="80" y="125" class="label">grid</text>
<rect x="105" y="150" width="180" height="170" class="box"/><text x="195" y="180" text-anchor="middle" class="label">bloque (0,0)</text>
<rect x="325" y="150" width="180" height="170" class="box"/><text x="415" y="180" text-anchor="middle" class="label">bloque (1,0)</text>
<g fill="#f3ddd4" stroke="#a95642"><circle cx="145" cy="220" r="18"/><circle cx="195" cy="220" r="18"/><circle cx="245" cy="220" r="18"/><circle cx="145" cy="270" r="18"/><circle cx="195" cy="270" r="18"/><circle cx="245" cy="270" r="18"/><circle cx="365" cy="220" r="18"/><circle cx="415" cy="220" r="18"/><circle cx="465" cy="220" r="18"/></g>
<text x="195" y="345" text-anchor="middle" class="small">hilos cooperan y sincronizan</text>
<rect x="640" y="110" width="230" height="230" class="green"/><text x="755" y="145" text-anchor="middle" class="label">tile compartido</text>
<path d="M690 180 H820 M690 225 H820 M690 270 H820 M720 165 V300 M765 165 V300" stroke="#5d7d65" stroke-width="2"/>
<path d="M555 230 H640" class="line"/><text x="598" y="215" text-anchor="middle" class="tiny">reutilizar</text>
'''),
    "topologia-hibrida.svg": svg(
        "Topología híbrida nodo–rank–hilo–GPU",
        "Dos nodos conectados; cada uno aloja procesos MPI, hilos OpenMP y un dispositivo asignado.",
        '''
<rect x="45" y="90" width="390" height="285" class="blue"/><text x="70" y="125" class="label">nodo 0</text>
<rect x="525" y="90" width="390" height="285" class="green"/><text x="550" y="125" class="label">nodo 1</text>
<rect x="80" y="155" width="195" height="90" class="box"/><text x="177" y="185" text-anchor="middle" class="label">rank 0</text><text x="177" y="218" text-anchor="middle" class="small">hilos 0…t−1</text>
<rect x="560" y="155" width="195" height="90" class="box"/><text x="657" y="185" text-anchor="middle" class="label">rank 1</text><text x="657" y="218" text-anchor="middle" class="small">hilos 0…t−1</text>
<rect x="300" y="155" width="105" height="150" class="accent"/><text x="352" y="215" text-anchor="middle" class="label">GPU 0</text><text x="352" y="242" text-anchor="middle" class="tiny">local</text>
<rect x="780" y="155" width="105" height="150" class="accent"/><text x="832" y="215" text-anchor="middle" class="label">GPU 1</text><text x="832" y="242" text-anchor="middle" class="tiny">local</text>
<path d="M275 200 H300 M755 200 H780 M435 230 H525" class="line"/><text x="480" y="215" text-anchor="middle" class="tiny">red MPI</text>
<text x="480" y="410" text-anchor="middle" class="small">Afinidad y asignación explícita evitan sobresuscripción y dos ranks sobre la misma GPU.</text>
'''),
    "metodo-rendimiento.svg": svg(
        "Método para interpretar rendimiento",
        "Cadena de mediciones repetidas, resumen robusto, perfil por fases, hipótesis y nuevo experimento.",
        '''
<rect x="55" y="150" width="160" height="80" class="blue"/><text x="135" y="182" text-anchor="middle" class="label">Repetir</text><text x="135" y="208" text-anchor="middle" class="small">datos crudos</text>
<rect x="270" y="150" width="160" height="80" class="green"/><text x="350" y="182" text-anchor="middle" class="label">Resumir</text><text x="350" y="208" text-anchor="middle" class="small">mediana + MAD</text>
<rect x="485" y="150" width="160" height="80" class="box"/><text x="565" y="182" text-anchor="middle" class="label">Perfilar</text><text x="565" y="208" text-anchor="middle" class="small">fases y costos</text>
<rect x="700" y="150" width="190" height="80" class="accent"/><text x="795" y="182" text-anchor="middle" class="label">Explicar</text><text x="795" y="208" text-anchor="middle" class="small">hipótesis verificable</text>
<path d="M215 190 H270 M430 190 H485 M645 190 H700" class="line"/>
<path d="M795 250 C795 345 135 345 135 250" class="dash"/><text x="465" y="335" text-anchor="middle" class="small">cambiar una variable y repetir el experimento</text>
<text x="480" y="105" text-anchor="middle" class="small">Una captura aislada no sustituye datos, contexto ni incertidumbre.</text>
'''),
}


README = """# Imágenes compartidas del curso

Esta carpeta contiene diagramas SVG reutilizables por todos los temas. Cada archivo incluye `title` y `desc` para accesibilidad, usa una paleta común y conserva texto legible sin depender de un notebook ejecutado.

Los diagramas son material conceptual: explican relaciones, jerarquías o secuencias, pero no sustituyen mediciones ni capturas auténticas de herramientas. Se generan con `python3 tools/generate_course_diagrams.py`; el modo `--check` impide que un notebook enlace una versión divergente o ausente.

| Archivo | Relación principal |
|---|---|
""" + "\n".join(f"| `{name}` | {content.split('<desc id=\"desc\">', 1)[1].split('</desc>', 1)[0]} |" for name, content in ASSETS.items()) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="No escribe; falla si un SVG o el README difieren")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = Path(__file__).resolve().parent.parent
    destination = repository / "curso" / "images"
    expected = {**ASSETS, "README.md": README}
    failures: list[str] = []
    for name, content in expected.items():
        target = destination / name
        if args.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != content:
                failures.append(str(target.relative_to(repository)))
        else:
            destination.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
    if args.check:
        actual = {path.name for path in destination.iterdir() if path.is_file()} if destination.is_dir() else set()
        unexpected = sorted(actual - set(expected))
        failures.extend(str((destination / name).relative_to(repository)) for name in unexpected)
    if failures:
        print("Imágenes compartidas ausentes o divergentes:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"Imágenes compartidas {'verificadas' if args.check else 'generadas'}: {len(ASSETS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
