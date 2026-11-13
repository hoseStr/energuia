# Análisis de errores del modelo (Fase 2)

Backtest walk-forward del 2025-11-01 al 2026-10-31: 12 folds mensuales y 8 760 horas. Cada pronóstico se emite un día antes, con la demanda publicada hasta 5 días antes de la emisión y `Gene` hasta 3 días antes. Corrido el 2026-11-13 con datos completos hasta el 2026-11-08.

Las tablas de las secciones 1, 2 y 4 y la lista de los peores días de la sección 5 salen de `python -m demand_forecast.cli backtest --model <modelo>`. Los cortes por festivo de la sección 3, la comparación día a día y los percentiles de la sección 5, el sesgo de la sección 6 y la importancia de la sección 7 se calcularon aparte sobre esos mismos pronósticos y no tienen un comando propio.

## 1. Resultado

| Modelo | MAE (MWh) | MAPE (%) |
|---|---|---|
| Ingenuo estacional | 338,6 | 3,40 |
| Promedio de 4 semanas | 300,5 | 3,03 |
| **LightGBM** | **194,5** | **1,96** |

El modelo baja el MAPE del mejor baseline un 35 %. La meta era un 15 % (MAPE ≤ 2,58 %).

Este backtest se corrió una sola vez. Las features, la variable que se ajusta y los hiperparámetros se decidieron mirando solo los 12 meses anteriores (2024-11-01 a 2025-10-31; ver `docs/decisions.md`). En esa ventana de selección el modelo había dado 2,09 % frente a 3,40 % del mejor baseline.

## 2. Por tipo de día

| Tipo de día | Ingenuo estacional | Promedio de 4 semanas | LightGBM |
|---|---|---|---|
| Laboral | 3,13 | 2,74 | 1,86 |
| Sábado | 2,67 | 2,37 | 2,04 |
| Domingo o festivo | 4,87 | 4,53 | 2,23 |

MAPE en %. La mayor ganancia está en domingos y festivos, que era donde los baselines fallaban. El sábado es donde menos mejora.

## 3. Festivos y fechas especiales

| Grupo | Días | Ingenuo estacional | Promedio de 4 semanas | LightGBM |
|---|---|---|---|---|
| Festivos entre semana | 19 | 11,25 | 11,00 | 2,54 |
| Domingos | 52 | 2,54 | 2,17 | 2,12 |
| Laboral posterior a un festivo | 15 | 3,94 | 3,97 | 2,79 |
| Laboral víspera de festivo | 5 | 4,39 | 5,90 | 2,63 |
| 20 de diciembre a 10 de enero | 22 | 5,67 | 5,84 | 1,65 |
| Semana Santa (29 de marzo a 5 de abril) | 8 | 8,91 | 7,45 | 1,99 |

- **Los festivos entre semana dejan de ser el problema.** Los baselines repiten un día laboral y se equivocan un 11 %; el modelo queda en 2,54 %. Frente al promedio de 4 semanas, el 25 de diciembre pasa de 22,3 % a 1,3 % y el Viernes Santo de 20,8 % a 1,4 %.
- **En los domingos el modelo casi no aporta** (2,12 % frente a 2,17 %). Un domingo se parece mucho al anterior y no queda mucho que corregir.
- **Los días pegados a un festivo siguen siendo los peores entre los laborales** (2,6 % a 2,8 % frente a 1,86 %). Hay features para la víspera y para el día siguiente, pero son pocos casos por año.
- **Peores festivos:** Corpus Christi (8 de junio, 5,6 %), San Pedro y San Pablo (29 de junio, 4,5 %) y el 1 de mayo (4,1 %). En los dos últimos el modelo pronosticó de menos: ese festivo bajó menos que un domingo típico.

## 4. Por hora y por mes

| Tramo | Ingenuo estacional | Promedio de 4 semanas | LightGBM |
|---|---|---|---|
| 00:00 a 05:59 | 3,04 | 2,70 | 2,16 |
| 06:00 a 11:59 | 3,70 | 3,35 | 1,82 |
| 12:00 a 17:59 | 3,77 | 3,36 | 1,94 |
| 18:00 a 23:59 | 3,09 | 2,73 | 1,91 |

- El peor tramo de los baselines era de 14:00 a 17:00 (hasta 4,11 %). El modelo lo deja en 2,0 %.
- **La madrugada es ahora el peor tramo del modelo** (hasta 2,30 % a la 01:00) y es donde menos mejora. El mejor es el pico de la noche (1,68 % a las 18:00).

| Mes | Ingenuo estacional | Promedio de 4 semanas | LightGBM |
|---|---|---|---|
| 2025-11 | 3,61 | 2,65 | 1,67 |
| 2025-12 | 4,57 | 4,15 | 2,24 |
| 2026-01 | 4,02 | 4,71 | 2,06 |
| 2026-02 | 2,86 | 2,34 | 2,50 |
| 2026-03 | 2,21 | 2,12 | 1,61 |
| 2026-04 | 4,89 | 3,82 | 1,64 |
| 2026-05 | 2,44 | 2,27 | 1,27 |
| 2026-06 | 4,31 | 3,37 | 2,91 |
| 2026-07 | 3,10 | 2,72 | 1,84 |
| 2026-08 | 3,62 | 3,18 | 2,15 |
| 2026-09 | 2,10 | 2,14 | 1,94 |
| 2026-10 | 3,08 | 2,86 | 1,73 |

- **Febrero de 2026 es el único mes en que el modelo pierde** contra el promedio de 4 semanas (2,50 % frente a 2,34 %). La causa son tres días seguidos (sección 5).
- **Junio de 2026 es el peor mes del modelo** (2,91 %), aunque sigue por debajo de los baselines.

## 5. Los peores días

Los 10 días con mayor MAPE del modelo:

| Día | Tipo | MAPE LightGBM | MAPE promedio de 4 semanas |
|---|---|---|---|
| 2026-02-02 lunes | laboral | 11,25 | 6,81 |
| 2026-02-03 martes | laboral | 8,13 | 5,08 |
| 2026-08-10 lunes | laboral | 7,79 | 2,63 |
| 2026-02-01 domingo | domingo o festivo | 7,62 | 6,43 |
| 2026-09-28 lunes | laboral | 6,38 | 5,60 |
| 2026-06-09 martes | laboral | 5,94 | 7,13 |
| 2026-06-08 lunes | domingo o festivo | 5,59 | 15,24 |
| 2026-04-07 martes | laboral | 5,53 | 2,06 |
| 2025-11-29 sábado | sábado | 5,53 | 4,71 |
| 2026-06-06 sábado | sábado | 5,50 | 5,77 |

- **Solo uno de los diez es festivo.** Los peores errores ya no vienen del calendario sino de caídas de la demanda que el modelo no puede ver con datos de hace 4 a 6 días.
- **1 al 3 de febrero de 2026.** La demanda cayó de golpe: el lunes 2 fue de 220,6 GWh frente a 241,7 el lunes anterior, y el domingo 1 de 202,6 frente a 215,0. No son festivos y la causa no está identificada (**por verificar**). El modelo pronosticó 245,4 GWh para el lunes. Los baselines también fallaron, pero menos.
- **6 al 10 de junio de 2026.** Cinco días seguidos con la demanda un 4 % a 6 % por debajo del pronóstico, alrededor del festivo de Corpus Christi. Tampoco tiene causa identificada.
- **10 de agosto de 2026.** Lunes posterior al festivo del viernes 7. La demanda fue de 239,0 GWh y el modelo pronosticó 257,8. El promedio de 4 semanas acertó por casualidad: dos de los cuatro lunes que promedia eran festivos.
- **Días con MAPE mayor al 5 %:** 13 del modelo, frente a 54 del promedio de 4 semanas y 60 del ingenuo estacional. El modelo gana al promedio de 4 semanas en el 66 % de los días.

| Percentil del MAPE diario | Ingenuo estacional | Promedio de 4 semanas | LightGBM |
|---|---|---|---|
| 50 | 2,24 | 2,15 | 1,58 |
| 90 | 7,90 | 6,21 | 3,70 |
| 95 | 10,61 | 8,41 | 4,34 |
| 99 | 18,69 | 15,94 | 6,83 |

## 6. Sesgo

- **El modelo pronostica de más un 0,34 % en promedio.** El sesgo es positivo en 8 de los 12 meses y llega a +1,60 % en septiembre de 2026.
- Por día de la semana va de +0,76 % el lunes a −0,22 % el viernes.
- No se corrigió: ajustar un sesgo medido en la ventana de evaluación sería ajustar la evaluación.

## 7. Qué pesa en el modelo

Importancia por ganancia del modelo del primer fold, entrenado con datos hasta el 2025-10-26:

| Feature | % de la ganancia |
|---|---|
| Día del año | 23,2 |
| Desviación del último día de `Gene` | 13,5 |
| Mismo día de la semana más reciente | 11,4 |
| Nivel de la última semana frente a la ventana | 10,1 |
| Desviación del último día de demanda | 9,8 |
| Desviación del penúltimo día de `Gene` | 7,9 |
| Las 14 restantes | 24,1 |

- **`Gene` aporta:** sus dos features de día suman el 21 % de la ganancia. Es la información más reciente que tiene el modelo.
- **El día del año es la feature de más peso.** Recoge el periodo de fin de año y la estacionalidad anual, pero con diez años de historia también puede memorizar fechas. Es un riesgo conocido.
- **El indicador de festivo casi no pesa (0,1 %)** porque el festivo ya está en el perfil contra el que se pronostica: el de los domingos y festivos recientes.

## 8. Limitaciones

- **El backtest usa los datos ya revisados.** En producción los días recientes llegan en versión preliminar; las revisiones observadas son de hasta un 0,04 % (`docs/data_notes.md` §3.2).
- **El retraso se supone fijo** (5 días para la demanda y 3 para `Gene`). Si XM no publica en fines de semana y festivos, el error real en esos días será mayor que el medido aquí. Lo deciden los snapshots del 14 al 17 de noviembre de 2026.
- **Sin clima.** Las caídas de varios días de la sección 5 son el tipo de error que una variable de temperatura podría explicar; queda como extensión (PLAN.md §9).
- **Los años de entrenamiento quedaron en el borde de la búsqueda** (10, el mayor valor probado). No se probó con más.
