"""Small built-in corpus used by benchmarks, tests, and demos.

The sentences are original, everyday Danish covering different sentence shapes.
"""

from __future__ import annotations

BENCHMARK_SENTENCES: tuple[str, ...] = (
    "God morgen, hvordan har du det i dag?",
    "Jeg vil gerne bestille en kop kaffe og en croissant.",
    "Toget til København afgår fra spor syv om ti minutter.",
    "Det er vigtigt, at vi husker at gemme arbejdet, før vi slukker computeren.",
    "Hun sagde, at hun ville komme igen i morgen eftermiddag.",
    "Kommunen har besluttet at renovere den gamle skole i løbet af næste år.",
    "Vi mangler stadig at underskrive kontrakten og sende den tilbage.",
    "Vejret bliver koldt i weekenden, så husk en varm jakke.",
    "Kan du forklare, hvordan denne funktion virker?",
    "Der er mange små virksomheder, som gerne vil bruge åbne værktøjer.",
)

DEMO_TEXT: str = (
    "Hej, og velkommen til Danglish. "
    "Det her er en lokal oversættelsespipeline, som kører på en almindelig computer. "
    "Den bruger en lille dansk-engelsk model og oversætter sætning for sætning, mens du skriver. "
    "Prøv at skrive en sætning, og se hvordan den engelske tekst opdateres med det samme."
)
