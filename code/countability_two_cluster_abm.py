#!/usr/bin/env python3
"""
Two-cluster countability sketch
===============================

A DIDACTIC MECHANISM SKETCH, not evidence. It keeps apart the two clusters
Chapter 9 of the book distinguishes:

- sem:  an agent's belief about how individuated a noun's referents are
        (semantic cluster), grounded in experience of the referents;
- gram: the noun's grammatical count status for that agent (morphosyntactic
        cluster), learned from usage and used for licensing.

Bidirectional inference couples them (rate `coupling`, both directions). For
most nouns the clusters agree. For object-mass nouns like *furniture* they
don't: referents are individuated, the grammar is non-count. The question
the sketch asks is what holds such a noun non-count against the coupling's
pull, and whether the same coupling restores a count noun's grammar after a
drop.

World facts per noun are TOY ASSUMPTIONS, not data:
    r   referent individuation (what learners experience)
    g0  starting grammatical status of the initial population only
New learners never inherit g0. They draw sem around r, and start gram either
from their own sem ('semantic': they expect grammar to follow meaning) or
uninformed at 0.5; then they learn from tutors' usage.

Mechanisms, each with a switch:
- usage learning (grammar from what is heard)                 lr_gram
- coupling (bidirectional inference)                          coupling
- measure-phrase outlet ("a piece of furniture")              outlets on/off
- singulative outlet ("a cow" for "cattle")                   singulatives on/off
- entrenchment (chunks bypass the threshold check above 0.7)  entrenchment_rate
- institutional correction of count frames on mass targets    institutional_targets
- regularization (forcing an unlicensed exact-count frame)    regularization_rate
- hearer response to forced forms ('evidence' or 'error')     hearer_response
- regression toward a looser frame's threshold                regress_on_loose (off)
- statistical preemption: a non-count or measure frame heard   preemption (0 = off)
  where the situation called for counting (the speaker meant one item,
  a few, or many) is extra evidence against count status
- superordinate referents: object-mass nouns denote             superordinate (off)
  heterogeneous aggregates, so what learners experience is less
  individuated (r_super instead of r)

Communicative intent comes from sem (talk is about referents); licensing
comes from gram. An exact-count intent the grammar won't license goes first
to a singulative, then to a measure phrase, then (sometimes) to forcing,
then to *much* or silence.

Author: Brett Reynolds / Claude Code, 2026-10-28.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

THRESH = {"a(n)": 0.9, "three": 0.85, "several": 0.7, "many": 0.5, "agreement": 0.3}
MUCH_MAX = 0.4
MUCH_TARGET = 0.2
INTENT_PREF = {
    "singleton": ["a(n)"],
    "exact_small": ["three"],
    "vague_many": ["many", "several"],
    "existence": ["agreement", "many"],
    "mass_quantity": ["much"],
}
INTENTS = list(INTENT_PREF)
EXACT = ("singleton", "exact_small")


def intent_weights(sem: float) -> List[float]:
    """Intent mix by semantic individuation (bands as in the original sketch)."""
    if sem >= 0.85:
        return [0.38, 0.25, 0.22, 0.13, 0.02]
    if sem >= 0.65:
        return [0.24, 0.20, 0.28, 0.18, 0.10]
    if sem >= 0.45:
        return [0.12, 0.12, 0.30, 0.26, 0.20]
    return [0.05, 0.05, 0.20, 0.30, 0.40]


@dataclass
class NounSpec:
    name: str
    r: float                      # referent individuation (toy assumption)
    g0: float                     # initial population's grammatical status
    singulative: Optional[str] = None
    measure_outlet: bool = False  # "a piece of X" available
    r_super: Optional[float] = None  # referent individuation under `superordinate`


DEFAULT_NOUNS = [
    NounSpec("book", r=0.95, g0=0.95),
    NounSpec("dog", r=0.95, g0=0.95),
    NounSpec("furniture", r=0.90, g0=0.25, measure_outlet=True, r_super=0.50),
    NounSpec("equipment", r=0.85, g0=0.20, measure_outlet=True, r_super=0.50),
    NounSpec("water", r=0.05, g0=0.10, measure_outlet=True),
    NounSpec("cattle", r=0.90, g0=0.55, singulative="cow"),
    NounSpec("cow", r=0.95, g0=0.95),
]


@dataclass
class Config:
    n_agents: int = 100
    n_institutional: int = 5
    lr_gram: float = 0.03
    lr_gram_learner: float = 0.06
    lr_sem: float = 0.05
    sem_noise: float = 0.10
    coupling: float = 0.01
    learner_gram_prior: str = "semantic"     # or 'uninformed'
    outlets: bool = True
    singulatives: bool = True
    entrenchment_rate: float = 0.02
    institutional_targets: Tuple[str, ...] = ("furniture", "equipment")
    correction_strength: float = 0.05
    regularization_rate: float = 0.0
    hearer_response: str = "evidence"
    regress_on_loose: bool = False
    turnover: float = 0.04
    tutor_observations: int = 120
    usage_learning: bool = True
    preemption: float = 0.0
    superordinate: bool = False


@dataclass
class Lex:
    sem: float
    gram: float
    ent: Dict[str, float] = field(default_factory=dict)


class Agent:
    def __init__(self, lex: Dict[str, Lex], institutional: bool, lr_gram: float):
        self.lex = lex
        self.institutional = institutional
        self.lr_gram = lr_gram

    def licenses(self, noun: str, c: str) -> bool:
        x = self.lex[noun]
        if x.ent.get(c, 0.0) > 0.7:
            return True
        if c == "much":
            return x.gram <= MUCH_MAX
        return x.gram >= THRESH[c]


class World:
    def __init__(self, cfg: Config, seed: int, nouns: Optional[List[NounSpec]] = None):
        random.seed(seed)
        self.cfg = cfg
        self.nouns = {n.name: n for n in (nouns or DEFAULT_NOUNS)}
        # Referent individuation actually experienced (world side).
        self.r = {n: (s.r_super if (cfg.superordinate and s.r_super is not None) else s.r)
                  for n, s in self.nouns.items()}
        self.agents = [self._initial_agent(i) for i in range(cfg.n_agents)]

    # -- construction --
    def _initial_agent(self, i: int) -> Agent:
        lex = {n: Lex(sem=self._clamp(self.r[n] + random.gauss(0, 0.05)),
                      gram=self._clamp(s.g0 + random.gauss(0, 0.05)))
               for n, s in self.nouns.items()}
        inst = i < self.cfg.n_institutional
        return Agent(lex, inst, 0.01 if inst else self.cfg.lr_gram)

    def _learner(self) -> Agent:
        lex = {}
        for n, s in self.nouns.items():
            sem = self._clamp(self.r[n] + random.gauss(0, self.cfg.sem_noise))
            if self.cfg.learner_gram_prior == "semantic":
                gram = self._clamp(sem + random.gauss(0, 0.10))
            else:
                gram = self._clamp(0.5 + random.gauss(0, 0.10))
            lex[n] = Lex(sem=sem, gram=gram)
        return Agent(lex, False, self.cfg.lr_gram_learner)

    @staticmethod
    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    # -- production --
    def produce(self, sp: Agent, noun: str) -> Optional[Tuple[str, str, str, str]]:
        """Returns (noun_heard, construction, kind, intent) with kind in
        {'normal', 'measure', 'forced'}, or None for silence."""
        cfg = self.cfg
        intent = random.choices(INTENTS, weights=intent_weights(sp.lex[noun].sem), k=1)[0]
        pref = INTENT_PREF[intent]
        for c in pref:
            if sp.licenses(noun, c):
                return (noun, c, "normal", intent)
        if intent != "mass_quantity":
            spec = self.nouns[noun]
            if cfg.singulatives and spec.singulative:
                for c in pref:
                    if sp.licenses(spec.singulative, c):
                        return (spec.singulative, c, "normal", intent)
            if intent in EXACT and cfg.outlets and spec.measure_outlet:
                return (noun, "much", "measure", intent)
            if (intent in EXACT and cfg.regularization_rate > 0
                    and sp.licenses(noun, "many")
                    and random.random() < cfg.regularization_rate):
                return (noun, pref[0], "forced", intent)
            if sp.licenses(noun, "much"):
                return (noun, "much", "normal", intent)
        return None

    # -- comprehension --
    def observe(self, h: Agent, noun: str, c: str, kind: str, intent: Optional[str] = None):
        cfg = self.cfg
        x = h.lex[noun]
        weight = 0.0 if (kind == "forced" and cfg.hearer_response == "error") else 1.0
        if cfg.usage_learning and weight > 0:
            lr = h.lr_gram
            if c == "much":
                x.gram -= lr * (x.gram - MUCH_TARGET) * 0.5
                if cfg.preemption > 0 and intent in ("singleton", "exact_small", "vague_many"):
                    # statistical preemption: a non-count frame where counting was called for
                    x.gram -= lr * cfg.preemption * (x.gram - MUCH_TARGET)
            else:
                t = THRESH[c]
                if x.gram < t:
                    x.gram += lr * (t - x.gram)
                elif cfg.regress_on_loose:
                    x.gram -= lr * (x.gram - t) * 0.5
            if kind != "measure":
                x.ent[c] = min(1.0, x.ent.get(c, 0.0) + cfg.entrenchment_rate)
        # bidirectional inference, then experience of the referent
        g, s = x.gram, x.sem
        x.gram += cfg.coupling * (s - g)
        x.sem += cfg.coupling * (g - s)
        r = self.r[noun]
        x.sem += cfg.lr_sem * (r + random.gauss(0, cfg.sem_noise) - x.sem)
        x.gram, x.sem = self._clamp(x.gram), self._clamp(x.sem)

    def correct(self, sp: Agent, noun: str, c: str):
        """Institutional hearers push count frames on mass targets back toward non-count."""
        if noun in self.cfg.institutional_targets and c in ("a(n)", "three"):
            x = sp.lex[noun]
            x.gram -= self.cfg.correction_strength * x.gram
            x.gram = self._clamp(x.gram)

    # -- dynamics --
    def step(self, interactions: int):
        names = list(self.nouns)
        for _ in range(interactions):
            sp, h = random.sample(self.agents, 2)
            noun = random.choice(names)
            out = self.produce(sp, noun)
            if out is None:
                continue
            heard, c, kind, intent = out
            self.observe(h, heard, c, kind, intent)
            if h.institutional and kind != "measure":
                self.correct(sp, heard, c)

    def turnover_round(self):
        cfg = self.cfg
        if cfg.turnover <= 0:
            return
        replaceable = [i for i, a in enumerate(self.agents) if not a.institutional]
        k = max(1, int(len(self.agents) * cfg.turnover))
        names = list(self.nouns)
        for idx in random.sample(replaceable, min(k, len(replaceable))):
            learner = self._learner()
            tutors = random.sample([i for i in replaceable if i != idx], 8)
            for _ in range(cfg.tutor_observations):
                tutor = self.agents[random.choice(tutors)]
                noun = random.choice(names)
                out = self.produce(tutor, noun)
                if out is not None:
                    self.observe(learner, *out)
            self.agents[idx] = learner

    def phase(self, steps: int = 35, interactions: int = 60):
        for _ in range(steps):
            self.step(interactions)
        self.turnover_round()

    # -- measurement --
    def mean_gram(self, noun: str) -> float:
        return sum(a.lex[noun].gram for a in self.agents) / len(self.agents)

    def mean_sem(self, noun: str) -> float:
        return sum(a.lex[noun].sem for a in self.agents) / len(self.agents)

    def profile(self, noun: str) -> Dict[str, float]:
        n = len(self.agents)
        prof = {c: sum(a.licenses(noun, c) for a in self.agents) / n for c in THRESH}
        prof["much"] = sum(a.licenses(noun, "much") for a in self.agents) / n
        return prof
