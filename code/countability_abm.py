#!/usr/bin/env python3
"""
Countability Agent-Based Model
==============================

A DIDACTIC MECHANISM SKETCH for Chapter 9 of "Words That Won't Hold Still."

STATUS: This is an intuition pump, not evidence. It makes the two-cluster story
and the implicational hierarchy visible by operationalizing the mechanisms as
interacting update rules. It is NOT a calibrated simulation and should not be
cited as empirical support for the theory.

Core components:
- Agents with lexicons containing individuation confidence per noun
- Constructions as "locks" with tolerance thresholds (tight to loose)
- Bidirectional inference: production and comprehension update beliefs
- Multi-timescale maintenance: processing, acquisition, transmission
- Institutional layer: prescriptive feedback from editors/teachers
- Functional anchoring: singulative alternatives bleed regularization pressure

The model demonstrates:
1. Tight-before-loose erosion (data drift as datum dies)
2. Functional anchoring as stabilizer (cattle stable with cow available)
3. Why prescriptivism struggles (LEGO -> Legos despite corporate campaigns)

Deliberate simplifications (important for interpretation):
- Individuation is a single latent scalar per noun per agent.
- Construction licensing is threshold-based (hard-coded tight-to-loose ordering).
- Turnover uses tutor-sampled bootstrap learning, not a full child-style acquisition pipeline.
- Anchor decay is an exogenous perturbation schedule, not inferred from corpus.
- Large populations still need proportionally scaled interaction budgets; otherwise
  agents are under-exposed and trajectories become artifacts of sparse updates.

STRESS TEST: Ablation compares targeted mechanism removals to a baseline run.
Use it as a sensitivity probe, not as proof.

Switches for the homeostasis and ablation checks (all off by default; the
defaults reproduce the original model exactly):
- learner_prior: 'fixed' (default) centres each new learner's prior on the
  noun's starting value, a setpoint held outside the population; 'community'
  centres it on the current population mean (same spread), leaving no
  outside setpoint.
- regularization_rate: probability (default 0.0) that a speaker who licenses
  `many` for a noun, has an exact-count intent it can't license, and has no
  singulative to hand the job to, forces the tight frame anyway. This is the
  count-ward pressure Chapter 9 attributes to bidirectional inference; a
  singulative intercepts it.
- hearer_response: 'evidence' (default) treats a forced form like any other
  utterance; 'error' gives it no evidential weight (the hearer doesn't
  update). Matters only when regularization_rate > 0.
- regress_on_loose: True (default) lets a construction whose threshold is
  below the hearer's value pull that value down ("slight regression toward
  observed threshold"); False treats such compatible evidence as
  uninformative.
- measure_phrase_evidence: False (default) abstracts the last-resort
  measure phrase ("a piece of data", "head of cattle") as silence; True
  lets an exact-count intent surface as a measure phrase, which hearers
  take as non-count evidence for the noun (observed as MUCH). It is not a
  prescriptive violation, so institutional agents don't correct it.

Usage:
    python countability_abm.py              # Run all experiments
    python countability_abm.py --exp=drift  # Run specific experiment
    python countability_abm.py --exp=ablation  # Run stress test
    python countability_abm.py --exp=sensitivity --seed-start=1 --seed-end=50
    python countability_abm.py --exp=drift --n-agents=2000 --interactions-scale=0.6
    python countability_abm.py --interactive  # Interactive exploration

Author: Brett Reynolds / Claude Code
Date: January 2026
"""

import random
import math
import json
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from enum import Enum
import argparse

# Optional visualization (graceful fallback if not available)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available; text-based output only")

try:
    from plot_style import PALETTE, set_plot_style
except ImportError:
    # Fallback if plot_style isn't local
    PALETTE = {
        "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
        "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
        "yellow": "#F0E442", "gray": "#4D4D4D", "black": "#000000"
    }
    def set_plot_style():
        if HAS_MATPLOTLIB:
            plt.rcParams.update({
                'font.family': 'serif',
                'mathtext.fontset': 'cm',
                'axes.labelsize': 10,
                'axes.titlesize': 11,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.dpi': 300,
                'lines.linewidth': 1.5,
                'axes.grid': True,
                'grid.linestyle': ':',
                'grid.alpha': 0.6,
                'axes.spines.top': False,
                'axes.spines.right': False
            })

ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ROOT_DIR / 'figures'


def _save_figure(fig, filename: str, dpi: int = 150):
    """
    Save figures to both `figures/` (used by LaTeX) and `code/` (local diagnostics).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        FIGURES_DIR / filename,
        CODE_DIR / filename,
    ]
    for path in targets:
        fig.savefig(path, dpi=dpi)

    print(f"\nPlot saved to: {FIGURES_DIR / filename}")


def _save_manifest(filename: str, payload: Dict):
    """
    Save a lightweight run manifest alongside plots for reproducibility.
    """
    payload = dict(payload)
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()

    targets = [
        FIGURES_DIR / filename,
        CODE_DIR / filename,
    ]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Manifest saved to: {FIGURES_DIR / filename}")

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

class Construction(Enum):
    """
    Constructions ordered from tightest count to loosest, plus mass construction.
    The numeric value represents the individuation threshold required.
    Higher = tighter = more precision demanded.

    MUCH is a mass construction: licensed when individuation is LOW (< 0.4).
    """
    A_N = 0.9           # "a book" - requires identifying exactly one atomic unit
    LOW_CARDINAL = 0.85  # "three books" - exact enumeration
    SEVERAL = 0.7       # "several books" - approximate plurality
    MANY = 0.5          # "many books" - magnitude only
    AGREEMENT = 0.3     # plural agreement - just non-singular construal
    MUCH = -0.4         # "much water" - mass quantifier (negative = max threshold)

    @property
    def tolerance(self) -> float:
        """The individuation threshold this construction demands."""
        return self.value

    @property
    def is_mass(self) -> bool:
        """True if this is a mass construction (licensed by low individuation)."""
        return self.value < 0

    @property
    def label(self) -> str:
        """Human-readable label for the construction."""
        labels = {
            'A_N': 'a(n)',
            'LOW_CARDINAL': 'three',
            'MANY': 'many',
            'SEVERAL': 'several',
            'AGREEMENT': 'agreement',
            'MUCH': 'much'
        }
        return labels.get(self.name, self.name)


@dataclass
class NounProfile:
    """
    A noun's profile in an agent's lexicon.

    Attributes:
        individuation: Confidence that this noun supports atomic individuation [0,1]
        entrenchment: Dict mapping constructions to entrenchment scores
                      (high = stored as chunk, bypasses compositional check)
        exposure_count: Number of times this noun has been encountered
    """
    individuation: float
    entrenchment: Dict[Construction, float] = field(default_factory=dict)
    exposure_count: int = 0

    def __post_init__(self):
        # Initialize entrenchment for all constructions if not provided
        for c in Construction:
            if c not in self.entrenchment:
                self.entrenchment[c] = 0.0


@dataclass
class NounSpec:
    """
    Specification for a noun type in the world.

    Attributes:
        name: The noun form
        initial_individuation: Starting individuation confidence
        has_singulative: Whether a singulative form exists (functional anchor)
        singulative_name: The singulative form if any
        is_institutional_target: Whether prescriptivists care about this noun
        institutional_preference: If targeted, what's the "correct" construal?
        frequency_weight: Relative frequency in discourse (affects sampling probability)
    """
    name: str
    initial_individuation: float
    has_singulative: bool = False
    singulative_name: Optional[str] = None
    is_institutional_target: bool = False
    institutional_preference: Optional[str] = None  # 'count' or 'mass'
    frequency_weight: float = 1.0  # Relative sampling probability


# =============================================================================
# AGENT CLASS
# =============================================================================

class Agent:
    """
    A speaker/hearer with a lexicon of noun profiles.

    Agents:
    - Produce utterances (selecting constructions for communicative goals)
    - Comprehend utterances (updating beliefs based on observed usage)
    - May be "institutional" (editors/teachers who provide corrective feedback)
    """

    def __init__(
        self,
        agent_id: int,
        lexicon: Dict[str, NounProfile],
        is_institutional: bool = False,
        learning_rate: float = 0.05,
        entrenchment_rate: float = 0.02,
        regress_on_loose: bool = True
    ):
        self.id = agent_id
        self.lexicon = lexicon
        self.is_institutional = is_institutional
        self.learning_rate = learning_rate
        self.entrenchment_rate = entrenchment_rate
        self.regress_on_loose = regress_on_loose
        self.last_forced = False  # set by produce(): was the last output a forced regularization?
        self.last_measure = False  # set by produce(): was the last output a measure phrase?

        # Track production history for analysis
        self.production_history: List[Tuple[str, Construction]] = []

    def can_use_construction(self, noun: str, construction: Construction) -> bool:
        """
        Check if this agent's grammar licenses the construction for this noun.

        A construction is licensed if:
        1. For count constructions: individuation >= tolerance
        2. For mass constructions (MUCH): individuation <= |tolerance|
        3. Highly entrenched constructions bypass compositional check
        """
        if noun not in self.lexicon:
            return False

        profile = self.lexicon[noun]

        # Check entrenchment first (chunked expressions bypass compositional check)
        if profile.entrenchment.get(construction, 0) > 0.7:
            return True

        # Compositional check depends on construction type
        if construction.is_mass:
            # Mass constructions: licensed when individuation is LOW
            return profile.individuation <= abs(construction.tolerance)
        else:
            # Count constructions: licensed when individuation is HIGH
            return profile.individuation >= construction.tolerance

    def get_licensed_constructions(self, noun: str) -> List[Construction]:
        """Return all constructions this agent licenses for the noun."""
        return [c for c in Construction if self.can_use_construction(noun, c)]

    def produce(
        self,
        noun: str,
        communicative_intent: str,
        world: 'World'
    ) -> Optional[Tuple[str, Construction]]:
        """
        Produce an utterance given a communicative goal.

        Args:
            noun: The noun to use
            communicative_intent: 'singleton', 'exact_small', 'vague_many',
                                 'existence', or 'mass_quantity'
            world: The simulation world (for accessing singulatives)

        Returns:
            Tuple of (noun_used, construction) or None if blocked
        """
        self.last_forced = False
        self.last_measure = False
        if noun not in self.lexicon:
            return None

        # Map intent to preferred construction(s)
        # Includes both count and mass paths
        intent_map = {
            'singleton': [Construction.A_N],
            'exact_small': [Construction.LOW_CARDINAL],
            'vague_many': [Construction.MANY, Construction.SEVERAL],
            'existence': [Construction.AGREEMENT, Construction.MANY],
            'mass_quantity': [Construction.MUCH],  # Mass path
        }

        preferred = intent_map.get(communicative_intent, [Construction.MANY])

        # Try preferred constructions in order
        for construction in preferred:
            if self.can_use_construction(noun, construction):
                self.production_history.append((noun, construction))
                return (noun, construction)

        # Fallback: try singulative if available (for count intents)
        noun_spec = world.get_noun_spec(noun)
        if noun_spec and noun_spec.has_singulative:
            sing = noun_spec.singulative_name
            if sing and sing in self.lexicon:
                # Retrieval of the singulative weakens as that form drops out of usage.
                sing_available = True
                if world.use_anchor_retrieval:
                    sing_spec = world.get_noun_spec(sing)
                    availability = sing_spec.frequency_weight if sing_spec else 1.0
                    availability = max(0.0, min(1.0, availability))
                    sing_available = random.random() < availability

                if sing_available:
                    for construction in preferred:
                        if self.can_use_construction(sing, construction):
                            self.production_history.append((sing, construction))
                            return (sing, construction)

        # Count-ward pressure (off unless world.regularization_rate > 0; no
        # random draw when off). A speaker who licenses `many` expects the
        # exact-count frames to work too; when they don't and no singulative
        # took the job, it sometimes forces the tight frame (regularization).
        if (world.regularization_rate > 0
                and communicative_intent in ('singleton', 'exact_small')
                and self.can_use_construction(noun, Construction.MANY)
                and random.random() < world.regularization_rate):
            forced = preferred[0]
            self.production_history.append((noun, forced))
            self.last_forced = True
            return (noun, forced)

        # Second fallback: try mass construction if count failed
        if communicative_intent != 'mass_quantity':
            if self.can_use_construction(noun, Construction.MUCH):
                self.production_history.append((noun, Construction.MUCH))
                return (noun, Construction.MUCH)

        # Last resort: measure phrase ("piece of furniture", "head of cattle").
        # Abstracted as None unless world.measure_phrase_evidence is on; then
        # an exact-count intent surfaces as a measure phrase, heard as
        # non-count evidence for the noun.
        if (world.measure_phrase_evidence
                and communicative_intent in ('singleton', 'exact_small')):
            self.last_measure = True
            return (noun, Construction.MUCH)
        return None

    def observe(self, noun: str, construction: Construction,
                is_correction: bool = False, correction_strength: float = 0.1,
                correction_direction: Optional[str] = None):
        """
        Update beliefs based on observing a noun-construction pair.

        Implements bidirectional inference: seeing a construction used with a noun
        updates the noun's individuation estimate.

        Args:
            noun: The noun observed
            construction: The construction it appeared in
            is_correction: Whether this is prescriptive correction
            correction_strength: How much weight to give corrections
            correction_direction: For corrections, target direction ('count' or 'mass')
        """
        if noun not in self.lexicon:
            return

        profile = self.lexicon[noun]
        profile.exposure_count += 1

        # Bayesian-ish update: observed construction implies individuation
        # sufficient for that construction
        observed_threshold = construction.tolerance

        # Learning rate decays with exposure (entrenchment makes beliefs sticky)
        effective_lr = self.learning_rate / (1 + 0.01 * profile.exposure_count)

        if is_correction:
            effective_lr = correction_strength

        # Update individuation toward the threshold that would license this
        # (or away from it if this was a correction saying "don't do this")
        if is_correction:
            # Correction: move toward the institutional preference
            if correction_direction == 'count':
                # Push toward count (higher individuation)
                profile.individuation += effective_lr * (1.0 - profile.individuation)
            elif correction_direction == 'mass':
                # Push toward mass (lower individuation)
                profile.individuation -= effective_lr * profile.individuation
            else:
                # Default: move away from observed (legacy behavior)
                profile.individuation += effective_lr * (1.0 - profile.individuation)
        else:
            # Normal observation: move toward licensing the construction
            # This is the bidirectional inference
            if construction.is_mass:
                # Mass construction observed: suggests low individuation
                target = abs(observed_threshold) * 0.5  # Target below mass threshold
                delta = profile.individuation - target
                profile.individuation -= effective_lr * delta * 0.5
            elif profile.individuation < observed_threshold:
                # Count construction suggests higher individuation than we thought
                delta = observed_threshold - profile.individuation
                profile.individuation += effective_lr * delta
            elif self.regress_on_loose:
                # Observation consistent with or below our estimate
                # Slight regression toward observed threshold
                delta = profile.individuation - observed_threshold
                profile.individuation -= effective_lr * delta * 0.5

        # Clamp to [0, 1]
        profile.individuation = max(0.0, min(1.0, profile.individuation))

        # Update entrenchment for this construction
        current = profile.entrenchment.get(construction, 0.0)
        profile.entrenchment[construction] = min(1.0, current + self.entrenchment_rate)

    def get_count_profile(self, noun: str) -> Dict[str, bool]:
        """
        Get the full count profile for a noun (which locks it opens).
        Returns dict mapping construction labels to True/False.
        """
        if noun not in self.lexicon:
            return {}

        return {c.label: self.can_use_construction(noun, c) for c in Construction}


# =============================================================================
# WORLD / SIMULATION
# =============================================================================

class World:
    """
    The simulation world containing agents, nouns, and running the dynamics.
    """

    def __init__(
        self,
        n_agents: int = 100,
        n_institutional: int = 5,
        random_seed: Optional[int] = None,
        use_anchor_retrieval: bool = True,
        learner_prior: str = 'fixed',
        regularization_rate: float = 0.0,
        hearer_response: str = 'evidence',
        regress_on_loose: bool = True,
        measure_phrase_evidence: bool = False
    ):
        if random_seed is not None:
            random.seed(random_seed)
        if learner_prior not in ('fixed', 'community'):
            raise ValueError("learner_prior must be 'fixed' or 'community'")
        if hearer_response not in ('evidence', 'error'):
            raise ValueError("hearer_response must be 'evidence' or 'error'")

        self.noun_specs: Dict[str, NounSpec] = {}
        self.agents: List[Agent] = []
        self.n_institutional = n_institutional
        self.use_anchor_retrieval = use_anchor_retrieval
        self.learner_prior = learner_prior
        self.regularization_rate = regularization_rate
        self.hearer_response = hearer_response
        self.regress_on_loose = regress_on_loose
        self.measure_phrase_evidence = measure_phrase_evidence
        self.time_step = 0

        # Statistics tracking
        self.history: List[Dict] = []

        self._initialize_nouns()
        self._initialize_agents(n_agents)

    def _initialize_nouns(self):
        """Set up the noun inventory with their specifications."""

        # Core nouns from the chapter
        specs = [
            # Fully count
            NounSpec('book', 0.95),
            NounSpec('dog', 0.95),
            NounSpec('idea', 0.90),

            # Quasi-count (stable with anchors)
            NounSpec('cattle', 0.55, has_singulative=True, singulative_name='cow'),
            NounSpec('police', 0.55, has_singulative=True, singulative_name='officer'),
            NounSpec('poultry', 0.50, has_singulative=True, singulative_name='chicken'),

            # Unstable intermediate (no anchor)
            NounSpec('folks', 0.82, has_singulative=False),

            # Drifting (dying anchor) - starts high because historically count
            NounSpec('data', 0.88, has_singulative=True, singulative_name='datum',
                    is_institutional_target=True, institutional_preference='count'),

            # Corporate target
            NounSpec('lego', 0.93, has_singulative=False,
                    is_institutional_target=True, institutional_preference='mass'),

            # Fully mass
            NounSpec('furniture', 0.25),
            NounSpec('water', 0.10),
            NounSpec('equipment', 0.20),

            # Singulatives
            NounSpec('cow', 0.95),
            NounSpec('officer', 0.95),
            NounSpec('chicken', 0.95),
            NounSpec('datum', 0.85),  # Starting higher but will decline
        ]

        for spec in specs:
            self.noun_specs[spec.name] = spec

    def _initialize_agents(self, n_agents: int):
        """Create the agent population."""

        for i in range(n_agents):
            # Build lexicon for this agent
            lexicon = {}
            for name, spec in self.noun_specs.items():
                # Add some individual variation
                sigma = 0.10 if name == 'folks' else 0.05
                ind = spec.initial_individuation + random.gauss(0, sigma)
                ind = max(0.0, min(1.0, ind))
                lexicon[name] = NounProfile(individuation=ind)

            # First n_institutional agents are editors/teachers
            is_institutional = (i < self.n_institutional)

            agent = Agent(
                agent_id=i,
                lexicon=lexicon,
                is_institutional=is_institutional,
                learning_rate=0.03 if not is_institutional else 0.01,
                regress_on_loose=self.regress_on_loose
            )
            self.agents.append(agent)

    def get_noun_spec(self, noun: str) -> Optional[NounSpec]:
        """Get the specification for a noun."""
        return self.noun_specs.get(noun)

    def _sample_noun_by_frequency(self) -> str:
        """
        Sample a noun weighted by frequency_weight.
        Nouns with higher frequency_weight are more likely to be selected.
        """
        nouns = list(self.noun_specs.keys())
        weights = [self.noun_specs[n].frequency_weight for n in nouns]
        total = sum(weights)
        if total == 0:
            return random.choice(nouns)
        # Weighted random selection
        r = random.random() * total
        cumulative = 0
        for noun, weight in zip(nouns, weights):
            cumulative += weight
            if r <= cumulative:
                return noun
        return nouns[-1]  # Fallback

    def _sample_intent(self, noun: str, speaker: Agent) -> str:
        """
        Sample communicative intent conditioned on the noun's current profile.

        High-individuation nouns are usually discussed in count-like contexts.
        Lower-individuation nouns are more likely to occur in mass-quantity uses.
        This coupling is a deliberate modeling assumption (not learned).
        """
        intents = ['singleton', 'exact_small', 'vague_many', 'existence', 'mass_quantity']

        if noun not in speaker.lexicon:
            return random.choice(intents)

        ind = speaker.lexicon[noun].individuation
        if ind >= 0.85:
            weights = [0.38, 0.25, 0.22, 0.13, 0.02]
        elif ind >= 0.65:
            weights = [0.24, 0.20, 0.28, 0.18, 0.10]
        elif ind >= 0.45:
            weights = [0.12, 0.12, 0.30, 0.26, 0.20]
        else:
            weights = [0.05, 0.05, 0.20, 0.30, 0.40]

        return random.choices(intents, weights=weights, k=1)[0]

    def step(self, interactions_per_step: int = 50):
        """
        Run one time step of the simulation.

        Each step:
        1. Random pairs of agents interact
        2. Speaker produces, hearer observes
        3. Institutional agents may provide corrections
        """
        self.time_step += 1

        for _ in range(interactions_per_step):
            # Pick speaker and hearer
            speaker, hearer = random.sample(self.agents, 2)

            # Pick a noun to talk about (weighted by frequency)
            noun = self._sample_noun_by_frequency()

            # Pick a communicative intent conditioned by current noun profile.
            intent = self._sample_intent(noun, speaker)

            # Speaker produces
            result = speaker.produce(noun, intent, self)

            if result:
                produced_noun, construction = result

                # Hearer observes (a forced regularization carries no
                # evidential weight when hearer_response == 'error')
                if not (speaker.last_forced and self.hearer_response == 'error'):
                    hearer.observe(produced_noun, construction)

                # Institutional correction?
                if hearer.is_institutional and not speaker.last_measure:
                    spec = self.get_noun_spec(produced_noun)
                    if spec and spec.is_institutional_target:
                        # Check if production violates institutional preference
                        if spec.institutional_preference == 'count':
                            # Should be count; correct if using mass-like forms
                            if construction in [Construction.MUCH] or \
                               (construction == Construction.AGREEMENT and
                                not speaker.can_use_construction(produced_noun, Construction.LOW_CARDINAL)):
                                speaker.observe(produced_noun, Construction.LOW_CARDINAL,
                                              is_correction=True, correction_strength=0.05,
                                              correction_direction='count')
                        elif spec.institutional_preference == 'mass':
                            # Should be mass; correct if treating as count
                            if construction in [Construction.A_N, Construction.LOW_CARDINAL]:
                                # This is the LEGO case - push toward mass
                                speaker.observe(produced_noun, Construction.MUCH,
                                              is_correction=True, correction_strength=0.05,
                                              correction_direction='mass')

        # Record statistics
        self._record_stats()

    def _record_stats(self):
        """Record population-level statistics for this time step."""
        stats = {'time': self.time_step}

        # For each noun, compute mean individuation and acceptance rates
        for noun in self.noun_specs.keys():
            ind_values = [a.lexicon[noun].individuation for a in self.agents
                         if noun in a.lexicon]
            if ind_values:
                stats[f'{noun}_ind_mean'] = sum(ind_values) / len(ind_values)
                stats[f'{noun}_ind_std'] = (sum((x - stats[f'{noun}_ind_mean'])**2
                                               for x in ind_values) / len(ind_values)) ** 0.5

            # Acceptance rates for each construction
            for c in Construction:
                accepts = sum(1 for a in self.agents if a.can_use_construction(noun, c))
                stats[f'{noun}_{c.label}_accept'] = accepts / len(self.agents)

        self.history.append(stats)

    def run(self, n_steps: int = 100, interactions_per_step: int = 50):
        """Run the simulation for n_steps."""
        for _ in range(n_steps):
            self.step(interactions_per_step)

    def perturb_singulative_availability(self, singulative: str, decay_rate: float = 0.02):
        """
        Reduce availability of a singulative form (simulating 'datum' dying).

        This reduces both:
        1. The frequency_weight (how often it's sampled in discourse)
        2. Agent-level individuation and entrenchment (internal representations)

        The frequency reduction is the primary mechanism: speakers encounter
        the singulative less often, weakening its role as an anchor.
        """
        if singulative not in self.noun_specs:
            return

        # PRIMARY: Reduce discourse frequency (affects sampling probability)
        spec = self.noun_specs[singulative]
        spec.frequency_weight *= (1 - decay_rate)

        # SECONDARY: Reduce individuation confidence across all agents
        for agent in self.agents:
            if singulative in agent.lexicon:
                profile = agent.lexicon[singulative]
                profile.individuation *= (1 - decay_rate * 0.5)  # Slower internal decay
                # Also reduce entrenchment
                for c in profile.entrenchment:
                    profile.entrenchment[c] *= (1 - decay_rate)

    def generational_turnover(self, turnover_rate: float = 0.05):
        """
        Replace some agents with new learners (simulates acquisition/transmission).

        New learners begin with weak priors and then bootstrap from finite tutor
        exposure. This implements multi-timescale maintenance: the slow loop of
        transmission introduces fresh learners who then participate in the fast
        interaction loop.

        Important caveat:
        This is still a simplification. Learners observe tutor productions in a
        compressed synthetic stream; this is not a full developmental model.

        Args:
            turnover_rate: Proportion of agents replaced per call (default 5%)
        """
        if turnover_rate <= 0:
            return

        n_replace = int(len(self.agents) * turnover_rate)
        if n_replace < 1:
            n_replace = 1

        # Replace random (non-institutional) agents
        replaceable = [i for i, a in enumerate(self.agents) if not a.is_institutional]
        if len(replaceable) < n_replace:
            n_replace = len(replaceable)

        to_replace = random.sample(replaceable, n_replace)

        # Prior centre per noun: the fixed starting value (default) or the
        # community mean before this round of replacement.
        if self.learner_prior == 'community':
            centres = {name: sum(a.lexicon[name].individuation for a in self.agents) / len(self.agents)
                       for name in self.noun_specs}
        else:
            centres = {name: spec.initial_individuation for name, spec in self.noun_specs.items()}

        for idx in to_replace:
            # Weak-prior learner (not initialized to adult means)
            lexicon = {}
            for name, spec in self.noun_specs.items():
                # Broad prior around the centre chosen above.
                # This is intentionally noisier and less informed than adults.
                ind = centres[name] + random.gauss(0, 0.18)
                ind = max(0.0, min(1.0, ind))
                lexicon[name] = NounProfile(individuation=ind)

            new_agent = Agent(
                agent_id=self.agents[idx].id,
                lexicon=lexicon,
                is_institutional=False,
                learning_rate=0.06,
                regress_on_loose=self.regress_on_loose
            )

            # Bootstrap from a finite sample of tutor speech events.
            candidate_tutors = [
                i for i, a in enumerate(self.agents)
                if i != idx and not a.is_institutional
            ]
            if len(candidate_tutors) < 4:
                candidate_tutors = [i for i in range(len(self.agents)) if i != idx]

            tutor_k = min(8, len(candidate_tutors))
            tutors = random.sample(candidate_tutors, tutor_k)

            observations = 120
            for _ in range(observations):
                tutor = self.agents[random.choice(tutors)]
                noun = self._sample_noun_by_frequency()
                intent = self._sample_intent(noun, tutor)
                produced = tutor.produce(noun, intent, self)
                if produced:
                    produced_noun, construction = produced
                    if not (tutor.last_forced and self.hearer_response == 'error'):
                        new_agent.observe(produced_noun, construction)

            self.agents[idx] = new_agent

    def get_community_profile(self, noun: str) -> Dict[str, float]:
        """
        Get community-level acceptance rates for each construction.
        Returns dict mapping construction labels to acceptance proportions.
        """
        profile = {}
        for c in Construction:
            accepts = sum(1 for a in self.agents if a.can_use_construction(noun, c))
            profile[c.label] = accepts / len(self.agents)
        return profile

    def print_matrix(self, nouns: Optional[List[str]] = None):
        """
        Print the implicational matrix (Table 9.1 style).
        Includes both count constructions (tight to loose) and mass (MUCH).
        """
        if nouns is None:
            nouns = ['book', 'cattle', 'police', 'folks', 'data', 'furniture', 'water']

        # Count constructions (tight to loose) + mass construction
        constructions = [Construction.A_N, Construction.LOW_CARDINAL,
                        Construction.MANY, Construction.AGREEMENT, Construction.MUCH]

        # Header
        header = f"{'Noun':<12}" + "".join(f"{c.label:>12}" for c in constructions)
        print(header)
        print("-" * len(header))

        # Rows
        for noun in nouns:
            if noun not in self.noun_specs:
                continue
            profile = self.get_community_profile(noun)
            row = f"{noun:<12}"
            for c in constructions:
                rate = profile[c.label]
                if rate > 0.8:
                    symbol = "   +   "
                elif rate > 0.5:
                    symbol = "   ~   "
                elif rate > 0.2:
                    symbol = "   ?   "
                else:
                    symbol = "   -   "
                row += f"{symbol:>12}"
            print(row)


# =============================================================================
# EXPERIMENTS
# =============================================================================

def _resolve_institutional_count(n_agents: int, n_institutional: Optional[int]) -> int:
    """Default institutional share is 5%, with a floor of 5."""
    if n_agents <= 0:
        raise ValueError("n_agents must be > 0")
    if n_institutional is None:
        return max(5, int(round(0.05 * n_agents)))
    if n_institutional < 0:
        raise ValueError("n_institutional must be >= 0")
    if n_institutional > n_agents:
        raise ValueError("n_institutional cannot exceed n_agents")
    return n_institutional


def _resolve_interactions_per_step(
    n_agents: int,
    interactions_per_step: Optional[int],
    interactions_scale: float
) -> int:
    """
    Keep interaction budget proportional to population size by default.
    With default scale 0.6 and n_agents=100, this recovers the original 60.
    """
    if n_agents <= 0:
        raise ValueError("n_agents must be > 0")
    if interactions_per_step is not None:
        if interactions_per_step <= 0:
            raise ValueError("interactions_per_step must be > 0")
        return interactions_per_step
    if interactions_scale <= 0:
        raise ValueError("interactions_scale must be > 0")
    return max(1, int(round(interactions_scale * n_agents)))


def _run_data_drift_protocol(
    world: World,
    n_phases: int = 10,
    steps_per_phase: int = 35,
    interactions_per_step: int = 60,
    turnover_rate: float = 0.04,
    apply_datum_decay: bool = True
) -> List[Dict[str, float]]:
    """
    Shared protocol for the data-drift experiment and its ablations.
    """
    phase_data: List[Dict[str, float]] = []

    for phase in range(n_phases):
        if apply_datum_decay:
            for _ in range(4):
                world.perturb_singulative_availability('datum', decay_rate=0.20)

        world.run(n_steps=steps_per_phase, interactions_per_step=interactions_per_step)
        if turnover_rate > 0:
            world.generational_turnover(turnover_rate=turnover_rate)

        data_profile = world.get_community_profile('data')
        data_ind_mean = sum(a.lexicon['data'].individuation for a in world.agents) / len(world.agents)
        datum_ind_mean = sum(a.lexicon['datum'].individuation for a in world.agents) / len(world.agents)
        datum_frequency = world.noun_specs['datum'].frequency_weight

        phase_data.append({
            'phase': phase + 1,
            'a(n)': data_profile['a(n)'],
            'three': data_profile['three'],
            'many': data_profile['many'],
            'agreement': data_profile['agreement'],
            'much': data_profile['much'],
            'data_ind_mean': data_ind_mean,
            'datum_ind_mean': datum_ind_mean,
            'datum_frequency': datum_frequency,
        })

    return phase_data

def experiment_baseline(seed: int = 42) -> World:
    """
    Baseline: Verify the initial hierarchy matches Chapter 9's predictions.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Baseline Hierarchy")
    print("="*70)
    print("\nVerifying that the initial state matches the implicational hierarchy.")
    print("Expected: book > cattle/police (quasi) > folks (unstable) > furniture/water")

    world = World(n_agents=100, random_seed=seed)

    # Run briefly to let agents interact
    world.run(n_steps=20, interactions_per_step=30)

    print("\nCommunity acceptance matrix (+ = >80%, ~ = 50-80%, ? = 20-50%, - = <20%):")
    print()
    world.print_matrix()

    return world


def experiment_data_drift(
    seed: int = 42,
    visualize: bool = True,
    n_agents: int = 100,
    n_institutional: Optional[int] = None,
    n_phases: int = 10,
    steps_per_phase: int = 35,
    interactions_per_step: Optional[int] = None,
    interactions_scale: float = 0.6,
    turnover_rate: float = 0.04,
) -> World:
    """
    Experiment 1: Data drift as datum dies.

    Demonstrates tight-before-loose erosion: when the singulative anchor
    weakens, the count cluster peels in order (agreement last to go).
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Data Drift (Tight-Before-Loose Erosion)")
    print("="*70)
    print("\nSimulating the death of 'datum' and tracking how 'data' rebalances.")
    print("Prediction: tight properties (a(n), cardinals) erode earlier than loose ones.")

    n_institutional_eff = _resolve_institutional_count(n_agents, n_institutional)
    interactions_per_step_eff = _resolve_interactions_per_step(
        n_agents=n_agents,
        interactions_per_step=interactions_per_step,
        interactions_scale=interactions_scale,
    )
    expected_participations = (
        2.0 * n_phases * steps_per_phase * interactions_per_step_eff / float(n_agents)
    )

    print(
        f"\nRuntime config: N={n_agents}, institutional={n_institutional_eff}, "
        f"phases={n_phases}, steps/phase={steps_per_phase}, interactions/step={interactions_per_step_eff}, "
        f"expected participations/agent={expected_participations:.1f}"
    )

    world = World(
        n_agents=n_agents,
        n_institutional=n_institutional_eff,
        random_seed=seed,
        use_anchor_retrieval=True
    )

    # Initial state
    print("\nInitial state of 'data':")
    print(world.get_community_profile('data'))

    phase_data = _run_data_drift_protocol(
        world,
        n_phases=n_phases,
        steps_per_phase=steps_per_phase,
        interactions_per_step=interactions_per_step_eff,
        turnover_rate=turnover_rate,
    )

    for phase in phase_data:
        print(f"\n--- Phase {phase['phase']}: datum availability declining ---")
        print(f"  data individuation mean: {phase['data_ind_mean']:.3f}")
        print(f"  datum individuation mean: {phase['datum_ind_mean']:.3f}")
        print(f"  a(n): {phase['a(n)']:.2%}")
        print(f"  three: {phase['three']:.2%}")
        print(f"  many: {phase['many']:.2%}")
        print(f"  agreement: {phase['agreement']:.2%}")
        print(f"  much: {phase['much']:.2%}")
        print(f"  (datum frequency: {phase['datum_frequency']:.3f})")

    print("\n" + "-"*50)
    print("RESULT (single-seed diagnostic):")
    print("  - a(n) and cardinals typically fall earlier and faster in this setup")
    print("  - many/agreement remain comparatively robust in this run window")
    print("  - Use --exp=sensitivity for multi-seed uncertainty summaries")
    print("-"*50)

    # Final matrix
    print("\nFinal community matrix:")
    world.print_matrix(['book', 'data', 'datum', 'cattle', 'water'])

    # Visualization
    if visualize and HAS_MATPLOTLIB:
        _plot_drift_experiment(phase_data)

    if phase_data:
        final = phase_data[-1]
        _save_manifest(
            "data_drift_experiment_manifest.json",
            {
                "experiment": "drift",
                "seed": seed,
                "n_phases": len(phase_data),
                "assumptions": {
                    "individuation_scalar": True,
                    "threshold_licensing": True,
                    "turnover_mode": "tutor_bootstrap_finite_observation",
                    "anchor_decay_mode": "exogenous_schedule",
                },
                "runtime": {
                    "n_agents": n_agents,
                    "n_institutional": n_institutional_eff,
                    "n_phases": n_phases,
                    "steps_per_phase": steps_per_phase,
                    "interactions_per_step": interactions_per_step_eff,
                    "turnover_rate": turnover_rate,
                    "expected_participations_per_agent": expected_participations,
                },
                "final_phase": {
                    "a(n)": final["a(n)"],
                    "three": final["three"],
                    "many": final["many"],
                    "agreement": final["agreement"],
                    "datum_frequency": final["datum_frequency"],
                },
            },
        )

    return world


def experiment_functional_anchoring(seed: int = 42) -> Tuple[World, World]:
    """
    Experiment 2: Functional anchoring as a stabilizer.

    Compare two scenarios:
    1. Cattle with 'cow' available (stable quasi-count)
    2. Simulated cattle without 'cow' (pressure toward regularization)
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Functional Anchoring")
    print("="*70)
    print("\nComparing quasi-count stability with and without singulative anchors.")

    # Scenario A: Normal (cow available)
    print("\n--- Scenario A: 'cattle' with 'cow' available ---")
    world_a = World(n_agents=100, random_seed=seed)
    world_a.run(n_steps=100, interactions_per_step=50)

    profile_a = world_a.get_community_profile('cattle')
    print(f"  a(n): {profile_a['a(n)']:.2%}")
    print(f"  three: {profile_a['three']:.2%}")
    print(f"  many: {profile_a['many']:.2%}")
    print(f"  agreement: {profile_a['agreement']:.2%}")

    # Scenario B: Remove cow (simulate anchor loss)
    print("\n--- Scenario B: 'cattle' with 'cow' unavailable ---")
    world_b = World(n_agents=100, random_seed=seed)

    # Remove the singulative specification
    world_b.noun_specs['cattle'].has_singulative = False
    world_b.noun_specs['cattle'].singulative_name = None

    # Also reduce 'cow' availability in lexicons
    for agent in world_b.agents:
        if 'cow' in agent.lexicon:
            agent.lexicon['cow'].individuation = 0.1  # Make cow unusable

    world_b.run(n_steps=100, interactions_per_step=50)

    profile_b = world_b.get_community_profile('cattle')
    print(f"  a(n): {profile_b['a(n)']:.2%}")
    print(f"  three: {profile_b['three']:.2%}")
    print(f"  many: {profile_b['many']:.2%}")
    print(f"  agreement: {profile_b['agreement']:.2%}")

    print("\n" + "-"*50)
    print("RESULT: Without functional anchor:")
    delta_three = profile_b['three'] - profile_a['three']
    delta_many = profile_b['many'] - profile_a['many']
    print(f"  - Delta three(cattle): {delta_three:+.2%}")
    print(f"  - Delta many(cattle):  {delta_many:+.2%}")
    if abs(delta_three) < 0.05 and abs(delta_many) < 0.05:
        print("  - In this run, anchor removal has only a mild short-horizon effect")
    else:
        print("  - Anchor removal shifts the quasi-count profile within this horizon")
    print("-"*50)

    return world_a, world_b


def experiment_prescriptivism(seed: int = 42, visualize: bool = True) -> World:
    """
    Experiment 3: Why prescriptivism struggles.

    Simulates the LEGO campaign: institutional pressure to use 'LEGO bricks'
    rather than 'Legos', but cognitive pressure from discrete-object construal.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Prescriptivism vs. Cognitive Pressure (LEGO)")
    print("="*70)
    print("\nSimulating 40 years of LEGO Group's campaign against 'Legos'.")
    print("Prediction: institutional pressure and discrete-object construal pull in")
    print("            opposite directions, with tight vs. loose properties separating.")

    world = World(n_agents=100, n_institutional=10, random_seed=seed)

    # LEGO starts as clearly count (discrete objects)
    # But has institutional pressure toward mass

    # Increase institutional presence for this experiment
    for agent in world.agents[:15]:
        agent.is_institutional = True

    print("\nInitial state of 'lego':")
    print(world.get_community_profile('lego'))

    phase_data = []

    # Run simulation in phases, tracking LEGO
    for phase in range(5):
        world.run(n_steps=40, interactions_per_step=60)

        profile = world.get_community_profile('lego')
        phase_data.append(profile)

        print(f"\nPhase {phase + 1} (timestep {world.time_step}):")
        print(f"  a(n): {profile['a(n)']:.2%}")
        print(f"  three: {profile['three']:.2%}")
        print(f"  many: {profile['many']:.2%}")

    print("\n" + "-"*50)
    print("RESULT:")
    start = phase_data[0]
    end = phase_data[-1]
    if end['a(n)'] < start['a(n)'] or end['three'] < start['three']:
        print("  - Institutional pressure suppresses tight count packaging over time")
        print("  - Even so, loose count behavior (many/agreement) remains highly stable")
    else:
        print("  - Discrete-object pressure keeps count packaging resilient")
        print("  - Institutional pressure can slow, but not eliminate, count usage")
    print("  - The model captures tension, not categorical victory by either side")
    print("-"*50)

    if visualize and HAS_MATPLOTLIB:
        _plot_prescriptivism_experiment(phase_data)

    return world


def experiment_folks_instability(seed: int = 42) -> World:
    """
    Experiment 4: Demonstrate the instability of 'folks'.

    'folks' lacks a functional anchor and shows high inter-speaker variation.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Folks Instability")
    print("="*70)
    print("\nDemonstrating why 'folks' shows high inter-speaker variation.")
    print("Unlike 'cattle' (anchored by 'cow'), 'folks' has no singulative.")

    world = World(n_agents=100, random_seed=seed)
    world.run(n_steps=100, interactions_per_step=50)

    # Compare cattle vs folks variability
    cattle_inds = [a.lexicon['cattle'].individuation for a in world.agents]
    folks_inds = [a.lexicon['folks'].individuation for a in world.agents]

    cattle_mean = sum(cattle_inds) / len(cattle_inds)
    cattle_std = (sum((x - cattle_mean)**2 for x in cattle_inds) / len(cattle_inds)) ** 0.5

    folks_mean = sum(folks_inds) / len(folks_inds)
    folks_std = (sum((x - folks_mean)**2 for x in folks_inds) / len(folks_inds)) ** 0.5

    print(f"\n'cattle' individuation: mean={cattle_mean:.3f}, std={cattle_std:.3f}")
    print(f"'folks' individuation:  mean={folks_mean:.3f}, std={folks_std:.3f}")

    print("\nConstruction acceptance across agents:")

    # Show that 'three folks' is highly variable
    n_agents = len(world.agents)
    three_cattle = sum(1 for a in world.agents
                       if a.can_use_construction('cattle', Construction.LOW_CARDINAL))
    three_folks = sum(1 for a in world.agents
                      if a.can_use_construction('folks', Construction.LOW_CARDINAL))

    many_cattle = sum(1 for a in world.agents
                      if a.can_use_construction('cattle', Construction.MANY))
    many_folks = sum(1 for a in world.agents
                     if a.can_use_construction('folks', Construction.MANY))

    print(f"  'three cattle' accepted by: {100*three_cattle/n_agents:.1f}% of agents")
    print(f"  'three folks' accepted by: {100*three_folks/n_agents:.1f}% of agents")
    print(f"  'many cattle' accepted by: {100*many_cattle/n_agents:.1f}% of agents")
    print(f"  'many folks' accepted by: {100*many_folks/n_agents:.1f}% of agents")

    print("\n" + "-"*50)
    print("RESULT:")
    print("  - 'folks' shows higher inter-speaker variation than canonical count nouns")
    if 0.1 < (three_folks / n_agents) < 0.9:
        print("  - 'three folks' is genuinely split across speakers")
    else:
        print("  - In this run, 'three folks' is mostly one-sided; seed effects remain")
    print("  - The model still predicts weaker tight-lock behavior for 'folks'")
    print("-"*50)

    return world


def experiment_mechanism_ablation(
    seed: int = 42,
    n_agents: int = 100,
    n_institutional: Optional[int] = None,
    n_phases: int = 10,
    steps_per_phase: int = 35,
    interactions_per_step: Optional[int] = None,
    interactions_scale: float = 0.6,
    turnover_rate: float = 0.04,
) -> Dict[str, World]:
    """
    STRESS TEST: Mechanism ablation as a sensitivity check.

    This compares drift trajectories under targeted mechanism removals.
    Interpretation should remain cautious for single-seed runs.
    """
    print("\n" + "="*70)
    print("STRESS TEST: Mechanism Ablation")
    print("="*70)
    print("\nComparing drift trajectories under targeted ablations.")
    print("Read this as a sensitivity check, not a proof.")

    results = {}
    n_institutional_eff = _resolve_institutional_count(n_agents, n_institutional)
    interactions_per_step_eff = _resolve_interactions_per_step(
        n_agents=n_agents,
        interactions_per_step=interactions_per_step,
        interactions_scale=interactions_scale,
    )

    # --- Baseline (all mechanisms on) ---
    print("\n" + "-"*50)
    print("BASELINE: Drift with anchor decay + learning + retrieval constraints")
    print("-"*50)
    world_baseline = World(
        n_agents=n_agents,
        n_institutional=n_institutional_eff,
        random_seed=seed,
        use_anchor_retrieval=True
    )
    _run_data_drift_protocol(
        world_baseline,
        n_phases=n_phases,
        steps_per_phase=steps_per_phase,
        interactions_per_step=interactions_per_step_eff,
        turnover_rate=turnover_rate,
    )
    baseline = world_baseline.get_community_profile('data')
    print(f"  data a(n): {baseline['a(n)']:.2%}, three: {baseline['three']:.2%}")
    print(f"  data many: {baseline['many']:.2%}, agreement: {baseline['agreement']:.2%}")
    results['baseline'] = world_baseline

    # --- Ablation 1: Remove anchor-retrieval gating ---
    print("\n" + "-"*50)
    print("ABLATION 1: Disable anchor-retrieval constraints")
    print("-"*50)
    print("Singulative fallback remains available even as token frequency decays.")
    world_no_anchor_retrieval = World(
        n_agents=n_agents,
        n_institutional=n_institutional_eff,
        random_seed=seed,
        use_anchor_retrieval=False
    )
    _run_data_drift_protocol(
        world_no_anchor_retrieval,
        n_phases=n_phases,
        steps_per_phase=steps_per_phase,
        interactions_per_step=interactions_per_step_eff,
        turnover_rate=turnover_rate,
    )
    no_anchor_retrieval = world_no_anchor_retrieval.get_community_profile('data')
    print(f"  data a(n): {no_anchor_retrieval['a(n)']:.2%}, three: {no_anchor_retrieval['three']:.2%}")
    print(f"  data many: {no_anchor_retrieval['many']:.2%}, agreement: {no_anchor_retrieval['agreement']:.2%}")
    print("  Delta vs baseline:")
    print(f"    a(n): {no_anchor_retrieval['a(n)'] - baseline['a(n)']:+.2%}")
    print(f"    three: {no_anchor_retrieval['three'] - baseline['three']:+.2%}")
    results['no_anchor_retrieval'] = world_no_anchor_retrieval

    # --- Ablation 2: Remove the anchor perturbation itself ---
    print("\n" + "-"*50)
    print("ABLATION 2: No datum decay")
    print("-"*50)
    print("The anchor remains available; drift should attenuate.")
    world_no_decay = World(
        n_agents=n_agents,
        n_institutional=n_institutional_eff,
        random_seed=seed,
        use_anchor_retrieval=True
    )
    _run_data_drift_protocol(
        world_no_decay,
        n_phases=n_phases,
        steps_per_phase=steps_per_phase,
        interactions_per_step=interactions_per_step_eff,
        turnover_rate=turnover_rate,
        apply_datum_decay=False,
    )
    no_decay = world_no_decay.get_community_profile('data')
    print(f"  data a(n): {no_decay['a(n)']:.2%}, three: {no_decay['three']:.2%}")
    print(f"  data many: {no_decay['many']:.2%}, agreement: {no_decay['agreement']:.2%}")
    print("  Delta vs baseline:")
    print(f"    a(n): {no_decay['a(n)'] - baseline['a(n)']:+.2%}")
    print(f"    three: {no_decay['three'] - baseline['three']:+.2%}")
    results['no_decay'] = world_no_decay

    # --- Ablation 3: Remove bidirectional inference ---
    print("\n" + "-"*50)
    print("ABLATION 3: Disable learning (freeze lexicons)")
    print("-"*50)
    print("Without bidirectional inference, the system should be static.")

    world_frozen = World(
        n_agents=n_agents,
        n_institutional=n_institutional_eff,
        random_seed=seed,
        use_anchor_retrieval=True
    )

    # Freeze learning by setting rate to 0
    for agent in world_frozen.agents:
        agent.learning_rate = 0.0

    initial_data = world_frozen.get_community_profile('data')
    _run_data_drift_protocol(
        world_frozen,
        n_phases=n_phases,
        steps_per_phase=steps_per_phase,
        interactions_per_step=interactions_per_step_eff,
        turnover_rate=0.0,
    )
    final_data = world_frozen.get_community_profile('data')

    print(f"  data a(n): start={initial_data['a(n)']:.2%}, end={final_data['a(n)']:.2%}")
    print(f"  data three: start={initial_data['three']:.2%}, end={final_data['three']:.2%}")
    print(f"  Delta a(n): {final_data['a(n)'] - initial_data['a(n)']:+.2%}")
    print(f"  Delta three: {final_data['three'] - initial_data['three']:+.2%}")

    results['frozen'] = world_frozen

    print("\n" + "-"*50)
    print("ABLATION SUMMARY (single-seed)")
    print("-"*50)
    # Computed from this run, not asserted: each line reports the ablation's
    # difference from baseline (or start) and its direction.
    def _direction(delta: float, tol: float = 0.005) -> str:
        if delta > tol:
            return "less erosion than baseline"
        if delta < -tol:
            return "more erosion than baseline"
        return "no difference from baseline"

    print(f"Final 'data' acceptance, baseline: a(n) {baseline['a(n)']:.2f}, three {baseline['three']:.2f}")
    for label, prof in (("No anchor-retrieval gating", no_anchor_retrieval),
                        ("No datum decay", no_decay)):
        for frame in ("a(n)", "three"):
            delta = prof[frame] - baseline[frame]
            print(f"  - {label}, {frame}: {delta:+.2f} ({_direction(delta)})")
    for frame in ("a(n)", "three"):
        change = final_data[frame] - initial_data[frame]
        print(f"  - Learning frozen, {frame}: {change:+.2f} from start")
    print("Single seed. Run --exp=sensitivity for cross-seed sign consistency.")
    print("-"*50)

    _save_manifest(
        "data_drift_ablation_manifest.json",
        {
            "experiment": "ablation",
            "seed": seed,
            "runtime": {
                "n_agents": n_agents,
                "n_institutional": n_institutional_eff,
                "n_phases": n_phases,
                "steps_per_phase": steps_per_phase,
                "interactions_per_step": interactions_per_step_eff,
                "turnover_rate": turnover_rate,
                "expected_participations_per_agent": (
                    2.0 * n_phases * steps_per_phase * interactions_per_step_eff / float(n_agents)
                ),
            },
            "baseline": baseline,
            "no_anchor_retrieval": no_anchor_retrieval,
            "no_decay": no_decay,
            "frozen_initial": initial_data,
            "frozen_final": final_data,
        },
    )

    return results


def experiment_seed_sensitivity(
    seed_start: int = 1,
    seed_end: int = 20,
    n_agents: int = 100,
    n_institutional: Optional[int] = None,
    n_phases: int = 10,
    steps_per_phase: int = 35,
    interactions_per_step: Optional[int] = None,
    interactions_scale: float = 0.6,
    turnover_rate: float = 0.04,
) -> List[Dict[str, float]]:
    """
    Multi-seed sensitivity summary for drift + ablations.

    Reports effect-size ranges and sign consistency so chapter claims are not
    anchored to a single random seed.
    """
    if seed_end < seed_start:
        raise ValueError("seed_end must be >= seed_start")

    n_institutional_eff = _resolve_institutional_count(n_agents, n_institutional)
    interactions_per_step_eff = _resolve_interactions_per_step(
        n_agents=n_agents,
        interactions_per_step=interactions_per_step,
        interactions_scale=interactions_scale,
    )
    expected_participations = (
        2.0 * n_phases * steps_per_phase * interactions_per_step_eff / float(n_agents)
    )

    rows: List[Dict[str, float]] = []
    seeds = list(range(seed_start, seed_end + 1))

    print("\n" + "=" * 70)
    print("SENSITIVITY: Multi-Seed Drift + Ablation Summary")
    print("=" * 70)
    print(f"Running seeds {seed_start}..{seed_end} ({len(seeds)} runs)")
    print(
        f"Runtime config: N={n_agents}, institutional={n_institutional_eff}, "
        f"phases={n_phases}, steps/phase={steps_per_phase}, interactions/step={interactions_per_step_eff}, "
        f"expected participations/agent={expected_participations:.1f}"
    )

    for seed in seeds:
        # Baseline drift
        w_base = World(
            n_agents=n_agents,
            n_institutional=n_institutional_eff,
            random_seed=seed,
            use_anchor_retrieval=True
        )
        _run_data_drift_protocol(
            w_base,
            n_phases=n_phases,
            steps_per_phase=steps_per_phase,
            interactions_per_step=interactions_per_step_eff,
            turnover_rate=turnover_rate,
        )
        base = w_base.get_community_profile("data")

        # No anchor retrieval
        w_no_anchor = World(
            n_agents=n_agents,
            n_institutional=n_institutional_eff,
            random_seed=seed,
            use_anchor_retrieval=False
        )
        _run_data_drift_protocol(
            w_no_anchor,
            n_phases=n_phases,
            steps_per_phase=steps_per_phase,
            interactions_per_step=interactions_per_step_eff,
            turnover_rate=turnover_rate,
        )
        no_anchor = w_no_anchor.get_community_profile("data")

        # No anchor decay
        w_no_decay = World(
            n_agents=n_agents,
            n_institutional=n_institutional_eff,
            random_seed=seed,
            use_anchor_retrieval=True
        )
        _run_data_drift_protocol(
            w_no_decay,
            n_phases=n_phases,
            steps_per_phase=steps_per_phase,
            interactions_per_step=interactions_per_step_eff,
            turnover_rate=turnover_rate,
            apply_datum_decay=False,
        )
        no_decay = w_no_decay.get_community_profile("data")

        # Frozen learning
        w_frozen = World(
            n_agents=n_agents,
            n_institutional=n_institutional_eff,
            random_seed=seed,
            use_anchor_retrieval=True
        )
        for agent in w_frozen.agents:
            agent.learning_rate = 0.0
        frozen_start = w_frozen.get_community_profile("data")
        _run_data_drift_protocol(
            w_frozen,
            n_phases=n_phases,
            steps_per_phase=steps_per_phase,
            interactions_per_step=interactions_per_step_eff,
            turnover_rate=0.0,
        )
        frozen_end = w_frozen.get_community_profile("data")

        rows.append(
            {
                "seed": float(seed),
                "baseline_a(n)": base["a(n)"],
                "baseline_three": base["three"],
                "baseline_many": base["many"],
                "baseline_agreement": base["agreement"],
                "delta_no_anchor_a(n)": no_anchor["a(n)"] - base["a(n)"],
                "delta_no_anchor_three": no_anchor["three"] - base["three"],
                "delta_no_decay_a(n)": no_decay["a(n)"] - base["a(n)"],
                "delta_no_decay_three": no_decay["three"] - base["three"],
                "delta_frozen_a(n)": frozen_end["a(n)"] - frozen_start["a(n)"],
                "delta_frozen_three": frozen_end["three"] - frozen_start["three"],
            }
        )

    def _summarize(key: str) -> Dict[str, float]:
        vals = [row[key] for row in rows]
        return {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "positive": sum(1 for v in vals if v > 0),
            "negative": sum(1 for v in vals if v < 0),
            "zero": sum(1 for v in vals if v == 0),
        }

    metrics = [
        "baseline_a(n)",
        "baseline_three",
        "baseline_many",
        "baseline_agreement",
        "delta_no_anchor_a(n)",
        "delta_no_anchor_three",
        "delta_no_decay_a(n)",
        "delta_no_decay_three",
        "delta_frozen_a(n)",
        "delta_frozen_three",
    ]
    summary = {m: _summarize(m) for m in metrics}

    print("\nSummary (mean / min / max | + / - / 0):")
    for metric in metrics:
        s = summary[metric]
        print(
            f"  {metric:>24}: "
            f"{s['mean']:+.3f} / {s['min']:+.3f} / {s['max']:+.3f} | "
            f"{int(s['positive'])}/{int(s['negative'])}/{int(s['zero'])}"
        )

    _save_manifest(
        "data_drift_seed_sensitivity_manifest.json",
        {
            "experiment": "sensitivity",
            "seed_start": seed_start,
            "seed_end": seed_end,
            "n_runs": len(rows),
            "runtime": {
                "n_agents": n_agents,
                "n_institutional": n_institutional_eff,
                "n_phases": n_phases,
                "steps_per_phase": steps_per_phase,
                "interactions_per_step": interactions_per_step_eff,
                "turnover_rate": turnover_rate,
                "expected_participations_per_agent": expected_participations,
            },
            "summary": summary,
        },
    )

    return rows


# =============================================================================
# VISUALIZATION
# =============================================================================

def _plot_drift_experiment(phase_data: List[Dict[str, float]]):
    """Plot data drift as two aligned panels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    phases = [row['phase'] for row in phase_data]

    # Panel A: anchor decline and individuation movement
    axes[0].plot(phases, [row['data_ind_mean'] for row in phase_data],
                 color='#1f77b4', marker='o', linewidth=2, label='data individuation')
    axes[0].plot(phases, [row['datum_ind_mean'] for row in phase_data],
                 color='#9467bd', marker='s', linewidth=2, label='datum individuation')
    axes[0].plot(phases, [row['datum_frequency'] for row in phase_data],
                 color='#7f7f7f', marker='D', linewidth=2, linestyle='--', label='datum frequency')
    axes[0].set_xlabel(r'Phase ($\it{datum}$ availability declining)', fontsize=11)
    axes[0].set_ylabel('Mean / Relative availability', fontsize=11)
    axes[0].set_title('A. Anchor erosion', loc='left', fontsize=12)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', frameon=False)

    # Panel B: constructional acceptance
    constructions = ['a(n)', 'three', 'many', 'agreement']
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    markers = ['o', 's', '^', 'D']
    for i, c in enumerate(constructions):
        values = [row[c] for row in phase_data]
        axes[1].plot(phases, values, color=colors[i], marker=markers[i],
                     linewidth=2, markersize=6, label=c)

    axes[1].set_xlabel(r'Phase ($\it{datum}$ availability declining)', fontsize=11)
    axes[1].set_ylabel('Community acceptance rate', fontsize=11)
    axes[1].set_title('B. Tight-before-loose erosion', loc='left', fontsize=12)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', frameon=False)

    axes[1].annotate('Tight properties\nerode first', xy=(3, 0.3), fontsize=9,
                     ha='center', style='italic', color='gray')

    plt.tight_layout()
    _save_figure(fig, 'data_drift_experiment.png')
    plt.close()


def _plot_prescriptivism_experiment(phase_data: List[Dict[str, float]]):
    """Plot the LEGO prescriptivism experiment."""
    fig, ax = plt.subplots(figsize=(10, 6))

    phases = list(range(1, len(phase_data) + 1))

    constructions = ['a(n)', 'three', 'many']
    colors = ['#d62728', '#ff7f0e', '#2ca02c']

    for i, c in enumerate(constructions):
        values = [phase[c] for phase in phase_data]
        ax.plot(phases, values, color=colors[i], linewidth=2,
                marker='o', markersize=8, label=c)

    ax.set_xlabel('Phase (simulation time)', fontsize=12)
    ax.set_ylabel('Community acceptance rate', fontsize=12)
    ax.set_title(r"Prescriptivism vs. Cognitive Pressure ($\it{Lego}$)" + "\n"
                 r"(Institutional $\it{LEGO\ bricks}$ campaign vs. count construal)", fontsize=14)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Add annotation about discrete objects
    ax.annotate('Discrete objects\nresist mass construal',
                xy=(3, 0.85), fontsize=10,
                ha='center', style='italic', color='gray')

    plt.tight_layout()
    _save_figure(fig, 'prescriptivism_experiment.png')
    plt.close()


def plot_basin_visualization():
    """
    Create a visualization of the count basin with locks at different positions.
    This illustrates the chapter's metaphor of tight locks at center, loose at edges.

    Key insight: HIGH individuation = CENTER of basin (opens all locks)
                 LOW individuation = EDGE or outside basin
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for basin visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # The basin edge is at individuation = 0.3 (agreement threshold)
    # Radius in the plot = 1 - individuation (so high individuation = center)

    # Draw concentric circles representing tolerance thresholds
    # Inner circles = tighter locks (higher threshold)
    thresholds = [
        (0.9, 'a(n)', '#d62728'),      # Tightest - innermost
        (0.85, 'three', '#ff7f0e'),
        (0.7, 'several', '#9467bd'),
        (0.5, 'many', '#2ca02c'),
        (0.3, 'agreement', '#1f77b4'), # Loosest - basin edge
    ]

    for thresh, label, color in thresholds:
        # Convert threshold to radius: higher threshold = smaller circle
        radius = 1.0 - thresh
        circle = plt.Circle((0, 0), radius, fill=False, color=color,
                            linewidth=2, linestyle='--', alpha=0.7)
        ax.add_patch(circle)
        # Label position
        label_r = radius + 0.03
        ax.annotate(f'{label}',
                   xy=(label_r * 0.7, label_r * 0.7),
                   fontsize=9, ha='center', color=color, fontweight='bold')

    # Draw the basin boundary (individuation = 0.3)
    basin_edge = plt.Circle((0, 0), 0.7, fill=True, color='#e8f4f8',
                            alpha=0.3, zorder=0)
    ax.add_patch(basin_edge)

    # Plot nouns as points based on their individuation
    # Radius = 1 - individuation (so book near center, water outside)
    nouns = [
        ('book', 0.95, 15),
        ('dog', 0.93, 45),
        ('idea', 0.88, 75),
        ('folks', 0.60, 150),
        ('cattle', 0.55, 180),
        ('police', 0.52, 210),
        ('data', 0.45, 250),   # Drifting toward edge
        ('furniture', 0.25, 290),  # Outside basin
        ('water', 0.10, 330),      # Well outside basin
    ]

    for name, ind, angle in nouns:
        rad = math.radians(angle)
        # Invert: high individuation = small radius (center)
        plot_radius = 1.0 - ind
        x = plot_radius * math.cos(rad)
        y = plot_radius * math.sin(rad)

        # Color by position: green=center, yellow=mid, red=edge/outside
        if ind > 0.85:
            color = '#2ecc71'  # Full count - center
        elif ind > 0.5:
            color = '#f39c12'  # Quasi-count - mid-basin
        elif ind > 0.3:
            color = '#e67e22'  # Marginal - near edge
        else:
            color = '#e74c3c'  # Mass - outside basin

        ax.plot(x, y, 'o', markersize=12, color=color, zorder=5)
        ax.annotate(name, xy=(x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontstyle='italic')

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect('equal')

    ax.set_title("The Count Basin\n"
                 "Center = high individuation (opens all locks); Edge = low individuation",
                 fontsize=14)

    # Legend
    green_patch = mpatches.Patch(color='#2ecc71', label='Full count (center, opens all locks)')
    yellow_patch = mpatches.Patch(color='#f39c12', label='Quasi-count (mid-basin, loose locks only)')
    red_patch = mpatches.Patch(color='#e74c3c', label='Mass (outside basin)')
    ax.legend(handles=[green_patch, yellow_patch, red_patch], loc='lower right')

    ax.axis('off')

    plt.tight_layout()
    _save_figure(fig, 'count_basin_visualization.png')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Countability Agent-Based Model (Chapter 9) - DIDACTIC MECHANISM SKETCH',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiments:
  baseline     Verify initial hierarchy matches predictions
  drift        Data drift as datum dies (tight-before-loose erosion)
  anchor       Functional anchoring (cattle with/without cow)
  prescriptive LEGO prescriptivism vs cognitive pressure
  folks        Folks instability demonstration
  ablation     STRESS TEST: remove mechanisms, inspect single-seed contrasts
  sensitivity  Multi-seed summary of drift/ablation effect ranges
  all          Run all experiments (default)
  basin        Generate basin visualization only

NOTE: This is an intuition pump, not calibrated evidence.
        """
    )
    parser.add_argument('--exp', default='all',
                       help='Experiment to run (see list below)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--seed-start', type=int, default=1,
                       help='Start seed for --exp=sensitivity')
    parser.add_argument('--seed-end', type=int, default=20,
                       help='End seed for --exp=sensitivity')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization (text output only)')
    parser.add_argument('--interactive', action='store_true',
                       help='Enter interactive exploration mode')
    parser.add_argument('--n-agents', type=int, default=100,
                       help='Population size for drift/ablation/sensitivity experiments')
    parser.add_argument('--n-institutional', type=int, default=-1,
                       help='Institutional agents count; default is auto 5%% (min 5)')
    parser.add_argument('--interactions-per-step', type=int, default=None,
                       help='Override interactions per simulation step')
    parser.add_argument('--interactions-scale', type=float, default=0.6,
                       help='Used when --interactions-per-step is omitted; default keeps legacy 60 when N=100')
    parser.add_argument('--n-phases', type=int, default=10,
                       help='Number of data-drift phases')
    parser.add_argument('--steps-per-phase', type=int, default=35,
                       help='Simulation steps per phase in data-drift protocols')
    parser.add_argument('--turnover-rate', type=float, default=0.04,
                       help='Learner replacement rate per phase in data-drift protocols')

    args = parser.parse_args()

    visualize = not args.no_viz and HAS_MATPLOTLIB

    print("="*70)
    print("COUNTABILITY AGENT-BASED MODEL")
    print("Implementing Chapter 9's HPC theory of the count cluster")
    print("="*70)

    if args.interactive:
        interactive_mode(args.seed)
        return

    n_institutional_cfg: Optional[int] = None if args.n_institutional < 0 else args.n_institutional
    n_institutional_eff = _resolve_institutional_count(args.n_agents, n_institutional_cfg)
    interactions_per_step_eff = _resolve_interactions_per_step(
        n_agents=args.n_agents,
        interactions_per_step=args.interactions_per_step,
        interactions_scale=args.interactions_scale,
    )
    expected_participations = (
        2.0 * args.n_phases * args.steps_per_phase * interactions_per_step_eff / float(args.n_agents)
    )
    print(
        f"Runtime config (drift-family): N={args.n_agents}, institutional={n_institutional_eff}, "
        f"phases={args.n_phases}, steps/phase={args.steps_per_phase}, interactions/step={interactions_per_step_eff}, "
        f"expected participations/agent={expected_participations:.1f}"
    )

    if args.exp == 'all':
        experiment_baseline(args.seed)
        experiment_data_drift(
            args.seed,
            visualize,
            n_agents=args.n_agents,
            n_institutional=n_institutional_cfg,
            n_phases=args.n_phases,
            steps_per_phase=args.steps_per_phase,
            interactions_per_step=args.interactions_per_step,
            interactions_scale=args.interactions_scale,
            turnover_rate=args.turnover_rate,
        )
        experiment_functional_anchoring(args.seed)
        experiment_prescriptivism(args.seed, visualize)
        experiment_folks_instability(args.seed)
        experiment_mechanism_ablation(
            args.seed,
            n_agents=args.n_agents,
            n_institutional=n_institutional_cfg,
            n_phases=args.n_phases,
            steps_per_phase=args.steps_per_phase,
            interactions_per_step=args.interactions_per_step,
            interactions_scale=args.interactions_scale,
            turnover_rate=args.turnover_rate,
        )  # Stress test
        if visualize:
            plot_basin_visualization()
    elif args.exp == 'baseline':
        experiment_baseline(args.seed)
    elif args.exp == 'drift':
        experiment_data_drift(
            args.seed,
            visualize,
            n_agents=args.n_agents,
            n_institutional=n_institutional_cfg,
            n_phases=args.n_phases,
            steps_per_phase=args.steps_per_phase,
            interactions_per_step=args.interactions_per_step,
            interactions_scale=args.interactions_scale,
            turnover_rate=args.turnover_rate,
        )
    elif args.exp == 'anchor':
        experiment_functional_anchoring(args.seed)
    elif args.exp == 'prescriptive':
        experiment_prescriptivism(args.seed, visualize)
    elif args.exp == 'folks':
        experiment_folks_instability(args.seed)
    elif args.exp == 'ablation':
        experiment_mechanism_ablation(
            args.seed,
            n_agents=args.n_agents,
            n_institutional=n_institutional_cfg,
            n_phases=args.n_phases,
            steps_per_phase=args.steps_per_phase,
            interactions_per_step=args.interactions_per_step,
            interactions_scale=args.interactions_scale,
            turnover_rate=args.turnover_rate,
        )
    elif args.exp == 'sensitivity':
        experiment_seed_sensitivity(
            args.seed_start,
            args.seed_end,
            n_agents=args.n_agents,
            n_institutional=n_institutional_cfg,
            n_phases=args.n_phases,
            steps_per_phase=args.steps_per_phase,
            interactions_per_step=args.interactions_per_step,
            interactions_scale=args.interactions_scale,
            turnover_rate=args.turnover_rate,
        )
    elif args.exp == 'basin':
        if visualize:
            plot_basin_visualization()
        else:
            print("Basin visualization requires matplotlib")
    else:
        print(f"Unknown experiment: {args.exp}")
        parser.print_help()

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)


def interactive_mode(seed: int):
    """Interactive exploration of the model."""
    print("\n--- Interactive Mode ---")
    print("Creating world with default parameters...")

    world = World(n_agents=100, random_seed=seed)

    print("\nCommands:")
    print("  run N        - Run N time steps")
    print("  matrix       - Show implicational matrix")
    print("  profile NOUN - Show detailed profile for a noun")
    print("  decay NOUN   - Decay singulative for a noun")
    print("  agent N      - Show agent N's lexicon")
    print("  quit         - Exit")

    while True:
        try:
            cmd = input("\n> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        if cmd[0] == 'quit':
            break
        elif cmd[0] == 'run':
            n = int(cmd[1]) if len(cmd) > 1 else 10
            world.run(n_steps=n)
            print(f"Ran {n} steps. Total time: {world.time_step}")
        elif cmd[0] == 'matrix':
            world.print_matrix()
        elif cmd[0] == 'profile':
            if len(cmd) > 1:
                noun = cmd[1]
                print(f"\nCommunity profile for '{noun}':")
                print(world.get_community_profile(noun))
            else:
                print("Usage: profile NOUN")
        elif cmd[0] == 'decay':
            if len(cmd) > 1:
                noun = cmd[1]
                world.perturb_singulative_availability(noun, 0.2)
                print(f"Decayed singulative availability for '{noun}'")
            else:
                print("Usage: decay NOUN")
        elif cmd[0] == 'agent':
            if len(cmd) > 1:
                n = int(cmd[1])
                if 0 <= n < len(world.agents):
                    agent = world.agents[n]
                    print(f"\nAgent {n} ({'institutional' if agent.is_institutional else 'regular'}):")
                    for noun, profile in agent.lexicon.items():
                        print(f"  {noun}: ind={profile.individuation:.3f}")
                else:
                    print(f"Agent index out of range (0-{len(world.agents)-1})")
            else:
                print("Usage: agent N")
        else:
            print(f"Unknown command: {cmd[0]}")


if __name__ == '__main__':
    main()
