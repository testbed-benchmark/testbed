"""Task models for experimental design."""

from .base import Task
from .cart_pole import CartPole
from .ces import CES
from .double_link import DoubleLink
from .gravimetry import Gravimetry
from .linear_regression import LinearRegression
from .location_finding import LocationFinding
from .michaelis_menten import MichaelisMenten
from .pharmacokinetic import Pharmacokinetic
from .prey_population import PreyPopulation
from .stochastic_pendulum import StochasticPendulum
from .temporal_discounting import TemporalDiscounting

# Old names, so that saved run configs still instantiate.
HyperbolicTemporalDiscounting = TemporalDiscounting
PharmacoKinetic = Pharmacokinetic
