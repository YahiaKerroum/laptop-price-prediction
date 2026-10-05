"""Price intelligence for the Algerian used-laptop market.

Sub-packages
------------
``cleaning``    normalisation of scraped CPU/GPU/RAM/storage/screen/price fields
``features``    the model-ready feature matrix and its schema
``models``      the end-to-end sklearn Pipeline, training, and the artifact registry
``anomaly``     scam detection, underpriced-deal ranking, spec-consistency checks
``evaluation``  metrics, baselines, and the three split strategies
``api``         FastAPI prediction service
``app``         Streamlit UI
"""

from laptop_price import paths
from laptop_price.config import CONFIG, Config, load_config

__version__ = "1.0.0"

__all__ = ["CONFIG", "Config", "load_config", "paths", "__version__"]
