from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProjectSettings:
    name: str
    version: str


@dataclass
class DataSettings:
    path: str
    target_column: str


@dataclass
class TaskSettings:
    type: str


@dataclass
class SplitSettings:
    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True


@dataclass
class MissingValueSettings:
    strategy: str = "median"


@dataclass
class EncodingSettings:
    strategy: str = "onehot"


@dataclass
class ScalingSettings:
    strategy: str = "standard"


@dataclass
class PreprocessingSettings:
    missing_values: MissingValueSettings
    categorical_encoding: EncodingSettings
    numerical_scaling: ScalingSettings


@dataclass
class FeatureSelectionSettings:
    enabled: bool = True
    method: str = "mutual_information"
    top_k: int = 10


@dataclass
class FeatureSettings:
    selection: FeatureSelectionSettings


@dataclass
class ModelSettings:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSettings:
    metrics: list[str] = field(default_factory=list)


@dataclass
class PersistenceSettings:
    model_path: str


@dataclass
class Settings:
    project: ProjectSettings
    data: DataSettings
    task: TaskSettings
    split: SplitSettings
    preprocessing: PreprocessingSettings
    features: FeatureSettings
    model: ModelSettings
    evaluation: EvaluationSettings
    persistence: PersistenceSettings