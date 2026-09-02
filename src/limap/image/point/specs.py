from dataclasses import dataclass


@dataclass
class PointDetectionOptions:
    method: str = "aliked-n16"


@dataclass
class PointMatcherOptions:
    method: str = "aliked+lightglue"
