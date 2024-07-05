from enum import Enum
from typing import ClassVar, List
from pydantic import BaseModel, Field, field_validator
import numpy as np


class CutEnum(str, Enum):
    FAIR = "Fair"
    GOOD = "Good"
    VERY_GOOD = "Very Good"
    IDEAL = "Ideal"
    PREMIUM = "Premium"

class ColorEnum(str, Enum):
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"
    J = "J"

class ClarityEnum(str, Enum):
    IF = "IF"
    VVS1 = "VVS1"
    VVS2 = "VVS2"
    VS1 = "VS1"
    VS2 = "VS2"
    SI1 = "SI1"
    SI2 = "SI2"
    I1 = "I1"


class Diamond(BaseModel):
    carat: float = Field(
        ...,
        gt=0,
        description="Carat weight of the diamond, must be greater than 0"
    )
    cut: CutEnum = Field(..., description="Cut quality of the diamond")
    color: ColorEnum = Field(..., description="Color grade of the diamond")
    clarity: ClarityEnum = Field(..., description="Clarity grade of the diamond")
    x: float = Field(..., gt=0, description="Dimension x of the diamond, must be greater than 0")

    CUT_MAPPING: ClassVar[dict] = {
        CutEnum.FAIR: 0,
        CutEnum.GOOD: 1,
        CutEnum.VERY_GOOD: 2,
        CutEnum.IDEAL: 3,
        CutEnum.PREMIUM: 4,
    }

    COLOR_MAPPING: ClassVar[dict] = {
        ColorEnum.D: 0,
        ColorEnum.E: 1,
        ColorEnum.F: 2,
        ColorEnum.G: 3,
        ColorEnum.H: 4,
        ColorEnum.I: 5,
        ColorEnum.J: 6
    }

    CLARITY_MAPPING: ClassVar[dict] = {
        ClarityEnum.IF: 0,
        ClarityEnum.VVS1: 1,
        ClarityEnum.VVS2: 2,
        ClarityEnum.VS1: 3,
        ClarityEnum.VS2: 4,
        ClarityEnum.SI1: 5,
        ClarityEnum.SI2: 6,
        ClarityEnum.I1: 7
    }

    # pylint: disable=no-self-argument
    @field_validator('cut')
    def validate_cut(cls, value):
        if value not in cls.CUT_MAPPING:
            raise ValueError('Invalid cut value')
        return value

    # pylint: disable=no-self-argument
    @field_validator('color')
    def validate_color(cls, value):
        if value not in cls.COLOR_MAPPING:
            raise ValueError('Invalid color value')
        return value

    # pylint: disable=no-self-argument
    @field_validator('clarity')
    def validate_clarity(cls, value):
        if value not in cls.CLARITY_MAPPING:
            raise ValueError('Invalid clarity value')
        return value

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_feature_array(self) -> List[np.ndarray]:
        cut_value = self.CUT_MAPPING[self.cut]
        color_value = self.COLOR_MAPPING[self.color]
        clarity_value = self.CLARITY_MAPPING[self.clarity]

        return [np.array([self.carat, cut_value, color_value, clarity_value, self.x], dtype=float)]
