from datetime import date, datetime
from typing import Literal
import math
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyproj import CRS, Transformer
from shapely.geometry import shape


class RasterMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation_id: str = Field(min_length=1, max_length=100)
    crs: str
    transform: tuple[float, float, float, float, float, float]
    model_version: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def valid_grid(self):
        a, b, c, d, e, f = self.transform
        if not all(math.isfinite(x) for x in self.transform) or abs(a * e - b * d) < 1e-12:
            raise ValueError("Некорректное преобразование пикселей")
        try:
            crs = CRS.from_user_input(self.crs)
            Transformer.from_crs(crs, 4326, always_xy=True)
        except Exception as exc:
            raise ValueError("Неизвестная система координат") from exc
        return self


class AFMeta(RasterMeta):
    acquisition_datetime: datetime

    @model_validator(mode="after")
    def timezone_required(self):
        if self.acquisition_datetime.tzinfo is None:
            raise ValueError("Для AF укажите часовой пояс, например Z")
        return self


class BSMeta(RasterMeta):
    date_pre: date
    date_post: date

    @model_validator(mode="after")
    def ordered_dates(self):
        if self.date_pre > self.date_post:
            raise ValueError("Дата до пожара не может быть позже даты после")
        return self


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=100)
    data_mode: Literal["synthetic", "real"]
    source: str = Field(min_length=1, max_length=300)
    active_fire: AFMeta
    burn_severity: BSMeta


class SubmissionChipMeta(RasterMeta):
    kind: Literal["af", "bs"]
    acquisition_datetime: datetime | None = None
    date_pre: date | None = None
    date_post: date | None = None

    @model_validator(mode="after")
    def valid_dates_for_kind(self):
        if self.kind == "af":
            if self.acquisition_datetime is None:
                raise ValueError("Для AF-чипа требуется acquisition_datetime")
            if self.acquisition_datetime.tzinfo is None:
                raise ValueError("Для AF укажите часовой пояс, например Z")
            if self.date_pre is not None or self.date_post is not None:
                raise ValueError("Для AF-чипа поля date_pre и date_post не применяются")
        else:
            if self.date_pre is None or self.date_post is None:
                raise ValueError("Для BS-чипа требуются date_pre и date_post")
            if self.date_pre > self.date_post:
                raise ValueError("Дата до пожара не может быть позже даты после")
            if self.acquisition_datetime is not None:
                raise ValueError("Для BS-чипа acquisition_datetime не применяется")
        return self


class SubmissionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=100)
    data_mode: Literal["synthetic", "real"]
    source: str = Field(min_length=1, max_length=300)
    chips: dict[str, SubmissionChipMeta] = Field(min_length=1)

    @model_validator(mode="after")
    def chip_ids_match_kinds(self):
        for chip_id, meta in self.chips.items():
            expected = "af" if chip_id.startswith("AF_") else "bs" if chip_id.startswith("BS_") else None
            if expected is None:
                raise ValueError(f"Неизвестный префикс chip_id: {chip_id}")
            if meta.kind != expected:
                raise ValueError(f"{chip_id}: kind должен быть {expected}")
        return self


class AnalysisQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    geometry: dict
    date_from: date
    date_to: date

    @model_validator(mode="after")
    def check_query(self):
        if self.date_from > self.date_to:
            raise ValueError("Начало периода позже конца")
        try:
            geometry = shape(self.geometry)
            if geometry.geom_type != "Polygon" or geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
                raise ValueError()
            x1, y1, x2, y2 = geometry.bounds
            if not all(math.isfinite(x) for x in geometry.bounds) or not (-180 <= x1 < x2 <= 180 and -85 <= y1 < y2 <= 85):
                raise ValueError()
            if len(str(self.geometry)) > 100000:
                raise ValueError()
        except Exception as exc:
            raise ValueError("Нужен валидный полигон WGS84: долгота, широта") from exc
        return self


class ExportQuery(BaseModel):
    result_id: str = Field(pattern=r"^[a-f0-9]{32}$")
