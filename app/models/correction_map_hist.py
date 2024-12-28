from datetime import datetime

from sqlalchemy.orm import mapped_column, Mapped

from app.config.db_session import Base
from decimal import Decimal


class CorrectionMapHist(Base):
    __tablename__ = "correction_map_hist"
    # __table_args__ = {'schema': 'public'}
    __table_args__ = {'schema': 'gateway'}

    correction_map_id: Mapped[str] = mapped_column(primary_key=True)
    longitude: Mapped[Decimal] = mapped_column(primary_key=True)
    latitude: Mapped[Decimal] = mapped_column(primary_key=True)
    longitude_length: Mapped[float] = mapped_column()
    latitude_length: Mapped[float] = mapped_column()
    wind_speed: Mapped[float] = mapped_column()
    wind_direction: Mapped[float] = mapped_column()
    wave_height: Mapped[float] = mapped_column()
    wave_direction: Mapped[float] = mapped_column()
    relative_humidity: Mapped[float] = mapped_column()
    total_column_water_vapor: Mapped[float] = mapped_column()
    skin_temperature: Mapped[float] = mapped_column()
