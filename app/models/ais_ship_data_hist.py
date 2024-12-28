from datetime import datetime
from sqlalchemy import String, Float, Integer, Boolean
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import mapped_column, Mapped
from app.config.db_session import Base


class AisShipDataHist(Base):
  __tablename__ = "ais_ship_data_hist"
  __table_args__ = {'schema': 'gateway'}

  mmsi_process: Mapped[str] = mapped_column(String(20), primary_key=True)
  timestamp: Mapped[datetime] = mapped_column(TIMESTAMP, primary_key=True)
  mmsi: Mapped[str] = mapped_column(String(20))
  imo_no: Mapped[str] = mapped_column(String(15))
  mmsi_origin: Mapped[str] = mapped_column(String(20))
  imo_no_origin: Mapped[str] = mapped_column(String(15))
  position_prcss_sttus_ty: Mapped[str] = mapped_column(String(2))
  static_prcss_sttus_ty: Mapped[str] = mapped_column(String(2))
  mapping_prcss_sttus_ty: Mapped[str] = mapped_column(String(2))
  clstr_total: Mapped[float] = mapped_column(Float)
  cret_dt: Mapped[datetime] = mapped_column(TIMESTAMP)
  updt_dt: Mapped[datetime] = mapped_column(TIMESTAMP)
  ais_ver: Mapped[float] = mapped_column(Float)
  call_sign: Mapped[str] = mapped_column(String(7))
  ship_nm: Mapped[str] = mapped_column(String(32))
  ship_type: Mapped[float] = mapped_column(Float)
  ship_dim_a: Mapped[float] = mapped_column(Float)
  ship_dim_b: Mapped[float] = mapped_column(Float)
  ship_dim_c: Mapped[float] = mapped_column(Float)
  ship_dim_d: Mapped[float] = mapped_column(Float)
  fixing_device_se: Mapped[float] = mapped_column(Float)
  eta: Mapped[datetime] = mapped_column(TIMESTAMP)
  max_draught: Mapped[float] = mapped_column(Float)
  destination: Mapped[str] = mapped_column(String(50))
  static_rcv_dt: Mapped[datetime] = mapped_column(TIMESTAMP)
  lc_rcv_dt: Mapped[datetime] = mapped_column(TIMESTAMP)
  static_message_no: Mapped[float] = mapped_column(Float)
  lc_message_no: Mapped[float] = mapped_column(Float)
  dte_flag: Mapped[bool] = mapped_column(Boolean)
  ais_channel: Mapped[str] = mapped_column(String(5))
  repeat_jdct: Mapped[float] = mapped_column(Float)
  nvg_sttus: Mapped[float] = mapped_column(Float)
  rot: Mapped[float] = mapped_column(Float)
  sog: Mapped[float] = mapped_column(Float)
  pos_accrcy: Mapped[float] = mapped_column(Float)
  lo_la: Mapped[float] = mapped_column(Float)
  la_lo: Mapped[float] = mapped_column(Float)
  heading: Mapped[float] = mapped_column(Float)
  special_manuv_jdct: Mapped[float] = mapped_column(Float)
  raim_flag: Mapped[bool] = mapped_column(Boolean)
  cmmnc_sttus: Mapped[str] = mapped_column(String(100))
  packet_id_s: Mapped[str] = mapped_column(String(400))
  record_dt: Mapped[datetime] = mapped_column(TIMESTAMP)
  ais_class: Mapped[str] = mapped_column(String(1))
  clstr_count: Mapped[float] = mapped_column(Float)
