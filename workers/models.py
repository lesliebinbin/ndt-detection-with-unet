from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Enum
from sqlalchemy.sql import func
from sqlalchemy.schema import PrimaryKeyConstraint
from sqlalchemy.orm import sessionmaker

from sqlalchemy import create_engine
import os
from enum import Enum as PyEnum


class WorkStatus(PyEnum):
    INIT = "init"
    ONGOING = "ongoing"
    ERROR = "error"
    FAILED = "failed"
    DONE = "DONE"


Base = declarative_base()


class RootFolder(Base):
    __tablename__ = "root_folders"
    id = Column(Integer, primary_key=True)
    path = Column(String, index=True, unique=True, nullable=False)


class SubFolder(Base):
    __tablename__ = "sub_folders"
    id = Column(Integer, primary_key=True)
    root_folder_id = Column(Integer, ForeignKey("root_folders.id"), nullable=False)
    path = Column(String, index=True, nullable=False)
    status = Column(Enum(WorkStatus), default=WorkStatus.INIT)
    created_at = Column(DateTime, index=True, default=func.now())
    updated_at = Column(DateTime, index=True, default=func.now(), onupdate=func.now())

    @classmethod
    def update_status(cls, folder, Sess, status):
        with Sess() as session:
            folder = session.query(cls).get(folder.id)
            folder.status = status
            session.commit()
            session.refresh(folder)
            return folder

class ClassificationResult(Base):
    __tablename__ = "classification_results"
    id = Column(Integer, primary_key=True)
    sub_folder_id = Column(Integer, ForeignKey("sub_folders.id"), nullable=False)
    file_name = Column(String, index=True, nullable=False)
    label = Column(String, index=True, nullable=False)
    remote_url = Column(String, nullable=False)
    created_at = Column(DateTime, index=True, default=func.now())
    is_filtered = Column(Boolean, default=False)


db_engine_path = "postgresql://dicom-classification:VUWkVxDsVAVe@dan.ccsp.basf.net:64819/dicom-classification"
engine = create_engine(db_engine_path, echo=False, pool_pre_ping=True)
Base.metadata.create_all(engine)
Sess = sessionmaker(bind=engine)
