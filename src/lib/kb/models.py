from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, INT4RANGE
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Note(Base):
    __tablename__ = "notes"

    id = Column(BigInteger, primary_key=True)
    title = Column(Text, nullable=True)
    body = Column(Text, nullable=False)
    source = Column(Text, nullable=True)
    lang = Column(String(8), nullable=True)
    hash = Column(Text, nullable=True, unique=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    units = relationship("Unit", back_populates="note", cascade="all, delete-orphan")


class Unit(Base):
    __tablename__ = "units"

    id = Column(BigInteger, primary_key=True)
    note_id = Column(BigInteger, ForeignKey("notes.id", ondelete="CASCADE"), nullable=False)
    utype = Column(String(32), nullable=False, default="semantic")
    text = Column(Text, nullable=False)
    span = Column(INT4RANGE, nullable=True)
    parent_id = Column(BigInteger, ForeignKey("units.id", ondelete="SET NULL"), nullable=True)
    embed = Column(Vector(768), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    note = relationship("Note", back_populates="units")
    parent = relationship("Unit", remote_side="Unit.id", backref="children")


class Entity(Base):
    __tablename__ = "entities"

    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    etype = Column(Text, nullable=True)

    outgoing = relationship(
        "Relation",
        foreign_keys="Relation.head_id",
        back_populates="head",
        cascade="all, delete-orphan",
    )
    incoming = relationship(
        "Relation",
        foreign_keys="Relation.tail_id",
        back_populates="tail",
        cascade="all, delete-orphan",
    )


class Relation(Base):
    __tablename__ = "relations"

    id = Column(BigInteger, primary_key=True)
    head_id = Column(BigInteger, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    rel = Column(Text, nullable=False)
    tail_id = Column(BigInteger, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    evidence_unit_id = Column(BigInteger, ForeignKey("units.id", ondelete="SET NULL"), nullable=True)

    head = relationship("Entity", foreign_keys=[head_id], back_populates="outgoing")
    tail = relationship("Entity", foreign_keys=[tail_id], back_populates="incoming")
    evidence_unit = relationship("Unit")


class Tag(Base):
    __tablename__ = "tags"

    id = Column(BigInteger, primary_key=True)
    skos_uri = Column(Text, nullable=True, unique=True)
    pref_label = Column(Text, nullable=False)
    alt_labels = Column(ARRAY(Text), nullable=True)

    units = relationship("UnitTag", back_populates="tag", cascade="all, delete-orphan")


class UnitTag(Base):
    __tablename__ = "unit_tags"

    unit_id = Column(BigInteger, ForeignKey("units.id", ondelete="CASCADE"), primary_key=True)
    tag_id = Column(BigInteger, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)

    unit = relationship("Unit", back_populates="tag_links")
    tag = relationship("Tag", back_populates="units")


# 双方向関係を後から設定
Unit.tag_links = relationship("UnitTag", back_populates="unit", cascade="all, delete-orphan")
