import enum


class MergeSpace(enum.Flag):
    BASE = enum.auto()
    DELTA = enum.auto()
