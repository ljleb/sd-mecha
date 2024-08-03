import enum


class MergeSpace(enum.Flag):
    BASE = enum.auto()
    DELTA = enum.auto()

    def __str__(self):
        return super().__str__().split(".")[1].lower().replace("base", "weight") + " space"
