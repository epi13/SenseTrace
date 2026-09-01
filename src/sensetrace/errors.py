"""Typed failures used at trust boundaries."""


class SenseTraceError(Exception):
    """Base class for expected SenseTrace failures."""


class ConfigError(SenseTraceError):
    """The configuration is invalid or internally inconsistent."""


class SchemaError(SenseTraceError):
    """An artifact does not conform to the expected schema."""


class IntegrityError(SenseTraceError):
    """A persisted artifact failed an integrity check."""


class JournalCorruptionError(IntegrityError):
    """The experiment journal contains an unrecoverable malformed record."""


class ForbiddenFeatureError(SchemaError):
    """A grouping/audit field was requested as a model feature."""
