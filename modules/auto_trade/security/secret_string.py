class SecretString:
    """Wrapper for sensitive strings to prevent accidental leakage in logs/dumps."""

    def __init__(self, value: str):
        if isinstance(value, SecretString):
            self._value = value.get_secret_value()
        else:
            self._value = value

    def get_secret_value(self) -> str:
        """Explicitly get the unredacted string value."""
        return self._value

    def __str__(self) -> str:
        return "***"

    def __repr__(self) -> str:
        return "SecretString('***')"

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, SecretString):
            return self._value == __value._value
        if isinstance(__value, str):
            return self._value == __value
        return False
