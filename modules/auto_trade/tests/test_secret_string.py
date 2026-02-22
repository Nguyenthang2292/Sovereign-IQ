import pytest
from modules.auto_trade.security.secret_string import SecretString


def test_secret_string_redaction():
    secret = SecretString("my_super_secret")

    assert str(secret) == "***"
    assert repr(secret) == "SecretString('***')"
    assert secret.get_secret_value() == "my_super_secret"


def test_secret_string_truthiness():
    secret1 = SecretString("test")
    assert bool(secret1) is True

    secret2 = SecretString("")
    assert bool(secret2) is False


def test_secret_string_nesting():
    secret = SecretString("secret")
    nested = SecretString(secret)
    assert nested.get_secret_value() == "secret"


def test_secret_string_equality():
    secret1 = SecretString("abc")
    secret2 = SecretString("abc")
    assert secret1 == secret2
    assert secret1 == "abc"
    assert secret1 != "def"
