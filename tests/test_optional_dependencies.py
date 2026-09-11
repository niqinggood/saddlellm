from importlib import metadata

import pytest

import saddlellm.utils.OptionalDependencies as optional_dependencies
from saddlellm.utils.OptionalDependencies import OptionalDependencyError, require_distribution


def test_require_distribution_returns_installed_version(monkeypatch):
    monkeypatch.setattr(optional_dependencies.metadata, "version", lambda name: f"{name}-version")

    assert require_distribution("example", extra="train", capability="training") == (
        "example-version"
    )


def test_require_distribution_names_the_missing_extra(monkeypatch):
    def missing(_name):
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(optional_dependencies.metadata, "version", missing)

    with pytest.raises(
        OptionalDependencyError,
        match=r"QLoRA requires bitsandbytes.*saddlellm\[posttrain,qlora\]",
    ):
        require_distribution(
            "bitsandbytes", extra="posttrain,qlora", capability="QLoRA"
        )
