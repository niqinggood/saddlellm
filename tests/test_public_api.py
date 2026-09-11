import importlib.util
from types import ModuleType

import saddlellm


def test_public_api_is_derived_from_lazy_registry():
    expected = ["__version__", *saddlellm._LAZY_IMPORT_MAP]

    assert saddlellm.__all__ == expected
    assert len(saddlellm.__all__) == len(set(saddlellm.__all__))
    assert set(saddlellm.__all__) <= set(dir(saddlellm))


def test_lazy_registry_only_references_packaged_modules():
    for public_name, (
        module_path,
        attribute_name,
    ) in saddlellm._LAZY_IMPORT_MAP.items():
        assert module_path.startswith("."), public_name
        assert importlib.util.find_spec(f"saddlellm{module_path}") is not None, (
            public_name
        )
        assert attribute_name is None or attribute_name.strip(), public_name


def test_easy_module_uses_the_same_lazy_resolution_path():
    assert isinstance(saddlellm.easy, ModuleType)
    assert saddlellm.easy.__name__ == "saddlellm.easy"


def test_compression_legacy_imports_forward_to_canonical_classes():
    from saddlellm.Lightweight import Lightweight as LegacyLightweight
    from saddlellm.ModelPruner import ModelPruner as LegacyModelPruner
    from saddlellm.ModelQuantizer import ModelQuantizer as LegacyModelQuantizer
    from saddlellm.compression import Lightweight, ModelPruner, ModelQuantizer

    assert LegacyLightweight is Lightweight
    assert LegacyModelPruner is ModelPruner
    assert LegacyModelQuantizer is ModelQuantizer


def test_data_pipeline_legacy_import_forwards_to_canonical_class():
    from saddlellm.DataPipeline import DataPipeline as LegacyDataPipeline
    from saddlellm.data import DataPipeline

    assert LegacyDataPipeline is DataPipeline
