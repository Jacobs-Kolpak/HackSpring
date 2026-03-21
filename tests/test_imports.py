import importlib
import pkgutil

import pytest


CORE_MODULES = [
    "backend.core.config",
    "backend.core.database",
    "backend.core.security",
]

UTIL_MODULES = [
    "backend.utils.llm",
    "backend.utils.document_reader",
    "backend.utils.embeddings",
    "backend.utils.chunker",
    "backend.utils.web_parser",
]

SERVICE_MODULES = [
    "backend.services.rag.service",
    "backend.services.content.summary",
    "backend.services.content.flashcards",
    "backend.services.content.mindmap",
    "backend.services.content.table",
    "backend.services.media.podcast",
    "backend.services.media.presentation",
    "backend.services.media.infographic",
    "backend.services.web.parser",
]

ROUTER_MODULES = [
    "backend.routers.auth.routes",
    "backend.routers.rag.routes",
    "backend.routers.content.summary",
    "backend.routers.content.flashcards",
    "backend.routers.content.mindmap",
    "backend.routers.content.table",
    "backend.routers.media.podcast",
    "backend.routers.media.presentation",
    "backend.routers.media.infographics",
    "backend.routers.web.parser",
]


@pytest.mark.parametrize("module_path", CORE_MODULES)
def test_core_imports(module_path):
    mod = importlib.import_module(module_path)
    assert mod is not None


@pytest.mark.parametrize("module_path", UTIL_MODULES)
def test_util_imports(module_path):
    mod = importlib.import_module(module_path)
    assert mod is not None


@pytest.mark.parametrize("module_path", SERVICE_MODULES)
def test_service_imports(module_path):
    mod = importlib.import_module(module_path)
    assert mod is not None


@pytest.mark.parametrize("module_path", ROUTER_MODULES)
def test_router_imports(module_path):
    mod = importlib.import_module(module_path)
    assert mod is not None
