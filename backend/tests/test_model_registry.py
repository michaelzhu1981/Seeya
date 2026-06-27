from app.model_registry import ModelRegistry


def test_registry_has_available_recommended_model() -> None:
    registry = ModelRegistry()
    models = registry.list_models()

    assert models
    assert registry.selected_model().available


def test_rejects_unavailable_model() -> None:
    registry = ModelRegistry()
    unavailable = next((model for model in registry.list_models() if not model.available), None)
    if unavailable is None:
        return

    try:
        registry.select_model(unavailable.id)
    except ValueError as exc:
        assert str(exc)
    else:
        raise AssertionError("Unavailable model should be rejected")
