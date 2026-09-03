from ml_system.config.loader import load_settings


def test_load_settings():
    settings = load_settings("config/config.yaml")

    assert settings.data.target_column == "Churn"
    assert settings.split.test_size == 0.20
    assert settings.model.name == "gradient_boosting"