from app.llm_interpreter import GeoInterpreter


def test_fallback_interpretation_is_conservative():
    text = GeoInterpreter(api_key="").interpret_score(
        lat=33.45,
        lon=-110.8,
        score=0.72,
        features={
            "log_cu_ppm": 3.8,
            "dist_fault_km": 2.0,
            "dist_deposit_km": 12.0,
        },
    )

    assert "Very High" in text
    assert "Recommendation" in text
    assert "resource estimate" not in text.lower()
