import pandas as pd

from app.geo_utils import score_point, within_az_nv


def test_within_az_nv_bounds():
    assert within_az_nv(33.45, -110.8)
    assert not within_az_nv(45.0, -110.8)
    assert not within_az_nv(33.45, -125.0)


def test_score_point_interpolates_and_tiers():
    predictions = pd.DataFrame(
        {
            "lat": [33.4, 33.5, 33.4, 33.5],
            "lon": [-110.8, -110.8, -110.9, -110.9],
            "prospectivity_score": [0.8, 0.7, 0.6, 0.5],
            "elevation_m": [1200, 1250, 1100, 1300],
            "log_cu_ppm": [4.0, 3.7, 3.2, 3.0],
            "dist_fault_km": [2.0, 3.0, 5.0, 8.0],
            "dist_deposit_km": [10.0, 12.0, 20.0, 30.0],
        }
    )

    result = score_point(33.45, -110.85, predictions)

    assert 0 <= result["score"] <= 1
    assert result["risk_tier"] in {"Low", "Moderate", "High", "Very High"}
    assert "log_cu_ppm" in result["features"]
    assert result["nearest_grid_distance_km"] > 0
