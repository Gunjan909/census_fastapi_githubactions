from fastapi.testclient import TestClient
from main import app  # make sure your FastAPI app is in main.py

client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "welcome" in response.json()["message"].lower()


def test_post_predict_low_income():
    payload = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "11th",
        "education-num": 7,
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in ['>50K', '<=50K']


def test_post_predict_high_income():
    payload = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in ['>50K', '<=50K']

