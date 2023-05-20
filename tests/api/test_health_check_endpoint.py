
from fastapi.testclient import TestClient
from pantanalise.main import app

def test_health_check():
    client = TestClient(app)
    response = client.get("/health-check")
    assert response.status_code == 200
    assert response.json() == {"message": "Deu bom"}
