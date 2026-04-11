import re
from src.index import generate_session_id


def test_generate_session_id_pattern():
    session_id = generate_session_id()
    assert session_id.startswith("session_")
    assert re.match(r"^session_\d{8}_\d{6}_[0-9a-f]{8}$", session_id)
