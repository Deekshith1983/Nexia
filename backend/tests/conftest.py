import pytest

@pytest.fixture(autouse=True)
def cleanup_db_connections():
    yield
    try:
        import app.db.connection as db_conn
        if db_conn._engine is not None:
            db_conn._engine.sync_engine.dispose()
    except Exception:
        pass
