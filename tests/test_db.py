from __future__ import annotations

from pdm.serving.db import AuditRecord, DriftRecord, PredictionRecord, get_session_factory, init_db


def test_in_memory_sqlite_keeps_tables_across_sessions() -> None:
    """Regression test for a real bug: without StaticPool, each new
    connection to sqlite:///:memory: gets a fresh, empty database, so the
    tables init_db() just created vanish before the next query runs."""
    engine = init_db("sqlite:///:memory:")
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        session.add(PredictionRecord(
            model_name="test", predicted_class="Classe A", probabilities={"Classe A": 1.0},
            conformal_set=["Classe A"], sensor_names=["Dados_1"], features={"Dados_1_rms": 0.1},
        ))
        session.commit()

    # A brand new session (and, without StaticPool, a brand new connection).
    with session_factory() as session:
        rows = session.query(PredictionRecord).all()
    assert len(rows) == 1
    assert rows[0].predicted_class == "Classe A"


def test_all_three_tables_are_created(tmp_path) -> None:
    engine = init_db(f"sqlite:///{tmp_path / 'test.db'}")
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        session.add(AuditRecord(report={"ok": True}))
        session.add(DriftRecord(n_alarms=0, n_reference=100, n_current=50, report=[]))
        session.commit()

    with session_factory() as session:
        assert session.query(AuditRecord).count() == 1
        assert session.query(DriftRecord).count() == 1
