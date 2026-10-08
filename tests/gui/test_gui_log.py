from tools.gui.gui_log import attach_session_log, diff_dicts, get_gui_logger, setup_gui_logging


def test_gui_log_writes_session_and_global(tmp_path, monkeypatch) -> None:
    import tools.gui.gui_log as gui_log

    monkeypatch.setattr(gui_log, "DEFAULT_SESSION_ROOT", tmp_path)
    log = get_gui_logger()
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    log = setup_gui_logging()
    session = tmp_path / "abc123"
    session.mkdir()
    attach_session_log(session)
    log.info("user changed width 1920 -> 1280")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("caught")
    for h in log.handlers:
        if hasattr(h, "flush"):
            h.flush()
    session_log = (session / "gui.log").read_text(encoding="utf-8")
    global_log = (tmp_path / "calibration_gui.log").read_text(encoding="utf-8")
    assert "user changed width" in session_log
    assert "RuntimeError: boom" in session_log
    assert "user changed width" in global_log


def test_diff_dicts() -> None:
    lines = diff_dicts({"width": "1920", "ae": True}, {"width": "1280", "ae": True})
    assert any("width" in line and "1920" in line and "1280" in line for line in lines)
    assert not any(line.startswith("ae:") for line in lines)
