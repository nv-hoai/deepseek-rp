"""SSE parser fixtures: minimal real shapes from POST /chat/completion."""

from dsk.sse import SseParser


def test_ready_then_response_then_close():
    parser = SseParser()
    chunks = []
    for line in [
        "event: ready",
        'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}',
        "event: update_session",
        'data: {"updated_at":1790389851.3}',
        'data: {"v":{"response":{"message_id":2,"fragments":[{"id":2,"type":"RESPONSE","content":"4"}]}}}',
        'data: {"p":"response/status","o":"SET","v":"FINISHED"}',
        "event: title",
        'data: {"content":"Just 4"}',
        "event: close",
        'data: {"click_behavior":"none","auto_resume":false}',
    ]:
        chunks.extend(parser.feed(line))
    by_type = [c.type for c in chunks]
    assert by_type == ["ready", "text", "title", "finish"]
    assert chunks[1].content == "4"
    assert chunks[1].fragment_type == "RESPONSE"
    assert chunks[3].finish_reason == "stop"


def test_think_streams_via_inherited_append():
    parser = SseParser()
    lines = [
        'data: {"v":{"response":{"message_id":2,"fragments":[{"id":2,"type":"THINK","content":"H"}]}}}',
        'data: {"p":"response/fragments/-1/content","o":"APPEND","v":"mm"}',
        'data: {"v":"!"}',  # inherits APPEND to last fragment content
        'data: {"p":"response/fragments","o":"APPEND","v":[{"id":3,"type":"RESPONSE","content":"Hi"}]}',
        'data: {"p":"response/fragments/-1/content","v":" there"}',
        "event: finish",
        "data: {}",
    ]
    texts = []
    for line in lines:
        for chunk in parser.feed(line):
            texts.append((chunk.type, chunk.content))
    assert texts == [
        ("thinking", "H"),
        ("thinking", "mm"),
        ("thinking", "!"),
        ("text", "Hi"),
        ("text", " there"),
        ("finish", ""),
    ]


def test_batch_updates_emit_nothing():
    parser = SseParser()
    assert parser.feed(
        'data: {"p":"response","o":"BATCH",'
        '"v":[{"p":"accumulated_token_usage","v":47}]}') == []
