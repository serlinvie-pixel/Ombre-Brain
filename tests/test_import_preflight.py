import asyncio
import json
import threading

import pytest

import web.import_api as import_api


class FakeMCP:
    def __init__(self):
        self.routes = {}

    def custom_route(self, path, methods):
        def decorator(fn):
            for method in methods:
                self.routes[(method, path)] = fn
            return fn

        return decorator


class BodyRequest:
    def __init__(self, body: str, filename: str = "upload.md"):
        self.headers = {}
        self.query_params = {"filename": filename}
        self._body = body.encode("utf-8")

    async def body(self):
        return self._body


class FakeDehydrator:
    api_available = True


class FakeImportEngine:
    is_running = False
    dehydrator = FakeDehydrator()


class QueryRequest:
    def __init__(self, **query_params):
        self.query_params = query_params


class FakeBucketManager:
    def __init__(self, buckets):
        self.buckets = buckets

    async def list_all(self, include_archive=False):
        assert include_archive is False
        return list(self.buckets)


def test_preview_import_counts_turns_chunks_and_estimated_calls():
    from import_memory import preview_import

    raw = "Human: 我喜欢茶\nAssistant: 我记住了\nUser: 明天提醒我整理导入体验"

    preview = preview_import(raw, filename="chat.md", human_label="阿立")

    assert preview["ok"] is True
    assert preview["detected_format"] == "markdown"
    assert preview["turns_count"] == 3
    assert preview["chunks_count"] == 1
    assert preview["estimated_api_calls"] == 1
    assert "[阿立]" in preview["first_chunk_preview"]


def test_preview_structured_memory_json_requires_no_llm():
    from import_memory import preview_import

    raw = json.dumps([
        {
            "name": "茶",
            "content": "用户喜欢乌龙茶。",
            "domain": ["饮食"],
            "valence": 0.7,
            "arousal": 0.2,
            "tags": ["茶"],
            "importance": 6,
        },
        {
            "name": "计划",
            "content": "用户计划周末整理书架。",
            "domain": ["计划"],
            "valence": 0.5,
            "arousal": 0.3,
            "tags": [],
            "importance": 5,
        },
    ], ensure_ascii=False)

    preview = preview_import(raw, filename="memories.json")

    assert preview["ok"] is True
    assert preview["detected_format"] == "structured_memory_json"
    assert preview["turns_count"] == 2
    assert preview["chunks_count"] == 2
    assert preview["estimated_api_calls"] == 0
    assert preview["requires_llm"] is False


def test_marshmallow_export_preserves_sender_roles_and_names():
    from import_memory import detect_and_parse, preview_import

    raw = json.dumps({
        "kind": "marshmallow-chat-export",
        "version": 1,
        "exportedAt": "2026-11-16T11:43:32+08:00",
        "conversation": {"type": "private", "name": "江玥影"},
        "messages": [
            {
                "id": "user-1",
                "senderId": "user",
                "senderName": "阿淸",
                "type": "text",
                "content": "这是用户说的话。",
                "timestamp": "2026-09-20T13:04:00+08:00",
            },
            {
                "id": "char-1",
                "senderId": "char_jiang",
                "senderName": "江玥影",
                "type": "text",
                "content": "这是角色说的话。",
                "timestamp": "2026-09-20T13:04:01+08:00",
            },
            {
                "id": "narrator-1",
                "senderId": "narrator",
                "senderName": "旁白",
                "type": "text",
                "content": "场景旁白。",
                "timestamp": "2026-09-20T13:04:02+08:00",
            },
            {
                "id": "system-1",
                "senderId": "system",
                "senderName": "系统",
                "type": "system",
                "content": "系统事件。",
                "timestamp": "2026-09-20T13:04:03+08:00",
            },
        ],
    }, ensure_ascii=False)

    turns = detect_and_parse(raw, filename="marshmallow.json")
    assert [turn["role"] for turn in turns] == [
        "user", "assistant", "narrator", "system"
    ]
    assert [turn["speaker"] for turn in turns] == [
        "阿淸", "江玥影", "旁白", "系统"
    ]
    assert turns[1]["timestamp"] == "2026-09-20T13:04:01+08:00"
    assert turns[1]["message_id"] == "char-1"

    preview = preview_import(raw, filename="marshmallow.json", human_label="阿淸")
    assert preview["detected_format"] == "marshmallow_json"
    assert preview["turns_count"] == 4
    assert "[阿淸]" in preview["first_chunk_preview"]
    assert "[江玥影]" in preview["first_chunk_preview"]
    assert "[旁白]" in preview["first_chunk_preview"]
    assert "[系统]" in preview["first_chunk_preview"]


def test_marshmallow_export_uses_inline_text_when_message_content_is_empty():
    from import_memory import detect_and_parse

    raw = json.dumps({
        "kind": "marshmallow-chat-export",
        "conversation": {"name": "江玥影"},
        "messages": [{
            "id": "sticker-1",
            "senderId": "char_jiang",
            "senderName": "江玥影",
            "type": "sticker",
            "content": "",
            "metadata": {"inlineText": "[发送贴纸：晚安]"},
            "timestamp": "2026-09-20T13:05:00+08:00",
        }],
    }, ensure_ascii=False)

    turns = detect_and_parse(raw, filename="marshmallow.json")
    assert len(turns) == 1
    assert turns[0]["role"] == "assistant"
    assert turns[0]["content"] == "[发送贴纸：晚安]"


def test_preview_claude_conversations_uses_claude_parser():
    from import_memory import preview_import

    raw = json.dumps([
        {
            "uuid": "conversation-1",
            "name": "茶话题",
            "chat_messages": [
                {
                    "sender": "human",
                    "text": "我喜欢乌龙茶。",
                    "created_at": "2024-01-01T00:00:00Z",
                },
                {
                    "sender": "assistant",
                    "text": "我记住了。",
                    "created_at": "2024-01-01T00:00:01Z",
                },
            ],
        }
    ], ensure_ascii=False)

    preview = preview_import(raw, filename="conversations.json")

    assert preview["ok"] is True
    assert preview["detected_format"] == "claude_json"
    assert preview["turns_count"] == 2
    assert preview["sample_turns"][0]["role"] == "human"


def test_preview_structured_memory_json_reports_exact_invalid_item():
    from import_memory import preview_import

    raw = json.dumps([
        {"name": "正常", "content": "可导入", "importance": 6},
        {"name": "缺正文", "importance": 5},
    ], ensure_ascii=False)

    preview = preview_import(raw, filename="memories.json")

    assert preview["ok"] is False
    assert preview["requires_llm"] is False
    assert preview["estimated_api_calls"] == 0
    assert "第 2 项" in preview["error"]
    assert "content" in preview["error"]


def test_preview_import_warns_when_invalid_json_falls_back_to_text():
    from import_memory import preview_import

    preview = preview_import("{not json", filename="bad.json")

    assert preview["ok"] is True
    assert preview["detected_format"] == "text"
    assert preview["turns_count"] == 1
    assert any("JSON" in warning for warning in preview["warnings"])


def test_preview_import_handles_deep_json_without_internal_error():
    from import_memory import preview_import

    raw = "[" * 100_000 + "0" + "]" * 100_000
    preview = preview_import(raw, filename="deep.json")

    assert preview["ok"] is True
    assert preview["detected_format"] == "text"
    assert any("JSON" in warning for warning in preview["warnings"])


def test_preview_import_handles_out_of_range_chatgpt_timestamp():
    from import_memory import preview_import

    raw = json.dumps({
        "mapping": {
            "node": {
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["仍应正常预检"]},
                    "create_time": 1e100,
                }
            }
        }
    })
    preview = preview_import(raw, filename="chatgpt.json")

    assert preview["ok"] is True
    assert preview["turns_count"] == 1
    assert preview["sample_turns"][0]["timestamp"] == ""


@pytest.mark.asyncio
async def test_import_preflight_route_returns_preview_with_runtime_readiness(monkeypatch):
    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", FakeImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {"human": "阿立"})

    mcp = FakeMCP()
    import_api.register(mcp)

    response = await mcp.routes[("POST", "/api/import/preflight")](
        BodyRequest("Human: hi\nAssistant: hello", filename="chat.md")
    )
    payload = json.loads(response.body)

    assert payload["ok"] is True
    assert payload["can_start"] is True
    assert payload["llm_ready"] is True
    assert payload["import_running"] is False
    assert payload["filename"] == "chat.md"
    assert payload["turns_count"] == 2
    assert payload["chunks_count"] == 1


@pytest.mark.asyncio
async def test_structured_preflight_can_start_when_llm_is_unavailable(monkeypatch):
    class OfflineDehydrator:
        api_available = False

    class OfflineImportEngine:
        is_running = False
        dehydrator = OfflineDehydrator()

    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", OfflineImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {"human": "阿立"})
    mcp = FakeMCP()
    import_api.register(mcp)
    raw = json.dumps([
        {"name": "来源明确", "content": "人工整理的记忆", "importance": 6}
    ], ensure_ascii=False)

    response = await mcp.routes[("POST", "/api/import/preflight")](
        BodyRequest(raw, filename="memories.json")
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["llm_ready"] is False
    assert payload["requires_llm"] is False
    assert payload["can_start"] is True


@pytest.mark.asyncio
async def test_import_preflight_route_hides_preview_worker_failure(monkeypatch):
    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", FakeImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {"human": "阿立"})

    def reject_preview(*_args, **_kwargs):
        raise RecursionError("malicious nesting detail")

    monkeypatch.setattr(import_api, "preview_import", reject_preview)
    mcp = FakeMCP()
    import_api.register(mcp)

    response = await mcp.routes[("POST", "/api/import/preflight")](
        BodyRequest("{}", filename="bad.json")
    )
    payload = json.loads(response.body)

    assert response.status_code == 400
    assert payload == {
        "ok": False,
        "error": "Import preview could not parse file",
    }
    assert "malicious nesting detail" not in response.body.decode("utf-8")


@pytest.mark.asyncio
async def test_preflight_is_off_loop_and_rejects_parallel_body_before_read(monkeypatch):
    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", FakeImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {"human": "阿立"})
    entered = threading.Event()
    release = threading.Event()

    def blocking_preview(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=2)
        return {"ok": True, "turns_count": 1, "chunks_count": 1}

    monkeypatch.setattr(import_api, "preview_import", blocking_preview)
    mcp = FakeMCP()
    import_api.register(mcp)
    preflight = mcp.routes[("POST", "/api/import/preflight")]

    class MustNotReadRequest(BodyRequest):
        async def body(self):
            raise AssertionError("parallel preflight must be rejected before body read")

    first = asyncio.create_task(preflight(BodyRequest("Human: one")))
    while not entered.is_set():
        await asyncio.sleep(0)

    second = await preflight(MustNotReadRequest("Human: two"))
    assert second.status_code == 409
    assert "active" in json.loads(second.body)["error"].lower()

    release.set()
    response = await asyncio.wait_for(first, timeout=2)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_import_results_filters_imported_buckets_and_paginates(monkeypatch):
    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", FakeImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {})
    monkeypatch.setattr(import_api.sh, "bucket_mgr", FakeBucketManager([
        {
            "id": "ordinary-newest",
            "content": "not imported",
            "metadata": {"name": "ordinary", "created": "2026-07-26T12:00:00"},
        },
        {
            "id": "import-new",
            "content": "new imported body",
            "metadata": {
                "name": "new import",
                "created": "2026-07-26T11:00:00",
                "imported": True,
            },
        },
        {
            "id": "import-legacy-source",
            "content": "legacy source marker",
            "metadata": {
                "name": "legacy import",
                "created": "2026-07-26T10:00:00",
                "source_tool": "import",
            },
        },
    ]))

    mcp = FakeMCP()
    import_api.register(mcp)
    route = mcp.routes[("GET", "/api/import/results")]

    first = json.loads((await route(QueryRequest(limit="1", offset="0"))).body)
    second = json.loads((await route(QueryRequest(limit="1", offset="1"))).body)

    assert first["total"] == 2
    assert first["has_more"] is True
    assert first["next_offset"] == 1
    assert [bucket["id"] for bucket in first["buckets"]] == ["import-new"]
    assert first["buckets"][0]["imported"] is True
    assert second["has_more"] is False
    assert second["next_offset"] is None
    assert [bucket["id"] for bucket in second["buckets"]] == [
        "import-legacy-source"
    ]


@pytest.mark.asyncio
async def test_import_results_rejects_negative_or_non_integer_offset(monkeypatch):
    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", FakeImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {})
    monkeypatch.setattr(import_api.sh, "bucket_mgr", FakeBucketManager([]))
    mcp = FakeMCP()
    import_api.register(mcp)
    route = mcp.routes[("GET", "/api/import/results")]

    for offset in ("-1", "nan"):
        response = await route(QueryRequest(offset=offset))
        assert response.status_code == 400


def test_marshmallow_export_preserves_special_speakers_and_message_types():
    from import_memory import detect_and_parse

    raw = json.dumps({
        "kind": "marshmallow-chat-records",
        "conversation": {"name": "江玥影"},
        "messages": [
            {
                "id": "narrator-1", "senderId": "system", "senderName": "旁白",
                "type": "system", "content": "场景旁白。", "timestamp": 1789880890819,
            },
            {
                "id": "guidance-1", "senderId": "guidance", "senderName": "本体",
                "type": "text", "content": "这是对角色逻辑的分析。", "timestamp": 1789880890820,
            },
            {
                "id": "action-1", "senderId": "system", "senderName": "聊天动作",
                "type": "chatAction", "content": "[聊天动作] 江玥影挥手",
                "metadata": {"actionText": "江玥影挥手"}, "timestamp": 1789880890821,
            },
            {
                "id": "voice-1", "senderId": "user", "senderName": "阿淸",
                "type": "voice", "content": "[语音消息]",
                "metadata": {"text": "语音转写正文"}, "timestamp": 1789880890822,
            },
            {
                "id": "image-1", "senderId": "user", "senderName": "阿淸",
                "type": "image", "content": "data:image/jpeg;base64,SHOULD_NOT_LEAK",
                "metadata": {"summary": "图片中的场景描述"}, "timestamp": 1789880890823,
            },
            {
                "id": "bundle-1", "senderId": "user", "senderName": "阿淸",
                "type": "chatBundle", "content": "[合并转发] 旧聊天",
                "metadata": {
                    "bundleTitle": "旧聊天",
                    "fromChatId": "chat_other",
                    "items": [
                        {
                            "senderId": "user", "senderName": "阿淸",
                            "type": "text", "content": "引用用户内容",
                        },
                        {
                            "senderId": "char_other", "senderName": "另一角色",
                            "type": "text", "content": "引用角色内容",
                        },
                    ],
                },
                "timestamp": 1789880890824,
            },
            {
                "id": "call-1", "senderId": "user", "senderName": "阿淸",
                "type": "voiceCall", "content": "视频通话",
                "metadata": {"transcript": "江玥影：通话转写内容"},
                "timestamp": 1789880890825,
            },
            {
                "id": "recalled-1", "senderId": "char_jiang", "senderName": "江玥影",
                "type": "text", "content": "这条消息后来撤回了。",
                "timestamp": 1789880890826, "recalled": True,
            },
        ],
    }, ensure_ascii=False)

    turns = detect_and_parse(raw, filename="marshmallow.json")

    assert [turn["role"] for turn in turns] == [
        "narrator", "guidance", "action", "user", "user", "user", "user", "assistant"
    ]
    assert turns[0]["speaker"] == "旁白"
    assert turns[1]["speaker"] == "本体"
    assert turns[2]["content"] == "江玥影挥手"
    assert turns[3]["content"] == "语音转写正文"
    assert turns[4]["content"] == "[图片描述：图片中的场景描述]"
    assert "SHOULD_NOT_LEAK" not in turns[4]["content"]
    assert "另一角色：引用角色内容" in turns[5]["content"]
    assert "来源会话：chat_other" in turns[5]["content"]
    assert "通话转写内容" in turns[6]["content"]
    assert turns[7]["content"].startswith("[已撤回消息]")
    assert turns[3]["message_id"] == "voice-1"
    assert turns[3]["timestamp"] == 1789880890822
    assert turns[3]["message_type"] == "voice"


def test_chunk_turns_retains_marshmallow_source_message_metadata():
    from import_memory import chunk_turns

    turns = [
        {
            "role": "user", "speaker": "阿淸", "content": "第一句",
            "timestamp": 1789880890819, "message_id": "msg-user",
            "message_type": "text", "sender_id": "user",
        },
        {
            "role": "assistant", "speaker": "江玥影", "content": "第二句",
            "timestamp": 1789880890820, "message_id": "msg-char",
            "message_type": "voice", "sender_id": "char_jiang",
        },
    ]

    chunks = chunk_turns(turns, target_tokens=100, human_label="阿淸")

    assert len(chunks) == 1
    assert chunks[0]["timestamp_start"] == 1789880890819
    assert chunks[0]["timestamp_end"] == 1789880890820
    assert [item["message_id"] for item in chunks[0]["source_messages"]] == [
        "msg-user", "msg-char"
    ]
    assert chunks[0]["source_messages"][1]["message_type"] == "voice"
