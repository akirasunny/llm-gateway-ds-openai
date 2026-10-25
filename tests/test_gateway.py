import asyncio
import json
from unittest.mock import AsyncMock
import pytest
from fastapi.testclient import TestClient
from gateway.api import create_app
from gateway.core import Gateway, GatewayError, Model
from gateway.models import LLMRequest, Usage
from gateway.providers import Event, Result


class Fake:
    def __init__(self, failures=0, text='{"answer":42}', chunks=None):
        self.failures, self.text, self.chunks = failures, text, chunks
        self.calls = 0
        self.messages = None
        self.closed = False

    async def close(self):
        self.closed = True

    async def complete(self, model, messages, fmt, timeout):
        self.calls += 1
        self.messages = messages
        if self.calls <= self.failures:
            raise ConnectionError('secret upstream text')
        return Result(self.text, Usage(input_tokens=12, output_tokens=4, cached_input_tokens=3))

    async def stream(self, model, messages, timeout):
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError('secret upstream text')
        for item in self.chunks or [Event(delta='hello'), Event(usage=Usage(input_tokens=12, output_tokens=1))]:
            if isinstance(item, Exception):
                raise item
            yield item


def setup(primary=None, backup=None, rpm=60, backoff=0):
    primary, backup = primary or Fake(), backup or Fake()
    return Gateway({'primary': Model('deepseek-v4-flash', primary, rpm, 'backup'),
                    'backup': Model('gpt-6-astra', backup)}, backoff=backoff), primary, backup


def req(**kwargs):
    return LLMRequest(model='primary', messages=[{'role':'user', 'content':'secret prompt'}], **kwargs)


async def test_complete_prompt_usage():
    g,p,b = setup()
    response = await g.complete(req(prompt={'name':'knowledge_decision','version':'v1','variables':{'product_name':'Demo'}}))
    assert response.usage.cached_input_tokens == 3
    assert 'Demo' in p.messages[0].content
    assert b.calls == 0
    assert g.traces[-1]['attempts'][0]['status'] == 'success'
    assert 'latency_ms' in g.traces[-1]['attempts'][0]
    assert 'secret prompt' not in json.dumps(list(g.traces))


async def test_retry_fallback():
    g,p,b = setup(Fake(failures=4))
    response = await g.complete(req())
    assert p.calls == 4 and b.calls == 1 and response.attempts == 5
    assert response.model == 'backup'


async def test_eight_attempts_exhausted():
    g,p,b = setup(Fake(failures=99), Fake(failures=99))
    with pytest.raises(GatewayError, match='exhausted'):
        await g.complete(req())
    assert p.calls == b.calls == 4
    assert g.traces[-1]['status'] == 'failed'
    assert 'secret upstream' not in json.dumps(list(g.traces))


async def test_exponential_backoff(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr('gateway.services.gateway.asyncio.sleep', sleep)
    monkeypatch.setattr('gateway.core.retry.random.uniform', lambda a,b: 1)
    g,_,_ = setup(Fake(failures=3), backoff=.25)
    await g.complete(req())
    assert [call.args[0] for call in sleep.call_args_list] == [.25,.5,1]


@pytest.mark.parametrize('text', ['no json', '[]', '{"answer":"wrong"}'])
async def test_invalid_json_no_retry(text):
    g,p,b=setup(Fake(text=text))
    with pytest.raises(GatewayError) as error:
        await g.complete(req(response_format={'type':'json_schema','json_schema':{'schema':{'type':'object','properties':{'answer':{'type':'integer'}},'required':['answer']}}}))
    assert error.value.code == 'invalid_structured_output'
    assert p.calls == 1 and b.calls == 0
    assert g.traces[-1]['usage']['input_tokens'] == 12


async def test_json_success():
    g,_,_=setup()
    result=await g.complete(req(response_format={'type':'json_object'}))
    assert result.parsed == {'answer':42}


async def test_deadline():
    class Slow(Fake):
        async def complete(self,*args):
            await asyncio.sleep(1)
    g,_,_=setup(Slow())
    with pytest.raises(GatewayError) as error:
        await g.complete(req(deadline_seconds=.01))
    assert error.value.code == 'deadline_exceeded'
    assert g.traces[-1]['error_code'] == 'deadline_exceeded'


async def test_unknown_model_trace():
    g,_,_=setup()
    request=req()
    request.model='unknown'
    with pytest.raises(GatewayError):
        await g.complete(request)
    assert g.traces[-1]['error_code'] == 'unknown_model'


def test_http_stream_usage():
    g,_,_=setup()
    with TestClient(create_app(g)) as client:
        response=client.post('/v1/llm',json=req(stream=True).model_dump())
        assert response.status_code == 200
        events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        assert [e['type'] for e in events] == ['content.delta','response.completed']
        assert len({e['request_id'] for e in events}) == 1
        assert events[-1]['usage']['input_tokens'] == 12
        assert g.traces[-1]['first_token_latency_ms'] is not None
        assert g.traces[-1]['attempts'][0]['status'] == 'success'


def test_stream_fallback():
    g,p,b=setup(Fake(failures=4))
    with TestClient(create_app(g)) as client:
        response=client.post('/v1/llm',json=req(stream=True).model_dump())
        assert response.status_code == 200
        assert 'backup' in response.text
        assert p.calls == 4 and b.calls == 1


def test_stream_no_restart_after_chunk():
    g,p,b=setup(Fake(chunks=[Event(delta='partial'), ConnectionError('secret')]))
    with TestClient(create_app(g)) as client:
        response=client.post('/v1/llm',json=req(stream=True).model_dump())
        assert 'response.failed' in response.text
        assert 'secret' not in response.text
        assert p.calls == 1 and b.calls == 0
        assert g.traces[-1]['status'] == 'failed'


@pytest.mark.parametrize('stream', [False,True])
def test_model_limit_http_429(stream):
    g,p,b=setup(rpm=1)
    with TestClient(create_app(g)) as client:
        assert client.post('/v1/llm',json=req().model_dump()).status_code == 200
        response=client.post('/v1/llm',json=req(stream=stream).model_dump())
        assert response.status_code == 429
        assert int(response.headers['retry-after']) > 0
        assert b.calls == 0
        body=req().model_dump()
        body['model']='backup'
        assert client.post('/v1/llm',json=body).status_code == 200


def test_validation_sanitized():
    g,_,_=setup()
    with TestClient(create_app(g)) as client:
        response=client.post('/v1/llm',json={'model':'primary','messages':[{'role':'user','content':'secret'}],'extra':'secret'})
        assert response.status_code == 422
        assert 'secret' not in response.text
        assert g.traces[-1]['error_code'] == 'invalid_request'


def test_auth(monkeypatch):
    monkeypatch.setenv('GATEWAY_API_KEY','test-token')
    g,_,_=setup()
    with TestClient(create_app(g)) as client:
        assert client.get('/v1/traces').status_code == 401
        assert client.get('/v1/traces',headers={'Authorization':'Bearer test-token'}).status_code == 200


def test_stream_structured_rejected():
    with pytest.raises(ValueError):
        req(stream=True,response_format={'type':'json_object'})


async def test_trace_bounded():
    g,_,_=setup()
    from collections import deque
    g.recorder.traces=deque(maxlen=2)
    g.traces = g.recorder.traces
    for _ in range(3):
        await g.complete(req())
    assert len(g.traces)==2


async def test_disconnect_closes_upstream():
    class Stream(Fake):
        async def stream(self,*args):
            try:
                yield Event(delta='hello')
                await asyncio.sleep(10)
            finally:
                self.closed=True
    provider=Stream()
    g,_,_=setup(provider)
    _, iterator=await g.start_stream(req(stream=True))
    await iterator.aclose()
    assert provider.closed
    assert g.traces[-1]['error_code']=='cancelled'
    assert g.traces[-1]['attempts'][0]['status']=='failed'


def test_stream_deadline_after_chunk():
    class Stream(Fake):
        async def stream(self,*args):
            yield Event(delta='hello')
            await asyncio.sleep(1)
    g,_,b=setup(Stream())
    with TestClient(create_app(g)) as client:
        response=client.post('/v1/llm',json=req(stream=True,deadline_seconds=.02).model_dump())
        assert 'deadline_exceeded' in response.text
        assert b.calls==0
        assert g.traces[-1]['error_code']=='deadline_exceeded'


def test_stream_deadline_before_chunk():
    class Stream(Fake):
        async def stream(self,*args):
            await asyncio.sleep(1)
            yield Event(delta='hello')
    g,_,_=setup(Stream())
    with TestClient(create_app(g)) as client:
        response=client.post('/v1/llm',json=req(stream=True,deadline_seconds=.02).model_dump())
        assert response.status_code==504


async def test_local_limit_counts_retries():
    g,p,b=setup(Fake(failures=4),rpm=2)
    with pytest.raises(GatewayError) as error:
        await g.complete(req())
    assert error.value.status==429
    assert p.calls==2 and b.calls==0


def test_external_schema_references_rejected():
    with pytest.raises(ValueError, match='local JSON Schema'):
        req(response_format={'type':'json_schema','json_schema':{'schema':{'type':'object','properties':{'secret':{'$ref':'https://example.test/private'}}}}})
