import json
import httpx
from openai import AsyncOpenAI
from gateway.models import Message, ResponseFormat
from gateway.providers import ChatCompletionsProvider, ResponsesProvider


async def test_chat_wire_format():
    captured=[]
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200,json={'id':'c','object':'chat.completion','created':0,'model':'deepseek-v4-flash','choices':[{'index':0,'message':{'role':'assistant','content':'{"ok":true}'},'finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12,'prompt_cache_hit_tokens':4}})
    p=ChatCompletionsProvider('test','https://example.test')
    await p.client.close()
    p.client=AsyncOpenAI(api_key='test',base_url='https://example.test',max_retries=0,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result=await p.complete('deepseek-v4-flash',[Message(role='user',content='hello')],ResponseFormat(type='json_object'),30)
    assert captured[0]['response_format']=={'type':'json_object'}
    assert result.usage.cached_input_tokens==4
    await p.close()


async def test_responses_wire_format():
    captured=[]
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200,json={'id':'r','object':'response','created_at':0,'model':'gpt-6-astra','status':'completed','output':[{'type':'message','id':'m','role':'assistant','status':'completed','content':[{'type':'output_text','text':'{"ok":true}','annotations':[]}]}],'usage':{'input_tokens':10,'output_tokens':2,'total_tokens':12,'input_tokens_details':{'cached_tokens':3},'output_tokens_details':{'reasoning_tokens':1}}})
    p=ResponsesProvider('test','https://example.test')
    await p.client.close()
    p.client=AsyncOpenAI(api_key='test',base_url='https://example.test',max_retries=0,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    fmt=ResponseFormat.model_validate({'type':'json_schema','json_schema':{'schema':{'type':'object','properties':{},'additionalProperties':False}}})
    result=await p.complete('gpt-6-astra',[Message(role='user',content='hello')],fmt,30)
    assert captured[0]['text']['format']['type']=='json_schema'
    assert captured[0]['store'] is False
    assert 'messages' not in captured[0]
    assert result.content=='{"ok":true}'
    assert result.usage.reasoning_output_tokens==1
    await p.close()


async def test_chat_stream_wire():
    events=[{'id':'c','object':'chat.completion.chunk','created':0,'model':'deepseek-v4-flash','choices':[{'index':0,'delta':{'content':'Hi'},'finish_reason':None}]},
            {'id':'c','object':'chat.completion.chunk','created':0,'model':'deepseek-v4-flash','choices':[{'index':0,'delta':{},'finish_reason':'stop'}]},
            {'id':'c','object':'chat.completion.chunk','created':0,'model':'deepseek-v4-flash','choices':[],'usage':{'prompt_tokens':2,'completion_tokens':1,'total_tokens':3}}]
    def handler(request):
        assert json.loads(request.content)['stream_options']['include_usage']
        return httpx.Response(200,headers={'content-type':'text/event-stream'},text=''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n')
    p=ChatCompletionsProvider('test','https://example.test')
    await p.client.close()
    p.client=AsyncOpenAI(api_key='test',base_url='https://example.test',max_retries=0,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    output=[e async for e in p.stream('deepseek-v4-flash',[Message(role='user',content='hello')],30)]
    assert output[0].delta=='Hi' and output[-1].usage.input_tokens==2
    await p.close()


async def test_responses_stream_wire():
    events=[{'type':'response.output_text.delta','delta':'Hi','item_id':'m','output_index':0,'content_index':0,'sequence_number':1},
            {'type':'response.completed','sequence_number':2,'response':{'id':'r','object':'response','created_at':0,'model':'gpt-6-astra','status':'completed','output':[],'usage':{'input_tokens':2,'output_tokens':1,'total_tokens':3,'input_tokens_details':{'cached_tokens':0},'output_tokens_details':{'reasoning_tokens':0}}}}]
    def handler(request):
        return httpx.Response(200,headers={'content-type':'text/event-stream'},text=''.join('event: '+e['type']+'\ndata: '+json.dumps(e)+'\n\n' for e in events))
    p=ResponsesProvider('test','https://example.test')
    await p.client.close()
    p.client=AsyncOpenAI(api_key='test',base_url='https://example.test',max_retries=0,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    output=[e async for e in p.stream('gpt-6-astra',[Message(role='user',content='hello')],30)]
    assert output[0].delta=='Hi' and output[-1].usage.output_tokens==1
    await p.close()
