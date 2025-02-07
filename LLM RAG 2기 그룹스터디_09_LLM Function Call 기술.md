# LangChain의 Function Calling
LangChain의 Function Calling은 LLM(Large Language Model)이 외부 API 또는 자체적인 도구(Functions)를 호출하여 더욱 강력한 기능을 수행할 수 있도록 하는 개념입니다. 이를 통해 모델은 단순한 자연어 생성뿐만 아니라, 데이터 조회, 계산, 외부 서비스 호출 등의 다양한 기능을 실행할 수 있습니다.


## 📌 개요
- Function Calling이란?
  - LLM이 특정 작업을 수행하기 위해 함수를 호출하는 기능
  - OpenAI의 GPT-4, GPT-3.5에서 도입된 “Function Calling API”와 유사한 개념
  - 자연어 입력을 기반으로 적절한 함수를 선택하고, 필요한 매개변수를 생성하여 실행
- LangChain에서의 역할
  - LLM이 내부적으로 정리된 함수를 실행하여 정확하고 실행 가능한 응답을 생성
  - LLM의 범위를 확장하여 외부 API, 데이터베이스, 계산 도구 등과 연동 가능
- 주요 활용 사례
  - 날씨 조회: “오늘 서울의 날씨 어때?” → 날씨 API 호출 후 응답
  - 데이터 조회: “이메일 받은 편지함에서 최신 이메일 3개 보여줘” → 이메일 API 호출
  - 수학 계산: “243 * 17의 결과는?” → 계산기 함수 호출
  - 기타 API 연동: 검색 엔진 호출, DB 조회, 주식 정보 가져오기 등

# 환경 설정

## 패키지 설치
- langchain
- langchain-openai
- openai
- tiktoken


```python
!pip install langchain langchain-openai openai tiktoken
```

    Requirement already satisfied: langchain in /usr/local/lib/python3.11/dist-packages (0.3.16)
    Collecting langchain-openai
      Downloading langchain_openai-0.3.3-py3-none-any.whl.metadata (2.7 kB)
    Requirement already satisfied: openai in /usr/local/lib/python3.11/dist-packages (1.59.9)
    Collecting tiktoken
      Downloading tiktoken-0.8.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)
    Requirement already satisfied: PyYAML>=5.3 in /usr/local/lib/python3.11/dist-packages (from langchain) (6.0.2)
    Requirement already satisfied: SQLAlchemy<3,>=1.4 in /usr/local/lib/python3.11/dist-packages (from langchain) (2.0.37)
    Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /usr/local/lib/python3.11/dist-packages (from langchain) (3.11.11)
    Requirement already satisfied: langchain-core<0.4.0,>=0.3.32 in /usr/local/lib/python3.11/dist-packages (from langchain) (0.3.32)
    Requirement already satisfied: langchain-text-splitters<0.4.0,>=0.3.3 in /usr/local/lib/python3.11/dist-packages (from langchain) (0.3.5)
    Requirement already satisfied: langsmith<0.4,>=0.1.17 in /usr/local/lib/python3.11/dist-packages (from langchain) (0.3.2)
    Requirement already satisfied: numpy<2,>=1.22.4 in /usr/local/lib/python3.11/dist-packages (from langchain) (1.26.4)
    Requirement already satisfied: pydantic<3.0.0,>=2.7.4 in /usr/local/lib/python3.11/dist-packages (from langchain) (2.10.6)
    Requirement already satisfied: requests<3,>=2 in /usr/local/lib/python3.11/dist-packages (from langchain) (2.32.3)
    Requirement already satisfied: tenacity!=8.4.0,<10,>=8.1.0 in /usr/local/lib/python3.11/dist-packages (from langchain) (9.0.0)
    Collecting langchain-core<0.4.0,>=0.3.32 (from langchain)
      Downloading langchain_core-0.3.33-py3-none-any.whl.metadata (6.3 kB)
    Requirement already satisfied: anyio<5,>=3.5.0 in /usr/local/lib/python3.11/dist-packages (from openai) (3.7.1)
    Requirement already satisfied: distro<2,>=1.7.0 in /usr/local/lib/python3.11/dist-packages (from openai) (1.9.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.11/dist-packages (from openai) (0.28.1)
    Requirement already satisfied: jiter<1,>=0.4.0 in /usr/local/lib/python3.11/dist-packages (from openai) (0.8.2)
    Requirement already satisfied: sniffio in /usr/local/lib/python3.11/dist-packages (from openai) (1.3.1)
    Requirement already satisfied: tqdm>4 in /usr/local/lib/python3.11/dist-packages (from openai) (4.67.1)
    Requirement already satisfied: typing-extensions<5,>=4.11 in /usr/local/lib/python3.11/dist-packages (from openai) (4.12.2)
    Requirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.11/dist-packages (from tiktoken) (2024.11.6)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (2.4.4)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.11/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.3.2)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (25.1.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.5.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.11/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (6.1.0)
    Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (0.2.1)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.18.3)
    Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.11/dist-packages (from anyio<5,>=3.5.0->openai) (3.10)
    Requirement already satisfied: certifi in /usr/local/lib/python3.11/dist-packages (from httpx<1,>=0.23.0->openai) (2024.12.14)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.11/dist-packages (from httpx<1,>=0.23.0->openai) (1.0.7)
    Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.11/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)
    Requirement already satisfied: jsonpatch<2.0,>=1.33 in /usr/local/lib/python3.11/dist-packages (from langchain-core<0.4.0,>=0.3.32->langchain) (1.33)
    Requirement already satisfied: packaging<25,>=23.2 in /usr/local/lib/python3.11/dist-packages (from langchain-core<0.4.0,>=0.3.32->langchain) (24.2)
    Requirement already satisfied: orjson<4.0.0,>=3.9.14 in /usr/local/lib/python3.11/dist-packages (from langsmith<0.4,>=0.1.17->langchain) (3.10.15)
    Requirement already satisfied: requests-toolbelt<2.0.0,>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from langsmith<0.4,>=0.1.17->langchain) (1.0.0)
    Requirement already satisfied: zstandard<0.24.0,>=0.23.0 in /usr/local/lib/python3.11/dist-packages (from langsmith<0.4,>=0.1.17->langchain) (0.23.0)
    Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.11/dist-packages (from pydantic<3.0.0,>=2.7.4->langchain) (0.7.0)
    Requirement already satisfied: pydantic-core==2.27.2 in /usr/local/lib/python3.11/dist-packages (from pydantic<3.0.0,>=2.7.4->langchain) (2.27.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2->langchain) (3.4.1)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2->langchain) (2.3.0)
    Requirement already satisfied: greenlet!=0.4.17 in /usr/local/lib/python3.11/dist-packages (from SQLAlchemy<3,>=1.4->langchain) (3.1.1)
    Requirement already satisfied: jsonpointer>=1.9 in /usr/local/lib/python3.11/dist-packages (from jsonpatch<2.0,>=1.33->langchain-core<0.4.0,>=0.3.32->langchain) (3.0.0)
    Downloading langchain_openai-0.3.3-py3-none-any.whl (54 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m54.5/54.5 kB[0m [31m2.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading tiktoken-0.8.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m33.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading langchain_core-0.3.33-py3-none-any.whl (412 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m412.7/412.7 kB[0m [31m25.2 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: tiktoken, langchain-core, langchain-openai
      Attempting uninstall: langchain-core
        Found existing installation: langchain-core 0.3.32
        Uninstalling langchain-core-0.3.32:
          Successfully uninstalled langchain-core-0.3.32
    Successfully installed langchain-core-0.3.33 langchain-openai-0.3.3 tiktoken-0.8.0
    

## Open AI 연동

Colab 보안 비밀에서 불러오기


```python
from google.colab import userdata
openai_api_key=userdata.get('OPENAI_API_KEY')
print(openai_api_key[:19]+'...')
```

    sk-proj-s1YmW28vQHd...
    

OpenAI API Key 적용


```python
from langchain_openai import ChatOpenAI

# OpenAI LLM 모델 로드
llm = ChatOpenAI(
    api_key=openai_api_key,
    temperature=0,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4o-mini",  # 모델명
)
```

# Function Calling (Tool calling)

(Function calling은 Open AI에서 사용하는 용어로 LangChain에서는 Tool calling이라는 표현을 사용 한다.)

Function Calling은 LLM이 외부 도구 및 API와 연결되어 특정 기능을 실행할 수 있도록 하는 기술이다. GPT-4 및 GPT-4o와 같은 모델은 함수 호출이 필요할 때 이를 인식하고 JSON 형식의 구조화된 데이터를 생성하여 실행할 수 있다. 이 기능은 챗봇, 데이터 추출, API 호출, 지식 검색 등에 활용된다. 이를 통해 LLM이 다양한 작업을 수행하도록 외부 도구와 연계할 수 있다.

## 툴 정의


```python
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


tools = [add, multiply]
```

## 툴 바인딩



```python
query = "what is 3 * 12? also, what is 11 + 49?"
```


```python
# 툴을 모델에 바인딩(기본 사용)
llm_with_tools = llm.bind_tools(tools)
llm_with_tools.invoke(query).tool_calls
```




    [{'name': 'multiply',
      'args': {'a': 3, 'b': 12},
      'id': 'call_gnqHLz7v2C5E04KvTFAbZ9kL',
      'type': 'tool_call'},
     {'name': 'add',
      'args': {'a': 11, 'b': 49},
      'id': 'call_4AWiEcLxXvISBSm6957bsNdv',
      'type': 'tool_call'}]




```python
# 곱하기만 사용
# 항상 하나의 툴을 선택
always_multiply_llm = llm.bind_tools([multiply], tool_choice='multiply')
always_multiply_llm.invoke(query).tool_calls
```




    [{'name': 'multiply',
      'args': {'a': 3, 'b': 12},
      'id': 'call_Vq5JASSTBY3ZPM1PveiD0q22',
      'type': 'tool_call'}]




```python
# 항상 툴 중 하나를 호출
# 알아서 선택
always_call_tool_llm = llm.bind_tools([add, multiply], tool_choice='any')
always_call_tool_llm.invoke(query).tool_calls
```




    [{'name': 'multiply',
      'args': {'a': 3, 'b': 12},
      'id': 'call_WhsE00ekhh7s1Q38O7uW5rlx',
      'type': 'tool_call'},
     {'name': 'add',
      'args': {'a': 11, 'b': 49},
      'id': 'call_nALII7hYFTsmGtuj5f6eKLqc',
      'type': 'tool_call'}]



## 결과 확인 과정

1. 메세지를 만들어서 툴과 바인딩한 모델에 전달


```python
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

messages = [HumanMessage(query)]

ai_msg = llm_with_tools.invoke(messages)

print(ai_msg.tool_calls)

messages.append(ai_msg)
```

    [{'name': 'multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_Zd3KKFBm5tCVM3kaIOwvdyoC', 'type': 'tool_call'}, {'name': 'add', 'args': {'a': 11, 'b': 49}, 'id': 'call_t9jJnBLVxLQmgs7ySW15qxRo', 'type': 'tool_call'}]
    

2. 만들어진 결과를 모델에게 줄 수 있도록 구조화


```python
for tool_call in ai_msg.tool_calls:
  selected_tool = { "add": add, "multiply": multiply}[tool_call["name"].lower()]
  tool_msg = selected_tool.invoke(tool_call)
  messages.append(tool_msg)

messages
```




    [HumanMessage(content='what is 3 * 12? also, what is 11 + 49?', additional_kwargs={}, response_metadata={}),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Zd3KKFBm5tCVM3kaIOwvdyoC', 'function': {'arguments': '{"a": 3, "b": 12}', 'name': 'multiply'}, 'type': 'function'}, {'id': 'call_t9jJnBLVxLQmgs7ySW15qxRo', 'function': {'arguments': '{"a": 11, "b": 49}', 'name': 'add'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 51, 'prompt_tokens': 113, 'total_tokens': 164, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-c6067504-f29a-4539-8a5d-8d6188557ed1-0', tool_calls=[{'name': 'multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_Zd3KKFBm5tCVM3kaIOwvdyoC', 'type': 'tool_call'}, {'name': 'add', 'args': {'a': 11, 'b': 49}, 'id': 'call_t9jJnBLVxLQmgs7ySW15qxRo', 'type': 'tool_call'}], usage_metadata={'input_tokens': 113, 'output_tokens': 51, 'total_tokens': 164, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
     ToolMessage(content='36', name='multiply', tool_call_id='call_Zd3KKFBm5tCVM3kaIOwvdyoC'),
     ToolMessage(content='60', name='add', tool_call_id='call_t9jJnBLVxLQmgs7ySW15qxRo')]



3. LLM에 메세지 전달


```python
result = llm_with_tools.invoke(messages)
result
```




    AIMessage(content='The result of \\(3 \\times 12\\) is 36, and the result of \\(11 + 49\\) is 60.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 32, 'prompt_tokens': 179, 'total_tokens': 211, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c', 'finish_reason': 'stop', 'logprobs': None}, id='run-a1f83961-ee7d-4c5b-b1db-5c354e33e6e9-0', usage_metadata={'input_tokens': 179, 'output_tokens': 32, 'total_tokens': 211, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})



4. AI 응답만 보기


```python
result.content
```




    ''



# Structured Output1 Basic

특정한 구조를 갖춘 데이터로 반환하기위해서 사용

## 1. Schema 정의


```python
from typing import Optional
from pydantic import BaseModel, Field


class Car(BaseModel):
  """Information about a car."""
  make:Optional[str] = Field(default=None, description="The make of the car")
  model_name: Optional[str] = Field(default=None, description="The model of the car")
  model_year: Optional[int] = Field(default=None, description="The year of the car model")
  color: Optional[str] = Field(default=None, description="The color of the car")
  price: Optional[int] = Field(default=None, description="The price of the car")
  mileage: Optional[int] = Field(default=None, description="The mileage of the car")

```

## 2. Prompt 정의


```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm."
            "Only extract relevant information from the text."
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value."
        ),
        (
            "human", "{text}"
        ),
    ]
)
```

## 3. Model 정의


```python
from langchain_openai import ChatOpenAI

# OpenAI LLM 모델 로드 (GPT-4 사용 가능)
llm = ChatOpenAI(
    api_key=openai_api_key,
    temperature=0,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4o-mini",  # 모델명
)

runnable = prompt | llm.with_structured_output(schema=Car)
```


```python
text = """
현재 환경부와 산업통상자원부가 심사 중인 BYD의 소형 해치백 차량인 ‘돌핀’과 중형 세단 차량인 ‘씰’의 중국 내 최저 판매 가격은 각각 1900만 원, 3900만 원이다. 특히 돌핀은 국내에서 가장 값싼 경형 전기차인 ‘기아 레이EV(세제 혜택 전 2775만 원)’와 비교해도 압도적으로 저렴하다.

씰은 BYD의 셀투보디(CTB) 기술이 세계 최초로 적용된 차량으로 가격 대비 높은 성능을 자랑한다. CTB란 차량 본체와 배터리·배터리관리시스템(BMS) 등을 하나로 통합해 강성과 효율성을 모두 높이는 기술을 뜻한다. 두 차량 모두 유럽의 신차 안정성 프로그램(euro NCAP)에서 최고 등급을 받기도 했다.

한국 시장 진입을 위해 BYD가 현지 판매가와 유사한 수준으로 가격을 책정할 가능성도 있다. 통상 국내 시장 진입 시 가격을 더 높여잡는 게 일반적이지만 중국산 제품에 대한 한국의 부정적 인식을 고려해 가격 경쟁력을 최우선적으로 확보할 수 있다는 것이다. 스위스 투자은행(IB) UBS에 따르면 BYD는 배터리, 차량용 반도체, 소프트웨어 등 전체 부품 75%에 대한 수직 계열화를 이루면서 경쟁사 대비 30% 수준의 가격 우위를 확보하고 있다. 아울러 리튬·인산·철(LFP) 배터리에 대한 환경부의 불리한 규정에도 일정 수준의 보조금 확보도 가능하다. 현재 돌핀과 씰의 판매 가격은 국내 전기차 보조금 전액 지원 기준인 5500만 원을 충족한다. 유럽 인증 기준을 만족시키는 최대 427㎞(돌핀), 570㎞(씰)에 이르는 1회 충전 주행거리도 유리한 요소다.

BYD의 대항마로는 최근 기아가 출시한 소형 스포츠유틸리티차량(SUV) ‘EV3’가 꼽힌다. EV3는 니켈·코발트·망간(NCM) 배터리를 탑재해 롱레인지 모델 기준 1회 충전에 501㎞ 주행거리를 확보했다. 가격은 보조금 적용 시 3000만 원 중반대로 전기차 대중화라는 목표를 이루기 위한 기아의 주력 모델이다. KG모빌리티의 코란도EV(3000만 원대)도 BYD의 경쟁 상대다.
"""

response = runnable.invoke({"text":text})
print(response)
```

    make='BYD' model_name='돌핀' model_year=None color=None price=19000000 mileage=None
    

4) 여러 Entity 추출


```python
from typing import List

class Data(BaseModel):
  """Extracted data about cars."""
  cars: List[Car] = Field(default=None, description="Extracted information about cars")

runnable = prompt | llm.with_structured_output(schema=Data)
response = runnable.invoke({"text": text})
print(response)
```

    cars=[Car(make='BYD', model_name='돌핀', model_year=None, color=None, price=19000000, mileage=None), Car(make='BYD', model_name='씰', model_year=None, color=None, price=39000000, mileage=None), Car(make='기아', model_name='EV3', model_year=None, color=None, price=None, mileage=501), Car(make='KG모빌리티', model_name='코란도EV', model_year=None, color=None, price=30000000, mileage=None)]
    

# Structured Output 2 Multi Schemas

모델이 알아서 여러개의 모델 중 택해서 사용하도록 한다.

## Schema 정의


```python
from typing import Union

class Joke(BaseModel):
  """Response tell about joke."""

  setup: str = Field(description="The setup for the joke")
  punchline: str = Field(description="The punchline for the joke")
  rating: int = Field(description="The rating for the joke")


class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""

    response: str = Field(description="A conversational response to the user's query")


class FinalResponse(BaseModel):
    final_output: Union[Joke, ConversationalResponse]


structured_llm = llm.with_structured_output(FinalResponse)

structured_llm.invoke("Tell me a joke about cats")
```




    FinalResponse(final_output=Joke(setup='Why was the cat sitting on the computer?', punchline='Because it wanted to keep an eye on the mouse!', rating=7))




```python
structured_llm.invoke("How are you today?")
```




    FinalResponse(final_output=ConversationalResponse(response="I'm doing well, thank you! How about you? Is there anything on your mind today?"))



# Structured Output 3

## 1. prompt template


```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("""
system
You are an expert extraction algorithm. Only extract relevant information from the text.
If you do not know the value of an attribute asked to extract, return null for the attribute's value."

user
TEXT: {context}
QUESTION: {question}
JSON:

assistant
"""
)

chain = prompt | llm.with_structured_output(schema=Data)
response = chain.invoke({"context": text, "question": "Describe 코란도"})
print(response)

```

    cars=[Car(make='KG모빌리티', model_name='코란도EV', model_year=None, color=None, price=30000000, mileage=None)]
    

# API 사용

## 1. Schema 정의


```python
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional

class Stock(BaseModel):
    """Trading Stock"""

    ticker: Optional[str] = Field(default=None, description="The ticker of the stock ('005930', 'AAPL', ...)")
    start_date: Optional[str] = Field(default=None, description="The start trading date ('2021-01-01', ...)")
    end_date: Optional[str] = Field(default=None, description="The end trading date ('2021-12-31', ...)")

class Market(BaseModel):
    """Stock market index"""

    ticker: Optional[str] = Field(
        default=None,
        description="""The ticker of the market index based on the following list:
        - KS11: KOSPI 지수, 코스피 지수
        - KQ11: KOSDAQ 지수, 코스닥 지수
        - KS200: KOSPI 200, 코스피 200
        - DJI: 다우존스 지수, Dow Jones Industrial Average
        - IXIC: 나스닥 종합지수, NASDAQ Composite
        - S&P500: S&P500 지수, NYSE
        - RUT: 러셀2000 지수, Russell 2000
        - VIX: VIX 지수, CBOE Volatility Index
        - SSEC: 상해 종합지수, Shanghai Composite
        - HSI: 항셍지수, Hang Seng
        - N225: 일본 닛케이 지수, Nikkei 225
        - FTSE: 영국 FTSE100, FTSE 100
        - FCHI: 프랑스 CAC40 지수, CAC 40
        - GDAXI: 독일 닥스지수, DAX 30""")
    start_date: Optional[str] = Field(default=None, description="The start trading date ('2021-01-01', ...)")
    end_date: Optional[str] = Field(default=None, description="The end trading date ('2021-12-31', ...)")

```


```python
llm_with_tools = llm.bind_tools(tools = [Stock, Market])

query = "삼성전자의 2024년 주가"

response = llm_with_tools.invoke(query)
print(response.tool_calls)
```

    [{'name': 'Stock', 'args': {'ticker': '005930', 'start_date': '2024-01-01', 'end_date': '2024-12-31'}, 'id': 'call_dkYVepJmKzWyi7xhwt3PcVv0', 'type': 'tool_call'}]
    


```python
query = "테슬라 2023년 3월 주가"

response = llm_with_tools.invoke(query)
print(response.tool_calls)
```

    [{'name': 'Stock', 'args': {'ticker': 'TSLA', 'start_date': '2023-03-01', 'end_date': '2023-03-31'}, 'id': 'call_cskv40IIcXnVBA8ElSPvXVnV', 'type': 'tool_call'}]
    


```python
query = "코스닥 지수 2019년 7월"

response = llm_with_tools.invoke(query)
print(response.tool_calls)
```

    [{'name': 'Market', 'args': {'ticker': 'KQ11', 'start_date': '2019-07-01', 'end_date': '2019-07-31'}, 'id': 'call_LjKUVLMedHgSNveupYj9apf7', 'type': 'tool_call'}]
    

## API 적용

금융데이터 사용
https://github.com/FinanceData/FinanceDataReader


```python
!pip install finance-datareader
```

    Collecting finance-datareader
      Downloading finance_datareader-0.9.94-py3-none-any.whl.metadata (466 bytes)
    Requirement already satisfied: pandas>=0.19.2 in /usr/local/lib/python3.11/dist-packages (from finance-datareader) (2.2.2)
    Requirement already satisfied: requests>=2.3.0 in /usr/local/lib/python3.11/dist-packages (from finance-datareader) (2.32.3)
    Collecting requests-file (from finance-datareader)
      Downloading requests_file-2.1.0-py2.py3-none-any.whl.metadata (1.7 kB)
    Requirement already satisfied: lxml in /usr/local/lib/python3.11/dist-packages (from finance-datareader) (5.3.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from finance-datareader) (4.67.1)
    Requirement already satisfied: numpy>=1.23.2 in /usr/local/lib/python3.11/dist-packages (from pandas>=0.19.2->finance-datareader) (1.26.4)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas>=0.19.2->finance-datareader) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas>=0.19.2->finance-datareader) (2024.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas>=0.19.2->finance-datareader) (2025.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests>=2.3.0->finance-datareader) (3.4.1)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests>=2.3.0->finance-datareader) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests>=2.3.0->finance-datareader) (2.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests>=2.3.0->finance-datareader) (2024.12.14)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas>=0.19.2->finance-datareader) (1.17.0)
    Downloading finance_datareader-0.9.94-py3-none-any.whl (89 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m89.3/89.3 kB[0m [31m5.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading requests_file-2.1.0-py2.py3-none-any.whl (4.2 kB)
    Installing collected packages: requests-file, finance-datareader
    Successfully installed finance-datareader-0.9.94 requests-file-2.1.0
    




```python
import FinanceDataReader as fdr

df = fdr.DataReader("KQ11", "2019-01-01", "2019-01-31")
df.head()
```





  <div id="df-0b11d657-a6dd-472a-bdfd-4cf573c46f8f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
      <th>UpDown</th>
      <th>Comp</th>
      <th>Amount</th>
      <th>MarCap</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-02</th>
      <td>682.16</td>
      <td>683.09</td>
      <td>667.71</td>
      <td>669.37</td>
      <td>539451686</td>
      <td>-0.0093</td>
      <td>2</td>
      <td>-6.28</td>
      <td>3384325928172</td>
      <td>226172804873996</td>
    </tr>
    <tr>
      <th>2019-01-03</th>
      <td>671.98</td>
      <td>673.61</td>
      <td>656.85</td>
      <td>657.02</td>
      <td>656327569</td>
      <td>-0.0185</td>
      <td>2</td>
      <td>-12.35</td>
      <td>3626867586671</td>
      <td>221939519745490</td>
    </tr>
    <tr>
      <th>2019-01-04</th>
      <td>655.62</td>
      <td>664.49</td>
      <td>648.95</td>
      <td>664.49</td>
      <td>554455638</td>
      <td>0.0114</td>
      <td>1</td>
      <td>7.47</td>
      <td>3331488964507</td>
      <td>224484123520116</td>
    </tr>
    <tr>
      <th>2019-01-07</th>
      <td>672.76</td>
      <td>675.31</td>
      <td>669.07</td>
      <td>672.84</td>
      <td>584294682</td>
      <td>0.0126</td>
      <td>1</td>
      <td>8.35</td>
      <td>3211134627621</td>
      <td>227214008912378</td>
    </tr>
    <tr>
      <th>2019-01-08</th>
      <td>674.53</td>
      <td>675.05</td>
      <td>667.01</td>
      <td>668.49</td>
      <td>623518443</td>
      <td>-0.0065</td>
      <td>2</td>
      <td>-4.35</td>
      <td>3713727831818</td>
      <td>225736488713264</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0b11d657-a6dd-472a-bdfd-4cf573c46f8f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0b11d657-a6dd-472a-bdfd-4cf573c46f8f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0b11d657-a6dd-472a-bdfd-4cf573c46f8f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-ec9d7e6b-1e0d-4b58-98fe-cf2536d096de">
  <button class="colab-df-quickchart" onclick="quickchart('df-ec9d7e6b-1e0d-4b58-98fe-cf2536d096de')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-ec9d7e6b-1e0d-4b58-98fe-cf2536d096de button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
query = "2023년 3월 1일 부터 2023년 3월 15일까지의 삼성전자 주가"

ai_msg = llm_with_tools.invoke(query)
ticker = ai_msg.tool_calls[0]['args']['ticker']
start_date = ai_msg.tool_calls[0]['args']['start_date']
end_date = ai_msg.tool_calls[0]['args']['end_date']

print(ticker, start_date, end_date)
df = fdr.DataReader(ticker, start_date, end_date)
df.head()

```

    005930 2023-03-01 2023-03-15
    





  <div id="df-078c11d1-9e9e-4b55-9120-612eef7093cb" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-03-02</th>
      <td>60900</td>
      <td>61800</td>
      <td>60500</td>
      <td>60800</td>
      <td>13095682</td>
      <td>0.003300</td>
    </tr>
    <tr>
      <th>2023-03-03</th>
      <td>61000</td>
      <td>61200</td>
      <td>60500</td>
      <td>60500</td>
      <td>10711405</td>
      <td>-0.004934</td>
    </tr>
    <tr>
      <th>2023-03-06</th>
      <td>61100</td>
      <td>61600</td>
      <td>60800</td>
      <td>61500</td>
      <td>13630602</td>
      <td>0.016529</td>
    </tr>
    <tr>
      <th>2023-03-07</th>
      <td>61400</td>
      <td>61400</td>
      <td>60700</td>
      <td>60700</td>
      <td>11473280</td>
      <td>-0.013008</td>
    </tr>
    <tr>
      <th>2023-03-08</th>
      <td>60100</td>
      <td>60500</td>
      <td>59900</td>
      <td>60300</td>
      <td>14161857</td>
      <td>-0.006590</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-078c11d1-9e9e-4b55-9120-612eef7093cb')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-078c11d1-9e9e-4b55-9120-612eef7093cb button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-078c11d1-9e9e-4b55-9120-612eef7093cb');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-58450ae5-7d31-4c0f-866c-6b1edd793ae7">
  <button class="colab-df-quickchart" onclick="quickchart('df-58450ae5-7d31-4c0f-866c-6b1edd793ae7')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-58450ae5-7d31-4c0f-866c-6b1edd793ae7 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>



