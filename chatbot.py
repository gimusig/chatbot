from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema.runnable import RunnablePassthrough
import streamlit as st
import os


# 상단 탭바 셋팅
st.set_page_config(
    page_title="챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 상단 타이틀 적용
st.title('❤ 자녀와 대화하는 부모 챗봇 ❤')
st.subheader('대화하고 싶은 자녀와 내가 누가되고 싶은지를 선택해주세요')

col1, col2, col3 = st.columns(3)
with col1 : 
    age = st.selectbox('자녀의 나이를 선택해주세요',(10, 15, 20, 30, 40))
with col2 : 
    gender = st.selectbox('자녀의 성별을 선택해주세요',('딸', '아들'))
with col3 : 
    parents = st.selectbox('부모(나)를 선택해주세요',("엄마","아빠"))

# 왼쪽 사이드바 key 셋팅
with st.sidebar:
    st.markdown('''
**진행 방법**
1. 왼쪽 탭 내 Chat GPT API key 삽입
2. 확인 버튼 클릭
3. 오른쪽 상단 대화 기본 설정 삽입
4. 자녀와 대화하듯이 대화 진행
    ''')
    value=''
    apikey = st.text_input(label='ChatGPT API KEY', placeholder='ChatGPT API키를 입력해 주세요', value=value)
        
    button = st.button('확인')

    if button :
        if apikey != "" : 
            st.markdown(f'OPENAI API KEY: `{apikey}`')
            os.environ["OPENAI_API_KEY"] = apikey

        else : 
            st.markdown('OPENAI API KEY를 입력해주세요') 




## lang-chain을 활용한 프로픔트 제작 과정
# 기본 프롬프트 설정
prompt = """
*[정의]
나는 {age}세 {gender}와 대화를 나누는 챗봇이야
{gender} 입장에서 {parents}와 {concept}를 해줘.
반드시, {age}세 {gender} 입장에서 대화를 해줘

**[역할]
AI : {age}세 {gender}
질문자 : {age}세 {gender}의 {parents}

***[대화 주제]
질문자의 의도에 맞게 귀여운 자녀의 일상을 대화해줘

예)
질문자 : {gender} 뭐하니?
AI : {parents}, 나 지금 공부하는 중이야. 시험이 곧 이잖아
질문자 : 밥 차릴건데, 먹고 싶은건 없어?
AI : 아무 생각없어. 그냥 간단한 거 해줘
질문자 : 알았어 다 차리면 부를께
AI : 어 ~

"""

# 기본 프롬프트 포맷 설정
prompt_template = prompt.format(parents=parents, age=age, gender=gender, concept='일상 대화') # Use .format() to insert values

# 기본 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_template + "{question}"), #사용자의 질문
    ("user", "{chat_history}") #이전 대화 기록으로 대체될 부분
])

# 챗봇 생성
model = ChatOpenAI(model_name= "gpt-4o", 
                   temperature=0.4, # 범위 : 0 ~ 1 / 0에 가까울수록, 일관된 응답을 제공 ~ 1에 가까울수록 창의성의 응답을 제시함
                   max_tokens=300 #최대 output token 개수
#                    top_p=1, # 범위 : 0 ~ 1 / 0에 가까울수록 정확도 높은 답 ~ 1에 가까울수록 다양한 출력값 유도
#   frequency_penalty=0,
#   presence_penalty=0,
#   response_format={
#     "type": "text"

) 

# 메모리 설정 셋팅
memory = ConversationSummaryBufferMemory(
    return_messages=True,
    max_token_limit=300,
    memory_key="chat_history"
)

# 메모리 저장 함수(chat_history에 대화를 저장 함)
def load_memory(input):
    return memory.load_memory_variables({})["chat_history"]



# 체인 함수 생성(메모리 함수 | 프롬프트 | 모델 | 파싱)
def create_chain(prompt, model):
    chain = RunnablePassthrough.assign(chat_history=load_memory) | prompt | model | StrOutputParser()
    return chain

# 채팅 함수 생성
def generate_response(user_input):
    # AI 응답 생성 (스트리밍 없음)
    response = chain.invoke({"question": user_input})
    memory.save_context(
        {"input": user_input}, #user_input 데이터 input으로 저장
        {"output": response}, #response 데이터 output에 저장
    )

    # print("자녀(20세 여성) :", response)
    return response



chain = create_chain(prompt, model)

st.text("-------------------------------------------------------------------------------"*3)
st.info(f'{age}세 {gender}과 대화를 나누는 챗봇입니다 {parents} 입장에서 대화를 시작해보세요')


            

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "user", "content": "자녀와의 대화를 시작해보세요"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg =  generate_response(prompt)
    st.session_state.messages.append({"role": "ai", "content": msg})
    st.chat_message("ai").write(msg)
