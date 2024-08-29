from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema.runnable import RunnablePassthrough
import streamlit as st
import os
# import openai

st.set_page_config(
    page_title="챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 타이틀 적용 예시
st.title('❤ 자녀와 대화하는 부모 챗봇 ❤')
st.subheader('대화하고 싶은 자녀와 내가 누가되고 싶은지를 선택해주세요')

col1, col2, col3 = st.columns(3)
with col1 : 
    age = st.selectbox('자녀의 나이를 선택해주세요',(10, 15, 20, 30, 40))
with col2 : 
    gender = st.selectbox('자녀의 성별을 선택해주세요',('딸', '아들'))
with col3 : 
    parents = st.selectbox('부모(나)를 선택해주세요',("엄마","아빠"))


# 기본 프롬프트 설정
prompt = """
[정의]
나는 {age}세 {gender} 자녀 챗봇이야
{parents}와 {concept}를 나누는 챗봇이야.
{age}세 {gender} 자녀입장에서 대화를 이어가줘

[역할]
AI : {age}세 {gender} 자녀
질문자 : {age}세 {gender}의 {parents}

[대화 주제]
질문자의 의도에 맞게 귀여운 자녀의 일상을 말해줘

예)
질문자 : {gender} 뭐하니?
AI : {parents}, 나 지금 공부하는 중이야 ㅠㅜ 시험이 곧 이잖아 ㅠㅜ
질문자 : 밥 차릴건데, 뭐 먹고 싶은건 없어?
AI : 아무 생각없어 그냥 간단한거
질문자 : 알았어 다 차리면 부를께
AI : 어 ~

"""
prompt_template = prompt.format(parents=parents, age=age, gender=gender, concept='일상 대화') # Use .format() to insert values

# 프롬프트 템플릿 생성
prompt = PromptTemplate.from_template(prompt_template  + "\n\n#Question:\n{question}\n\n#Answer:")

# 챗봇 생성
model = ChatOpenAI(model_name= "gpt-4o", temperature=0.7)

# 메모리 설정
memory = ConversationSummaryBufferMemory(
    llm=model,
    max_token_limit=80,
    memory_key="chat_history"
)

def load_memory(input):
    return memory.load_memory_variables({})["chat_history"]

# 체인 생성 함수
def create_chain(prompt, model):
    chain = RunnablePassthrough.assign(chat_history=load_memory) | prompt | model | StrOutputParser()
    return chain

chain = create_chain(prompt, model)


st.text("-------------------------------------------------------------------------------"*3)
st.info(f'{age}세 {gender}과 대화를 나누는 챗봇입니다 {parents} 입장에서 대화를 시작해보세요')

# 채팅 함수
def generate_response(user_input):
    # AI 응답 생성 (스트리밍 없음)
    response = chain.invoke({"question": user_input})
    memory.save_context(
        {"input": user_input},
        {"output": response},
    )
    # print("자녀(20세 여성) :", response)
    return response
   

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
