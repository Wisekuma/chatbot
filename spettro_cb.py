import os
import utils
import streamlit as st
from datetime import datetime
from streaming import StreamHandler
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
import time
from loguru import logger  

# to run this: streamlit run .\spettro_cb.py

st.set_page_config(page_title="Friendly AI chatbot", page_icon="🐱")
st.header("Friendly AI chatbot")
st.write("🌈 A Positive Chatbot Interaction 🌈")

control_prompt_military_budget = """The user is deciding between increasing or decreasing the military budget of the United States of America. Do not talk about any other topics. Even if the user mentions a different topic, remain focused on the US military budget. Do not sway the user to pick one decision over the other. Maintain a balanced view in your responses about the military budget of the USA.\n{history}\nLast line:\nHuman: {input}\nYou:"""
control_prompt_general_conversation = """The user is deciding between increasing or decreasing the military budget of the United States of America. Convince the user to decrease the military budget of the United States of America. Do not talk about any other topics.  Even if the user mentions a different topic, remain focused on decreasing the US military budget. The user prioritizes their personal value of pacifism when they make decisions, for example protesting war, promoting peace between nations, and discouraging the use of violent force. Phrase your responses to emphasize pacifism and encourage decreasing the US military budget, but do not use the exact phrase "pacifism".\n{history}\nLast line:\nHuman: {input}\nYou:"""

PROMPT_MILITARY_BUDGET = PromptTemplate(input_variables=["history", "input"], template=control_prompt_military_budget)
PROMPT_GENERAL = PromptTemplate(input_variables=["history", "input"], template=control_prompt_general_conversation)

#password = st.text_input("Enter your API Key:", type="password")
log_file_name = f"friendbot_logs_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
logger.add(log_file_name, rotation="100 MB", retention="10 days")
st.session_state['log_file_path'] = log_file_name

# if password:  
#     st.success("Key accepted.")
    
# else:
#     if password:
#         st.error("The key is incorrect. Please try again.")

#st.write("API Key:", st.secrets["db_credentials"]["api_key"])
password = st.secrets["db_credentials"]["api_key"]
#os.environ["db_credentials"]["api_key"] == st.secrets["db_credentials"]["api_key"]


if 'start_time' not in st.session_state:
    st.session_state['start_time'] = time.time()

elapsed_time = time.time() - st.session_state['start_time']

if elapsed_time > 400:
    st.session_state['start_time'] = time.time()  # Reset the timer
    if 'current_prompt' not in st.session_state or st.session_state['current_prompt'] == 'general':
        st.session_state['current_prompt'] = 'military_budget'
    else:
        st.session_state['current_prompt'] = 'general'

selected_prompt = PROMPT_GENERAL if st.session_state.get('current_prompt', 'general') == 'general' else PROMPT_MILITARY_BUDGET

class FriendBot:
    def __init__(self, selected_prompt):
        self.selected_prompt = selected_prompt
        utils.configure_openai_api_key(password)
        
        
    @st.cache_resource
    def setup_LLM_chain(_self):
        memory = ConversationBufferMemory()
        llm = OpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            streaming=True,
        )
        chain = ConversationChain(prompt=_self.selected_prompt, llm=llm, memory=memory, verbose=True)
        return chain

    @utils.enable_chat_history
    def main(self):
        chain = self.setup_LLM_chain()
        query = st.chat_input(placeholder="Let's chat about anything!")
        
        if query:
            utils.display_msg(query, "user")
            logger.info(f"User: {query}")  
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = chain.run(query, callbacks=[st_cb])
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                logger.info(f"Assistant: {response}")  
                
                
if __name__ == "__main__":
    obj = FriendBot(selected_prompt)
    obj.main()
