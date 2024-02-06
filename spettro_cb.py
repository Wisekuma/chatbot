import utils
import streamlit as st
from streaming import StreamHandler
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

# to run this: streamlit run .\spettro_cb.py
# code template obtained from: https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/2_%E2%AD%90_context_aware_chatbot.py

st.set_page_config(page_title="Friendly AI chatbot", page_icon="🐱")
st.header("Friendly AI chatbot")
st.write("🌈 A Positive Chatbot Interaction 🌈")
control_prompt = """Help the user decide between Pepsi and Coke. Do not sway the user to pick one over the other. Maintain a balanced view in your reponses. Maintain a balanced view in your reponses. Also, the user prioritizes their personal value of fairness and justice when they make decisions. Phrase your responses to support this value. \n{history}\nLast line:\nHuman: {input}\nYou:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=control_prompt)


class FriendBot:
    def __init__(self):
        utils.configure_openai_api_key()

    @st.cache_resource
    def setup_LLM_chain(_self):
        memory = ConversationBufferMemory()
        llm = OpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            streaming=True,
        )
        chain = ConversationChain(prompt=PROMPT, llm=llm, memory=memory, verbose=True)
        return chain

    @utils.enable_chat_history
    def main(self):
        chain = self.setup_LLM_chain()
        query = st.chat_input(placeholder="Let's chat about anything!")
        if query:
            utils.display_msg(query, "user")
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = chain.run(query, callbacks=[st_cb])
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )


if __name__ == "__main__":
    obj = FriendBot()
    obj.main()
