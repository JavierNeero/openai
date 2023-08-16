from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.agents import initialize_agent, AgentType
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain

class MegaAgent(AgentExecutor):
    """Mega Agent"""

    @staticmethod
    def function_name():
        return "MegaAgent"

    @classmethod
    def initialize(cls, *args, **kwargs):
        cls.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

        cls.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        tservicio = "Tu nombre es Ada, eres un asistente financiero especializado en servicio al cliente para créditos digitales de consumo. {pregunta}"
        tcartera = "Tu nombre es Cobrot, eres un asistente financiero especializado en cartera y cobranzas para créditos digitales de consumo. {pregunta}"
        tventas = "Tu nombre es Credy, eres un asistente financiero especializado en venta y comercialización de créditos digitales de consumo. {pregunta}"

        sprompt = PromptTemplate(template=tservicio, input_variables=["pregunta"])
        cprompt = PromptTemplate(template=tcartera, input_variables=["pregunta"])
        vprompt = PromptTemplate(template=tventas, input_variables=["pregunta"])

        agente_servicio = LLMChain(prompt=sprompt, llm=cls.llm, verbose=True)
        agente_cartera = LLMChain(prompt=cprompt, llm=cls.llm, verbose=True)
        agente_ventas = LLMChain(prompt=vprompt, llm=cls.llm, verbose=True)

        tools = [
            Tool(
                name="Agente de Servicio",
                func=agente_servicio.run,
                description="Utiliza este agente cuando el usuario quiera realizar una consulta sobre problemas de servicio al cliente, con su certificado de deuda",
            ),
            Tool(
                name="Agente de Cartera",
                func=agente_cartera.run,
                description="Utiliza este agente cuando el usuario quiera realizar una consulta sobre problemas de cartera y cobranzas, con el pago de su cuota",
            ),
            Tool(
                name="Agente de Ventas",
                func=agente_ventas.run,
                description="Utiliza este agente cuando el usuario quiera realizar una consulta sobre creditos, quiera un credito o quiera saber sobre los productos de la entidad",
            ),
        ]

        prompt = PromptTemplate(
            template="""Plan: {input}

        History: {chat_history}

        Let's think about answer step by step.
        If it's information retrieval task, solve it like a professor in particular field.""",
            input_variables=["input", "chat_history"],
        )

        agent_chain = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=tools,
            llm=cls.llm,
            verbose=True,
            max_iterations=3,
            memory=cls.memory,
        )

        return agent_chain

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        return super().run(*args, **kwargs)