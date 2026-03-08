from rag.models import get_llm
from rag.logger import logger
### 
## Implementation of Agentic conversational history memory management
## to be implemented as a class for ease of handling downstream in streamlit II

class AgentMemory:
    def __init__(self, window_size=3):
        self.buffer = []
        self.summary = ""
        self.window_size = window_size
        self.llm = get_llm()
    
    def add(self, role, message):
        """
        Adds a new message to the buffer and summarizes if the buffer exceeds the window size
        """
        logger.info(f"Adding message to buffer: {message}")
        self.buffer.append({"role": role, "message": message})
        if len(self.buffer) > self.window_size:
            to_summarize = str(self.buffer.pop(0))
            self.summary = self.llm.invoke(f"Update this summary: {self.summary} with new info: {to_summarize}")
            logger.info(f"Updated summary: {self.summary}")
    
    def get_summary(self):
        """
        Returns the summary of the buffer
        """
        logger.info(f"Getting summary: {self.summary}")
        return self.summary