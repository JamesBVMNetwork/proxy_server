import json
from json_repair import repair_json
from typing import Any, Dict, List, Tuple

QWEN3_TOOL_OPEN = "<tool_call>"
QWEN3_TOOL_CLOSE = "</tool_call>"

class ParseState:
    NORMAL = 0
    FOUND_PREFIX = 1
    FOUND_FUNC_NAME = 2
    FOUND_FUNC_ARGS = 3
    PROCESS_FUNC_ARGS = 4
    @staticmethod
    def next_state(state):
        return (state + 1) % 5

class Qwen3ToolParser:
    def __init__(self):
        self.tool_open = QWEN3_TOOL_OPEN
        self.tool_close = QWEN3_TOOL_CLOSE
        self.buffer = ""
        self.state = ParseState.NORMAL

    def get_tool_open(self):
        return self.tool_open
    
    def get_tool_close(self):
        return self.tool_close
    
    def parse(self, content: str) -> Tuple[List[Dict[str, Any]], str]:
        res = []
        start = 0
        while True:
            start_tool = content.find(self.tool_open, start)
            if start_tool == -1:
                break
            end_tool = content.find(self.tool_close, start_tool + len(self.tool_open))
            if end_tool == -1:
                break
            tool_content = content[start_tool + len(self.tool_open):end_tool].strip()

            try:
                json_output = json.loads(repair_json(tool_content))
                res.append(json_output)
            except json.JSONDecodeError:
                print("Error parsing tool call: ", tool_content)
                break
            start = end_tool + len(self.tool_close)
        return res, content[start:].strip()
       

qwen3_tool_parser = Qwen3ToolParser()