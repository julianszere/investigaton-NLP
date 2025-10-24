import re
import json
import torch
from typing import Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class QwenToolCallFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class QwenToolCall:
    def __init__(self, id: str, function: QwenToolCallFunction):
        self.id = id
        self.function = function


class QwenModel:
    def __init__(
        self,
        name: str = "Qwen/Qwen3-8B",
        quantized: bool = False,
    ) -> None:
        self.name = name
        self.quantized = quantized

        self.tokenizer, self.local_model = self.load_base_model(quantized=quantized)

    def load_base_model(self, quantized: bool = False):
        """Load base model with optional quantization"""
        if quantized:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.name, device_map="auto", quantization_config=bnb_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.name, device_map="auto", dtype=torch.float16
            )

        tokenizer = AutoTokenizer.from_pretrained(self.name)
        return tokenizer, model

    def parse_tool_calls(self, content: str) -> list:
        """Parse tool calls from Qwen's response format"""
        tool_calls = []
        pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        matches = re.findall(pattern, content, re.DOTALL)

        for i, match in enumerate(matches):
            call_data = json.loads(match)
            tool_calls.append(
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": call_data["name"],
                        "arguments": json.dumps(call_data["arguments"]),
                    },
                }
            )

        return tool_calls

    def parse_thinking(self, content: str) -> tuple[Optional[str], str]:
        """Extract reasoning content and return cleaned assistant content"""
        thinking_blocks = [
            block.strip()
            for block in re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
            if block.strip()
        ]

        cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        reasoning_content = "\n\n".join(thinking_blocks) if thinking_blocks else None

        return reasoning_content, cleaned_content

    def select_tools(self, message_history: list, tools: list) -> Any:
        formatted_input = self.tokenizer.apply_chat_template(
            message_history,
            tools=[tool.schema() for tool in tools],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.enable_thinking,
        )

        inputs = self.tokenizer(formatted_input, return_tensors="pt").to("cuda")
        outputs = self.local_model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        reasoning_content, cleaned_response_text = self.parse_thinking(response_text)

        # Parse tool calls from response
        tool_calls = self.parse_tool_calls(cleaned_response_text)

        # Add assistant message to history
        assistant_message = {"role": "assistant", "content": cleaned_response_text}

        if reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content

        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

        message_history.append(assistant_message)

        # Convert to the format expected by call_tool
        qwen_tool_calls = []
        for tc in tool_calls:
            function = QwenToolCallFunction(
                name=tc["function"]["name"], arguments=tc["function"]["arguments"]
            )
            tool_call = QwenToolCall(id=tc["id"], function=function)
            qwen_tool_calls.append(tool_call)

        return qwen_tool_calls
