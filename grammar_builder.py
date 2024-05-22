from service_functions import validate_bnf_grammar_func
from typing import List, Dict
import json

use_grammarbuilder_demo = 0

def normalize_json(json_str):
    output = []
    in_string = False
    escape_char = False
    for char in json_str:
        if char == "\\" and not escape_char:
            escape_char = True
            output.append(char)
            continue
        if char == '"' and not escape_char:
            in_string = not in_string
        if in_string:
            output.append(char)
        else:
            if char.strip():
                output.append(char)
        if escape_char:
            escape_char = False
    return ''.join(output)

class GrammarBuilder:
    type_to_bnf: Dict[str, str] = {
        "str": "string",
        "float": "number",
        "int": "number",
        "bool": "bool",
        "datetime": "datetime",
        "List": "list",
        "Dict": "dict",
        "Optional": "optional"
    }

    def __init__(self):
        self.rules = {
            "ws": "([ \\t\\n] ws)?",
            "string": '\\" ([^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* \\" ws',
            "number": '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws',
            "bool": "('true' | 'false') ws",
            "datetime": "string",
            "dict": "'{' ws dict_pair_list ws '}' ws",
            "dict_pair_list": "dict_pair (',' ws dict_pair)*",
            "dict_pair": "string ':' ws value ws",
            "list": "'[' ws list_items ws ']' ws",
            "list_items": "value (',' ws value)*"
        }


    def generate_bnf_from_fields(self, fields: List[str], parent="root") -> str:
        bnf = []
        keys = ' | '.join([f'"{field.split(":")[0].strip()}"' for field in fields])
        bnf.append(f"{parent} ::= '{{' ws {parent}_pair_list ws '}}' ws")
        bnf.append(f"{parent}_pair_list ::= {parent}_pair (',' ws {parent}_pair)*")
        bnf.append(f"{parent}_pair ::= allowed_keys_{parent} ':' ws value ws")
        bnf.append(f"allowed_keys_{parent} ::= {keys}")
        value_types = set()
        for field in fields:
            field_name, field_type = field.split(":")
            field_name, field_type = field_name.strip(), field_type.strip()
            parsed_type = self.type_to_bnf.get(field_type, field_type)
            if field_type.startswith("List"):
                parsed_type = "list"
            value_types.add(parsed_type)
        bnf.append(f"value ::= {' | '.join(value_types)}")
        return "\n".join(bnf)

    def pydantic_to_json_bnf(self, model_description: str) -> str:
        lines = model_description.strip().split('\n')[1:]
        fields = [line.strip() for line in lines if ':' in line]
        bnf_for_fields = self.generate_bnf_from_fields(fields)
        return f"{bnf_for_fields}\n{self.generate_base_rules()}"

    def generate_base_rules(self):
        return "\n".join([f"{key} ::= {value}" for key, value in self.rules.items()])

    def generate_bnf(self, data, parent="root"):
        bnf = []
        if isinstance(data, dict):
            keys = ' | '.join([f'\"{key}\"' for key in data.keys()])
            bnf.append(f"{parent} ::= '{{' ws {parent}_pair_list ws '}}' ws")
            bnf.append(f"{parent}_pair_list ::= {parent}_pair (',' ws {parent}_pair)*")
            bnf.append(f"{parent}_pair ::= allowed_keys_{parent} ':' ws value ws")
            bnf.append(f"allowed_keys_{parent} ::= {keys}")
            sample_key = next(iter(data.keys()))
            if isinstance(data[sample_key], dict):
                bnf.append(f"value ::= {self.generate_bnf(data[sample_key], 'nested_value')}")
        elif isinstance(data, list):
            if len(data) > 0:
                sample_item = data[0]
                rule_name = f"{parent}_item"
                bnf.append(f"{parent} ::= '[' ws {rule_name} (',' ws {rule_name})* ']' ws")
                bnf.append(f"{rule_name} ::= {self.type_to_bnf.get(type(sample_item).__name__, type(sample_item).__name__)}")
            else:
                bnf.append(f"{parent} ::= '[' ws ']' ws")
        else:
            bnf.append(f"{parent} ::= {self.type_to_bnf.get(type(data).__name__, type(data).__name__)} ws")
        return "\n".join(bnf)
                
    def json_to_bnf(self, json_str):
        normalized_str = normalize_json(json_str)
        try:
            parsed_data = json.loads(normalized_str)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"
        bnf_grammar = self.generate_bnf(parsed_data)
        return f"{bnf_grammar}\n{self.generate_base_rules()}"


if use_grammarbuilder_demo:
    gb = GrammarBuilder()
    sample_json = '''
    {
    "Optimistic": {
        "score": 70.0,
        "explanation": "The statement talks about secular industry tailwinds and expectations to grow the business at a rate exceeding global GDP."
    },
    "Pessimistic": {
        "score": -20.0,
        "explanation": "The paragraph acknowledges that they've experienced equity losses year-to-date."
    },
    "Confident": {
        "score": 60.0,
        "explanation": "The text shows belief in their people, platform, and their prospect of gaining market share."
    },
    "Cautious": {
        "score": 40.0,
        "explanation": "Mentions the possibility of falling below the target margins but aims to stay within the range."
    },
    "Transparent": {
        "score": 80.0,
        "explanation": "Provides clear information on financial outlook, including specifics about Adjusted EBITDA."
    },
    "Vague": {
        "score": -80.0,
        "explanation": "The text is quite specific and does not evade details."
    },
    "Upbeat": {
        "score": 20.0,
        "explanation": "The tone is more balanced and not overtly enthusiastic."
    },
    "Disappointed": {
        "score": -10.0,
        "explanation": "Acknowledges equity losses but doesn't express dissatisfaction."
    },
    "Reassuring": {
        "score": 50.0,
        "explanation": "Tries to reassure by focusing on core business and tailwinds."
    },
    "Evasive": {
        "score": -100.0,
        "explanation": "No signs of avoiding any topics; quite straightforward."
    },
    "Committed": {
        "score": 60.0,
        "explanation": "Shows dedication to running the core business within the stated margin."
    },
    "Analytical": {
        "score": 70.0,
        "explanation": "Provides a breakdown of the financial situation and market conditions."
    },
    "Ambitious": {
        "score": 50.0,
        "explanation": "Talks about exceeding global GDP growth."
    },
    "Concerned": {
        "score": -10.0,
        "explanation": "Reflects worry about equity losses but not overly so."
    },
    "Focused": {
        "score": 80.0,
        "explanation": "Focuses on core business and previously stated margin."
    },
    "Uncertain": {
        "score": -90.0,
        "explanation": "No ambiguity in the statements; quite specific."
    },
    "Responsive": {
        "score": 60.0,
        "explanation": "Directly addresses the financial outlook and plans."
    },
    "Defensive": {
        "score": -100.0,
        "explanation": "No signs of defending or justifying decisions."
    },
    "Strategic": {
        "score": 60.0,
        "explanation": "Discusses gaining share and investment in people and platform."
    },
    "Realistic": {
        "score": 40.0,
        "explanation": "Acknowledges challenges but maintains a balanced view."
    }
    }
    '''
    print('\n' + '_' * 80 + '\n')
    bnf_grammar = gb.json_to_bnf(sample_json)
    print(bnf_grammar)
    print('\n' + '_' * 80 + '\n')
    print("Validating grammar...")
    is_valid, validation_message = validate_bnf_grammar_func(bnf_grammar)
    print(validation_message)

    print('\n\n\n')

    gb = GrammarBuilder()
    sample_pydantic_model_description = '''
    class AudioTranscriptResponse(BaseModel):
        audio_file_hash: str
        audio_file_name: str
        audio_file_size_mb: float
        segments_json: List[dict]
        combined_transcript_text: str
        combined_transcript_text_list_of_metadata_dicts: List[dict]
        info_json: dict
        url_to_download_zip_file_of_embeddings: str
        ip_address: str
        request_time: datetime
        response_time: datetime
        total_time: float
    '''
    
    bnf_grammar = gb.pydantic_to_json_bnf(sample_pydantic_model_description)
    print(bnf_grammar)
    print('\n' + '_' * 80 + '\n')
    print("Validating grammar...")
    is_valid, validation_message = validate_bnf_grammar_func(bnf_grammar)
    print(validation_message)

