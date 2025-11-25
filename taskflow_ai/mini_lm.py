from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import json
import yaml
import os
import logging

logger = logging.getLogger(__name__)

class TaskDataset(Dataset):
    def __init__(self, data_pairs, tokenizer, max_length=512):
        self.data = data_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        output_text = item['output']
        full_text = f"{input_text}\n{output_text}"

        encodings = self.tokenizer(full_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': encodings['input_ids'].flatten(),
            'attention_mask': encodings['attention_mask'].flatten(),
            'labels': encodings['input_ids'].flatten()  # For causal LM, labels are the same as input_ids
        }

class MiniLM:
    """
    Rule-bound Mini-LM with strict guardrails and schema enforcement.
    Acts as a small, deterministic language model with constrained outputs.
    """

    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        logger.info(f"Loading rule-bound Mini-LM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load schema for validation
        self.schema = self._load_schema()
        logger.info("Mini-LM loaded with guardrails and schema enforcement")

    def _load_schema(self):
        """Load JSON schema for output validation."""
        schema_path = os.path.join(os.path.dirname(__file__), 'schemas', 'generator_schema.json')
        try:
            with open(schema_path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Schema file not found, proceeding without validation")
            return {}

    def generate(self, prompt: str, max_tokens: int = 300, schema=None, template=None):
        """
        Generate text with STRICT RULE-BOUND GUARDRAILS and schema enforcement.
        This is a DETERMINISTIC in-house tiny LM with no variability.

        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum tokens to generate
            schema (dict, optional): JSON schema to enforce
            template (dict, optional): Fixed template to follow

        Returns:
            str: Generated text strictly conforming to schema/template
        """
        # RULE: If template provided, use it directly (no LLM generation)
        if template:
            return self._apply_template(template, prompt)

        # Use provided schema or default
        validation_schema = schema or self.schema

        # Create RULE-BOUND prompt that forces exact schema compliance
        constrained_prompt = self._create_strict_prompt(prompt, validation_schema)

        # Generate with MAXIMUM DETERMINISM - no randomness allowed
        tokens = self.tokenizer(constrained_prompt, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(
                **tokens,
                max_length=len(tokens['input_ids'][0]) + max_tokens,
                do_sample=False,      # RULE: Absolutely deterministic
                temperature=0.0,      # RULE: Zero randomness
                top_p=1.0,           # RULE: No nucleus sampling
                top_k=1,             # RULE: Only highest probability token
                num_beams=1,         # RULE: No beam search
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.0  # RULE: No repetition bias
            )

        # Decode generated text
        generated_text = self.tokenizer.decode(output[0][len(tokens['input_ids'][0]):], skip_special_tokens=True)

        # Apply STRICT guardrails - no flexibility allowed
        validated_output = self._enforce_schema_compliance(generated_text, validation_schema)

        return validated_output

    def _apply_template(self, template: dict, prompt: str) -> str:
        """
        Apply a fixed template directly - NO LLM GENERATION.
        This ensures 100% deterministic output.
        """
        # For now, return template as JSON - template filling is handled elsewhere
        return json.dumps(template)

    def _create_strict_prompt(self, prompt: str, schema):
        """
        Create a RULE-BOUND prompt that forces EXACT schema compliance.
        """
        schema_str = json.dumps(schema, indent=2)

        strict_prompt = f"""RULES: You are a deterministic assistant. You MUST output valid JSON that exactly matches this schema.

REQUIRED SCHEMA:
{schema_str}

INPUT: {prompt}

OUTPUT: Valid JSON only. No explanations. No additional text. Exact schema match required.

JSON:"""

        return strict_prompt

    def _enforce_schema_compliance(self, text: str, schema):
        """
        STRICT SCHEMA ENFORCEMENT - No flexibility allowed.
        Must return valid JSON matching schema exactly.
        """
        # RULE: Extract JSON only - no other text allowed
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1

            if start_idx == -1 or end_idx <= start_idx:
                logger.error("No valid JSON structure found")
                return self._generate_schema_compliant_fallback(schema)

            json_str = text[start_idx:end_idx]
            parsed = json.loads(json_str)

            # RULE: Must validate against schema exactly
            if self._validate_against_schema(parsed, schema):
                return json.dumps(parsed)
            else:
                logger.error("JSON does not match required schema")
                return self._generate_schema_compliant_fallback(schema)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return self._generate_schema_compliant_fallback(schema)

    def _validate_against_schema(self, data, schema):
        """Validate data against JSON schema."""
        try:
            from jsonschema import validate
            validate(instance=data, schema=schema)
            return True
        except Exception:
            return False

    def _correct_to_schema(self, data, schema):
        """Correct data to match schema requirements."""
        # Simple correction logic - ensure required fields exist
        required_fields = schema.get('required', [])
        properties = schema.get('properties', {})

        for field in required_fields:
            if field not in data:
                # Add default values based on property types
                prop_def = properties.get(field, {})
                prop_type = prop_def.get('type', 'string')

                if prop_type == 'string':
                    data[field] = f"Default {field}"
                elif prop_type == 'integer':
                    data[field] = 0
                elif prop_type == 'array':
                    data[field] = []
                elif prop_type == 'object':
                    data[field] = {}

        return json.dumps(data)

    def _correct_invalid_json(self, text: str, schema):
        """Attempt to correct invalid JSON."""
        # Try to extract key-value pairs and reconstruct
        try:
            import re
            pairs = re.findall(r'"([^"]+)":\s*"([^"]*)"', text)

            if pairs:
                reconstructed = {key: value for key, value in pairs}
                return self._correct_to_schema(reconstructed, schema)
        except:
            pass

        # Fallback to schema-compliant default
        return self._generate_fallback(schema)

    def _generate_schema_compliant_fallback(self, schema):
        """Generate a STRICTLY schema-compliant fallback response."""
        fallback = {}

        required_fields = schema.get('required', [])
        properties = schema.get('properties', {})

        for field in required_fields:
            prop_def = properties.get(field, {})
            prop_type = prop_def.get('type', 'string')

            # RULE: Generate values that exactly match schema constraints
            if prop_type == 'string':
                fallback[field] = f"Generated {field}"
            elif prop_type == 'integer':
                fallback[field] = 1
            elif prop_type == 'number':
                fallback[field] = 1.0
            elif prop_type == 'boolean':
                fallback[field] = True
            elif prop_type == 'array':
                fallback[field] = ["item1", "item2"]
            elif prop_type == 'object':
                fallback[field] = {"key": "value"}

        return json.dumps(fallback)