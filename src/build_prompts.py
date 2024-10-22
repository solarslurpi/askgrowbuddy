from dataclasses import dataclass
from src.soil_test_models import M3Report, SPReport
from src.guidance import instruction, properties
from src.hybrid_retriever import HybridRetriever

@dataclass
class SoilTestProperty:
    name: str
    pH: float
    nitrogen_content: float
    phosphorus_content: float
    potassium_content: float

@dataclass
class PromptDict:
    soil_test_property: SoilTestProperty
    prompt: str
    # Add other fields as needed


def build_prompts(retriever: HybridRetriever, m3_results:M3Report, sp_results: SPReport) -> list[str]:
    '''
    The soil test results have been loaded into M3Report and SPReport Pydantic objects. This means the fields have been validated. The `properties` list in `guidance.py` has each soil characteristic that will be evaluated.  For each characteristic, a prompt is created.
    '''
    prompt_list = []

    for soil_test_property in properties:
        property_name = soil_test_property["name"]
        value = _get_property_value(property_name, m3_results, sp_results)
        optimal_range = soil_test_property["optimal_range"]
        context_prompt = f"Cannabis cultivation guidance for {property_name} at value {value}; optimal range is {optimal_range}. Includes actionable advice on adjustments if current value is outside optimal conditions."
        nodes_with_score = retriever.retrieve(context_prompt)
        context = "\n".join([node.get_content() for node in nodes_with_score])
        prompt_str = instruction.format(
            context=context,
            property=property_name,
            value=value,
            optimal_range=optimal_range
        )
        prompt_list.append(prompt_str)

    return prompt_list

def _get_property_value(property_name: str, m3_report:M3Report, sp_report:SPReport) -> float:
    # Try getting the value from M3Report first, if not found, then from SPReport
    if hasattr(m3_report, property_name.lower()):
        return getattr(m3_report, property_name.lower())
    elif hasattr(sp_report, property_name.lower()):
        return getattr(sp_report, property_name.lower())
    else:
        raise ValueError(f"Property {property_name} not found in either report.")
