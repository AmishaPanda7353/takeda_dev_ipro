from typing import Optional

import yaml
from pydantic import BaseModel, Field, validator


# Define Pydantic models for the configuration
class ModelParams(BaseModel):
    engine: Optional[str] = Field(None, description="Engine type for OpenAI models")
    temperature: float
    max_tokens: Optional[int]
    n: int
    stop: Optional[str]
    function: Optional[str] = Field(None, description="Function to use")
    timeout: int
    max_tries: int

    @validator("temperature")
    def temperature_range(cls, v):
        if not (0 <= v <= 2):
            raise ValueError("temperature must be between 0 and 2 inclusive")
        return v


class OpenAIModelConfig(BaseModel):
    model_params: ModelParams


class FollowupQuestionTagging(BaseModel):
    openai: OpenAIModelConfig


class TextToQuery(BaseModel):
    openai: OpenAIModelConfig


class QueryToChartType(BaseModel):
    openai: OpenAIModelConfig


class QuertToChartCode(BaseModel):
    openai: OpenAIModelConfig


class TableToInsightQuestion(BaseModel):
    openai: OpenAIModelConfig


class QuetionToInsights(BaseModel):
    openai: OpenAIModelConfig


class SummarizeInsights(BaseModel):
    openai: OpenAIModelConfig


class SummarizeTables(BaseModel):
    openai: OpenAIModelConfig


class InsightQuestionToCode(BaseModel):
    openai: OpenAIModelConfig


class ParamDict(BaseModel):
    text_to_query: TextToQuery
    followup_question_tagging: FollowupQuestionTagging
    query_to_chart_type: QueryToChartType
    query_to_chart_code: QuertToChartCode
    table_to_insight_questions: TableToInsightQuestion
    questions_to_insights: QuetionToInsights
    summarize_insights: SummarizeInsights
    summarize_tables: SummarizeTables
    insight_questions_to_code: InsightQuestionToCode


class Config(BaseModel):
    param_dict: ParamDict


# Function to load and validate configuration
def validate_model_prompts(config_file: str, model_file: str):
    # Load the model type from model.yaml
    with open(model_file, "r") as f:
        model_config = yaml.safe_load(f)
    llm_model_type = model_config.get("llm_model_type")
    # Load the configuration file
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    if llm_model_type not in [
        "openai",
        "aws_llama",
        "aws_mistral",
        "aws_claude_2",
        "aws_claude_3",
        "aws_claude_3_5",
    ]:
        raise ValueError(f"Unsupported model type: {llm_model_type}")

    # Validate the configuration for the specified model type
    config = Config(**config_data)

    if llm_model_type == "openai":
        config.param_dict.text_to_query.openai.model_params
        config.param_dict.followup_question_tagging.openai.model_params
        config.param_dict.query_to_chart_type.openai.model_params
        config.param_dict.query_to_chart_code.openai.model_params
        config.param_dict.table_to_insight_questions.openai.model_params
        config.param_dict.questions_to_insights.openai.model_params
        config.param_dict.summarize_insights.openai.model_params
        config.param_dict.summarize_tables.openai.model_params
        config.param_dict.insight_questions_to_code.openai.model_params
    # elif llm_model_type == "open_source_llama":
    #     config.param_dict.text_to_query.open_source_llama.model_params
    #     config.param_dict.followup_question_tagging.open_source_llama.model_params
    # config.param_dict.query_to_chart_type.open_source_llama.model_params

    return


# model_params = "configs/model/model_params.yaml"

# # with open("configs/model.yaml", "r") as file:
# model_file = "configs/model.yaml"

# config = validate_model_prompts(model_params, model_file)

# print("Configuration validated successfully!")
