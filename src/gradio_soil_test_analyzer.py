import logging
from typing import Optional

import gradio as gr
from llama_index.core import Document as LlamaIndexDocument
from llama_index.core import PropertyGraphIndex
from src.hybrid_retriever import HybridRetriever
from src.write_report import write_report
from src.ingest_service import IngestService
from src.soil_test_processors import M3Processor, SPProcessor
from src.soil_test_models import M3Report, SPReport

from src.build_prompts import build_prompts, PromptDict
from src.logging_config import setup_logging
from src.ollama_stuff import ask_question, QueryResult

setup_logging()
logger = logging.getLogger(__name__)



class SoilTestAnalyzer:
    def __init__(self) -> None:
        self.ingest_service: IngestService = IngestService()
        self.m3_report: Optional[M3Report] = None
        self.sp_report: Optional[SPReport] = None
        self.prompt_dicts: list[PromptDict] = []
        self.results: list[Optional[QueryResult]] = []
        self.current_property_index: int = 0

    def setup_retriever(self, nodes: list[LlamaIndexDocument]) -> HybridRetriever:
        vector_index = self._load_vector_index(nodes, create_vector_index=False)
        kg_index = self._load_kg_index(nodes, create_kg_index=False)
        bm25_retriever = self._load_bm_retriever(nodes, create_bm_retriever=False)
        retriever = HybridRetriever(
            vector_index=vector_index,
            kg_index=kg_index,
            bm25_retriever=bm25_retriever,
        )
        return retriever

    def _load_vector_index(self, nodes: list[LlamaIndexDocument], create_vector_index: bool = False):
        # Create a vector_index.
        if create_vector_index:
            vector_index = self.ingest_service.build_vector_index(nodes,embed_model_name='nomic-embed-text', collection_name='soil_test_knowledge', persist_dir="soil_test_knowledge")
        else:
            vector_index = self.ingest_service.get_vector_index('soil_test_knowledge')
        return vector_index

    def _load_kg_index(self, nodes: list[LlamaIndexDocument], create_kg_index: bool = False) -> PropertyGraphIndex:
        # Create a knowledge graph index.
        if create_kg_index:
            kg_index = self.ingest_service.build_knowledge_graph(nodes)
        else:
            kg_index = self.ingest_service.get_knowledge_graph()
        return kg_index

    def _load_bm_retriever(self, nodes: list[LlamaIndexDocument], create_bm_retriever: bool = False):
        if create_bm_retriever:
            bm_retriever = self.ingest_service.build_bm25_retriever(nodes)
        else:
            bm_retriever = self.ingest_service.get_bm25_retriever()
        return bm_retriever


    def load_reports(self, m3_pdf, sp_pdf):
        '''Logan Labs has sent the reports back as PDF files. There is a PDF file for the mehlic-3 and one for the saturated paste.
        '''
        logger.debug(f"m3_pdf: {m3_pdf}")
        logger.debug(f"sp_pdf: {sp_pdf}")
        m3 = M3Processor()
        self.m3_report = m3.process_pdf(m3_pdf.name)
        sp = SPProcessor()
        self.sp_report = sp.process_pdf(sp_pdf.name)
        # retriever = self.setup_retriever(nodes)
        # self.prompt_dicts = build_prompts(retriever=retriever, m3_results=self.m3_report, sp_results=self.sp_report)
        # # Create a list to store the results of the analysis. The list is as long as the number of properties to analyze and initialized to None.
        # self.results = [None] * len(self.prompt_dicts)
        # self.current_property_index = 0
        return "Reports loaded successfully. Ready to start analysis."

    def get_current_property(self) -> str:
        if self.current_property_index < len(self.prompt_dicts):
            return self.prompt_dicts[self.current_property_index].soil_test_property.name
        return "Analysis complete"

    def analyze_current_property(self):
        if self.current_property_index >= len(self.prompt_dicts):
            return "Analysis complete", "No more properties to analyze"

        prompt_dict = self.prompt_dicts[self.current_property_index]
        result = ask_question(prompt_dict.prompt, model_name='llama3.2')
        self.results[self.current_property_index] = result
        return prompt_dict.soil_test_property.name, result

    def provide_feedback(self, feedback, new_prompt):
        if self.current_property_index >= len(self.prompt_dicts):
            return "Analysis complete", "No more properties to analyze"

        if feedback == "accept":
            self.current_property_index += 1
            if self.current_property_index < len(self.prompt_dicts):
                return self.analyze_current_property()
            else:
                return "Analysis complete", "All properties analyzed"
        elif feedback == "modify":
            result = ask_question(new_prompt, model_name='llama3.2')
            self.results[self.current_property_index] = result
            return self.prompt_dicts[self.current_property_index]['soil_test_property']['name'], result
        else:
            feedback_prompt = f"Original analysis: {self.results[self.current_property_index]}\n\nHuman feedback: {feedback}\n\nPlease update the analysis based on the feedback."
            result = ask_question(feedback_prompt, model_name='llama3.2')
            self.results[self.current_property_index] = result
            return self.prompt_dicts[self.current_property_index]['soil_test_property']['name'], result

    def generate_report(self):
        write_report(self.results, self.prompt_dicts)
        return "Report generated successfully."

analyzer = SoilTestAnalyzer()

with gr.Blocks() as ui:
    gr.Markdown("# Soil Test Analyzer")

    with gr.Row():
        m3_file = gr.File(label="Upload M3 Report PDF")
        sp_file = gr.File(label="Upload SP Report PDF")
        load_button = gr.Button("Load Reports")

    status_output = gr.Textbox(label="Status")

    with gr.Row():
        current_property = gr.Textbox(label="Current Property")
        analyze_button = gr.Button("Analyze Current Property")

    analysis_output = gr.Textbox(label="Analysis Result")

    with gr.Row():
        feedback_input = gr.Textbox(label="Feedback")
        new_prompt_input = gr.Textbox(label="New Prompt (if modifying)")

    with gr.Row():
        accept_button = gr.Button("Accept and Continue")
        modify_button = gr.Button("Modify Prompt")
        update_button = gr.Button("Update with Feedback")

    generate_report_button = gr.Button("Generate Final Report")
    report_status = gr.Textbox(label="Report Status")

    # Event handlers
    load_button.click( # type: ignore
        fn=analyzer.load_reports,
        inputs=[m3_file, sp_file],
        outputs=status_output
    )

    analyze_button.click( # type: ignore
        fn=analyzer.analyze_current_property,
        outputs=[current_property, analysis_output]
    )

    accept_button.click( # type: ignore
        fn=lambda: analyzer.provide_feedback("accept", ""),
        outputs=[current_property, analysis_output]
    )

    modify_button.click( # type: ignore
        fn=lambda: analyzer.provide_feedback("modify", new_prompt_input.value),
        inputs=new_prompt_input,
        outputs=[current_property, analysis_output]
    )

    update_button.click( # type: ignore
        fn=lambda: analyzer.provide_feedback(feedback_input.value, ""),
        inputs=feedback_input,
        outputs=[current_property, analysis_output]
    )

    generate_report_button.click( # type: ignore
        fn=analyzer.generate_report,
        outputs=report_status
    )

if __name__ == "__main__":
    ui.launch(show_error=True)
