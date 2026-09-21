import os
import unittest
import pandas as pd
import numpy as np

from src.data_engine.loader import DataLoader
from src.data_engine.profiler import DataProfiler
from src.data_engine.cleaner import DataCleaner
from src.data_engine.validator import DataValidator
from src.data_engine.eda import EDAEngine
from src.data_engine.statistical_analyzer import StatisticalAnalyzer
from src.data_engine.sql_engine import SQLEngine
from src.data_engine.viz_engine import VisualizationEngine
from src.data_engine.kpi_engine import KPIEngine
from src.data_engine.ml_engine import MLEngine
from src.data_engine.db_connectors import DatabaseConnector
from src.data_engine.duckdb_accelerator import DuckDBAccelerator
from src.data_engine.forecasting_engine import ForecastingEngine
from src.agent_core.text_to_sql import TextToSQLAssistant
from src.agent_core.insight_generator import InsightGenerator
from src.agent_core.recommendation_engine import RecommendationEngine
from src.agent_core.report_builder import ReportBuilder
from src.agent_core.web_search import extract_search_results
from src.exports.pdf_exporter import PrintablePDFExporter

class TestAIDataAnalystEnterprisePipeline(unittest.TestCase):

    def setUp(self):
        self.sample_path = "sample_data/sample_ecommerce.csv"
        self.assertTrue(os.path.exists(self.sample_path), "Sample dataset missing.")
        self.df_original, self.metadata = DataLoader.load_file(self.sample_path)

    def test_01_ingestion_and_metadata(self):
        self.assertGreater(self.metadata["num_rows"], 0)
        self.assertGreater(self.metadata["num_cols"], 0)
        self.assertIn("customer_age", self.metadata["column_buckets"]["numerical_columns"])

    def test_02_duckdb_accelerator(self):
        acc = DuckDBAccelerator(self.df_original, table_name="test_acc")
        res_df, q_meta = acc.query("SELECT COUNT(*) AS cnt FROM test_acc;")
        self.assertTrue(q_meta["success"])
        self.assertEqual(res_df.iloc[0]["cnt"], len(self.df_original))
        acc.close()

    def test_03_text_to_sql(self):
        res_df, q_meta = TextToSQLAssistant.ask_data(
            "Show top 5 product categories by total spend",
            self.df_original,
            self.metadata,
            table_name="test_sql"
        )
        self.assertTrue(q_meta["success"])
        self.assertIn("SELECT", q_meta["generated_sql"])

    def test_04_forecasting_and_shap(self):
        cleaner = DataCleaner(self.df_original, self.metadata)
        df_cleaned, _ = cleaner.run_full_cleaning_pipeline()

        fc_res = ForecastingEngine.forecast_trend(
            df_cleaned,
            date_col="transaction_date",
            value_col="total_spend",
            periods_ahead=4
        )
        self.assertIn("forecast_data", fc_res)
        self.assertEqual(len(fc_res["forecast_data"]), 4)

        ml_res = MLEngine.train_baseline_model(df_cleaned, target_col="is_returned")
        shap_res = ForecastingEngine.explain_feature_attribution(ml_res)
        self.assertIsInstance(shap_res, list)

    def test_05_printable_pdf_exporter(self):
        cleaner = DataCleaner(self.df_original, self.metadata)
        df_cleaned, trans_log = cleaner.run_full_cleaning_pipeline()
        profiling = DataProfiler.profile_dataset(self.df_original, self.metadata)
        validation = DataValidator.validate_cleaning(self.df_original, df_cleaned, trans_log)
        report_md = ReportBuilder.build_markdown_report("test_sample", self.metadata, profiling, validation, trans_log, {}, [], {}, [], [], [])

        out_path = "output_report/test_pdf.html"
        os.makedirs("output_report", exist_ok=True)
        res_path = PrintablePDFExporter.generate_printable_report("test_sample", self.metadata, profiling, report_md, out_path)
        self.assertTrue(os.path.exists(res_path))

    def test_06_web_search_result_parsing(self):
        html = '''
        <html><body>
        <a rel="nofollow" class="result-link" href="https://example.com/retail-benchmark">Retail benchmark</a>
        <div class="result-snippet">Average ecommerce conversion rate is 3.2% for retail brands.</div>
        <a rel="nofollow" class="result-link" href="https://example.com/market">Market trend</a>
        <div class="result-snippet">Demand is strongest in digital channels and loyalty programs.</div>
        </body></html>
        '''
        results = extract_search_results(html, max_results=2)
        self.assertGreaterEqual(len(results), 2)
        self.assertIn("Retail benchmark", results[0]["title"])
        self.assertIn("conversion rate", results[0]["snippet"].lower())

if __name__ == "__main__":
    unittest.main()
